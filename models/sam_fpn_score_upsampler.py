from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize(x: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    """Safe resize that handles nearest mode (no align_corners)."""
    if mode in ("bilinear", "bicubic", "trilinear"):
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


def build_upsample_targets(
    low_hw: Tuple[int, int],
    final_hw: Tuple[int, int],
    num_stages: int = 4,
) -> List[Tuple[int, int]]:
    """
    Build exactly `num_stages` upsample targets from low_hw to final_hw.

    For ViT-L-14 (grid 16×16 → 288×288):
        16 → 32 → 64 → 144 → 288  (4 stages)

    For ViT-L-14@336 (grid 24×24 → 288×288):
        24 → 48 → 96 → 192 → 288   (4 stages)
    """
    low_h, low_w = int(low_hw[0]), int(low_hw[1])
    final_h, final_w = int(final_hw[0]), int(final_hw[1])

    # Fast path for known configurations.
    if (low_h, low_w) == (16, 16) and (final_h, final_w) == (288, 288):
        return [(32, 32), (64, 64), (144, 144), (288, 288)]
    if (low_h, low_w) == (24, 24) and (final_h, final_w) == (288, 288):
        return [(48, 48), (96, 96), (192, 192), (288, 288)]

    # Fallback: first N-1 stages double, final stage snaps to final_hw.
    targets = []
    cur_h, cur_w = low_h, low_w
    for _ in range(num_stages - 1):
        next_h = min(cur_h * 2, final_h)
        next_w = min(cur_w * 2, final_w)
        if (next_h, next_w) == (cur_h, cur_w):
            break
        targets.append((next_h, next_w))
        cur_h, cur_w = next_h, next_w

    if len(targets) == 0 or targets[-1] != (final_h, final_w):
        targets.append((final_h, final_w))

    return targets


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 2D feature maps [N, C, H, W]."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, H, W] → [N, H, W, C] → norm → [N, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DoubleConv(nn.Module):
    """Conv3x3 → Norm → Act → Conv3x3 → Norm → Act."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str = "group_norm",
        act: str = "gelu",
    ):
        super().__init__()
        if norm == "group_norm":
            num_groups = min(8, out_ch)
            if out_ch % num_groups != 0:
                num_groups = 1
            norm_layer = lambda c: nn.GroupNorm(num_groups, c)
        elif norm == "layer_norm":
            norm_layer = lambda c: LayerNorm2d(c)
        elif norm == "batch_norm":
            norm_layer = lambda c: nn.BatchNorm2d(c)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        act_layer = nn.GELU if act == "gelu" else nn.ReLU

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            act_layer(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FpnScoreConcatUpBlock(nn.Module):
    """Single upsample block: resize x, concat SAM FPN + score guidance, DoubleConv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sam_fpn_ch: int = 32,
        fpn_in_ch: int = 256,
        score_ch: int = 4,
        score_in_ch: int = 2,
        upsample_mode: str = "bilinear",
        norm: str = "group_norm",
        act: str = "gelu",
    ):
        super().__init__()
        self.out_ch = out_ch
        self.upsample_mode = upsample_mode

        self.in_proj = None
        if in_ch != out_ch:
            self.in_proj = nn.Conv2d(in_ch, out_ch, 1)

        self.fpn_proj = nn.Sequential(
            nn.Conv2d(fpn_in_ch, sam_fpn_ch, 1),
            nn.GroupNorm(min(8, sam_fpn_ch), sam_fpn_ch) if sam_fpn_ch >= 8 else nn.Identity(),
            nn.GELU(),
        )

        self.score_proj = nn.Sequential(
            nn.Conv2d(score_in_ch, score_ch, 1),
            nn.GroupNorm(min(4, score_ch), score_ch) if score_ch >= 4 else nn.Identity(),
            nn.GELU(),
        )

        total_ch = out_ch + sam_fpn_ch + score_ch
        self.conv = DoubleConv(total_ch, out_ch, norm=norm, act=act)

    def forward(
        self,
        x: torch.Tensor,
        sam_fpn: torch.Tensor,
        score_input: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        B, C, _, H, W = x.shape
        x_flat = x.reshape(B * C, -1, H, W)

        if (H, W) != target_hw:
            x_flat = _resize(x_flat, target_hw, self.upsample_mode)

        if self.in_proj is not None:
            x_flat = self.in_proj(x_flat)

        sam_fpn_proj = self.fpn_proj(sam_fpn)
        if sam_fpn_proj.shape[-2:] != target_hw:
            sam_fpn_proj = _resize(sam_fpn_proj, target_hw, self.upsample_mode)
        sam_fpn_proj = sam_fpn_proj[:, None].expand(B, C, -1, -1, -1).reshape(B * C, -1, *target_hw)

        score_flat = score_input.reshape(B * C, -1, *score_input.shape[-2:])
        score_proj = self.score_proj(score_flat)
        if score_proj.shape[-2:] != target_hw:
            score_proj = _resize(score_proj, target_hw, self.upsample_mode)

        y = torch.cat([x_flat, sam_fpn_proj, score_proj], dim=1)
        y = self.conv(y)
        return y.reshape(B, C, self.out_ch, *target_hw)


class SamFpnScoreConcatUpsampler(nn.Module):
    """
    GSNet-style upsampler.

    Block count and channel configs are fixed at __init__ so that all
    parameters are registered before the optimizer is built.
    Target spatial sizes are computed dynamically on the first forward pass.
    """

    def __init__(
        self,
        in_ch: int = 256,
        decoder_channels: Optional[List[int]] = None,
        sam_guidance_channels: Optional[List[int]] = None,
        score_channels: Optional[List[int]] = None,
        score_input: str = "score_and_tanh_logit",
        upsample_mode: str = "bilinear",
        norm: str = "group_norm",
        act: str = "gelu",
        class_chunk_size: int = 4,
    ):
        super().__init__()
        self.score_input = str(score_input)
        self.class_chunk_size = int(class_chunk_size)
        self.upsample_mode = upsample_mode

        if decoder_channels is None:
            decoder_channels = [256, 128, 96, 64, 32]
        if sam_guidance_channels is None:
            sam_guidance_channels = [32, 24, 16, 8]
        if score_channels is None:
            score_channels = [8, 4, 4, 4]

        self.num_stages = len(decoder_channels) - 1  # first entry is start, rest = per-stage out
        start_ch = decoder_channels[0]
        stage_out_chs = decoder_channels[1:]
        expected_stages = len(stage_out_chs)

        if len(sam_guidance_channels) != expected_stages:
            raise ValueError(
                f"sam_guidance_channels len ({len(sam_guidance_channels)}) != num stages ({expected_stages})"
            )
        if len(score_channels) != expected_stages:
            raise ValueError(
                f"score_channels len ({len(score_channels)}) != num stages ({expected_stages})"
            )

        if self.score_input == "score_and_tanh_logit":
            score_in_ch = 2
        elif self.score_input == "score":
            score_in_ch = 1
        else:
            raise ValueError(
                f"Unknown score_input={self.score_input!r}. Supported: 'score_and_tanh_logit', 'score'."
            )

        prev_ch = start_ch
        self.blocks = nn.ModuleList()
        for out_ch, fpn_ch, score_ch in zip(stage_out_chs, sam_guidance_channels, score_channels):
            self.blocks.append(
                FpnScoreConcatUpBlock(
                    in_ch=prev_ch,
                    out_ch=out_ch,
                    sam_fpn_ch=fpn_ch,
                    fpn_in_ch=256,
                    score_ch=score_ch,
                    score_in_ch=score_in_ch,
                    upsample_mode=upsample_mode,
                    norm=norm,
                    act=act,
                )
            )
            prev_ch = out_ch

        self.out_ch = prev_ch

        # Resolved on first forward; re-computed if spatial sizes change.
        self._target_hws: Optional[List[Tuple[int, int]]] = None
        self._resolved_low_hw: Optional[Tuple[int, int]] = None
        self._resolved_final_hw: Optional[Tuple[int, int]] = None

    def _select_fpn_feature(
        self,
        sam_fpn_features: List[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if len(sam_fpn_features) == 0:
            raise ValueError("sam_fpn_features must be non-empty.")
        for feat in sam_fpn_features:
            if feat.shape[1] != 256:
                raise ValueError(
                    f"sam_fpn_features channel must be 256, got {feat.shape[1]}"
                )
        best_idx = 0
        best_dist = float("inf")
        for i, feat in enumerate(sam_fpn_features):
            h, w = int(feat.shape[-2]), int(feat.shape[-1])
            dist = abs(h - target_hw[0]) + abs(w - target_hw[1])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return sam_fpn_features[best_idx]

    def _build_score_input(
        self,
        semantic_logits: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        B, C, H, W = semantic_logits.shape
        if (H, W) != target_hw:
            logits = _resize(
                semantic_logits.reshape(B * C, 1, H, W),
                target_hw,
                "bilinear",
            ).reshape(B, C, 1, *target_hw)
        else:
            logits = semantic_logits.unsqueeze(2)

        if self.score_input == "score_and_tanh_logit":
            return torch.cat([torch.sigmoid(logits), torch.tanh(logits)], dim=2)
        return torch.sigmoid(logits)

    def forward(
        self,
        x_low_refined: torch.Tensor,
        semantic_logits: torch.Tensor,
        sam_fpn_features: List[torch.Tensor],
    ) -> torch.Tensor:
        low_hw = (int(x_low_refined.shape[-2]), int(x_low_refined.shape[-1]))
        final_hw = (int(semantic_logits.shape[-2]), int(semantic_logits.shape[-1]))

        if (
            self._target_hws is None
            or low_hw != self._resolved_low_hw
            or final_hw != self._resolved_final_hw
        ):
            self._target_hws = build_upsample_targets(low_hw, final_hw, num_stages=len(self.blocks))
            self._resolved_low_hw = low_hw
            self._resolved_final_hw = final_hw
            if len(self._target_hws) != len(self.blocks):
                raise ValueError(
                    f"build_upsample_targets produced {len(self._target_hws)} stages, "
                    f"but upsampler has {len(self.blocks)} blocks."
                )

        x = x_low_refined
        for block, target_hw in zip(self.blocks, self._target_hws):
            sam_fpn = self._select_fpn_feature(sam_fpn_features, target_hw)
            score_input = self._build_score_input(semantic_logits, target_hw)
            x = block(x, sam_fpn, score_input, target_hw)
        return x


class FinalMaskConvHead(nn.Module):
    """Shared 3x3 conv head. Produces per-class mask logits with optional class chunking."""

    def __init__(self, in_ch: int = 32, class_chunk_size: int = 4):
        super().__init__()
        self.in_ch = int(in_ch)
        self.class_chunk_size = int(class_chunk_size)
        self.conv = nn.Conv2d(self.in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        chunk_size = min(self.class_chunk_size, C)
        chunks = []
        for start in range(0, C, chunk_size):
            end = min(start + chunk_size, C)
            x_chunk = x[:, start:end]
            Bc, Cc = x_chunk.shape[0], x_chunk.shape[1]
            logits = self.conv(x_chunk.reshape(Bc * Cc, D, H, W))
            chunks.append(logits.reshape(Bc, Cc, H, W))
        return torch.cat(chunks, dim=1)