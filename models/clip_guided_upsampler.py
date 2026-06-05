from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize(x: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    if mode in ("bilinear", "bicubic", "trilinear"):
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


def build_upsample_targets(
    low_hw: Tuple[int, int],
    final_hw: Tuple[int, int],
    num_stages: int = 4,
) -> List[Tuple[int, int]]:
    low_h, low_w = int(low_hw[0]), int(low_hw[1])
    final_h, final_w = int(final_hw[0]), int(final_hw[1])

    if (low_h, low_w) == (16, 16) and (final_h, final_w) == (288, 288):
        return [(32, 32), (64, 64), (144, 144), (288, 288)]

    if (low_h, low_w) == (24, 24) and (final_h, final_w) == (288, 288):
        return [(48, 48), (96, 96), (192, 192), (288, 288)]

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
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "group_norm":
        num_groups = min(8, channels)
        if channels % num_groups != 0:
            num_groups = 1
        return nn.GroupNorm(num_groups, channels)

    if norm == "layer_norm":
        return LayerNorm2d(channels)

    if norm == "batch_norm":
        return nn.BatchNorm2d(channels)

    raise ValueError(f"Unknown norm: {norm}")


def _make_act(act: str) -> nn.Module:
    if act == "gelu":
        return nn.GELU()
    if act == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown act: {act}")


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str = "group_norm",
        act: str = "gelu",
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(norm, out_ch),
            _make_act(act),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(norm, out_ch),
            _make_act(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ClipGuidedUpBlock(nn.Module):
    """
    One upsample block.

    Input:
        x:        [B, C, D_in, H, W]
        sam_fpn:  [B, 256, Hs, Ws]
        clip_mid: optional [B, D_clip_native, Hc, Wc]

    Output:
        y:        [B, C, D_out, target_h, target_w]
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sam_fpn_ch: int,
        clip_mid_in_ch: int,
        clip_mid_ch: int,
        use_clip_mid: bool,
        fpn_in_ch: int = 256,
        upsample_mode: str = "bilinear",
        norm: str = "group_norm",
        act: str = "gelu",
        class_chunk_size: int = 4,
    ):
        super().__init__()
        self.out_ch = int(out_ch)
        self.upsample_mode = str(upsample_mode)
        self.use_clip_mid = bool(use_clip_mid)
        self.class_chunk_size = int(class_chunk_size)

        if self.class_chunk_size <= 0:
            raise ValueError(
                f"class_chunk_size must be positive, got {class_chunk_size}."
            )

        self.in_proj = None
        if int(in_ch) != int(out_ch):
            self.in_proj = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1)

        self.fpn_proj = nn.Sequential(
            nn.Conv2d(int(fpn_in_ch), int(sam_fpn_ch), kernel_size=1, bias=False),
            _make_norm("group_norm", int(sam_fpn_ch)) if int(sam_fpn_ch) > 0 else nn.Identity(),
            nn.GELU(),
        )

        if self.use_clip_mid:
            self.clip_proj = nn.Sequential(
                nn.ConvTranspose2d(
                    int(clip_mid_in_ch),
                    int(clip_mid_ch),
                    kernel_size=2,
                    stride=2,
                    bias=False,
                ),
                _make_norm("group_norm", int(clip_mid_ch)),
                nn.GELU(),
            )
        else:
            self.clip_proj = None

        total_ch = int(out_ch) + int(sam_fpn_ch)
        if self.use_clip_mid:
            total_ch += int(clip_mid_ch)

        self.conv = DoubleConv(total_ch, int(out_ch), norm=norm, act=act)

    def forward(
        self,
        x: torch.Tensor,
        sam_fpn: torch.Tensor,
        clip_mid: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"x must be [B, C, D, H, W], got {tuple(x.shape)}."
            )

        B, C, _, H, W = x.shape
        target_hw = (int(target_hw[0]), int(target_hw[1]))

        if sam_fpn.dim() != 4:
            raise ValueError(
                f"sam_fpn must be [B, 256, Hs, Ws], got {tuple(sam_fpn.shape)}."
            )

        if int(sam_fpn.shape[0]) != B:
            raise ValueError(
                f"sam_fpn batch mismatch: expected {B}, got {sam_fpn.shape[0]}."
            )

        if self.use_clip_mid:
            if clip_mid is None:
                raise ValueError("clip_mid is required for this upsample stage.")
            if clip_mid.dim() != 4:
                raise ValueError(
                    f"clip_mid must be [B, D_clip, Hc, Wc], got {tuple(clip_mid.shape)}."
                )
            if int(clip_mid.shape[0]) != B:
                raise ValueError(
                    f"clip_mid batch mismatch: expected {B}, got {clip_mid.shape[0]}."
                )

        # Project class-independent guidance once.
        sam_fpn_proj = self.fpn_proj(sam_fpn)
        if tuple(sam_fpn_proj.shape[-2:]) != target_hw:
            sam_fpn_proj = _resize(
                sam_fpn_proj,
                target_hw,
                self.upsample_mode,
            )

        if self.use_clip_mid:
            clip_proj = self.clip_proj(clip_mid)
            if tuple(clip_proj.shape[-2:]) != target_hw:
                clip_proj = _resize(
                    clip_proj,
                    target_hw,
                    self.upsample_mode,
                )
        else:
            clip_proj = None

        chunk_outputs: list[torch.Tensor] = []

        for start in range(0, C, self.class_chunk_size):
            end = min(start + self.class_chunk_size, C)
            chunk_c = end - start

            x_chunk = x[:, start:end]
            x_flat = x_chunk.reshape(B * chunk_c, -1, H, W)

            if (H, W) != target_hw:
                x_flat = _resize(
                    x_flat,
                    target_hw,
                    self.upsample_mode,
                )

            if self.in_proj is not None:
                x_flat = self.in_proj(x_flat)

            sam_chunk = (
                sam_fpn_proj[:, None]
                .expand(B, chunk_c, -1, -1, -1)
                .reshape(B * chunk_c, -1, *target_hw)
            )

            parts = [x_flat, sam_chunk]

            if self.use_clip_mid:
                clip_chunk = (
                    clip_proj[:, None]
                    .expand(B, chunk_c, -1, -1, -1)
                    .reshape(B * chunk_c, -1, *target_hw)
                )
                parts.append(clip_chunk)

            y_chunk = torch.cat(parts, dim=1)
            y_chunk = self.conv(y_chunk)
            y_chunk = y_chunk.reshape(
                B,
                chunk_c,
                self.out_ch,
                *target_hw,
            )

            chunk_outputs.append(y_chunk)

        return torch.cat(chunk_outputs, dim=1).contiguous()


class StagewisePresenceRefiner(nn.Module):
    """
    Update per-class text guidance after each upsample stage.

    For each stage:
        1. each class guidance cross-attends to its own class feature map
        2. all class guidance vectors self-attend across classes

    Input:
        class_text_guidance: [B, C, D_t]
        x_stage:             [B, C, D_x, H, W]

    Output:
        updated_guidance: [B, C, D_t]
    """

    def __init__(
        self,
        text_dim: int,
        stage_channels: Sequence[int],
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.text_dim = int(text_dim)
        self.num_heads = int(num_heads)

        if self.text_dim % self.num_heads != 0:
            raise ValueError(
                f"text_dim={text_dim} must be divisible by num_heads={num_heads}."
            )

        self.stage_feature_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(ch), self.text_dim, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, self.text_dim), self.text_dim),
                nn.GELU(),
            )
            for ch in stage_channels
        ])

        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.text_dim,
                num_heads=self.num_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            for _ in stage_channels
        ])

        self.cross_norm = nn.ModuleList([
            nn.LayerNorm(self.text_dim)
            for _ in stage_channels
        ])

        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.text_dim,
                num_heads=self.num_heads,
                dropout=float(dropout),
                batch_first=True,
            )
            for _ in stage_channels
        ])

        self.self_norm = nn.ModuleList([
            nn.LayerNorm(self.text_dim)
            for _ in stage_channels
        ])

        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.text_dim),
                nn.Linear(self.text_dim, self.text_dim * 4),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.text_dim * 4, self.text_dim),
                nn.Dropout(float(dropout)),
            )
            for _ in stage_channels
        ])

    def forward_stage(
        self,
        class_text_guidance: torch.Tensor,
        x_stage: torch.Tensor,
        stage_idx: int,
        presence_pool_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if class_text_guidance.dim() != 3:
            raise ValueError(
                f"class_text_guidance must be [B, C, D], "
                f"got {tuple(class_text_guidance.shape)}."
            )

        if x_stage.dim() != 5:
            raise ValueError(
                f"x_stage must be [B, C, D_x, H, W], got {tuple(x_stage.shape)}."
            )

        B, C, D_t = class_text_guidance.shape
        Bx, Cx, D_x, H, W = x_stage.shape

        if (Bx, Cx) != (B, C):
            raise ValueError(
                f"x_stage batch/class mismatch: expected {(B, C)}, got {(Bx, Cx)}."
            )

        if D_t != self.text_dim:
            raise ValueError(
                f"class_text_guidance dim mismatch: expected {self.text_dim}, got {D_t}."
            )

        if stage_idx < 0 or stage_idx >= len(self.stage_feature_projs):
            raise IndexError(
                f"stage_idx={stage_idx} out of range for "
                f"{len(self.stage_feature_projs)} presence stages."
            )

        pool_h = int(presence_pool_hw[0])
        pool_w = int(presence_pool_hw[1])

        if pool_h <= 0 or pool_w <= 0:
            raise ValueError(
                f"presence_pool_hw must be positive, got {presence_pool_hw}."
            )

        x_flat = x_stage.reshape(B * C, D_x, H, W)

        # Critical memory fix:
        # pool BEFORE projection and attention.
        if (H, W) != (pool_h, pool_w):
            x_flat = F.adaptive_avg_pool2d(
                x_flat,
                output_size=(pool_h, pool_w),
            )

        kv = self.stage_feature_projs[stage_idx](x_flat)
        kv = kv.flatten(2).transpose(1, 2).contiguous()
        # [B*C, pool_h*pool_w, D_t]

        q = class_text_guidance.reshape(B * C, 1, D_t)

        cross_out, _ = self.cross_attn[stage_idx](
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        guidance_after_cross = self.cross_norm[stage_idx](
            q + cross_out
        ).reshape(B, C, D_t)

        self_out, _ = self.self_attn[stage_idx](
            query=guidance_after_cross,
            key=guidance_after_cross,
            value=guidance_after_cross,
            need_weights=False,
        )

        guidance_after_self = self.self_norm[stage_idx](
            guidance_after_cross + self_out
        )

        updated = guidance_after_self + self.ffn[stage_idx](guidance_after_self)
        return updated.contiguous()


class SamFpnClipGuidedUpsampler(nn.Module):
    """
    Four-stage upsampler.

    stage 1:
        x + SAM FPN + CLIP mid feature 0

    stage 2:
        x + SAM FPN + CLIP mid feature 1

    stage 3/4:
        x + SAM FPN only

    No semantic score guidance is used here.
    """

    def __init__(
        self,
        in_ch: int = 256,
        decoder_channels: Optional[Sequence[int]] = None,
        sam_guidance_channels: Optional[Sequence[int]] = None,
        clip_mid_in_ch: int = 1024,
        clip_guidance_channels: Optional[Sequence[int]] = None,
        clip_guidance_stage_indices: Optional[Sequence[int]] = None,
        upsample_mode: str = "bilinear",
        norm: str = "group_norm",
        act: str = "gelu",
        class_chunk_size: int = 4,
        presence_text_dim: int = 256,
        presence_num_heads: int = 8,
        presence_dropout: float = 0.1,
    ):
        super().__init__()

        self.class_chunk_size = int(class_chunk_size)
        if self.class_chunk_size <= 0:
            raise ValueError(
                f"class_chunk_size must be positive, got {class_chunk_size}."
            )

        if decoder_channels is None:
            decoder_channels = [256, 128, 96, 64, 32]

        if sam_guidance_channels is None:
            sam_guidance_channels = [32, 24, 16, 8]

        if clip_guidance_channels is None:
            clip_guidance_channels = [32, 24]

        if clip_guidance_stage_indices is None:
            clip_guidance_stage_indices = [0, 1]

        decoder_channels = list(decoder_channels)
        sam_guidance_channels = list(sam_guidance_channels)
        clip_guidance_channels = list(clip_guidance_channels)
        clip_guidance_stage_indices = [int(x) for x in clip_guidance_stage_indices]

        self.num_stages = len(decoder_channels) - 1
        stage_out_chs = decoder_channels[1:]

        if len(sam_guidance_channels) != self.num_stages:
            raise ValueError(
                f"sam_guidance_channels length {len(sam_guidance_channels)} "
                f"must equal num_stages {self.num_stages}."
            )

        if len(clip_guidance_channels) != len(clip_guidance_stage_indices):
            raise ValueError(
                "clip_guidance_channels length must match "
                "clip_guidance_stage_indices length."
            )

        clip_ch_by_stage = {
            int(stage_idx): int(ch)
            for stage_idx, ch in zip(clip_guidance_stage_indices, clip_guidance_channels)
        }

        prev_ch = int(decoder_channels[0])
        self.blocks = nn.ModuleList()

        for stage_idx, (out_ch, sam_ch) in enumerate(
            zip(stage_out_chs, sam_guidance_channels)
        ):
            use_clip_mid = stage_idx in clip_ch_by_stage
            clip_mid_ch = clip_ch_by_stage.get(stage_idx, 0)

            self.blocks.append(
                ClipGuidedUpBlock(
                    in_ch=prev_ch,
                    out_ch=int(out_ch),
                    sam_fpn_ch=int(sam_ch),
                    clip_mid_in_ch=int(clip_mid_in_ch),
                    clip_mid_ch=int(clip_mid_ch),
                    use_clip_mid=use_clip_mid,
                    fpn_in_ch=256,
                    upsample_mode=upsample_mode,
                    norm=norm,
                    act=act,
                    class_chunk_size=self.class_chunk_size,
                )
            )

            prev_ch = int(out_ch)

        self.out_ch = int(prev_ch)
        self.clip_guidance_stage_indices = tuple(clip_guidance_stage_indices)

        self.presence_refiner = StagewisePresenceRefiner(
            text_dim=int(presence_text_dim),
            stage_channels=stage_out_chs,
            num_heads=int(presence_num_heads),
            dropout=float(presence_dropout),
        )

        self._target_hws: Optional[List[Tuple[int, int]]] = None
        self._resolved_low_hw: Optional[Tuple[int, int]] = None
        self._resolved_final_hw: Optional[Tuple[int, int]] = None

    @staticmethod
    def _select_fpn_feature(
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
            dist = abs(h - int(target_hw[0])) + abs(w - int(target_hw[1]))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return sam_fpn_features[best_idx]

    def _get_clip_mid_for_stage(
        self,
        clip_mid_features: Optional[List[torch.Tensor]],
        stage_idx: int,
    ) -> Optional[torch.Tensor]:
        if stage_idx not in self.clip_guidance_stage_indices:
            return None

        if clip_mid_features is None or len(clip_mid_features) == 0:
            raise ValueError(
                f"clip_mid_features are required for upsample stage {stage_idx}."
            )

        local_idx = list(self.clip_guidance_stage_indices).index(stage_idx)

        if local_idx >= len(clip_mid_features):
            raise ValueError(
                f"Need clip_mid_features[{local_idx}] for stage {stage_idx}, "
                f"but only got {len(clip_mid_features)} features."
            )

        return clip_mid_features[local_idx]

    def forward(
        self,
        x_low_refined: torch.Tensor,
        class_text_guidance: torch.Tensor,
        final_hw: Tuple[int, int],
        sam_fpn_features: List[torch.Tensor],
        clip_mid_features: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_hw = (
            int(x_low_refined.shape[-2]),
            int(x_low_refined.shape[-1]),
        )
        final_hw = (
            int(final_hw[0]),
            int(final_hw[1]),
        )

        if (
            self._target_hws is None
            or low_hw != self._resolved_low_hw
            or final_hw != self._resolved_final_hw
        ):
            self._target_hws = build_upsample_targets(
                low_hw,
                final_hw,
                num_stages=len(self.blocks),
            )
            self._resolved_low_hw = low_hw
            self._resolved_final_hw = final_hw

            if len(self._target_hws) != len(self.blocks):
                raise ValueError(
                    f"build_upsample_targets produced {len(self._target_hws)} stages, "
                    f"but upsampler has {len(self.blocks)} blocks."
                )

        x = x_low_refined
        stage_text_guidance_history: list[torch.Tensor] = []

        for stage_idx, (block, target_hw) in enumerate(
            zip(self.blocks, self._target_hws)
        ):
            sam_fpn = self._select_fpn_feature(sam_fpn_features, target_hw)
            clip_mid = self._get_clip_mid_for_stage(
                clip_mid_features=clip_mid_features,
                stage_idx=stage_idx,
            )

            x = block(
                x=x,
                sam_fpn=sam_fpn,
                clip_mid=clip_mid,
                target_hw=target_hw,
            )

            class_text_guidance = self.presence_refiner.forward_stage(
                class_text_guidance=class_text_guidance,
                x_stage=x,
                stage_idx=stage_idx,
                presence_pool_hw=low_hw,
            )

            stage_text_guidance_history.append(class_text_guidance)

        stage_text_guidance_history = torch.stack(
            stage_text_guidance_history,
            dim=2,
        ).contiguous()

        return x.contiguous(), stage_text_guidance_history