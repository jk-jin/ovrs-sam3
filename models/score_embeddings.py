from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


# ---------------------------------------------------------------------------
# ClipScoreEmbedding
# ---------------------------------------------------------------------------


class ClipScoreEmbedding(nn.Module):
    """
    Build CLIP-based score embedding at 36×36 from RemoteCLIP dense features.

    Input:
        remoteclip_feat_map: [B, D_clip, 36, 36]
        template_text:       [C, 32, D_clip]

    Process:
        template_text × remoteclip_feat_map → score_maps [B, C, 32, 36, 36]
        → two-stage conv → clip_score_embed [B, C, 256, 36, 36]
    """

    def __init__(
        self,
        clip_text_encoder,
        prompt_templates: list[str],
        normalize_label: bool = True,
        clip_output_dim: int = 768,
        score_embed_dim: int = 256,
        conv_kernel: int = 7,
    ):
        super().__init__()

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.prompt_templates = list(prompt_templates)
        self.normalize_label = bool(normalize_label)
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_prompt_templates = len(self.prompt_templates)

        if self.num_prompt_templates != 32:
            raise ValueError(
                f"Expected 32 prompt templates, got {self.num_prompt_templates}."
            )

        padding = int(conv_kernel) // 2

        self.score_conv = nn.Sequential(
            nn.Conv2d(
                self.num_prompt_templates,
                self.score_embed_dim,
                kernel_size=int(conv_kernel),
                stride=1,
                padding=padding,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
            nn.Conv2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self._text_feature_cache: dict[tuple, torch.Tensor] = {}

    def _has_trainable_clip_text_params(self) -> bool:
        return any(
            p.requires_grad for p in self.clip_text_encoder.parameters()
        )

    def _make_text_cache_key(
        self, class_names: list[str], device: torch.device
    ) -> tuple:
        return (tuple(class_names), str(device))

    def clear_text_cache(self) -> None:
        self._text_feature_cache.clear()

    def _encode_template_text(
        self, class_names: list[str], device: torch.device
    ) -> torch.Tensor:
        trainable = self._has_trainable_clip_text_params()

        if not trainable:
            cache_key = self._make_text_cache_key(class_names, device)
            if cache_key in self._text_feature_cache:
                return self._text_feature_cache[cache_key].to(device=device)

        result = self.clip_text_encoder.encode_prompt_templates(
            class_names=class_names,
            templates=self.prompt_templates,
            device=device,
            normalize_label=self.normalize_label,
            normalize=False,
        )

        if not trainable:
            self._text_feature_cache[cache_key] = result.detach().contiguous()

        return result

    def forward(
        self,
        class_names: list[str],
        remoteclip_feat_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            class_names:          list of C class names.
            remoteclip_feat_map:  [B, D_clip, 36, 36]

        Returns:
            clip_score_embed_36: [B, C, score_embed_dim, 36, 36]
            score_maps_36:       [B, C, 32, 36, 36]
            template_clip_text:  [C, 32, D_clip]
        """
        batch_size, image_clip_dim, H, W = remoteclip_feat_map.shape

        if image_clip_dim != self.clip_output_dim:
            raise ValueError(
                f"CLIP dimension mismatch: expected {self.clip_output_dim}, "
                f"got {image_clip_dim}."
            )
        if (H, W) != (36, 36):
            raise ValueError(
                f"Expected 36×36 feature map, got {(H, W)}."
            )

        num_classes = len(class_names)
        if num_classes == 0:
            raise ValueError("class_names is empty.")

        template_clip_text = self._encode_template_text(
            class_names=class_names,
            device=remoteclip_feat_map.device,
        )

        if tuple(template_clip_text.shape) != (
            num_classes,
            self.num_prompt_templates,
            self.clip_output_dim,
        ):
            raise ValueError(
                f"template_clip_text shape mismatch: expected "
                f"({num_classes}, {self.num_prompt_templates}, {self.clip_output_dim}), "
                f"got {tuple(template_clip_text.shape)}."
            )

        text_norm = F.normalize(
            template_clip_text.to(
                device=remoteclip_feat_map.device,
                dtype=remoteclip_feat_map.dtype,
            ),
            dim=-1,
        )
        image_norm = F.normalize(remoteclip_feat_map, dim=1)

        score_maps_36 = torch.einsum(
            "ckd,bdhw->bckhw",
            text_norm,
            image_norm,
        ) * 20.0

        score_flat = score_maps_36.reshape(
            batch_size * num_classes,
            self.num_prompt_templates,
            H,
            W,
        )

        clip_score_flat = self.score_conv(score_flat)

        clip_score_embed_36 = clip_score_flat.reshape(
            batch_size, num_classes, self.score_embed_dim, H, W
        ).contiguous()

        return (
            clip_score_embed_36,
            score_maps_36.contiguous(),
            template_clip_text.contiguous(),
        )


# ---------------------------------------------------------------------------
# ScoreFpnFusion
# ---------------------------------------------------------------------------


class ScoreFpnFusion(nn.Module):
    """
    Fuse CLIP-only score embedding with SAM FPN feature before refiner layers.

    Inputs:
        score_embed_36: [B, C, score_dim, 36, 36]
        sam_fpn_36:     [B, fpn_dim,   36, 36]

    Output:
        fused_score_embed_36: [B, C, fused_dim, 36, 36]
    """

    def __init__(
        self,
        score_dim: int = 256,
        fpn_dim: int = 256,
        fused_dim: int = 256,
    ):
        super().__init__()
        self.score_dim = int(score_dim)
        self.fpn_dim = int(fpn_dim)
        self.fused_dim = int(fused_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.score_dim + self.fpn_dim,
                self.fused_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.fused_dim),
            nn.GELU(),
            nn.Conv2d(
                self.fused_dim,
                self.fused_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.fused_dim),
            nn.GELU(),
        )

    def forward(
        self,
        score_embed_36: torch.Tensor,
        sam_fpn_36: torch.Tensor,
    ) -> torch.Tensor:
        B, C, score_dim, H, W = score_embed_36.shape

        if tuple(score_embed_36.shape) != (B, C, self.score_dim, H, W):
            raise ValueError(
                f"score_embed_36 must be [{B}, {C}, {self.score_dim}, {H}, {W}], "
                f"got {tuple(score_embed_36.shape)}."
            )

        if tuple(sam_fpn_36.shape) != (B, self.fpn_dim, H, W):
            raise ValueError(
                f"sam_fpn_36 must be [{B}, {self.fpn_dim}, {H}, {W}], "
                f"got {tuple(sam_fpn_36.shape)}."
            )

        fpn = (
            sam_fpn_36[:, None]
            .expand(B, C, self.fpn_dim, H, W)
            .contiguous()
        )

        x = torch.cat([score_embed_36, fpn], dim=2)
        x = x.reshape(B * C, self.score_dim + self.fpn_dim, H, W)

        out = self.fuse(x)
        return out.reshape(B, C, self.fused_dim, H, W).contiguous()
