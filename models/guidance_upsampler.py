from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


def get_mid_feature_by_layer(
    clip_mid_features: List[torch.Tensor],
    clip_mid_layer_indices: tuple[int, ...],
    target_layer: int,
) -> torch.Tensor:
    """Retrieve a CLIP middle feature by its ViT layer index."""
    for feat, layer_idx in zip(clip_mid_features, clip_mid_layer_indices):
        if int(layer_idx) == int(target_layer):
            return feat
    available = list(clip_mid_layer_indices)
    raise ValueError(
        f"CLIP middle feature for layer {target_layer} not found. "
        f"Available layers: {available}."
    )


class ClipGuidanceUpsampler(nn.Module):
    """
    Upsamples refined score embeddings from 18→36→72, fusing
    CLIP middle features and SAM3 FPN features along the way.

    Input:
        refined_score_embed_18: [B, C, 256, 18, 18]
        sam_fpn_feat_36:        [B, C_sam, 36, 36]
        sam_fpn_feat_72:        [B, C_sam, 72, 72]
        clip_mid_features:      list[[B, D_native, Hc, Wc]]
        clip_mid_layer_indices: tuple[int, ...]

    Output:
        clip_guidance_36: [B, C, 256, 36, 36]
        clip_guidance_72: [B, C, 256, 72, 72]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        clip_native_dim: int = 1024,
        sam_fpn_dim: int = 256,
        mid15_target_layer: int = 15,
        mid7_target_layer: int = 7,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mid15_target_layer = int(mid15_target_layer)
        self.mid7_target_layer = int(mid7_target_layer)

        # Projections for CLIP mid features (D_native -> hidden_dim).
        self.clip_mid15_proj = nn.Sequential(
            nn.Conv2d(int(clip_native_dim), self.hidden_dim, kernel_size=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )
        self.clip_mid7_proj = nn.Sequential(
            nn.Conv2d(int(clip_native_dim), self.hidden_dim, kernel_size=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )

        # Projections for SAM3 FPN features.
        self.sam36_proj = nn.Sequential(
            nn.Conv2d(int(sam_fpn_dim), self.hidden_dim, kernel_size=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )
        self.sam72_proj = nn.Sequential(
            nn.Conv2d(int(sam_fpn_dim), self.hidden_dim, kernel_size=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )

        # Fusion conv blocks (18→36 and 36→72).
        self.fuse_18_to_36 = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, kernel_size=3, padding=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )

        self.fuse_36_to_72 = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, kernel_size=3, padding=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        refined_score_embed_18: torch.Tensor,
        sam_fpn_feat_36: torch.Tensor,
        sam_fpn_feat_72: torch.Tensor,
        clip_mid_features: List[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            clip_guidance_36: [B, C, 256, 36, 36]
            clip_guidance_72: [B, C, 256, 72, 72]
        """
        B, C, D, H18, W18 = refined_score_embed_18.shape

        if (H18, W18) != (18, 18):
            raise ValueError(
                f"refined_score_embed_18 must have spatial size 18x18, "
                f"got {(H18, W18)}."
            )

        # ---- 18 → 36 ----
        score_36_flat = F.interpolate(
            refined_score_embed_18.reshape(B * C, D, H18, W18),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, C, D, 36, 36)

        # CLIP mid15.
        clip_mid15_raw = get_mid_feature_by_layer(
            clip_mid_features, clip_mid_layer_indices, self.mid15_target_layer,
        )
        clip_mid15_36 = F.interpolate(
            clip_mid15_raw.to(dtype=score_36_flat.dtype),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )
        clip_mid15_36 = self.clip_mid15_proj(clip_mid15_36)  # [B, D, 36, 36]

        # SAM3 FPN 36.
        sam36_proj = self.sam36_proj(sam_fpn_feat_36.to(dtype=score_36_flat.dtype))  # [B, D, 36, 36]

        # Broadcast image features to C dim and fuse.
        clip_mid15_36_c = clip_mid15_36.unsqueeze(1).expand(B, C, D, 36, 36)
        sam36_proj_c = sam36_proj.unsqueeze(1).expand(B, C, D, 36, 36)

        fused_36_flat = torch.cat(
            [score_36_flat, clip_mid15_36_c, sam36_proj_c], dim=2,
        ).reshape(B * C, D * 3, 36, 36)

        clip_guidance_36 = self.fuse_18_to_36(fused_36_flat).reshape(B, C, D, 36, 36)
        clip_guidance_36 = clip_guidance_36 + score_36_flat  # residual

        # ---- 36 → 72 ----
        score_72_flat = F.interpolate(
            clip_guidance_36.reshape(B * C, D, 36, 36),
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, C, D, 72, 72)

        # CLIP mid7.
        clip_mid7_raw = get_mid_feature_by_layer(
            clip_mid_features, clip_mid_layer_indices, self.mid7_target_layer,
        )
        clip_mid7_72 = F.interpolate(
            clip_mid7_raw.to(dtype=score_72_flat.dtype),
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        )
        clip_mid7_72 = self.clip_mid7_proj(clip_mid7_72)  # [B, D, 72, 72]

        # SAM3 FPN 72.
        sam72_proj = self.sam72_proj(sam_fpn_feat_72.to(dtype=score_72_flat.dtype))  # [B, D, 72, 72]

        clip_mid7_72_c = clip_mid7_72.unsqueeze(1).expand(B, C, D, 72, 72)
        sam72_proj_c = sam72_proj.unsqueeze(1).expand(B, C, D, 72, 72)

        fused_72_flat = torch.cat(
            [score_72_flat, clip_mid7_72_c, sam72_proj_c], dim=2,
        ).reshape(B * C, D * 3, 72, 72)

        clip_guidance_72 = self.fuse_36_to_72(fused_72_flat).reshape(B, C, D, 72, 72)
        clip_guidance_72 = clip_guidance_72 + score_72_flat  # residual

        return clip_guidance_36, clip_guidance_72
