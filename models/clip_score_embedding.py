from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipScoreEmbeddingBuilder(nn.Module):
    """
    Build 3-scale CLIP score embeddings from CLIP text and image feature map.

    Input:
        clip_text_features: [B, C, Q, D_clip]  (learned query or fixed templates)
        clip_image_feat_map: [B, D_clip, Hc, Wc]  (Hc=Wc=16 for ViT-L/14)

    Flow (baseline):
        clip_image_feat_map → bilinear to 18×18
        → text-image dot product → score_maps_18: [B, C, Q, 18, 18]
        → Conv2d(Q → D_score, 7×7)          → score_embed_18: [B, C, D_score, 18, 18]
        → ConvTranspose2d ×1 (learnable 2×)  → score_embed_36: [B, C, D_score, 36, 36]
        → ConvTranspose2d ×1 (learnable 2×)  → score_embed_72: [B, C, D_score, 72, 72]

    When score_upsample_fuse_clip_mid=True:
        score_embed_18 → bilinear to 36 → concat proj(CLIP layer 15) → conv → score_embed_36
        score_embed_36 → bilinear to 72 → concat proj(CLIP layer 7)  → conv → score_embed_72
    """

    def __init__(
        self,
        clip_output_dim: int = 768,
        score_embed_dim: int = 128,
        num_query_tokens: int = 32,
        conv_kernel: int = 7,
        base_hw: int = 18,
        score_upsample_fuse_clip_mid: bool = False,
        score_mid_proj_dim: int = 64,
        clip_mid_native_dim: int = 1024,
        clip_mid_layer_for_36: int = 15,
        clip_mid_layer_for_72: int = 7,
    ):
        super().__init__()
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.base_hw = int(base_hw)

        self.score_upsample_fuse_clip_mid = bool(score_upsample_fuse_clip_mid)
        self.score_mid_proj_dim = int(score_mid_proj_dim)
        self.clip_mid_layer_for_36 = int(clip_mid_layer_for_36)
        self.clip_mid_layer_for_72 = int(clip_mid_layer_for_72)

        padding = int(conv_kernel) // 2
        num_groups = min(8, self.score_embed_dim)
        if self.score_embed_dim % num_groups != 0:
            num_groups = 1

        self.score_conv_18 = nn.Sequential(
            nn.Conv2d(
                self.num_query_tokens,
                self.score_embed_dim,
                kernel_size=int(conv_kernel),
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups, self.score_embed_dim),
            nn.GELU(),
        )

        self.score_up_18_to_36 = nn.Sequential(
            nn.ConvTranspose2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, self.score_embed_dim),
            nn.GELU(),
        )

        self.score_up_36_to_72 = nn.Sequential(
            nn.ConvTranspose2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, self.score_embed_dim),
            nn.GELU(),
        )

        # CLIP mid-feature fusion modules (Experiment 2)
        if self.score_upsample_fuse_clip_mid:
            mid_groups_15 = min(8, self.score_mid_proj_dim)
            if self.score_mid_proj_dim % mid_groups_15 != 0:
                mid_groups_15 = 1

            self.clip_mid15_proj = nn.Sequential(
                nn.Conv2d(int(clip_mid_native_dim), self.score_mid_proj_dim, kernel_size=1, bias=False),
                nn.GroupNorm(mid_groups_15, self.score_mid_proj_dim),
                nn.GELU(),
            )

            self.clip_mid7_proj = nn.Sequential(
                nn.Conv2d(int(clip_mid_native_dim), self.score_mid_proj_dim, kernel_size=1, bias=False),
                nn.GroupNorm(mid_groups_15, self.score_mid_proj_dim),
                nn.GELU(),
            )

            fuse_groups_18_36 = min(8, self.score_embed_dim)
            if self.score_embed_dim % fuse_groups_18_36 != 0:
                fuse_groups_18_36 = 1

            self.fuse_18_to_36 = nn.Sequential(
                nn.Conv2d(self.score_embed_dim + self.score_mid_proj_dim,
                          self.score_embed_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(fuse_groups_18_36, self.score_embed_dim),
                nn.GELU(),
            )

            self.fuse_36_to_72 = nn.Sequential(
                nn.Conv2d(self.score_embed_dim + self.score_mid_proj_dim,
                          self.score_embed_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(fuse_groups_18_36, self.score_embed_dim),
                nn.GELU(),
            )

    def _get_clip_mid_feature(
        self,
        clip_mid_features: list[torch.Tensor] | None,
        clip_mid_layer_indices: list[int] | None,
        target_layer: int,
    ) -> torch.Tensor:
        """Extract a specific CLIP mid feature by layer index."""
        if clip_mid_features is None or clip_mid_layer_indices is None:
            raise ValueError(
                f"clip_mid_features or clip_mid_layer_indices is None, "
                f"but score_upsample_fuse_clip_mid=True requires layer {target_layer}."
            )
        for feat, idx in zip(clip_mid_features, clip_mid_layer_indices):
            if int(idx) == target_layer:
                return feat
        raise ValueError(
            f"CLIP mid layer {target_layer} not found in clip_mid_layer_indices="
            f"{clip_mid_layer_indices}. Required for score_upsample_fuse_clip_mid=True."
        )

    def forward(
        self,
        clip_text_features: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        clip_mid_features: list[torch.Tensor] | None = None,
        clip_mid_layer_indices: list[int] | None = None,
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            clip_text_features:  [B, C, Q, D_clip]
            clip_image_feat_map: [B, D_clip, Hc, Wc]
            clip_mid_features:   list of [B, D_native, Hc, Wc] (optional)
            clip_mid_layer_indices: list[int] (optional)

        Returns:
            clip_score_embeds:
                {
                    "scale_18": [B, C, D_score, 18, 18],
                    "scale_36": [B, C, D_score, 36, 36],
                    "scale_72": [B, C, D_score, 72, 72],
                }
            clip_score_maps_18: [B, C, Q, 18, 18]
        """
        batch_size, num_classes, num_queries, clip_dim = clip_text_features.shape
        image_batch_size, image_clip_dim, _, _ = clip_image_feat_map.shape

        if image_batch_size != batch_size:
            raise ValueError(
                f"clip_image_feat_map batch mismatch: "
                f"expected {batch_size}, got {image_batch_size}."
            )

        if clip_dim != image_clip_dim:
            raise ValueError(
                f"CLIP dimension mismatch: text={clip_dim}, image={image_clip_dim}."
            )

        clip_image_feat_18 = F.interpolate(
            clip_image_feat_map,
            size=(self.base_hw, self.base_hw),
            mode="bilinear",
            align_corners=False,
        )

        text_norm = F.normalize(clip_text_features, dim=-1)
        image_norm = F.normalize(clip_image_feat_18, dim=1)

        score_maps_18 = torch.einsum(
            "bcqd,bdhw->bcqhw",
            text_norm,
            image_norm,
        ) * 20.0

        score_embed_18_flat = self.score_conv_18(
            score_maps_18.reshape(
                batch_size * num_classes,
                num_queries,
                self.base_hw,
                self.base_hw,
            )
        )

        # --- 18 → 36 ---
        if self.score_upsample_fuse_clip_mid:
            score_embed_18_bc = score_embed_18_flat.reshape(
                batch_size, num_classes, self.score_embed_dim, self.base_hw, self.base_hw,
            )
            score_embed_36_bc = F.interpolate(
                score_embed_18_bc.reshape(batch_size * num_classes, self.score_embed_dim,
                                          self.base_hw, self.base_hw),
                size=(self.base_hw * 2, self.base_hw * 2),
                mode="bilinear",
                align_corners=False,
            )

            mid15 = self._get_clip_mid_feature(
                clip_mid_features, clip_mid_layer_indices, self.clip_mid_layer_for_36,
            )
            mid15_proj = self.clip_mid15_proj(mid15)
            mid15_proj = F.interpolate(
                mid15_proj,
                size=(self.base_hw * 2, self.base_hw * 2),
                mode="bilinear",
                align_corners=False,
            )[:, None].expand(batch_size, num_classes, self.score_mid_proj_dim,
                              self.base_hw * 2, self.base_hw * 2)

            fused_36 = torch.cat(
                [score_embed_36_bc.reshape(batch_size, num_classes, self.score_embed_dim,
                                           self.base_hw * 2, self.base_hw * 2),
                 mid15_proj],
                dim=2,
            )
            score_embed_36_flat = self.fuse_18_to_36(
                fused_36.reshape(batch_size * num_classes,
                                 self.score_embed_dim + self.score_mid_proj_dim,
                                 self.base_hw * 2, self.base_hw * 2)
            )
        else:
            score_embed_36_flat = self.score_up_18_to_36(score_embed_18_flat)

        # --- 36 → 72 ---
        if self.score_upsample_fuse_clip_mid:
            score_embed_36_bc = score_embed_36_flat.reshape(
                batch_size, num_classes, self.score_embed_dim, self.base_hw * 2, self.base_hw * 2,
            )
            score_embed_72_bc = F.interpolate(
                score_embed_36_bc.reshape(batch_size * num_classes, self.score_embed_dim,
                                          self.base_hw * 2, self.base_hw * 2),
                size=(self.base_hw * 4, self.base_hw * 4),
                mode="bilinear",
                align_corners=False,
            )

            mid7 = self._get_clip_mid_feature(
                clip_mid_features, clip_mid_layer_indices, self.clip_mid_layer_for_72,
            )
            mid7_proj = self.clip_mid7_proj(mid7)
            mid7_proj = F.interpolate(
                mid7_proj,
                size=(self.base_hw * 4, self.base_hw * 4),
                mode="bilinear",
                align_corners=False,
            )[:, None].expand(batch_size, num_classes, self.score_mid_proj_dim,
                              self.base_hw * 4, self.base_hw * 4)

            fused_72 = torch.cat(
                [score_embed_72_bc.reshape(batch_size, num_classes, self.score_embed_dim,
                                           self.base_hw * 4, self.base_hw * 4),
                 mid7_proj],
                dim=2,
            )
            score_embed_72_flat = self.fuse_36_to_72(
                fused_72.reshape(batch_size * num_classes,
                                 self.score_embed_dim + self.score_mid_proj_dim,
                                 self.base_hw * 4, self.base_hw * 4)
            )
        else:
            score_embed_72_flat = self.score_up_36_to_72(score_embed_36_flat)

        clip_score_embed_18 = score_embed_18_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            self.base_hw,
            self.base_hw,
        ).contiguous()

        clip_score_embed_36 = score_embed_36_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            self.base_hw * 2,
            self.base_hw * 2,
        ).contiguous()

        clip_score_embed_72 = score_embed_72_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            self.base_hw * 4,
            self.base_hw * 4,
        ).contiguous()

        return {
            "scale_18": clip_score_embed_18,
            "scale_36": clip_score_embed_36,
            "scale_72": clip_score_embed_72,
        }, score_maps_18.contiguous()
