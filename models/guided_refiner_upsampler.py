from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_refiner_blocks import (
    GuidedWindowAttention,
    flatten_batch_class,
    safe_group_norm,
    unflatten_batch_class,
)


class GuidedUpsampleBlock(nn.Module):
    """
    One guided upsample stage:

        1. bilinear interpolate ×2 + conv
        2. project sam_image_last + clip_mid feature to guidance_embed_dim
        3. regular GuidedWindowAttention (guidance = sam_proj + clip_proj)
        4. shifted GuidedWindowAttention
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        guidance_embed_dim: int = 128,
        clip_native_dim: int = 1024,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.guidance_embed_dim = int(guidance_embed_dim)

        # Upsample: bilinear + conv
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )

        # SAM image guidance projection
        self.sam_guidance_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.guidance_embed_dim, kernel_size=1, bias=False),
            safe_group_norm(self.guidance_embed_dim),
            nn.GELU(),
        )

        # CLIP mid feature guidance projection
        self.clip_mid_proj = nn.Sequential(
            nn.Conv2d(int(clip_native_dim), self.guidance_embed_dim, kernel_size=1, bias=False),
            safe_group_norm(self.guidance_embed_dim),
            nn.GELU(),
        )

        # Guidance is sam_proj + clip_proj concatenated along channel dim
        combined_guidance_dim = self.guidance_embed_dim * 2

        self.attn_regular = GuidedWindowAttention(
            hidden_dim=self.hidden_dim,
            guidance_dim=combined_guidance_dim,
            num_heads=int(num_heads),
            window_size=int(window_size),
            shift_size=0,
            dropout=float(dropout),
        )

        self.attn_shifted = GuidedWindowAttention(
            hidden_dim=self.hidden_dim,
            guidance_dim=combined_guidance_dim,
            num_heads=int(num_heads),
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=float(dropout),
        )

    def forward(
        self,
        x_low: torch.Tensor,
        sam_image_high: torch.Tensor,
        clip_mid_high: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            x_low:           [B, C, D, H_low, W_low]
            sam_image_high:  [B, D, H_high, W_high]
            clip_mid_high:   [B, D_native, H_high, W_high]
            target_hw:       (H_high, W_high)

        Returns:
            x_high: [B, C, D, H_high, W_high]
        """
        B, C, D, H_low, W_low = x_low.shape

        # 1. Bilinear upsample + conv
        x_flat, batch_size, num_classes = flatten_batch_class(x_low)
        x_flat = F.interpolate(
            x_flat,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        x_flat = self.upsample_conv(x_flat)
        x_high = unflatten_batch_class(x_flat, batch_size, num_classes)

        H_high, W_high = target_hw

        # 2. Project guidance features
        sam_high = F.interpolate(
            sam_image_high,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        sam_guidance = self.sam_guidance_proj(sam_high)

        clip_high = F.interpolate(
            clip_mid_high,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        clip_guidance = self.clip_mid_proj(clip_high)

        guidance = torch.cat([sam_guidance, clip_guidance], dim=1)

        # 3 & 4. Regular + shifted guided window attention
        x_high = self.attn_regular(
            encoder_features=x_high,
            guidance=guidance,
        )
        x_high = self.attn_shifted(
            encoder_features=x_high,
            guidance=guidance,
        )

        return x_high


class GuidedRefinerUpsampler(nn.Module):
    """
    Two-stage guided upsampler: 18→36→72.

    Stage 1 (18→36): sam_image_last_36 + CLIP layer 15 feature
    Stage 2 (36→72): sam_image_last_72 + CLIP layer 7 feature
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        clip_native_dim: int = 1024,
        guidance_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
        upsample_clip_layer_36: int = 15,
        upsample_clip_layer_72: int = 7,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.upsample_clip_layer_36 = int(upsample_clip_layer_36)
        self.upsample_clip_layer_72 = int(upsample_clip_layer_72)

        self.block_18_to_36 = GuidedUpsampleBlock(
            hidden_dim=self.hidden_dim,
            guidance_embed_dim=int(guidance_embed_dim),
            clip_native_dim=int(clip_native_dim),
            num_heads=int(num_heads),
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=float(dropout),
        )

        self.block_36_to_72 = GuidedUpsampleBlock(
            hidden_dim=self.hidden_dim,
            guidance_embed_dim=int(guidance_embed_dim),
            clip_native_dim=int(clip_native_dim),
            num_heads=int(num_heads),
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=float(dropout),
        )

    def forward(
        self,
        refined_features_18: torch.Tensor,
        sam_image_last_72: torch.Tensor,
        clip_mid_features: list[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Args:
            refined_features_18:    [B, C, D, 18, 18]
            sam_image_last_72:      [B, D, 72, 72]
            clip_mid_features:      list of [B, D_native, Hc, Wc]
            clip_mid_layer_indices: e.g. (7, 15)

        Returns:
            refined_features_72: [B, C, D, 72, 72]
        """
        B, C, D, H, W = refined_features_18.shape
        if (H, W) != (18, 18):
            raise ValueError(
                f"GuidedRefinerUpsampler expects 18×18 input, got {(H, W)}."
            )
        if sam_image_last_72.ndim != 4 or tuple(sam_image_last_72.shape[-2:]) != (72, 72):
            raise ValueError(
                f"sam_image_last_72 must end with (72, 72), got {tuple(sam_image_last_72.shape)}"
            )
        if len(clip_mid_features) != len(clip_mid_layer_indices):
            raise ValueError(
                f"clip_mid_features and clip_mid_layer_indices must have same length, "
                f"got {len(clip_mid_features)} vs {len(clip_mid_layer_indices)}"
            )

        mid_feature_by_layer = {
            int(layer_idx): feat
            for layer_idx, feat in zip(clip_mid_layer_indices, clip_mid_features)
        }

        # Stage 1: 18 → 36
        if self.upsample_clip_layer_36 not in mid_feature_by_layer:
            raise ValueError(
                f"Required CLIP mid layer {self.upsample_clip_layer_36} not found in "
                f"clip_mid_layer_indices={clip_mid_layer_indices}"
            )
        clip_mid_15 = mid_feature_by_layer[self.upsample_clip_layer_36]

        sam_image_last_36 = F.interpolate(
            sam_image_last_72,
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )

        x_36 = self.block_18_to_36(
            x_low=refined_features_18,
            sam_image_high=sam_image_last_36,
            clip_mid_high=clip_mid_15,
            target_hw=(36, 36),
        )

        # Stage 2: 36 → 72
        if self.upsample_clip_layer_72 not in mid_feature_by_layer:
            raise ValueError(
                f"Required CLIP mid layer {self.upsample_clip_layer_72} not found in "
                f"clip_mid_layer_indices={clip_mid_layer_indices}"
            )
        clip_mid_7 = mid_feature_by_layer[self.upsample_clip_layer_72]

        x_72 = self.block_36_to_72(
            x_low=x_36,
            sam_image_high=sam_image_last_72,
            clip_mid_high=clip_mid_7,
            target_hw=(72, 72),
        )

        return x_72
