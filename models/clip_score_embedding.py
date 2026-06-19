from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipScoreEmbeddingBuilder18(nn.Module):
    """
    Build 18×18 CLIP score embedding from dynamic CLIP text and image feature map.

    Input:
        dynamic_clip_text:  [B, C, Q, D_clip]
        clip_image_feat_map: [B, D_clip, Hc, Wc]  (Hc=Wc=16 for ViT-L/14)

    Flow:
        clip_image_feat_map → bilinear to 18×18
        → text-image dot product → score_maps_18: [B, C, Q, 18, 18]
        → Conv2d(Q → D_score, 7×7) → score_embed_18: [B, C, D_score, 18, 18]
    """

    def __init__(
        self,
        clip_output_dim: int = 768,
        score_embed_dim: int = 128,
        num_query_tokens: int = 32,
        conv_kernel: int = 7,
        base_hw: int = 18,
    ):
        super().__init__()
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.base_hw = int(base_hw)

        padding = int(conv_kernel) // 2
        num_groups = min(8, self.score_embed_dim)
        if self.score_embed_dim % num_groups != 0:
            num_groups = 1

        self.score_conv = nn.Sequential(
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

    def forward(
        self,
        dynamic_clip_text: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dynamic_clip_text:  [B, C, Q, D_clip]
            clip_image_feat_map: [B, D_clip, Hc, Wc]

        Returns:
            clip_score_embed_18: [B, C, D_score, 18, 18]
            clip_score_maps_18:  [B, C, Q, 18, 18]
        """
        batch_size, num_classes, num_queries, clip_dim = dynamic_clip_text.shape
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

        if num_queries != self.num_query_tokens:
            raise ValueError(
                f"Query count mismatch: expected {self.num_query_tokens}, got {num_queries}."
            )

        clip_image_feat_18 = F.interpolate(
            clip_image_feat_map,
            size=(self.base_hw, self.base_hw),
            mode="bilinear",
            align_corners=False,
        )

        text_norm = F.normalize(dynamic_clip_text, dim=-1)
        image_norm = F.normalize(clip_image_feat_18, dim=1)

        text_flat = text_norm.reshape(
            batch_size * num_classes * num_queries,
            clip_dim,
        )

        image_expanded = (
            image_norm[:, None, None]
            .expand(
                batch_size,
                num_classes,
                num_queries,
                image_clip_dim,
                self.base_hw,
                self.base_hw,
            )
            .reshape(
                batch_size * num_classes * num_queries,
                image_clip_dim,
                self.base_hw * self.base_hw,
            )
        )

        score_maps_18 = torch.bmm(
            text_flat.unsqueeze(1),
            image_expanded,
        ).reshape(
            batch_size,
            num_classes,
            num_queries,
            self.base_hw,
            self.base_hw,
        ) * 20.0

        score_embed_flat = self.score_conv(
            score_maps_18.reshape(
                batch_size * num_classes,
                num_queries,
                self.base_hw,
                self.base_hw,
            )
        )

        clip_score_embed_18 = score_embed_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            self.base_hw,
            self.base_hw,
        ).contiguous()

        return clip_score_embed_18, score_maps_18.contiguous()
