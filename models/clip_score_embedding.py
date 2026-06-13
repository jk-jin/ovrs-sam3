from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


def _conv_norm_gelu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int | None = None,
) -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        _make_group_norm(out_channels),
        nn.GELU(),
    )


def _deconv_up_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        _make_group_norm(channels),
        nn.GELU(),
        _conv_norm_gelu(channels, channels, kernel_size=3),
        _conv_norm_gelu(channels, channels, kernel_size=3),
    )


class ClipScoreEmbeddingBuilder(nn.Module):
    """
    Build 3-scale CLIP score embeddings from dynamic CLIP text and image feature map.

    Input:
        dynamic_clip_text: [B, C, Q, D_clip]
        clip_image_feat_map: [B, D_clip, Hc, Wc]  (Hc=Wc=16 for ViT-L/14)

    Flow:
        clip_image_feat_map → bilinear to 18×18
        → text-image dot product → score_maps_18: [B, C, Q, 18, 18]
        → enhanced 3-stage conv: Q → 64 → 128 → 256
        → enhanced 18→36 deconv up block
        → enhanced 36→72 deconv up block
    """

    def __init__(
        self,
        clip_output_dim: int = 768,
        score_embed_dim: int = 256,
        num_query_tokens: int = 32,
        conv_kernel: int = 7,
        base_hw: int = 18,
    ):
        super().__init__()
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.base_hw = int(base_hw)

        if self.score_embed_dim != 256:
            raise ValueError(
                "Current design requires score_embed_dim=256."
            )

        score_hidden_dim_1 = self.score_embed_dim // 4
        score_hidden_dim_2 = self.score_embed_dim // 2

        self.score_conv_18 = nn.Sequential(
            _conv_norm_gelu(
                self.num_query_tokens,
                score_hidden_dim_1,
                kernel_size=int(conv_kernel),
            ),
            _conv_norm_gelu(
                score_hidden_dim_1,
                score_hidden_dim_2,
                kernel_size=3,
            ),
            _conv_norm_gelu(
                score_hidden_dim_2,
                self.score_embed_dim,
                kernel_size=3,
            ),
        )

        self.score_up_18_to_36 = _deconv_up_block(self.score_embed_dim)
        self.score_up_36_to_72 = _deconv_up_block(self.score_embed_dim)

    def forward(
        self,
        dynamic_clip_text: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            dynamic_clip_text:  [B, C, Q, D_clip]
            clip_image_feat_map: [B, D_clip, Hc, Wc]

        Returns:
            clip_score_embeds:
                {
                    "scale_18": [B, C, D_score, 18, 18],
                    "scale_36": [B, C, D_score, 36, 36],
                    "scale_72": [B, C, D_score, 72, 72],
                }
            clip_score_maps_18: [B, C, Q, 18, 18]
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

        score_embed_18_flat = self.score_conv_18(
            score_maps_18.reshape(
                batch_size * num_classes,
                num_queries,
                self.base_hw,
                self.base_hw,
            )
        )

        score_embed_36_flat = self.score_up_18_to_36(score_embed_18_flat)
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
