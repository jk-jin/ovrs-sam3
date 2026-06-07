from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipScoreEmbeddingBuilder(nn.Module):
    """
    Build CLIP score embedding from dynamic CLIP text and image feature map.

    Input:
        dynamic_clip_text: [B, C, Q, D_clip]
        clip_image_feat_map: [B, D_clip, Hc, Wc]  (Hc=Wc=16 for ViT-L/14)

    Flow:
        score_maps [B, C, Q, Hc, Wc]
        → Conv2d(Q → D_score) → [B, C, D_score, Hc, Wc]
        → upsample to mid_hw → bilinear to encoder_hw
        → clip_score_embed [B, C, D_score, encoder_hw, encoder_hw]
    """

    def __init__(
        self,
        clip_output_dim: int = 768,
        score_embed_dim: int = 32,
        num_query_tokens: int = 32,
        conv_kernel: int = 7,
        mid_hw: int = 32,
        encoder_hw: int = 36,
    ):
        super().__init__()
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.mid_hw = int(mid_hw)
        self.encoder_hw = int(encoder_hw)

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
        Returns:
            clip_score_embed: [B, C, D_score, encoder_hw, encoder_hw]
            clip_score_maps:  [B, C, Q, Hc, Wc]
        """
        B, C, Q, D_clip = dynamic_clip_text.shape
        _, D_img, Hc, Wc = clip_image_feat_map.shape

        if D_clip != D_img:
            raise ValueError(
                f"CLIP dimension mismatch: text={D_clip}, image={D_img}"
            )

        if Q != self.num_query_tokens:
            raise ValueError(
                f"Query count mismatch: expected {self.num_query_tokens}, got {Q}"
            )

        text_norm = F.normalize(dynamic_clip_text, dim=-1)
        img_norm = F.normalize(clip_image_feat_map, dim=1)

        text_flat = text_norm.reshape(B * C * Q, D_clip)

        img_expanded = (
            img_norm[:, None, None]
            .expand(B, C, Q, D_img, Hc, Wc)
            .reshape(B * C * Q, D_img, Hc * Wc)
        )
        score_maps = torch.bmm(
            text_flat.unsqueeze(1), img_expanded
        ).reshape(B * C * Q, Hc, Wc) * 20.0

        clip_score_maps = score_maps.reshape(B, C, Q, Hc, Wc)

        # Conv with Q as input channels: [B*C, Q, Hc, Wc] → [B*C, D_score, Hc, Wc]
        score_flat = clip_score_maps.reshape(B * C, Q, Hc, Wc)
        score_embed = self.score_conv(score_flat)
        score_embed = score_embed.reshape(B, C, self.score_embed_dim, Hc, Wc)

        # Upsample: Hc×Wc → mid_hw×mid_hw → encoder_hw×encoder_hw
        if (Hc, Wc) != (self.mid_hw, self.mid_hw):
            score_embed = score_embed.reshape(B * C, self.score_embed_dim, Hc, Wc)
            score_embed = F.interpolate(
                score_embed,
                size=(self.mid_hw, self.mid_hw),
                mode="bilinear",
                align_corners=False,
            )
            score_embed = score_embed.reshape(
                B, C, self.score_embed_dim, self.mid_hw, self.mid_hw
            )

        if (self.mid_hw, self.mid_hw) != (self.encoder_hw, self.encoder_hw):
            score_embed = score_embed.reshape(
                B * C, self.score_embed_dim, self.mid_hw, self.mid_hw
            )
            score_embed = F.interpolate(
                score_embed,
                size=(self.encoder_hw, self.encoder_hw),
                mode="bilinear",
                align_corners=False,
            )
            score_embed = score_embed.reshape(
                B, C, self.score_embed_dim, self.encoder_hw, self.encoder_hw
            )

        return score_embed.contiguous(), clip_score_maps.contiguous()
