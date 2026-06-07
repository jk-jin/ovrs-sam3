from __future__ import annotations

from typing import Optional, Tuple

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
        → Conv2d(Q → D_score, 7×7)         → [B*C, D_score, 16, 16]
        → ConvTranspose2d ×2 (learnable 4×) → [B*C, D_score, 64, 64]
        → bilinear interpolate to target_hw  → [B*C, D_score, Th, Tw]
    """

    def __init__(
        self,
        clip_output_dim: int = 768,
        score_embed_dim: int = 32,
        num_query_tokens: int = 32,
        conv_kernel: int = 7,
        encoder_hw: int = 72,
    ):
        super().__init__()
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.encoder_hw = int(encoder_hw)

        padding = int(conv_kernel) // 2
        num_groups = min(8, self.score_embed_dim)
        if self.score_embed_dim % num_groups != 0:
            num_groups = 1

        # Q-channel 7×7 conv: Q score maps → D_score channels.
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

        # Learnable 4× upsampling: 16×16 → 32×32 → 64×64.
        # kernel_size=4, stride=2, padding=1 gives exact 2× size doubling.
        self.score_upsampler = nn.Sequential(
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

    def forward(
        self,
        dynamic_clip_text: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dynamic_clip_text:  [B, C, Q, D_clip]
            clip_image_feat_map: [B, D_clip, Hc, Wc]
            target_hw:          (Th, Tw) — final spatial size.
                                If None, falls back to (encoder_hw, encoder_hw).

        Returns:
            clip_score_embed: [B, C, D_score, Th, Tw]
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

        # Q-channel 7×7 conv: [B*C, Q, 16, 16] → [B*C, D_score, 16, 16]
        score_flat = clip_score_maps.reshape(B * C, Q, Hc, Wc)
        score_embed = self.score_conv(score_flat)

        # Learnable 4× upsampling: 16→32→64
        score_embed = self.score_upsampler(score_embed)
        # [B*C, D_score, 64, 64]

        # Final bilinear interpolation to align with encoder feature size.
        th, tw = (
            (int(target_hw[0]), int(target_hw[1]))
            if target_hw is not None
            else (self.encoder_hw, self.encoder_hw)
        )

        if score_embed.shape[-2:] != (th, tw):
            score_embed = F.interpolate(
                score_embed,
                size=(th, tw),
                mode="bilinear",
                align_corners=False,
            )

        score_embed = score_embed.reshape(
            B, C, self.score_embed_dim, th, tw
        )

        return score_embed.contiguous(), clip_score_maps.contiguous()
