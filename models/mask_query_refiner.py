from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    while num_groups > 1 and int(num_channels) % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, int(num_channels))


class SemanticMaskQueryRefiner(nn.Module):
    """
    Refine per-class semantic masks with learnable mask queries.

    Input:
        pixel_embed: [B, C, D, H, W]
        init_logits: [B, C, H, W]

    Output:
        final_logits: [B, C, H, W]

    Main idea:
        1. Use sigmoid(init_logits) as a mask-guided attention bias.
        2. Let Q learnable queries attend to downsampled pixel features.
        3. Match updated queries back to full-resolution pixel features.
        4. Pool Q query scores with logsumexp.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_downsample: int = 4,
        mask_gate_floor: float = 0.05,
        mask_bias_scale: float = 2.0,
        query_pool_temperature: float = 1.0,
        logit_scale_init: float = 5.0,
        logit_scale_max: float = 50.0,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.attn_downsample = int(attn_downsample)
        self.mask_gate_floor = float(mask_gate_floor)
        self.mask_bias_scale = float(mask_bias_scale)
        self.query_pool_temperature = float(query_pool_temperature)
        self.logit_scale_max = float(logit_scale_max)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by "
                f"num_heads={self.num_heads}."
            )
        if self.attn_downsample <= 0:
            raise ValueError("attn_downsample must be positive.")
        if not 0.0 < self.mask_gate_floor <= 1.0:
            raise ValueError("mask_gate_floor must be in (0, 1].")
        if self.mask_bias_scale < 0.0:
            raise ValueError("mask_bias_scale must be non-negative.")
        if self.query_pool_temperature <= 0.0:
            raise ValueError("query_pool_temperature must be positive.")
        if logit_scale_init <= 0.0:
            raise ValueError("logit_scale_init must be positive.")
        if self.logit_scale_max <= 0.0:
            raise ValueError("logit_scale_max must be positive.")

        self.query_embed = nn.Parameter(
            torch.randn(1, 1, self.num_queries, self.hidden_dim) * 0.02
        )

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.pixel_norm = _safe_group_norm(self.hidden_dim)

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(float(logit_scale_init)), dtype=torch.float32)
        )

    def forward(
        self,
        pixel_embed: torch.Tensor,
        init_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_embed: [B, C, D, H, W]
            init_logits: [B, C, H, W]

        Returns:
            final_logits: [B, C, H, W]
        """
        if pixel_embed.ndim != 5:
            raise ValueError(
                f"pixel_embed must be [B, C, D, H, W], got {tuple(pixel_embed.shape)}."
            )
        if init_logits.ndim != 4:
            raise ValueError(
                f"init_logits must be [B, C, H, W], got {tuple(init_logits.shape)}."
            )

        B, C, D, H, W = pixel_embed.shape

        if D != self.hidden_dim:
            raise ValueError(
                f"pixel_embed channel dim mismatch: expected {self.hidden_dim}, got {D}."
            )
        if tuple(init_logits.shape) != (B, C, H, W):
            raise ValueError(
                f"init_logits must be [{B}, {C}, {H}, {W}], "
                f"got {tuple(init_logits.shape)}."
            )

        # Shared learnable queries, expanded to each image-class pair.
        queries = self.query_embed.expand(B, C, self.num_queries, D)
        queries = self.query_norm(queries)

        # Normalize pixel features once and use them for both attention and final matching.
        pixel_flat = pixel_embed.contiguous().reshape(B * C, D, H, W)
        pixel_normed_flat = self.pixel_norm(pixel_flat)
        pixel_normed = pixel_normed_flat.reshape(B, C, D, H, W)

        # Build mask probability from initial SAM3 semantic logits.
        mask_prob = torch.sigmoid(init_logits).contiguous().reshape(B * C, 1, H, W)

        # Downsample by configured factor.
        # For H=W=288 and attn_downsample=4, attention happens at 72×72.
        attn_h = max(1, H // self.attn_downsample)
        attn_w = max(1, W // self.attn_downsample)

        if (attn_h, attn_w) != (H, W):
            pixel_attn = F.interpolate(
                pixel_normed_flat,
                size=(attn_h, attn_w),
                mode="bilinear",
                align_corners=False,
            )
            mask_attn = F.interpolate(
                mask_prob,
                size=(attn_h, attn_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pixel_attn = pixel_normed_flat
            mask_attn = mask_prob

        N = attn_h * attn_w
        head_dim = D // self.num_heads

        q = self.q_proj(queries.reshape(B * C, self.num_queries, D))

        kv = pixel_attn.flatten(2).transpose(1, 2)  # [B*C, N, D]
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = q.reshape(B * C, self.num_queries, self.num_heads, head_dim)
        q = q.transpose(1, 2)  # [B*C, heads, Q, head_dim]

        k = k.reshape(B * C, N, self.num_heads, head_dim)
        k = k.transpose(1, 2)  # [B*C, heads, N, head_dim]

        v = v.reshape(B * C, N, self.num_heads, head_dim)
        v = v.transpose(1, 2)  # [B*C, heads, N, head_dim]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        # attn_logits: [B*C, heads, Q, N]

        # More aggressive mask-guided additive bias.
        # mask_bias = mask_bias_scale * log(max(mask_prob, mask_gate_floor) + eps)
        mask_gate = mask_attn.flatten(2)
        mask_gate = mask_gate.to(dtype=attn_logits.dtype)
        mask_gate = mask_gate.clamp_min(self.mask_gate_floor)

        mask_bias = self.mask_bias_scale * torch.log(mask_gate + 1e-6)
        # mask_bias: [B*C, 1, N]
        attn_logits = attn_logits + mask_bias.unsqueeze(1)
        # broadcast to [B*C, heads, Q, N]

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        updated = torch.matmul(attn, v)
        updated = updated.transpose(1, 2).reshape(B * C, self.num_queries, D)
        updated = self.out_proj(updated)

        base_queries = queries.reshape(B * C, self.num_queries, D)
        updated = self.out_norm(base_queries + self.dropout(updated))
        updated = updated.reshape(B, C, self.num_queries, D)

        # Query-to-full-resolution-pixel similarity.
        norm_queries = F.normalize(updated, dim=-1)
        norm_pixels = F.normalize(pixel_normed, dim=2)

        query_scores = torch.einsum(
            "bcqd,bcdhw->bcqhw",
            norm_queries,
            norm_pixels,
        )
        # query_scores: [B, C, Q, H, W]

        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
        scale = scale.to(device=query_scores.device, dtype=query_scores.dtype)
        query_scores = query_scores * scale

        # Train and inference both use logsumexp pooling.
        tau = self.query_pool_temperature
        final_logits = tau * (
            torch.logsumexp(query_scores / tau, dim=2)
            - math.log(self.num_queries)
        )

        return final_logits.contiguous()
