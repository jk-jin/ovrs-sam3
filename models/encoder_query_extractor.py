from __future__ import annotations

import torch
import torch.nn as nn


class EncoderQueryExtractor(nn.Module):
    """
    Extract per-class query tokens from encoder visual features.

    Input:
        e: [B, C, D, H, W]  — encoder last-layer visual features

    Output:
        class_query_tokens: [B, C, Q, D]

    Design:
        learnable_query [1, Q, D]
        → cross-attend e (flattened spatially) → class_query_tokens
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_query_tokens: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.num_heads = int(num_heads)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )

        self.query_embed = nn.Parameter(
            torch.zeros(1, self.num_query_tokens, self.hidden_dim)
        )
        nn.init.normal_(self.query_embed, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = e.shape

        if D != self.hidden_dim:
            raise ValueError(
                f"Channel mismatch: expected {self.hidden_dim}, got {D}"
            )

        N = H * W
        e_flat = e.flatten(3).permute(0, 1, 3, 2)  # [B, C, N, D]
        e_bc = e_flat.reshape(B * C, N, D)

        query = self.query_embed.expand(B * C, self.num_query_tokens, D)
        query = query.to(device=e_bc.device, dtype=e_bc.dtype)

        attn_out, _ = self.cross_attn(
            query=query,
            key=e_bc,
            value=e_bc,
            need_weights=False,
        )
        class_query_tokens = self.norm(query + self.dropout(attn_out))
        return class_query_tokens.reshape(B, C, self.num_query_tokens, D).contiguous()
