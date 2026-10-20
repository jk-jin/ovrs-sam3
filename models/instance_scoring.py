from __future__ import annotations

import math

import torch
from torch import nn


class QueryCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        update = self.attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )[0]
        return queries + update


class InstanceScoreAdapter(nn.Module):
    """Two residual cross-attentions; original mask queries remain untouched."""

    def __init__(self, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if (
            isinstance(num_heads, bool)
            or not isinstance(num_heads, int)
            or num_heads <= 0
            or 256 % num_heads
        ):
            raise ValueError("instance num_heads must be a positive divisor of 256.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("instance dropout must be finite and in [0, 1).")
        self.layers = nn.ModuleList(
            [QueryCrossAttention(256, num_heads, dropout) for _ in range(2)]
        )

    def forward(self, queries, presence_features, fused_features):
        # Inputs are frozen observations. Only this adapter receives score loss gradients.
        tokens = torch.cat([presence_features.detach(), queries.detach()], dim=1)
        memory = fused_features.detach().flatten(2).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens, memory)
        return tokens[:, 1:], tokens[:, :1]


def aggregate_instance_probabilities(
    mask_logits, score_logits, presence_logits, threshold
):
    """Filter by score * presence; retain unweighted soft mask probabilities."""
    joint_scores = score_logits.sigmoid() * presence_logits.sigmoid().unsqueeze(-1)
    keep = joint_scores > threshold
    return mask_logits.sigmoid().masked_fill(~keep[..., None, None], 0).amax(dim=2)
