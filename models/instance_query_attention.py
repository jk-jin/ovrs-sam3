"""Update spatial features using all frozen, same-prompt instance queries."""

import torch
from torch import nn


class InstanceQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True,
        )

    def forward(self, features: torch.Tensor, instance_queries: torch.Tensor):
        # Each batch row is a single image-prompt pair, never mixed across classes.
        tokens = features.flatten(2).transpose(1, 2)
        queries = self.query_norm(instance_queries.detach())
        update = self.attention(
            self.feature_norm(tokens), queries, queries, need_weights=False,
        )[0]
        return (tokens + update).transpose(1, 2).reshape_as(features)
