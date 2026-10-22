"""Class masks from shared learnable vectors and frozen SAM3 mask queries."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, queries, memory):
        memory = self.memory_norm(memory)
        update = self.attention(
            self.query_norm(queries), memory, memory, need_weights=False
        )[0]
        queries = queries + self.dropout(update)
        return queries + self.ffn(self.ffn_norm(queries))


class LearnableInstanceBranch(nn.Module):
    def __init__(self, num_heads=8, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.vectors = nn.Embedding(256, 256)
        nn.init.normal_(self.vectors.weight, std=0.02)
        self.feature_attention = CrossAttentionBlock(256, num_heads, dropout)
        self.query_attention = CrossAttentionBlock(256, num_heads, dropout)
        self.mask_projection = nn.Conv2d(256, 1, kernel_size=1)
        self.use_checkpoint = use_checkpoint

    def _forward(self, fused_feature, mask_queries, instance_features):
        vectors = self.vectors.weight.unsqueeze(0).expand(fused_feature.shape[0], -1, -1)
        vectors = self.feature_attention(vectors, fused_feature.flatten(2).transpose(1, 2))
        vectors = self.query_attention(vectors, mask_queries)
        responses = torch.einsum("nqd,ndhw->nqhw", vectors, instance_features)
        return self.mask_projection(responses)

    def forward(self, fused_feature, mask_queries, instance_features):
        if mask_queries.shape != (fused_feature.shape[0], 200, 256):
            raise ValueError("mask_queries must be [N, 200, 256].")
        if self.training and self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                self._forward, fused_feature, mask_queries, instance_features,
                use_reentrant=False,
            )
        return self._forward(fused_feature, mask_queries, instance_features)


@torch.no_grad()
def aggregate_teacher_masks(mask_logits, matching_probabilities, threshold):
    """Keep scores strictly above threshold; an empty pair is marked invalid."""
    keep = matching_probabilities > threshold
    valid_pairs = keep.any(dim=-1)
    probabilities = mask_logits.sigmoid().masked_fill(~keep[..., None, None], 0.0)
    return probabilities.amax(dim=-3), valid_pairs
