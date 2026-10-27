from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MaskFeedbackLayer(nn.Module):
    """Inter-class attention at each pixel; only the feature supplies values."""

    def __init__(self, num_heads: int, dropout: float):
        super().__init__()
        if num_heads <= 0 or 256 % num_heads:
            raise ValueError("num_heads must be positive and divide 256")
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder_norm = nn.LayerNorm(256)
        self.feature_norm = nn.LayerNorm(256)
        self.mask_norm = nn.LayerNorm(256)
        self.q_proj = nn.Linear(768, 256)
        self.k_proj = nn.Linear(768, 256)
        self.v_proj = nn.Linear(256, 256)
        self.out_proj = nn.Linear(256, 256)
        self.ffn_norm = nn.LayerNorm(256)
        self.ffn = nn.Sequential(
            nn.Linear(256, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.Dropout(dropout),
        )

    def forward(self, feature, encoder, mask_embed):
        batch, prompts, channels, height, width = feature.shape

        def tokens(x):
            return x.permute(0, 3, 4, 1, 2).reshape(-1, prompts, channels)

        current = tokens(feature)
        normalized = self.feature_norm(current)
        qk = torch.cat((self.encoder_norm(tokens(encoder)), normalized,
                        self.mask_norm(tokens(mask_embed))), dim=-1)

        def heads(x):
            return x.reshape(-1, prompts, self.num_heads, channels // self.num_heads).transpose(1, 2)

        update = F.scaled_dot_product_attention(
            heads(self.q_proj(qk)), heads(self.k_proj(qk)),
            heads(self.v_proj(normalized)),
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).reshape(-1, prompts, channels)
        current = current + F.dropout(self.out_proj(update), self.dropout, self.training)
        current = current + self.ffn(self.ffn_norm(current))
        return current.reshape(batch, height, width, prompts, channels).permute(0, 3, 4, 1, 2).contiguous()


class MaskFeedbackRefiner(nn.Module):
    """Detached first masks guide two independent 72x72 inter-class layers."""

    def __init__(self, num_heads=8, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.mask_embedding = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256), nn.GELU(),
        )
        self.layers = nn.ModuleList([MaskFeedbackLayer(num_heads, dropout) for _ in range(2)])

    def _run(self, module, *inputs):
        if self.training and self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, feature, encoder, first_logits, chunk_size):
        batch, prompts = feature.shape[:2]
        expected = (batch, prompts, 256, 72, 72)
        if tuple(feature.shape) != expected or tuple(encoder.shape) != expected:
            raise ValueError(f"feature and encoder must both have shape {expected}")
        if tuple(first_logits.shape) != (batch, prompts, 288, 288):
            raise ValueError("first_logits must be [B, P, 288, 288]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        embeddings = []
        for start in range(0, prompts, chunk_size):
            probabilities = first_logits[:, start:start + chunk_size].detach().sigmoid()
            count = probabilities.shape[1]
            embedded = self._run(self.mask_embedding, probabilities.reshape(-1, 1, 288, 288))
            embeddings.append(embedded.reshape(batch, count, 256, 72, 72))
        mask_embed = torch.cat(embeddings, dim=1)
        for layer in self.layers:
            feature = self._run(layer, feature, encoder, mask_embed)
        return feature
