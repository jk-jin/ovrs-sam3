from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .position_encoding import PositionEmbeddingSine


class PostFusionAttention(nn.Module):
    """Global spatial attention for one prompt chunk, with asymmetric QK/V.

    Query blocks attend to ALL keys. Checkpointing each block bounds saved
    attention activations even when SDPA falls back to the math backend.
    """

    def __init__(self, qk_dim=64, value_dim=256, num_heads=4, dropout=0.1,
                 query_chunk_size=256, use_checkpoint=True):
        super().__init__()
        for name, value in (("qk_dim", qk_dim), ("value_dim", value_dim),
                            ("num_heads", num_heads), ("query_chunk_size", query_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if qk_dim % num_heads or value_dim % num_heads:
            raise ValueError("QK and value dimensions must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.value_dim = value_dim
        self.query_chunk_size = query_chunk_size
        self.use_checkpoint = use_checkpoint
        self.dropout = float(dropout)
        self.feature_norm = nn.LayerNorm(256)
        self.encoder_norm = nn.LayerNorm(256)
        self.fpn_norm = nn.LayerNorm(256)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=768)
        self.q_proj = nn.Linear(768, qk_dim)
        self.k_proj = nn.Linear(768, qk_dim)
        self.v_proj = nn.Linear(256, value_dim)
        self.out_proj = nn.Linear(value_dim, 256)
        self.output_dropout = nn.Dropout(dropout)

    def _attend(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )

    def forward(self, feature, encoder, fpn):
        if feature.ndim != 4 or feature.shape[1] != 256 or encoder.shape != feature.shape:
            raise ValueError("feature and encoder must both be [N, 256, H, W].")
        if fpn.ndim != 4 or fpn.shape[1:] != feature.shape[1:]:
            raise ValueError("fpn must be [B, 256, H, W].")
        pairs, _, height, width = feature.shape
        batch = fpn.shape[0]
        if batch <= 0 or pairs <= 0 or pairs % batch:
            raise ValueError("N must be positive and divisible by B.")
        prompts = pairs // batch
        tokens = height * width
        f = self.feature_norm(feature.flatten(2).transpose(1, 2))
        e = self.encoder_norm(encoder.flatten(2).transpose(1, 2))
        # Normalize the image-level FPN before broadcasting over prompts.
        g = self.fpn_norm(fpn.flatten(2).transpose(1, 2))
        f = f.reshape(batch, prompts, tokens, 256)
        e = e.reshape(batch, prompts, tokens, 256)
        g = g[:, None].expand(-1, prompts, -1, -1)
        pos = self.position_encoding(feature[:1]).to(device=f.device, dtype=f.dtype)
        qk_input = torch.cat((f, e, g), dim=-1).reshape(pairs, tokens, 768)
        qk_input = qk_input + pos.flatten(2).transpose(1, 2)
        f = f.reshape(pairs, tokens, 256)

        def heads(x, dim):
            return x.reshape(pairs, tokens, self.num_heads, dim // self.num_heads).transpose(1, 2)

        q = heads(self.q_proj(qk_input), self.qk_dim)
        k = heads(self.k_proj(qk_input), self.qk_dim)
        v = heads(self.v_proj(f), self.value_dim)
        blocks = []
        for q_block in q.split(self.query_chunk_size, dim=-2):
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                out = checkpoint(self._attend, q_block, k, v, use_reentrant=False)
            else:
                out = self._attend(q_block, k, v)
            blocks.append(out)
        out = torch.cat(blocks, dim=-2).transpose(1, 2).reshape(pairs, tokens, self.value_dim)
        update = self.output_dropout(self.out_proj(out)).transpose(1, 2)
        return feature + update.reshape(pairs, 256, height, width)
