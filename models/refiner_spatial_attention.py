from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GlobalFeatureAttention(nn.Module):
    """Per-class 18×18 self-attention context, with 2D relative position bias.

    The output is context only: there is no residual into the main stream.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by positive num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_dropout = nn.Dropout(dropout)

        size = 18
        self.relative_position_bias_table = nn.Parameter(torch.empty((2 * size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")).flatten(1)
        offsets = coords[:, :, None] - coords[:, None, :]
        index = (offsets[0] + size - 1) * (2 * size - 1) + offsets[1] + size - 1
        self.register_buffer("relative_position_index", index, persistent=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, classes, channels, height, width = feature.shape
        if channels != self.hidden_dim or (height, width) != (36, 36):
            raise ValueError("GlobalFeatureAttention expects [B, C, hidden_dim, 36, 36]")
        # Each pooled position represents a contiguous 2×2 area of feature36.
        pooled = F.avg_pool2d(feature.reshape(batch * classes, channels, height, width), 2)
        tokens = self.norm(pooled.flatten(2).transpose(1, 2))
        qkv = self.qkv(tokens).reshape(batch * classes, 324, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_position_bias_table[self.relative_position_index].permute(2, 0, 1)
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(dtype=q.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch * classes, 324, channels)
        context = self.out_dropout(self.out_proj(context)).transpose(1, 2).reshape(batch * classes, channels, 18, 18)
        context = F.interpolate(context, size=(36, 36), mode="bilinear", align_corners=False)
        return context.reshape(batch, classes, channels, 36, 36)


class LocalScoreAttention(nn.Module):
    """Strict sliding 3×3 attention; one weight map updates both streams.

    Q/K: feature + score + global context. Feature value: feature + context.
    Score value: score only. Returns updates, without adding residuals.
    """

    def __init__(self, hidden_dim: int = 256, score_embed_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by positive num_heads")
        self.hidden_dim = hidden_dim
        self.score_embed_dim = score_embed_dim
        self.num_heads = num_heads
        self.norm_feature = nn.LayerNorm(hidden_dim)
        self.norm_score = nn.LayerNorm(score_embed_dim)
        self.norm_context = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(2 * hidden_dim + score_embed_dim, hidden_dim)
        self.k_proj = nn.Linear(2 * hidden_dim + score_embed_dim, hidden_dim)
        self.v_feature_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.v_score_proj = nn.Linear(score_embed_dim, hidden_dim)
        self.out_feature_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_score_proj = nn.Linear(hidden_dim, score_embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Nine offsets from a central query to its contiguous neighbours.
        self.relative_position_bias = nn.Parameter(torch.empty(num_heads, 9))
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def _neighbours(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        maps = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return F.unfold(maps, kernel_size=3, padding=1).reshape(
            batch, self.num_heads, channels // self.num_heads, 9, height * width,
        ).permute(0, 1, 4, 3, 2)

    def forward(self, feature: torch.Tensor, score: torch.Tensor,
                context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, classes, channels, height, width = feature.shape
        if channels != self.hidden_dim or context.shape != feature.shape:
            raise ValueError("feature/context must share [B, C, hidden_dim, H, W]")
        if score.shape != (batch, classes, self.score_embed_dim, height, width):
            raise ValueError("score shape does not match the feature stream")
        def tokens(x):
            return x.permute(0, 1, 3, 4, 2).reshape(batch * classes, height * width, x.shape[2])

        f = self.norm_feature(tokens(feature))
        s = self.norm_score(tokens(score))
        g = self.norm_context(tokens(context))
        qk_input = torch.cat((f, s, g), dim=-1)
        head_dim = channels // self.num_heads
        q = self.q_proj(qk_input).reshape(batch * classes, height * width, self.num_heads, head_dim).transpose(1, 2)
        k = self._neighbours(self.k_proj(qk_input), height, width)
        weights = (q.unsqueeze(-2) * k).sum(-1) * head_dim ** -0.5
        weights = weights + self.relative_position_bias[None, :, None, :].to(weights.dtype)
        # Padded positions are excluded from softmax, including at corners.
        valid = F.unfold(feature.new_ones(1, 1, height, width), kernel_size=3, padding=1)
        weights = weights.masked_fill(~valid.transpose(1, 2).bool().unsqueeze(1), float("-inf"))
        weights = self.dropout(weights.softmax(dim=-1))

        def update(value, projection, out_channels):
            neighbours = self._neighbours(value, height, width)
            out = (weights.unsqueeze(-1) * neighbours).sum(-2)
            out = out.transpose(1, 2).reshape(batch * classes, height * width, channels)
            out = self.dropout(projection(out))
            return out.reshape(batch, classes, height, width, out_channels).permute(0, 1, 4, 2, 3).contiguous()

        feature_update = update(self.v_feature_proj(torch.cat((f, g), dim=-1)), self.out_feature_proj, channels)
        score_update = update(self.v_score_proj(s), self.out_score_proj, self.score_embed_dim)
        return feature_update, score_update
