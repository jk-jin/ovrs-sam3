from __future__ import annotations

import torch
import torch.nn as nn


def apply_layer_norm_bcdhw(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Normalize channels of [batch, prompt, channel, height, width]."""
    return norm(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()


class GuidedAttention(nn.Module):
    """Q/K use target, other stream and fixed score; V uses target only."""

    def __init__(self, hidden_dim=256, score_embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.score_embed_dim = score_embed_dim
        self.num_heads = num_heads
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.guidance_norm = nn.LayerNorm(hidden_dim)
        self.score_norm = nn.LayerNorm(score_embed_dim)
        qk_dim = 2 * hidden_dim + score_embed_dim
        self.q_proj = nn.Linear(qk_dim, hidden_dim)
        self.k_proj = nn.Linear(qk_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _normalize(self, target, guidance, score):
        if target.ndim != 5 or target.shape[2] != self.hidden_dim:
            raise ValueError("target must be [B, P, hidden_dim, H, W]")
        if guidance.shape != target.shape:
            raise ValueError("guidance must have the same shape as target")
        expected_score = (*target.shape[:2], self.score_embed_dim, *target.shape[-2:])
        if tuple(score.shape) != expected_score:
            raise ValueError(f"score must have shape {expected_score}")
        return (
            apply_layer_norm_bcdhw(target, self.target_norm),
            apply_layer_norm_bcdhw(guidance, self.guidance_norm),
            apply_layer_norm_bcdhw(score, self.score_norm),
        )

    def _attend(self, target, guidance, score, bias=None):
        # Inputs are [independent group, token, channel].
        qk = torch.cat((target, guidance, score), dim=-1)
        groups, tokens, channels = target.shape
        head_dim = channels // self.num_heads

        def split_heads(x):
            return x.reshape(groups, tokens, self.num_heads, head_dim).transpose(1, 2)

        q, k = split_heads(self.q_proj(qk)), split_heads(self.k_proj(qk))
        v = split_heads(self.v_proj(target))
        logits = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
        if bias is not None:
            logits = logits + bias
        weights = self.dropout(logits.softmax(dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(groups, tokens, channels)
        return self.dropout(self.out_proj(out))


class InterClassAttention(GuidedAttention):
    """Aggregate all prompts, including self, at each image/spatial position."""

    def forward(self, target, guidance, score):
        target, guidance, score = self._normalize(target, guidance, score)
        batch, prompts, channels, height, width = target.shape

        def flatten(x):
            return x.permute(0, 3, 4, 1, 2).reshape(batch * height * width, prompts, -1)

        out = self._attend(flatten(target), flatten(guidance), flatten(score))
        return (
            out.reshape(batch, height, width, prompts, channels)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )


class WindowAttention(GuidedAttention):
    """Per-prompt regular/shifted windows with relative bias and shift mask."""

    def __init__(
        self,
        hidden_dim=256,
        score_embed_dim=256,
        num_heads=8,
        dropout=0.1,
        window_size=12,
        shift_size=0,
    ):
        super().__init__(hidden_dim, score_embed_dim, num_heads, dropout)
        if window_size <= 0 or not 0 <= shift_size < window_size:
            raise ValueError(
                "Require window_size > 0 and 0 <= shift_size < window_size"
            )
        self.window_size = window_size
        self.shift_size = shift_size
        self.relative_position_bias_table = nn.Parameter(
            torch.empty((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size), torch.arange(window_size), indexing="ij"
            )
        ).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        relative = relative + window_size - 1
        self.register_buffer(
            "relative_position_index",
            relative[..., 0] * (2 * window_size - 1) + relative[..., 1],
        )

    def _partition(self, x):
        batch, channels, height, width = x.shape
        ws = self.window_size
        return (
            x.reshape(batch, channels, height // ws, ws, width // ws, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, ws * ws, channels)
        )

    def _shift_mask(self, height, width, pairs, device):
        if not self.shift_size:
            return None
        ws, shift = self.window_size, self.shift_size
        regions = torch.zeros(1, 1, height, width, device=device)
        slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        for row, hs in enumerate(slices):
            for col, vs in enumerate(slices):
                regions[:, :, hs, vs] = row * 3 + col
        windows = self._partition(regions).squeeze(-1)
        blocked = windows[:, :, None] != windows[:, None, :]
        return blocked.repeat(pairs, 1, 1).unsqueeze(1)

    def forward(self, target, guidance, score):
        target, guidance, score = self._normalize(target, guidance, score)
        batch, prompts, channels, height, width = target.shape
        ws, shift = self.window_size, self.shift_size
        if height % ws or width % ws:
            raise ValueError("Spatial dimensions must be divisible by window_size")
        pairs = batch * prompts

        def partition(x):
            x = x.reshape(pairs, -1, height, width)
            if shift:
                x = torch.roll(x, (-shift, -shift), (-2, -1))
            return self._partition(x)

        tokens = ws * ws
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = (
            bias.reshape(tokens, tokens, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        )
        mask = self._shift_mask(height, width, pairs, target.device)
        if mask is not None:
            bias = bias.masked_fill(mask, float("-inf"))
        out = self._attend(
            partition(target), partition(guidance), partition(score), bias
        )
        out = (
            out.reshape(pairs, height // ws, width // ws, ws, ws, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(pairs, channels, height, width)
        )
        if shift:
            out = torch.roll(out, (shift, shift), (-2, -1))
        return out.reshape(batch, prompts, channels, height, width)


class StreamFFN(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        update = self.layers(self.norm(x.permute(0, 1, 3, 4, 2)))
        return x + update.permute(0, 1, 4, 2, 3)


class EncoderRefinerLayer(nn.Module):
    """Always intra then inter; odd/even layers exchange the target streams."""

    def __init__(
        self,
        hidden_dim=256,
        score_embed_dim=256,
        num_heads=8,
        window_size=12,
        shift_size=6,
        dropout=0.1,
        intra_target="fpn",
    ):
        super().__init__()
        if intra_target not in ("fpn", "encoder"):
            raise ValueError("intra_target must be 'fpn' or 'encoder'")
        self.intra_target = intra_target
        args = dict(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.window_regular = WindowAttention(
            **args, window_size=window_size, shift_size=0
        )
        self.window_shifted = WindowAttention(
            **args, window_size=window_size, shift_size=shift_size
        )
        self.inter_attention = InterClassAttention(**args)
        self.encoder_ffn = StreamFFN(hidden_dim, dropout)
        self.fpn_ffn = StreamFFN(hidden_dim, dropout)

    def forward(self, refined_encoder_36, refined_fpn_36, score_embed_36):
        if self.intra_target == "fpn":
            target, other = refined_fpn_36, refined_encoder_36
        else:
            target, other = refined_encoder_36, refined_fpn_36
        target = target + self.window_regular(target, other, score_embed_36)
        target = target + self.window_shifted(target, other, score_embed_36)
        other = other + self.inter_attention(other, target, score_embed_36)
        if self.intra_target == "fpn":
            encoder, fpn = other, target
        else:
            encoder, fpn = target, other
        return self.encoder_ffn(encoder), self.fpn_ffn(fpn)
