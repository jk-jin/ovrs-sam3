from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import WINDOW_SCALES


def flatten_batch_class(
    features: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """[B, P, D, H, W] → [B*P, D, H, W]"""
    batch_size, num_prompts, channels, height, width = features.shape
    return (
        features.reshape(batch_size * num_prompts, channels, height, width),
        batch_size,
        num_prompts,
    )


def unflatten_batch_class(
    features: torch.Tensor,
    batch_size: int,
    num_prompts: int,
) -> torch.Tensor:
    """[B*P, D, H, W] → [B, P, D, H, W]"""
    _, channels, height, width = features.shape
    return features.reshape(
        batch_size, num_prompts, channels, height, width
    ).contiguous()


def resize_prompt_features(features: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinearly resize each image-prompt feature map independently."""
    if features.shape[-2:] == (size, size):
        return features
    flat, batch_size, num_prompts = flatten_batch_class(features)
    resized = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
    return unflatten_batch_class(resized, batch_size, num_prompts)


def apply_layer_norm_bcdhw(
    x: torch.Tensor,
    norm: nn.LayerNorm,
) -> torch.Tensor:
    """Apply LayerNorm on the channel dim of [B, P, D, H, W]."""
    return norm(
        x.permute(0, 1, 3, 4, 2)
    ).permute(0, 1, 4, 2, 3).contiguous()


def validate_guidance(guidance, feature_state, guidance_dim):
    if guidance_dim == 0:
        if guidance is not None:
            raise ValueError("This attention has no guidance input")
        return
    B, P, _, H, W = feature_state.shape
    if guidance is None or tuple(guidance.shape) != (B, P, guidance_dim, H, W):
        raise ValueError(f"guidance must have shape {(B, P, guidance_dim, H, W)}")


class CrossClassAttention(nn.Module):
    """Single-value cross-class attention at each spatial position.

    Q/K concatenate pre-normalized semantic state and feature state. Only the
    encoder feature supplies values; the returned update is added by
    the outer block.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        guidance_dim: int = 0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.num_heads = int(num_heads)
        if self.feature_dim % self.num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.value_dim = self.feature_dim
        self.guidance_dim = int(guidance_dim)
        qk_in_dim = self.feature_dim + self.semantic_dim + self.guidance_dim
        self.q_proj = nn.Linear(qk_in_dim, self.feature_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.value_dim, self.feature_dim)
        self.out_proj = nn.Linear(self.feature_dim, self.value_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        feature_state: torch.Tensor,
        semantic_state: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the encoder feature update; inputs are [B, P, D, H, W]."""
        B, P, D, H, W = feature_state.shape
        validate_guidance(guidance, feature_state, self.guidance_dim)
        D_semantic = self.semantic_dim
        if tuple(semantic_state.shape) != (B, P, D_semantic, H, W):
            raise ValueError(
                "semantic_state shape must match feature_state's "
                "batch/prompt/grid"
            )
        N = H * W
        f_flat = feature_state.permute(0, 3, 4, 1, 2).reshape(B * N, P, D)
        s_flat = semantic_state.permute(0, 3, 4, 1, 2).reshape(
            B * N, P, D_semantic
        )
        qk_input = torch.cat([s_flat, f_flat], dim=-1)
        if guidance is not None:
            g_flat = guidance.permute(0, 3, 4, 1, 2).reshape(
                B * N, P, self.guidance_dim
            )
            qk_input = torch.cat([qk_input, g_flat], dim=-1)
        value = f_flat
        head_dim = D // self.num_heads
        q = self.q_proj(qk_input).reshape(
            B * N, P, self.num_heads, head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(qk_input).reshape(
            B * N, P, self.num_heads, head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(
            B * N, P, self.num_heads, head_dim
        ).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B * N, P, D)
        out = self.dropout(self.out_proj(out))
        return out.reshape(B, H, W, P, self.value_dim).permute(
            0, 3, 4, 1, 2
        ).contiguous()


class WindowSpatialAttention(nn.Module):
    """Single-value intra-class window attention with relative position bias.

    Q/K concatenate pre-normalized semantic state and feature state. Only the
    encoder feature supplies values. Each window covers a fixed region of the grid.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        dropout: float = 0.1,
        guidance_dim: int = 0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)

        if self.feature_dim % self.num_heads != 0:
            raise ValueError(
                f"feature_dim={feature_dim} not divisible by "
                f"num_heads={num_heads}"
            )
        self.value_dim = self.feature_dim

        self.guidance_dim = int(guidance_dim)
        qk_in_dim = self.feature_dim + self.semantic_dim + self.guidance_dim

        self.q_proj = nn.Linear(qk_in_dim, self.feature_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.feature_dim)

        self.v_proj = nn.Linear(self.value_dim, self.feature_dim)
        self.out_proj = nn.Linear(self.feature_dim, self.value_dim)

        self.dropout = nn.Dropout(float(dropout))

        # Relative position bias (GSNet / Swin style).
        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)          # [2, ws*ws]

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, ws*ws, ws*ws]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def _window_partition(self, x: torch.Tensor):
        B, D, H, W = x.shape
        ws = self.window_size
        x = x.reshape(B, D, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, D)
        return x

    def _window_reverse(self, x: torch.Tensor, B: int, H: int, W: int):
        ws = self.window_size
        D = x.shape[-1]
        x = x.reshape(B, H // ws, W // ws, ws, ws, D)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, D, H, W)
        return x

    def _get_relative_position_bias(self) -> torch.Tensor:
        """Return [num_heads, N, N], N = window_size * window_size."""
        ws = self.window_size
        N = ws * ws

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            N, N, self.num_heads
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()
        return relative_position_bias

    def forward(
        self,
        feature_state: torch.Tensor,
        semantic_state: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the encoder feature update; inputs are [B, P, D, H, W]."""
        B, P, D, H, W = feature_state.shape
        validate_guidance(guidance, feature_state, self.guidance_dim)
        D_semantic = self.semantic_dim

        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(
                "WindowSpatialAttention expects H/W divisible by "
                f"window_size={self.window_size}, got H={H}, W={W}."
            )
        if tuple(semantic_state.shape) != (B, P, D_semantic, H, W):
            raise ValueError(
                f"semantic_state must be [{B}, {P}, {D_semantic}, {H}, {W}], "
                f"got {tuple(semantic_state.shape)}"
            )

        bc = B * P
        f_windows = self._window_partition(feature_state.reshape(bc, D, H, W))
        s_windows = self._window_partition(semantic_state.reshape(bc, D_semantic, H, W))

        g_windows = None
        if guidance is not None:
            g_flat = guidance.reshape(bc, self.guidance_dim, H, W)
            g_windows = self._window_partition(g_flat)

        # Q/K from both streams; values from encoder features only.
        qk_input = torch.cat([s_windows, f_windows], dim=-1)
        if g_windows is not None:
            qk_input = torch.cat([qk_input, g_windows], dim=-1)

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        value = f_windows
        v = self.v_proj(value)

        head_dim = D // self.num_heads
        num_win, N = q.shape[0], q.shape[1]

        q = q.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)

        rel_pos_bias = self._get_relative_position_bias().to(
            device=attn.device, dtype=attn.dtype
        )
        attn = attn + rel_pos_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(num_win, N, D)
        out = self.dropout(self.out_proj(out))
        out = self._window_reverse(out, bc, H, W)
        return out.reshape(B, P, self.value_dim, H, W).contiguous()


class WindowAttentionUpdate(nn.Module):
    """One pre-normalized window attention update for a single stream."""

    def __init__(
        self,
        feature_dim: int,
        semantic_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        guidance_dim: int,
    ):
        super().__init__()
        self.attn = WindowSpatialAttention(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            guidance_dim=guidance_dim,
        )
        self.norm_feature = nn.LayerNorm(feature_dim)
        self.norm_semantic = nn.LayerNorm(semantic_dim)
        self.norm_guidance = nn.LayerNorm(guidance_dim)

    def forward(
        self,
        feature_state: torch.Tensor,
        semantic_state: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(
            feature_state=apply_layer_norm_bcdhw(
                feature_state, self.norm_feature
            ),
            semantic_state=apply_layer_norm_bcdhw(
                semantic_state, self.norm_semantic
            ),
            guidance=apply_layer_norm_bcdhw(guidance, self.norm_guidance),
        )


class MultiscaleWindowUpdate(nn.Module):
    """Fuse aligned window updates with learned softmax weights over three scales."""

    def __init__(self, share_weights: bool, **window_kwargs):
        super().__init__()
        self.share_weights = share_weights
        self.windows = nn.ModuleList(
            WindowAttentionUpdate(**window_kwargs)
            for _ in range(1 if share_weights else len(WINDOW_SCALES))
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(WINDOW_SCALES)))

    def forward(
        self,
        features: torch.Tensor,
        semantic_scales: tuple[torch.Tensor, ...],
        spatial_scales: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_weights = self.scale_logits.softmax(dim=0)
        update_sum = None
        for index, size in enumerate(WINDOW_SCALES):
            window = self.windows[0 if self.share_weights else index]
            update = window(
                resize_prompt_features(features, size), semantic_scales[index], spatial_scales[index],
            )
            aligned = resize_prompt_features(update, features.shape[-1])
            weighted = scale_weights[index] * aligned
            update_sum = weighted if update_sum is None else update_sum + weighted
        return update_sum


class AggregationAttentionBlock(nn.Module):
    """Add a cross-class residual followed by softmax-fused spatial updates."""

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        dropout: float = 0.1,
        guidance_dim: int = 256,
        share_window_weights_across_scales: bool = True,
    ):
        super().__init__()
        self.cross_attention = CrossClassAttention(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_norm_feature = nn.LayerNorm(feature_dim)
        self.cross_norm_semantic = nn.LayerNorm(semantic_dim)
        self.window_update = MultiscaleWindowUpdate(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            guidance_dim=guidance_dim,
            share_weights=share_window_weights_across_scales,
        )

    def forward(
        self,
        feature_state_36: torch.Tensor,
        semantic_scales: tuple[torch.Tensor, ...],
        spatial_scales: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        cross_update = self.cross_attention(
            feature_state=apply_layer_norm_bcdhw(feature_state_36, self.cross_norm_feature),
            semantic_state=apply_layer_norm_bcdhw(semantic_scales[0], self.cross_norm_semantic),
        )
        feature_state_36 = feature_state_36 + cross_update
        return feature_state_36 + self.window_update(
            feature_state_36, semantic_scales, spatial_scales
        )
