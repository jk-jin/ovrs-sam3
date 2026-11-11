from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    selected value stream supplies values; the returned update is added by
    the outer block.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        value_stream: str = "feature",
        guidance_dim: int = 0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.num_heads = int(num_heads)
        if self.feature_dim % self.num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if value_stream not in ("feature", "semantic"):
            raise ValueError("value_stream must be 'feature' or 'semantic'")
        self.value_stream = value_stream
        self.value_dim = (
            self.feature_dim
            if value_stream == "feature"
            else self.semantic_dim
        )
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
        """Return the update for the value stream; inputs are [B, P, D, H, W]."""
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
        value = f_flat if self.value_stream == "feature" else s_flat
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
    selected value stream supplies values. Regular and shifted windows follow
    the original implementation.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 0,
        dropout: float = 0.1,
        value_stream: str = "feature",
        guidance_dim: int = 0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        if self.feature_dim % self.num_heads != 0:
            raise ValueError(
                f"feature_dim={feature_dim} not divisible by "
                f"num_heads={num_heads}"
            )
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                f"shift_size={shift_size} must be in "
                f"[0, window_size={window_size})"
            )

        if value_stream not in ("feature", "semantic"):
            raise ValueError("value_stream must be 'feature' or 'semantic'")
        self.value_stream = value_stream
        self.value_dim = (
            self.feature_dim
            if value_stream == "feature"
            else self.semantic_dim
        )

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

    @staticmethod
    def _pad_to_window(x: torch.Tensor, window_size: int):
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, H, W
        return F.pad(x, (0, pad_w, 0, pad_h)), H, W

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

    def _build_shift_attn_mask(
        self,
        padded_h: int,
        padded_w: int,
        bc: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        ws = self.window_size
        shift = self.shift_size

        img_mask = torch.zeros(
            (1, padded_h, padded_w), device=device, dtype=torch.float32
        )

        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask.unsqueeze(0))
        mask_windows = mask_windows.squeeze(-1)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        win_per_img = attn_mask.shape[0]
        attn_mask = attn_mask.unsqueeze(0).expand(
            bc, win_per_img, ws * ws, ws * ws
        )
        attn_mask = attn_mask.reshape(bc * win_per_img, ws * ws, ws * ws)

        return attn_mask.to(dtype=dtype)

    def forward(
        self,
        feature_state: torch.Tensor,
        semantic_state: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the update for the value stream; inputs are [B, P, D, H, W]."""
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
        ws = self.window_size

        f_flat = feature_state.reshape(bc, D, H, W)
        s_flat = semantic_state.reshape(bc, D_semantic, H, W)

        f_flat, orig_h, orig_w = self._pad_to_window(f_flat, ws)
        s_flat, _, _ = self._pad_to_window(s_flat, ws)

        pad_h, pad_w = f_flat.shape[-2], f_flat.shape[-1]

        shift = self.shift_size
        if shift > 0:
            f_flat = torch.roll(f_flat, shifts=(-shift, -shift), dims=(-2, -1))
            s_flat = torch.roll(s_flat, shifts=(-shift, -shift), dims=(-2, -1))

        f_windows = self._window_partition(f_flat)   # [num_win, ws*ws, D]
        s_windows = self._window_partition(s_flat)   # [num_win, ws*ws, D_sem]

        g_windows = None
        if guidance is not None:
            g_flat = guidance.reshape(bc, self.guidance_dim, H, W)
            g_flat, _, _ = self._pad_to_window(g_flat, ws)
            if shift > 0:
                g_flat = torch.roll(
                    g_flat, shifts=(-shift, -shift), dims=(-2, -1)
                )
            g_windows = self._window_partition(g_flat)

        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=feature_state.device,
            dtype=feature_state.dtype,
        )

        # Q/K from both streams; values from the selected stream only.
        qk_input = torch.cat([s_windows, f_windows], dim=-1)
        if g_windows is not None:
            qk_input = torch.cat([qk_input, g_windows], dim=-1)

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        value = f_windows if self.value_stream == "feature" else s_windows
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

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(num_win, N, D)
        out = self.dropout(self.out_proj(out))
        out = self._window_reverse(out, bc, pad_h, pad_w)
        if shift > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(-2, -1))
        out = out[:, :, :orig_h, :orig_w]
        return out.reshape(B, P, self.value_dim, H, W).contiguous()


class CrossClassAggregationBlock(nn.Module):
    """Cross-class block with one semantic update then one feature update.

    Each step has its own pre-norm, projections and dropout. Q/K concatenate
    [semantic_state, feature_state] at each spatial position; the value and
    the residual target are the stream being updated. Prompts attend to each
    other independently per image and spatial position.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.semantic_attention = CrossClassAttention(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            dropout=dropout,
            value_stream="semantic",
        )
        self.feature_attention = CrossClassAttention(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            dropout=dropout,
            value_stream="feature",
        )
        self.semantic_attention_norm_feature = nn.LayerNorm(feature_dim)
        self.semantic_attention_norm_semantic = nn.LayerNorm(semantic_dim)
        self.feature_attention_norm_feature = nn.LayerNorm(feature_dim)
        self.feature_attention_norm_semantic = nn.LayerNorm(semantic_dim)

    def forward(
        self,
        feature_state_36: torch.Tensor,
        semantic_state_36: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_state_36 = semantic_state_36 + self.semantic_attention(
            feature_state=apply_layer_norm_bcdhw(
                feature_state_36, self.semantic_attention_norm_feature
            ),
            semantic_state=apply_layer_norm_bcdhw(
                semantic_state_36, self.semantic_attention_norm_semantic
            ),
        )
        feature_state_36 = feature_state_36 + self.feature_attention(
            feature_state=apply_layer_norm_bcdhw(
                feature_state_36, self.feature_attention_norm_feature
            ),
            semantic_state=apply_layer_norm_bcdhw(
                semantic_state_36, self.feature_attention_norm_semantic
            ),
        )
        return feature_state_36, semantic_state_36


class WindowAttentionUpdate(nn.Module):
    """One pre-normalized window attention update for a single stream."""

    def __init__(
        self,
        feature_dim: int,
        semantic_dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        dropout: float,
        value_stream: str,
        guidance_dim: int,
    ):
        super().__init__()
        self.attn = WindowSpatialAttention(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
            value_stream=value_stream,
            guidance_dim=guidance_dim,
        )
        self.norm_feature = nn.LayerNorm(feature_dim)
        self.norm_semantic = nn.LayerNorm(semantic_dim)
        self.norm_guidance = nn.LayerNorm(guidance_dim)

    def forward(
        self,
        feature_state_36: torch.Tensor,
        semantic_state_36: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(
            feature_state=apply_layer_norm_bcdhw(
                feature_state_36, self.norm_feature
            ),
            semantic_state=apply_layer_norm_bcdhw(
                semantic_state_36, self.norm_semantic
            ),
            guidance=apply_layer_norm_bcdhw(guidance, self.norm_guidance),
        )


class IntraClassAggregationBlock(nn.Module):
    """Intra-class block: two semantic window steps, then two feature steps.

    Semantic steps use RemoteCLIP spatial guidance; feature steps use SAM3
    FPN36 guidance. Regular and shifted windows each own independent
    parameters, and every step re-normalizes and re-projects its inputs.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 6,
        dropout: float = 0.1,
        guidance_dim: int = 256,
    ):
        super().__init__()
        step_kwargs = dict(
            feature_dim=feature_dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            guidance_dim=guidance_dim,
        )
        self.semantic_regular = WindowAttentionUpdate(
            shift_size=0, value_stream="semantic", **step_kwargs
        )
        self.semantic_shifted = WindowAttentionUpdate(
            shift_size=shift_size, value_stream="semantic", **step_kwargs
        )
        self.feature_regular = WindowAttentionUpdate(
            shift_size=0, value_stream="feature", **step_kwargs
        )
        self.feature_shifted = WindowAttentionUpdate(
            shift_size=shift_size, value_stream="feature", **step_kwargs
        )

    def forward(
        self,
        feature_state_36: torch.Tensor,
        semantic_state_36: torch.Tensor,
        clip_spatial_guidance_36: torch.Tensor,
        sam3_spatial_guidance_36: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_state_36 = semantic_state_36 + self.semantic_regular(
            feature_state_36, semantic_state_36, clip_spatial_guidance_36
        )
        semantic_state_36 = semantic_state_36 + self.semantic_shifted(
            feature_state_36, semantic_state_36, clip_spatial_guidance_36
        )
        feature_state_36 = feature_state_36 + self.feature_regular(
            feature_state_36, semantic_state_36, sam3_spatial_guidance_36
        )
        feature_state_36 = feature_state_36 + self.feature_shifted(
            feature_state_36, semantic_state_36, sam3_spatial_guidance_36
        )
        return feature_state_36, semantic_state_36
