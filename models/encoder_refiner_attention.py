from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def flatten_batch_class(
    features: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """[B, C, D, H, W] → [B*C, D, H, W]"""
    batch_size, num_classes, channels, height, width = features.shape
    return (
        features.reshape(batch_size * num_classes, channels, height, width),
        batch_size,
        num_classes,
    )


def unflatten_batch_class(
    features: torch.Tensor,
    batch_size: int,
    num_classes: int,
) -> torch.Tensor:
    """[B*C, D, H, W] → [B, C, D, H, W]"""
    _, channels, height, width = features.shape
    return features.reshape(
        batch_size, num_classes, channels, height, width
    ).contiguous()


def apply_layer_norm_bcdhw(
    x: torch.Tensor,
    norm: nn.LayerNorm,
) -> torch.Tensor:
    """Apply LayerNorm on the channel dim of [B, C, D, H, W]."""
    return norm(
        x.permute(0, 1, 3, 4, 2)
    ).permute(0, 1, 4, 2, 3).contiguous()


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


# ---------------------------------------------------------------------------
# ClassScoreAttention
# ---------------------------------------------------------------------------


class ClassScoreAttention(nn.Module):
    """Single-value inter-class attention at each spatial position.

    Q/K concatenate pre-normalized score and feature. Only value_stream
    supplies values; the returned update is added by the outer layer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        value_stream: str = "feature",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if value_stream not in ("feature", "score"):
            raise ValueError("value_stream must be 'feature' or 'score'")
        self.value_stream = value_stream
        self.value_dim = (
            self.hidden_dim if value_stream == "feature" else self.score_embed_dim
        )
        qk_in_dim = self.hidden_dim + self.score_embed_dim
        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.value_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.value_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        feature: torch.Tensor,
        score_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Return the update for value_stream; inputs are [B, C, D, H, W]."""
        B, C, D, H, W = feature.shape
        D_score = self.score_embed_dim
        if tuple(score_embed.shape) != (B, C, D_score, H, W):
            raise ValueError("score_embed shape must match feature's batch/class/grid")
        N = H * W
        f_flat = feature.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)
        s_flat = score_embed.permute(0, 3, 4, 1, 2).reshape(B * N, C, D_score)
        qk_input = torch.cat([s_flat, f_flat], dim=-1)
        value = f_flat if self.value_stream == "feature" else s_flat
        head_dim = D // self.num_heads
        q = self.q_proj(qk_input).reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(qk_input).reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B * N, C, D)
        out = self.dropout(self.out_proj(out))
        return out.reshape(B, H, W, C, self.value_dim).permute(0, 3, 4, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# WindowScoreAttention
# ---------------------------------------------------------------------------


class WindowScoreAttention(nn.Module):
    """Single-value intra-class window attention with relative position bias.

    Q/K concatenate pre-normalized score and feature. Only value_stream
    supplies values. Regular/shifted windows follow the original implementation.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 0,
        dropout: float = 0.1,
        value_stream: str = "feature",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                f"shift_size={shift_size} must be in [0, window_size={window_size})"
            )

        if value_stream not in ("feature", "score"):
            raise ValueError("value_stream must be 'feature' or 'score'")
        self.value_stream = value_stream
        self.value_dim = (
            self.hidden_dim if value_stream == "feature" else self.score_embed_dim
        )

        qk_in_dim = self.hidden_dim + self.score_embed_dim  # 256+256=512

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)

        self.v_proj = nn.Linear(self.value_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.value_dim)

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
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]

        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1

        relative_position_index = relative_coords.sum(-1)  # [N, N]
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
        """
        Returns:
            relative_position_bias: [num_heads, N, N] where N = window_size * window_size
        """
        ws = self.window_size
        N = ws * ws

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.view(N, N, self.num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias  # [num_heads, N, N]

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
        feature: torch.Tensor,
        score_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Return the update for value_stream; inputs are [B, C, D, H, W]."""
        B, C, D, H, W = feature.shape
        D_score = self.score_embed_dim

        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(
                f"WindowScoreAttention expects H/W divisible by window_size={self.window_size}, "
                f"got H={H}, W={W}."
            )
        if tuple(score_embed.shape) != (B, C, D_score, H, W):
            raise ValueError(
                f"score_embed must be [{B}, {C}, {D_score}, {H}, {W}], "
                f"got {tuple(score_embed.shape)}"
            )

        bc = B * C
        ws = self.window_size

        f_flat = feature.reshape(bc, D, H, W)
        s_flat = score_embed.reshape(bc, D_score, H, W)

        f_flat, orig_h, orig_w = self._pad_to_window(f_flat, ws)
        s_flat, _, _ = self._pad_to_window(s_flat, ws)

        pad_h, pad_w = f_flat.shape[-2], f_flat.shape[-1]

        shift = self.shift_size
        if shift > 0:
            f_flat = torch.roll(f_flat, shifts=(-shift, -shift), dims=(-2, -1))
            s_flat = torch.roll(s_flat, shifts=(-shift, -shift), dims=(-2, -1))

        f_windows = self._window_partition(f_flat)   # [num_win, ws*ws, D]
        s_windows = self._window_partition(s_flat)   # [num_win, ws*ws, D_score]

        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=feature.device,
            dtype=feature.dtype,
        )

        # Q/K from both streams; values from the selected stream only.
        qk_input = torch.cat([s_windows, f_windows], dim=-1)  # [num_win, N, 512]

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

        # Add relative position bias.
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
        return out.reshape(B, C, self.value_dim, H, W).contiguous()


# ---------------------------------------------------------------------------
# LocalScoreAttention
# ---------------------------------------------------------------------------


class LocalScoreAttention(nn.Module):
    """Single-value intra-class attention over each pixel's contiguous 3×3 grid.

    Inputs are pre-normalized. Q/K concatenate score and feature; V and
    the output come only from value_stream. Out-of-image keys are masked.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        value_stream: str = "feature",
    ):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if value_stream not in ("feature", "score"):
            raise ValueError("value_stream must be 'feature' or 'score'")
        self.hidden_dim = hidden_dim
        self.score_embed_dim = score_embed_dim
        self.num_heads = num_heads
        self.value_stream = value_stream
        self.value_dim = hidden_dim if value_stream == "feature" else score_embed_dim
        self.q_proj = nn.Linear(hidden_dim + score_embed_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim + score_embed_dim, hidden_dim)
        self.v_proj = nn.Linear(self.value_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.value_dim)
        # Row-major offsets: top-left, top, ..., center, ..., bottom-right.
        self.relative_position_bias_table = nn.Parameter(torch.empty(9, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, feature: torch.Tensor, score_embed: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = feature.shape
        if D != self.hidden_dim or tuple(score_embed.shape) != (B, C, self.score_embed_dim, H, W):
            raise ValueError("feature/score_embed channels and batch/class/grid must match")
        bc, n = B * C, H * W
        head_dim = D // self.num_heads
        f = feature.permute(0, 1, 3, 4, 2).reshape(bc, n, D)
        s = score_embed.permute(0, 1, 3, 4, 2).reshape(bc, n, self.score_embed_dim)
        qk_input = torch.cat([s, f], dim=-1)
        q = self.q_proj(qk_input).transpose(1, 2).reshape(bc, self.num_heads, head_dim, n)

        def neighbors(x):
            x = x.transpose(1, 2).reshape(bc, D, H, W)
            return F.unfold(x, kernel_size=3, padding=1).reshape(
                bc, self.num_heads, head_dim, 9, n
            )

        k = neighbors(self.k_proj(qk_input))
        v = neighbors(self.v_proj(f if self.value_stream == "feature" else s))
        logits = (q.unsqueeze(3) * k).sum(dim=2) * (head_dim ** -0.5)
        bias = self.relative_position_bias_table.T.to(dtype=logits.dtype)
        logits = logits + bias[None, :, :, None]
        valid = F.unfold(feature.new_ones(1, 1, H, W), kernel_size=3, padding=1)
        logits = logits.masked_fill(~valid.bool().unsqueeze(1), float("-inf"))
        attention = self.dropout(F.softmax(logits, dim=2))
        out = (attention.unsqueeze(2) * v).sum(dim=3)
        out = out.reshape(bc, D, n).transpose(1, 2)
        out = self.dropout(self.out_proj(out))
        return out.reshape(B, C, H, W, self.value_dim).permute(0, 1, 4, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# EncoderRefinerLayer
# ---------------------------------------------------------------------------


class EncoderRefinerLayer(nn.Module):
    """Alternating single-value Refiner layer with pre-norm/direct residuals.

    score_attention_type="intra": configured intra-class attention updates score,
    then inter-class attention updates feature.
    score_attention_type="inter": inter-class attention updates score,
    then configured intra-class attention updates feature.
    Each stream independently chooses two windows or four local 3×3 steps.
    Both finish with independent Feature FFN and Score FFN.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 6,
        dropout: float = 0.1,
        score_attention_type: str = "intra",
        score_intra_attn_type: str = "window",
        feature_intra_attn_type: str = "window",
    ):
        super().__init__()
        if score_attention_type not in ("intra", "inter"):
            raise ValueError("score_attention_type must be 'intra' or 'inter'")
        self.score_attention_type = score_attention_type
        for name, kind in (("score_intra_attn_type", score_intra_attn_type),
                           ("feature_intra_attn_type", feature_intra_attn_type)):
            if kind not in ("window", "local_3x3"):
                raise ValueError(f"{name} must be 'window' or 'local_3x3'")
        self.intra_attn_type = (
            score_intra_attn_type if score_attention_type == "intra" else feature_intra_attn_type
        )
        class_value_stream = "feature" if score_attention_type == "intra" else "score"
        intra_value_stream = "score" if score_attention_type == "intra" else "feature"
        self.class_attn = ClassScoreAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            value_stream=class_value_stream,
        )
        if self.intra_attn_type == "window":
            self.window_attn_regular = WindowScoreAttention(
                hidden_dim=hidden_dim, score_embed_dim=score_embed_dim,
                num_heads=num_heads, window_size=window_size, shift_size=0,
                dropout=dropout, value_stream=intra_value_stream,
            )
            self.window_attn_shifted = WindowScoreAttention(
                hidden_dim=hidden_dim, score_embed_dim=score_embed_dim,
                num_heads=num_heads, window_size=window_size, shift_size=shift_size,
                dropout=dropout, value_stream=intra_value_stream,
            )
        else:
            self.local_attn = nn.ModuleList([
                LocalScoreAttention(
                    hidden_dim=hidden_dim, score_embed_dim=score_embed_dim,
                    num_heads=num_heads, dropout=dropout, value_stream=intra_value_stream,
                )
                for _ in range(4)
            ])
        self.class_norm_feature = nn.LayerNorm(hidden_dim)
        self.class_norm_score = nn.LayerNorm(score_embed_dim)
        if self.intra_attn_type == "window":
            self.regular_norm_feature = nn.LayerNorm(hidden_dim)
            self.regular_norm_score = nn.LayerNorm(score_embed_dim)
            self.shifted_norm_feature = nn.LayerNorm(hidden_dim)
            self.shifted_norm_score = nn.LayerNorm(score_embed_dim)
        else:
            self.local_norm_feature = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])
            self.local_norm_score = nn.ModuleList([nn.LayerNorm(score_embed_dim) for _ in range(4)])
        self.ffn_norm_feature = nn.LayerNorm(hidden_dim)
        self.ffn_norm_score = nn.LayerNorm(score_embed_dim)

        self.ffn_fc1_feature = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn_fc2_feature = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_dropout_feature = nn.Dropout(float(dropout))
        self.ffn_fc1_score = nn.Linear(score_embed_dim, score_embed_dim * 4)
        self.ffn_fc2_score = nn.Linear(score_embed_dim * 4, score_embed_dim)
        self.ffn_dropout_score = nn.Dropout(float(dropout))

    def _class_update(
        self,
        feature: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        return self.class_attn(
            feature=apply_layer_norm_bcdhw(feature, self.class_norm_feature),
            score_embed=apply_layer_norm_bcdhw(score, self.class_norm_score),
        )

    def _update_intra(
        self,
        feature: torch.Tensor,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = (
            ((self.window_attn_regular, self.regular_norm_feature, self.regular_norm_score),
             (self.window_attn_shifted, self.shifted_norm_feature, self.shifted_norm_score))
            if self.intra_attn_type == "window"
            else zip(self.local_attn, self.local_norm_feature, self.local_norm_score)
        )
        for attention, feature_norm, score_norm in steps:
            update = attention(
                feature=apply_layer_norm_bcdhw(feature, feature_norm),
                score_embed=apply_layer_norm_bcdhw(score, score_norm),
            )
            if self.score_attention_type == "intra":
                score = score + update
            else:
                feature = feature + update
        return feature, score

    def _ffn_feature_update(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token FFN for pre-normalized feature. Returns update only."""
        B, C, D, H, W = feature.shape
        x = feature.permute(0, 1, 3, 4, 2)
        x = self.ffn_fc2_feature(
            self.ffn_dropout_feature(F.gelu(self.ffn_fc1_feature(x)))
        )
        x = self.ffn_dropout_feature(x)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def _ffn_score_update(
        self,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token FFN for pre-normalized score. Returns update only."""
        B, C, Ds, H, W = score.shape
        x = score.permute(0, 1, 3, 4, 2)
        x = self.ffn_fc2_score(
            self.ffn_dropout_score(F.gelu(self.ffn_fc1_score(x)))
        )
        x = self.ffn_dropout_score(x)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def forward(
        self,
        feature_36: torch.Tensor,
        score_embed_36: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update [B, C, 256, 36, 36] streams, score first then feature."""
        if self.score_attention_type == "intra":
            feature_36, score_embed_36 = self._update_intra(feature_36, score_embed_36)
            feature_36 = feature_36 + self._class_update(feature_36, score_embed_36)
        else:
            score_embed_36 = score_embed_36 + self._class_update(feature_36, score_embed_36)
            feature_36, score_embed_36 = self._update_intra(feature_36, score_embed_36)

        feature_36 = feature_36 + self._ffn_feature_update(
            apply_layer_norm_bcdhw(feature_36, self.ffn_norm_feature)
        )
        score_embed_36 = score_embed_36 + self._ffn_score_update(
            apply_layer_norm_bcdhw(score_embed_36, self.ffn_norm_score)
        )
        return feature_36, score_embed_36
