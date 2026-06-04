from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassCodeGuidedClassAttention(nn.Module):
    """
    Class-wise attention where q/k concatenate class_code for guidance.
    class_code only enters q/k, not v.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}")

        self.q_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, class_code: torch.Tensor) -> torch.Tensor:
        B, C, D, Hc, Wc = x.shape
        N = Hc * Wc

        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)
        cc_expanded = class_code[:, None].expand(B, N, C, D).reshape(B * N, C, D)

        q = self.q_proj(torch.cat([x_flat, cc_expanded], dim=-1))
        k = self.k_proj(torch.cat([x_flat, cc_expanded], dim=-1))
        v = self.v_proj(x_flat)

        head_dim = self.hidden_dim // self.num_heads
        q = q.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B * N, C, D)
        out = self.out_proj(out)
        out = self.norm(x_flat + self.dropout(out))
        return out.reshape(B, Hc, Wc, C, D).permute(0, 3, 4, 1, 2).contiguous()


class SamClassFeatureWindowAttention(nn.Module):
    """
    Window attention within each class.

    Supports regular (shift_size=0) and shifted-window (shift_size>0) modes.
    In shifted mode an attention mask prevents cross-region leakage.

    q/k concatenate class_feature_low for guidance.
    v uses x only. No semantic-score logit bias is applied.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}")
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(f"shift_size={shift_size} must be in [0, window_size={window_size})")

        self.q_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_window(x: torch.Tensor, window_size: int):
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, H, W
        return F.pad(x, (0, pad_w, 0, pad_h)), H, W

    def _window_partition(self, x: torch.Tensor):
        # [B, D, H, W] → [num_win, ws*ws, D]
        B, D, H, W = x.shape
        ws = self.window_size
        x = x.reshape(B, D, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, D)
        return x

    def _window_reverse(self, x: torch.Tensor, B: int, H: int, W: int):
        ws = self.window_size
        D = self.hidden_dim
        x = x.reshape(B, H // ws, W // ws, ws, ws, D)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, D, H, W)
        return x

    # ------------------------------------------------------------------
    # Shifted-window attention mask
    # ------------------------------------------------------------------

    def _build_shift_attn_mask(
        self,
        padded_h: int,
        padded_w: int,
        bc: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """
        Build per-window attention mask for shifted-window mode.

        Returns [total_windows, ws*ws, ws*ws] where -inf masks out
        cross-region token pairs.
        """
        if self.shift_size == 0:
            return None

        ws = self.window_size
        shift = self.shift_size

        # Assign a unique ID to each pre-shift contiguous window region.
        img_mask = torch.zeros((1, padded_h, padded_w), device=device, dtype=torch.float32)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w] = cnt
                cnt += 1

        # Roll and partition the mask the same way as features.
        img_mask = torch.roll(img_mask, shifts=(-shift, -shift), dims=(1, 2))
        mask_windows = self._window_partition(img_mask.unsqueeze(0))  # [win_per_img, ws*ws, 1]
        mask_windows = mask_windows.squeeze(-1)                        # [win_per_img, ws*ws]

        # Per-window mask: same-ID tokens attend, different-ID tokens get -inf.
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        # Repeat for all (B*C) images.
        win_per_img = attn_mask.shape[0]
        attn_mask = attn_mask.unsqueeze(0).expand(bc, win_per_img, ws * ws, ws * ws)
        attn_mask = attn_mask.reshape(bc * win_per_img, ws * ws, ws * ws)

        return attn_mask.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        class_feature_low: torch.Tensor,
    ) -> torch.Tensor:
        B, C, D, Hc, Wc = x.shape
        bc = B * C

        x_flat = x.reshape(bc, D, Hc, Wc)
        cf_flat = class_feature_low.reshape(bc, D, Hc, Wc)

        # Pad to window multiples.
        x_flat, orig_h, orig_w = self._pad_to_window(x_flat, self.window_size)
        cf_flat, _, _ = self._pad_to_window(cf_flat, self.window_size)
        pad_h, pad_w = x_flat.shape[-2], x_flat.shape[-1]

        # Shifted window: roll before partitioning.
        if self.shift_size > 0:
            shift = self.shift_size
            x_flat = torch.roll(x_flat, shifts=(-shift, -shift), dims=(-2, -1))
            cf_flat = torch.roll(cf_flat, shifts=(-shift, -shift), dims=(-2, -1))

        # Partition into windows.
        x_windows = self._window_partition(x_flat)
        cf_windows = self._window_partition(cf_flat)

        # Build attention mask for shifted mode.
        attn_mask = self._build_shift_attn_mask(pad_h, pad_w, bc, x.device, x.dtype)

        # Q/K/V projections.
        q = self.q_proj(torch.cat([x_windows, cf_windows], dim=-1))
        k = self.k_proj(torch.cat([x_windows, cf_windows], dim=-1))
        v = self.v_proj(x_windows)

        head_dim = D // self.num_heads
        num_win, N = q.shape[0], q.shape[1]
        q = q.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(num_win, N, D)
        out = self.out_proj(out)
        out = self.norm(x_windows + self.dropout(out))

        out = self._window_reverse(out, bc, pad_h, pad_w)

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(-2, -1))

        out = out[:, :, :orig_h, :orig_w]
        return out.reshape(B, C, D, Hc, Wc).contiguous()


class ClassFFN(nn.Module):
    """Per-class FFN with residual connection."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, Hc, Wc = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * Hc * Wc, C, D)
        residual = x_flat
        x_flat = self.norm(x_flat)
        x_flat = self.fc2(self.dropout(F.gelu(self.fc1(x_flat))))
        x_flat = residual + self.dropout(x_flat)
        return x_flat.reshape(B, Hc, Wc, C, D).permute(0, 3, 4, 1, 2).contiguous()


class LowResClassGuidedAggregatorLayer(nn.Module):
    """
    One aggregator layer:
        1. ClassCodeGuidedClassAttention
        2. SamClassFeatureWindowAttention  (regular window)
        3. SamClassFeatureWindowAttention  (shifted window)
        4. ClassFFN
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.class_attn = ClassCodeGuidedClassAttention(hidden_dim, num_heads, dropout)
        self.local_window_attn = SamClassFeatureWindowAttention(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=0,
            dropout=dropout,
        )
        self.shifted_window_attn = SamClassFeatureWindowAttention(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=shift_size,
            dropout=dropout,
        )
        self.ffn = ClassFFN(hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        class_feature_low: torch.Tensor,
        class_code: torch.Tensor,
    ) -> torch.Tensor:
        x = self.class_attn(x, class_code)
        x = self.local_window_attn(x, class_feature_low)
        x = self.shifted_window_attn(x, class_feature_low)
        x = self.ffn(x)
        return x


class LowResClassGuidedAggregator(nn.Module):
    """Multi-layer low-res aggregator."""

    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            LowResClassGuidedAggregatorLayer(
                hidden_dim,
                num_heads,
                window_size,
                shift_size,
                dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        class_feature_low: torch.Tensor,
        class_code: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, class_feature_low, class_code)
        return x