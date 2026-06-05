from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedClassAttention(nn.Module):
    """
    Class-wise attention guided by per-class text guidance vector.

    Args:
        x:
            [B, C, D, Hc, Wc]

        class_text_guidance:
            [B, C, D]

    Design:
        q/k = concat(x, class_text_guidance)
        v   = x
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )

        self.q_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        class_text_guidance: torch.Tensor,
    ) -> torch.Tensor:
        B, C, D, Hc, Wc = x.shape

        if class_text_guidance.shape != (B, C, D):
            raise ValueError(
                f"class_text_guidance must be [B, C, D] = {(B, C, D)}, "
                f"got {tuple(class_text_guidance.shape)}."
            )

        N = Hc * Wc

        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)
        guidance_expanded = (
            class_text_guidance[:, None]
            .expand(B, N, C, D)
            .reshape(B * N, C, D)
        )

        q = self.q_proj(torch.cat([x_flat, guidance_expanded], dim=-1))
        k = self.k_proj(torch.cat([x_flat, guidance_expanded], dim=-1))
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


class ClipSamGuidedWindowAttention(nn.Module):
    """
    Window attention within each class.

    q/k use:
        x + SAM3 class feature + CLIP dense feature

    v uses:
        x only
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
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                f"shift_size={shift_size} must be in [0, window_size={window_size})"
            )

        self.q_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _pad_to_window(x: torch.Tensor, window_size: int):
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, H, W
        return F.pad(x, (0, pad_w, 0, pad_h)), H, W

    def _window_partition(self, x: torch.Tensor):
        # [B, D, H, W] -> [num_win, ws*ws, D]
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
            (1, padded_h, padded_w),
            device=device,
            dtype=torch.float32,
        )

        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w] = cnt
                cnt += 1

        img_mask = torch.roll(
            img_mask,
            shifts=(-shift, -shift),
            dims=(1, 2),
        )

        mask_windows = self._window_partition(img_mask.unsqueeze(0))
        mask_windows = mask_windows.squeeze(-1)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        win_per_img = attn_mask.shape[0]
        attn_mask = attn_mask.unsqueeze(0).expand(
            bc,
            win_per_img,
            ws * ws,
            ws * ws,
        )
        attn_mask = attn_mask.reshape(
            bc * win_per_img,
            ws * ws,
            ws * ws,
        )

        return attn_mask.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        class_feature_low: torch.Tensor,
        clip_dense_low: torch.Tensor,
    ) -> torch.Tensor:
        B, C, D, Hc, Wc = x.shape

        if tuple(class_feature_low.shape) != (B, C, D, Hc, Wc):
            raise ValueError(
                "class_feature_low must have same shape as x: "
                f"expected {(B, C, D, Hc, Wc)}, got {tuple(class_feature_low.shape)}."
            )

        if tuple(clip_dense_low.shape) != (B, D, Hc, Wc):
            raise ValueError(
                "clip_dense_low must be [B, D, Hc, Wc]: "
                f"expected {(B, D, Hc, Wc)}, got {tuple(clip_dense_low.shape)}."
            )

        bc = B * C

        x_flat = x.reshape(bc, D, Hc, Wc)
        cf_flat = class_feature_low.reshape(bc, D, Hc, Wc)

        clip_flat = (
            clip_dense_low[:, None]
            .expand(B, C, D, Hc, Wc)
            .reshape(bc, D, Hc, Wc)
        )

        x_flat, orig_h, orig_w = self._pad_to_window(x_flat, self.window_size)
        cf_flat, _, _ = self._pad_to_window(cf_flat, self.window_size)
        clip_flat, _, _ = self._pad_to_window(clip_flat, self.window_size)

        pad_h, pad_w = x_flat.shape[-2], x_flat.shape[-1]

        if self.shift_size > 0:
            shift = self.shift_size
            x_flat = torch.roll(x_flat, shifts=(-shift, -shift), dims=(-2, -1))
            cf_flat = torch.roll(cf_flat, shifts=(-shift, -shift), dims=(-2, -1))
            clip_flat = torch.roll(clip_flat, shifts=(-shift, -shift), dims=(-2, -1))

        x_windows = self._window_partition(x_flat)
        cf_windows = self._window_partition(cf_flat)
        clip_windows = self._window_partition(clip_flat)

        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=x.device,
            dtype=x.dtype,
        )

        qk_input = torch.cat(
            [x_windows, cf_windows, clip_windows],
            dim=-1,
        )

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
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


class TextGuidedAggregatorLayer(nn.Module):
    """
    One aggregator layer:
        1. TextGuidedClassAttention
        2. ClipSamGuidedWindowAttention  (regular window)
        3. ClipSamGuidedWindowAttention  (shifted window)
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
        self.class_attn = TextGuidedClassAttention(hidden_dim, num_heads, dropout)
        self.local_window_attn = ClipSamGuidedWindowAttention(
            hidden_dim,
            num_heads,
            window_size,
            shift_size=0,
            dropout=dropout,
        )
        self.shifted_window_attn = ClipSamGuidedWindowAttention(
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
        clip_dense_low: torch.Tensor,
        class_text_guidance: torch.Tensor,
    ) -> torch.Tensor:
        x = self.class_attn(
            x=x,
            class_text_guidance=class_text_guidance,
        )

        x = self.local_window_attn(
            x=x,
            class_feature_low=class_feature_low,
            clip_dense_low=clip_dense_low,
        )

        x = self.shifted_window_attn(
            x=x,
            class_feature_low=class_feature_low,
            clip_dense_low=clip_dense_low,
        )

        x = self.ffn(x)
        return x


class TextGuidedLowResAggregator(nn.Module):
    """Multi-layer low-res aggregator guided by fused text vector."""

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
            TextGuidedAggregatorLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        ])

    def forward(
        self,
        x: torch.Tensor,
        class_feature_low: torch.Tensor,
        clip_dense_low: torch.Tensor,
        class_text_guidance: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                class_feature_low=class_feature_low,
                clip_dense_low=clip_dense_low,
                class_text_guidance=class_text_guidance,
            )
        return x