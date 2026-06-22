from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


class ConvDownsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            _safe_group_norm(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvFuse2d(nn.Module):
    """Fuse two feature maps by conv after channel-wise concatenation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        padding = int(kernel_size) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=int(kernel_size),
                padding=padding,
                bias=False,
            ),
            _safe_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# TextConditionedClassAttention
# ---------------------------------------------------------------------------


class TextConditionedClassAttention(nn.Module):
    """
    Inter-class attention at each spatial position.

    q/k = concat(encoder_features, sam_text_mean)
    v   = encoder_features

    Attention happens across C classes at every spatial position.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )

        qk_in_dim = self.hidden_dim * 2

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        encoder_features: torch.Tensor,
        sam_text_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: [B, C, D, H, W]
            sam_text_mean:    [B, C, D]

        Returns:
            encoder_features: [B, C, D, H, W]
        """
        B, C, D, H, W = encoder_features.shape

        if sam_text_mean.ndim != 3:
            raise ValueError(
                f"sam_text_mean must be 3D [B, C, D], got {tuple(sam_text_mean.shape)}"
            )
        if tuple(sam_text_mean.shape) != (B, C, D):
            raise ValueError(
                f"sam_text_mean must be [{B}, {C}, {D}], "
                f"got {tuple(sam_text_mean.shape)}"
            )

        N = H * W

        e_flat = encoder_features.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)

        sam_text_broadcast = (
            sam_text_mean.to(device=e_flat.device, dtype=e_flat.dtype)[:, None]
            .expand(B, N, C, D)
            .reshape(B * N, C, D)
        )

        qk_input = torch.cat([e_flat, sam_text_broadcast], dim=-1)

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(e_flat)

        head_dim = D // self.num_heads
        q = q.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B * N, C, D)
        out = self.out_proj(out)
        out = self.norm(e_flat + self.dropout(out))

        return out.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# ScoreGuidedWindowAttention
# ---------------------------------------------------------------------------


class ScoreGuidedWindowAttention(nn.Module):
    """
    Intra-class window attention.

    q/k = concat(encoder_features, score_embed)
    v   = encoder_features
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 0,
        dropout: float = 0.1,
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

        qk_in_dim = self.hidden_dim + self.score_embed_dim

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)
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

        img_mask = torch.roll(img_mask, shifts=(-shift, -shift), dims=(1, 2))

        mask_windows = self._window_partition(img_mask.unsqueeze(0))
        mask_windows = mask_windows.squeeze(-1)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        win_per_img = attn_mask.shape[0]
        attn_mask = attn_mask.unsqueeze(0).expand(
            bc, win_per_img, ws * ws, ws * ws
        )
        attn_mask = attn_mask.reshape(bc * win_per_img, ws * ws, ws * ws)

        return attn_mask.to(dtype=dtype)

    def forward(
        self,
        encoder_features: torch.Tensor,
        score_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: [B, C, D, H, W]
            score_embed:      [B, C, D_score, H, W]

        Returns:
            encoder_features: [B, C, D, H, W]
        """
        B, C, D, H, W = encoder_features.shape

        if tuple(score_embed.shape) != (B, C, self.score_embed_dim, H, W):
            raise ValueError(
                f"score_embed must be "
                f"[{B}, {C}, {self.score_embed_dim}, {H}, {W}], "
                f"got {tuple(score_embed.shape)}"
            )

        bc = B * C

        e_flat = encoder_features.reshape(bc, D, H, W)
        score_flat = score_embed.reshape(bc, self.score_embed_dim, H, W)

        e_flat, orig_h, orig_w = self._pad_to_window(e_flat, self.window_size)
        score_flat, _, _ = self._pad_to_window(score_flat, self.window_size)

        pad_h, pad_w = e_flat.shape[-2], e_flat.shape[-1]

        shift = self.shift_size

        if shift > 0:
            e_flat = torch.roll(e_flat, shifts=(-shift, -shift), dims=(-2, -1))
            score_flat = torch.roll(score_flat, shifts=(-shift, -shift), dims=(-2, -1))

        e_windows = self._window_partition(e_flat)
        score_windows = self._window_partition(score_flat)

        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=encoder_features.device,
            dtype=encoder_features.dtype,
        )

        qk_input = torch.cat([e_windows, score_windows], dim=-1)

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(e_windows)

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
        out = self.norm(e_windows + self.dropout(out))

        out = self._window_reverse(out, bc, pad_h, pad_w)

        if shift > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(-2, -1))

        out = out[:, :, :orig_h, :orig_w]
        return out.reshape(B, C, D, H, W).contiguous()


# ---------------------------------------------------------------------------
# LowResScoreSpatialRefiner
# ---------------------------------------------------------------------------


class LowResScoreSpatialRefiner(nn.Module):
    """
    Single-scale low-resolution spatial refiner.

    Flow:
        72x72 -> downsample to 18x18
        -> score-guided window attention (regular + shifted) at 18x18
        -> bilinear upsample to 36x36 + 36x36 guide (from sam_image_last_72)
        -> 7x7 conv fusion
        -> bilinear upsample to 72x72 + 72x72 guide (sam_image_last_72)
        -> 7x7 conv fusion
        -> residual add to original 72x72
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
        spatial_fusion_kernel: int = 7,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.down_72_to_36 = ConvDownsample2d(self.hidden_dim)
        self.down_36_to_18 = ConvDownsample2d(self.hidden_dim)

        self.sam_guide_down_72_to_36 = ConvDownsample2d(self.hidden_dim)

        self.attn_18_regular = ScoreGuidedWindowAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            dropout=dropout,
        )

        self.attn_18_shifted = ScoreGuidedWindowAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
        )

        self.fuse_18_to_36 = ConvFuse2d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            kernel_size=spatial_fusion_kernel,
        )

        self.fuse_36_to_72 = ConvFuse2d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            kernel_size=spatial_fusion_kernel,
        )

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        sam_image_last_72: torch.Tensor,
        score_embed_18: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features_72: [B, C, D, 72, 72]
            sam_image_last_72:   [B, D, 72, 72]
            score_embed_18:      [B, C, D_score, 18, 18]

        Returns:
            refined_features_72: [B, C, D, 72, 72]
        """
        B, C, D, H, W = encoder_features_72.shape
        assert (H, W) == (72, 72)

        # 72 -> 36 -> 18
        x = encoder_features_72.reshape(B * C, D, 72, 72)
        x36 = self.down_72_to_36(x)
        x18 = self.down_36_to_18(x36)
        x18 = x18.reshape(B, C, D, 18, 18)

        # Score-guided window attention at 18x18
        x18 = self.attn_18_regular(x18, score_embed_18)
        x18 = self.attn_18_shifted(x18, score_embed_18)

        # 18 -> 36, fuse with self-built 36 guide
        x36 = x18.reshape(B * C, D, 18, 18)
        x36 = F.interpolate(x36, size=(36, 36), mode="bilinear", align_corners=False)

        guide36 = self.sam_guide_down_72_to_36(sam_image_last_72)
        guide36 = (
            guide36[:, None]
            .expand(B, C, D, 36, 36)
            .reshape(B * C, D, 36, 36)
        )

        x36 = self.fuse_18_to_36(torch.cat([x36, guide36], dim=1))

        # 36 -> 72, fuse with SAM3 retained 72 guide
        x72 = F.interpolate(x36, size=(72, 72), mode="bilinear", align_corners=False)

        guide72 = (
            sam_image_last_72[:, None]
            .expand(B, C, D, 72, 72)
            .reshape(B * C, D, 72, 72)
        )

        delta72 = self.fuse_36_to_72(torch.cat([x72, guide72], dim=1))
        delta72 = delta72.reshape(B, C, D, 72, 72)

        # Residual add
        out = encoder_features_72 + delta72
        return out.contiguous()


# ---------------------------------------------------------------------------
# TextScoreEncoderRefinerLayer
# ---------------------------------------------------------------------------


class TextScoreEncoderRefinerLayer(nn.Module):
    """
    One refiner layer:

        1. TextConditionedClassAttention at 72x72
        2. LowResScoreSpatialRefiner (72->18 window attn + guided upsample back to 72)
        3. FFN
        4. LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
        spatial_fusion_kernel: int = 7,
    ):
        super().__init__()

        self.class_attn = TextConditionedClassAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.spatial_refiner = LowResScoreSpatialRefiner(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
            spatial_fusion_kernel=spatial_fusion_kernel,
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn_fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_dropout = nn.Dropout(float(dropout))

        self.output_norm = nn.LayerNorm(hidden_dim)

    def _output_layer_norm(self, encoder_features: torch.Tensor) -> torch.Tensor:
        return self.output_norm(
            encoder_features.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()

    def _ffn(self, encoder_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, hidden_dim, height, width = encoder_features.shape

        features_flat = encoder_features.permute(
            0, 3, 4, 1, 2,
        ).reshape(
            batch_size * height * width,
            num_classes,
            hidden_dim,
        )

        residual = features_flat
        features_flat = self.ffn_norm(features_flat)
        features_flat = self.ffn_fc2(
            self.ffn_dropout(F.gelu(self.ffn_fc1(features_flat)))
        )
        features_flat = residual + self.ffn_dropout(features_flat)

        return features_flat.reshape(
            batch_size,
            height,
            width,
            num_classes,
            hidden_dim,
        ).permute(0, 3, 4, 1, 2).contiguous()

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        sam_text_mean: torch.Tensor,
        sam_image_last_72: torch.Tensor,
        score_embed_18: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features_72: [B, C, D, 72, 72]
            sam_text_mean:       [B, C, D]
            sam_image_last_72:   [B, D, 72, 72]
            score_embed_18:      [B, C, D_score, 18, 18]

        Returns:
            encoder_features_72: [B, C, D, 72, 72]
        """
        x = self.class_attn(
            encoder_features=encoder_features_72,
            sam_text_mean=sam_text_mean,
        )

        x = self.spatial_refiner(
            encoder_features_72=x,
            sam_image_last_72=sam_image_last_72,
            score_embed_18=score_embed_18,
        )

        x = self._ffn(x)
        return self._output_layer_norm(x)
