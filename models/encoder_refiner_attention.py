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
    return features.reshape(batch_size * num_classes, channels, height, width), batch_size, num_classes


def unflatten_batch_class(
    features: torch.Tensor,
    batch_size: int,
    num_classes: int,
) -> torch.Tensor:
    """[B*C, D, H, W] → [B, C, D, H, W]"""
    _, channels, height, width = features.shape
    return features.reshape(batch_size, num_classes, channels, height, width).contiguous()


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


class BilinearConvUpsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _safe_group_norm(channels),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(x)


# ---------------------------------------------------------------------------
# ClassTokenScoreClassAttention
# ---------------------------------------------------------------------------


class ClassTokenScoreClassAttention(nn.Module):
    """
    Inter-class attention at each spatial position.

    q/k = concat(encoder_features, sam_text_mean, clip_score_embed)
    v   = encoder_features

    Attention happens across C classes at every spatial position.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_heads = int(num_heads)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )

        qk_in_dim = self.hidden_dim * 2 + self.score_embed_dim

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
        clip_score_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: [B, C, D, H, W]
            sam_text_mean:    [B, C, D]
            clip_score_embed: [B, C, D_score, H, W]

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
        if tuple(clip_score_embed.shape) != (B, C, self.score_embed_dim, H, W):
            raise ValueError(
                f"clip_score_embed must be [{B}, {C}, {self.score_embed_dim}, {H}, {W}], "
                f"got {tuple(clip_score_embed.shape)}"
            )

        N = H * W

        e_flat = encoder_features.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)

        sam_text_broadcast = (
            sam_text_mean.to(device=e_flat.device, dtype=e_flat.dtype)[:, None]
            .expand(B, N, C, D)
            .reshape(B * N, C, D)
        )

        score_flat = (
            clip_score_embed
            .permute(0, 3, 4, 1, 2)
            .reshape(B * N, C, self.score_embed_dim)
        )

        qk_input = torch.cat([e_flat, sam_text_broadcast, score_flat], dim=-1)

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
# WindowAttention2d (generic intra-class spatial window attention)
# ---------------------------------------------------------------------------


class WindowAttention2d(nn.Module):
    """
    Generic intra-class spatial window attention.

    q/k are built from qk_features.
    v is built from value_features.

    Args:
        qk_features:    [B, C, D_qk, H, W]
        value_features: [B, C, D, H, W]

    Returns:
        [B, C, D, H, W]
    """

    def __init__(
        self,
        qk_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qk_dim = int(qk_dim)
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

        self.q_proj = nn.Linear(self.qk_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.qk_dim, self.hidden_dim)
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
        qk_features: torch.Tensor,
        value_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            qk_features:    [B, C, D_qk, H, W]
            value_features: [B, C, D, H, W]

        Returns:
            [B, C, D, H, W]
        """
        B, C, _, H, W = qk_features.shape
        D = self.hidden_dim

        if tuple(value_features.shape) != (B, C, D, H, W):
            raise ValueError(
                f"value_features must be [{B}, {C}, {D}, {H}, {W}], "
                f"got {tuple(value_features.shape)}"
            )

        bc = B * C

        qk_flat = qk_features.reshape(bc, self.qk_dim, H, W)
        v_flat = value_features.reshape(bc, D, H, W)

        qk_flat, orig_h, orig_w = self._pad_to_window(qk_flat, self.window_size)
        v_flat, _, _ = self._pad_to_window(v_flat, self.window_size)

        pad_h, pad_w = qk_flat.shape[-2], qk_flat.shape[-1]

        shift = self.shift_size

        if shift > 0:
            qk_flat = torch.roll(qk_flat, shifts=(-shift, -shift), dims=(-2, -1))
            v_flat = torch.roll(v_flat, shifts=(-shift, -shift), dims=(-2, -1))

        qk_windows = self._window_partition(qk_flat)
        v_windows = self._window_partition(v_flat)

        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=qk_features.device,
            dtype=qk_features.dtype,
        )

        q = self.q_proj(qk_windows)
        k = self.k_proj(qk_windows)
        v = self.v_proj(v_windows)

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
        out = self.norm(v_windows + self.dropout(out))

        out = self._window_reverse(out, bc, pad_h, pad_w)

        if shift > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(-2, -1))

        out = out[:, :, :orig_h, :orig_w]
        return out.reshape(B, C, D, H, W).contiguous()


# ---------------------------------------------------------------------------
# MultiScaleImageScoreWindowAttention (serial: 18 → 36 → 72)
# ---------------------------------------------------------------------------


class MultiScaleImageScoreWindowAttention(nn.Module):
    """
    Serial multi-scale intra-class window attention.

    Flow:
        72 → conv down → 36 → conv down → 18

        18×18: regular + shifted window attention
        → upsample to 36 → add to features_36 → LayerNorm

        36×36: regular + shifted window attention
        → upsample to 72 → LayerNorm → return
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)

        self.down_72_to_36 = ConvDownsample2d(self.hidden_dim)
        self.down_36_to_18 = ConvDownsample2d(self.hidden_dim)

        self.up_18_to_36 = BilinearConvUpsample2d(self.hidden_dim)
        self.up_36_to_72 = BilinearConvUpsample2d(self.hidden_dim)

        image_score_qk_dim = self.hidden_dim * 2 + self.score_embed_dim

        self.attn_18_regular = WindowAttention2d(
            qk_dim=image_score_qk_dim,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            dropout=dropout,
        )
        self.attn_18_shifted = WindowAttention2d(
            qk_dim=image_score_qk_dim,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
        )

        self.attn_36_regular = WindowAttention2d(
            qk_dim=image_score_qk_dim,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            dropout=dropout,
        )
        self.attn_36_shifted = WindowAttention2d(
            qk_dim=image_score_qk_dim,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
        )

        self.norm_18_fused_36 = nn.LayerNorm(self.hidden_dim)
        self.norm_36_to_72 = nn.LayerNorm(self.hidden_dim)

    def _build_image_score_qk(
        self,
        encoder_features: torch.Tensor,
        sam_image_features: torch.Tensor,
        clip_score_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build qk input: concat(encoder_features, sam_image_features, clip_score_embed).

        Args:
            encoder_features:   [B, C, D, H, W]
            sam_image_features: [B, D, H, W]
            clip_score_embed:   [B, C, D_score, H, W]

        Returns:
            qk_features: [B, C, D + D + D_score, H, W]
        """
        B, C, D, H, W = encoder_features.shape

        sam_expanded = (
            sam_image_features[:, None]
            .expand(B, C, D, H, W)
        )

        qk_features = torch.cat(
            [encoder_features, sam_expanded, clip_score_embed],
            dim=2,
        )
        return qk_features

    def _downsample_encoder_features(
        self,
        encoder_features_72: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features_72_flat, batch_size, num_classes = flatten_batch_class(
            encoder_features_72,
        )

        features_36_flat = self.down_72_to_36(features_72_flat)
        features_18_flat = self.down_36_to_18(features_36_flat)

        features_36 = unflatten_batch_class(
            features_36_flat,
            batch_size,
            num_classes,
        )
        features_18 = unflatten_batch_class(
            features_18_flat,
            batch_size,
            num_classes,
        )

        return features_36, features_18

    @staticmethod
    def _norm_bcdhw(
        x: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        return norm(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()

    def _upsample_18_to_36(
        self,
        features_18: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        features_18_flat, batch_size, num_classes = flatten_batch_class(features_18)
        features_36_flat = self.up_18_to_36(features_18_flat, target_hw=target_hw)
        return unflatten_batch_class(features_36_flat, batch_size, num_classes)

    def _upsample_36_to_72(
        self,
        features_36: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        features_36_flat, batch_size, num_classes = flatten_batch_class(features_36)
        features_72_flat = self.up_36_to_72(features_36_flat, target_hw=target_hw)
        return unflatten_batch_class(features_72_flat, batch_size, num_classes)

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        sam_image_last_72: torch.Tensor,
        clip_score_embeds: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            encoder_features_72: [B, C, D, 72, 72]
            sam_image_last_72:   [B, D, 72, 72]
            clip_score_embeds:   {"scale_18": ..., "scale_36": ..., "scale_72": ...}

        Returns:
            spatial_features_72: [B, C, D, 72, 72]
        """
        batch_size, num_classes, hidden_dim, height, width = encoder_features_72.shape

        if (height, width) != (72, 72):
            raise ValueError(
                f"MultiScaleImageScoreWindowAttention expects 72x72 input, "
                f"got {(height, width)}."
            )

        features_36, features_18 = self._downsample_encoder_features(
            encoder_features_72,
        )

        sam_image_last_36 = F.interpolate(
            sam_image_last_72,
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )
        sam_image_last_18 = F.interpolate(
            sam_image_last_72,
            size=(18, 18),
            mode="bilinear",
            align_corners=False,
        )

        clip_score_embed_18 = clip_score_embeds["scale_18"]
        clip_score_embed_36 = clip_score_embeds["scale_36"]

        # 18×18 stage: regular + shifted window attention
        qk_18 = self._build_image_score_qk(
            encoder_features=features_18,
            sam_image_features=sam_image_last_18,
            clip_score_embed=clip_score_embed_18,
        )
        features_18 = self.attn_18_regular(
            qk_features=qk_18,
            value_features=features_18,
        )

        qk_18 = self._build_image_score_qk(
            encoder_features=features_18,
            sam_image_features=sam_image_last_18,
            clip_score_embed=clip_score_embed_18,
        )
        features_18 = self.attn_18_shifted(
            qk_features=qk_18,
            value_features=features_18,
        )

        # Upsample 18 result to 36, add to features_36, norm
        features_18_to_36 = self._upsample_18_to_36(
            features_18,
            target_hw=(36, 36),
        )

        features_36 = features_36 + features_18_to_36
        features_36 = self._norm_bcdhw(features_36, self.norm_18_fused_36)

        # 36×36 stage: regular + shifted window attention
        qk_36 = self._build_image_score_qk(
            encoder_features=features_36,
            sam_image_features=sam_image_last_36,
            clip_score_embed=clip_score_embed_36,
        )
        features_36 = self.attn_36_regular(
            qk_features=qk_36,
            value_features=features_36,
        )

        qk_36 = self._build_image_score_qk(
            encoder_features=features_36,
            sam_image_features=sam_image_last_36,
            clip_score_embed=clip_score_embed_36,
        )
        features_36 = self.attn_36_shifted(
            qk_features=qk_36,
            value_features=features_36,
        )

        # Upsample 36 result to 72, norm
        features_72 = self._upsample_36_to_72(
            features_36,
            target_hw=(72, 72),
        )
        features_72 = self._norm_bcdhw(features_72, self.norm_36_to_72)

        return features_72


# ---------------------------------------------------------------------------
# EncoderRefinerLayer
# ---------------------------------------------------------------------------


class EncoderRefinerLayer(nn.Module):
    """
    One refiner layer:

        1. ClassTokenScoreClassAttention at 72x72
        2. MultiScaleImageScoreWindowAttention (serial 18→36→72)
        3. FFN
        4. LayerNorm

    No layer-level residual inside the refiner layers.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.class_attn = ClassTokenScoreClassAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.multiscale_spatial_attn = MultiScaleImageScoreWindowAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
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
        clip_score_embed_18: torch.Tensor,
        clip_score_embed_36: torch.Tensor,
        clip_score_embed_72: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features_72: [B, C, D, 72, 72]
            sam_text_mean:       [B, C, D]
            sam_image_last_72:   [B, D, 72, 72]
            clip_score_embed_18: [B, C, D_score, 18, 18]
            clip_score_embed_36: [B, C, D_score, 36, 36]
            clip_score_embed_72: [B, C, D_score, 72, 72]

        Returns:
            encoder_features_72: [B, C, D, 72, 72]
        """
        class_attended_features_72 = self.class_attn(
            encoder_features=encoder_features_72,
            sam_text_mean=sam_text_mean,
            clip_score_embed=clip_score_embed_72,
        )

        clip_score_embeds = {
            "scale_18": clip_score_embed_18,
            "scale_36": clip_score_embed_36,
            "scale_72": clip_score_embed_72,
        }

        spatial_features_72 = self.multiscale_spatial_attn(
            encoder_features_72=class_attended_features_72,
            sam_image_last_72=sam_image_last_72,
            clip_score_embeds=clip_score_embeds,
        )

        spatial_fused_features_72 = class_attended_features_72 + spatial_features_72
        output_features_72 = self._ffn(spatial_fused_features_72)

        return self._output_layer_norm(output_features_72)
