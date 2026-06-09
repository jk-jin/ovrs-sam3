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


class ConvDownsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BilinearConvUpsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
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

    q/k = concat(encoder_features, class_token_mean, clip_score_embed)
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

        self.class_token_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
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
        class_query_tokens: torch.Tensor,
        clip_score_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features:   [B, C, D, H, W]
            class_query_tokens: [B, C, Q, D]
            clip_score_embed:   [B, C, D_score, H, W]

        Returns:
            encoder_features: [B, C, D, H, W]
        """
        B, C, D, H, W = encoder_features.shape
        Q = class_query_tokens.shape[2]

        if class_query_tokens.ndim != 4:
            raise ValueError(
                f"class_query_tokens must be 4D [B, C, Q, D], "
                f"got {tuple(class_query_tokens.shape)}"
            )
        if class_query_tokens.shape[0] != B or class_query_tokens.shape[1] != C:
            raise ValueError(
                f"class_query_tokens batch/class mismatch: "
                f"expected [{B}, {C}], got [{class_query_tokens.shape[0]}, {class_query_tokens.shape[1]}]"
            )
        if class_query_tokens.shape[-1] != D:
            raise ValueError(
                f"class_query_tokens last dim must be {D}, "
                f"got {class_query_tokens.shape[-1]}"
            )
        if tuple(clip_score_embed.shape) != (B, C, self.score_embed_dim, H, W):
            raise ValueError(
                f"clip_score_embed must be [{B}, {C}, {self.score_embed_dim}, {H}, {W}], "
                f"got {tuple(clip_score_embed.shape)}"
            )

        N = H * W

        e_flat = encoder_features.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)

        class_token_mean = self.class_token_proj(class_query_tokens).mean(dim=2)
        class_token_broadcast = (
            class_token_mean[:, None]
            .expand(B, N, C, D)
            .reshape(B * N, C, D)
        )

        score_flat = (
            clip_score_embed
            .permute(0, 3, 4, 1, 2)
            .reshape(B * N, C, self.score_embed_dim)
        )

        qk_input = torch.cat([e_flat, class_token_broadcast, score_flat], dim=-1)

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
# ImageScoreWindowAttention
# ---------------------------------------------------------------------------


class ImageScoreWindowAttention(nn.Module):
    """
    Intra-class window attention.

    q/k = concat(encoder_features, sam_image_features, clip_image_features, clip_score_embed)
    v   = encoder_features
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
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

        self.clip_image_proj = nn.Sequential(
            nn.Conv2d(self.clip_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
        )

        qk_in_dim = self.hidden_dim * 3 + self.score_embed_dim

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

    def _prepare_clip_image_features(
        self,
        clip_image_features: torch.Tensor,
        target_hw: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        H, W = target_hw
        clip_image_features = clip_image_features.to(dtype=dtype)

        if clip_image_features.shape[-2:] != (H, W):
            clip_image_features = F.interpolate(
                clip_image_features,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        return self.clip_image_proj(clip_image_features)

    def forward(
        self,
        encoder_features: torch.Tensor,
        sam_image_features: torch.Tensor,
        clip_image_features: torch.Tensor,
        clip_score_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features:    [B, C, D, H, W]
            sam_image_features:  [B, D, H, W]
            clip_image_features: [B, D_clip, Hc, Wc]
            clip_score_embed:    [B, C, D_score, H, W]

        Returns:
            encoder_features: [B, C, D, H, W]
        """
        B, C, D, H, W = encoder_features.shape

        if tuple(sam_image_features.shape) != (B, D, H, W):
            raise ValueError(
                f"sam_image_features must be [{B}, {D}, {H}, {W}], "
                f"got {tuple(sam_image_features.shape)}"
            )
        if clip_image_features.ndim != 4:
            raise ValueError(
                f"clip_image_features must be [B, D_clip, Hc, Wc], "
                f"got {tuple(clip_image_features.shape)}"
            )
        if clip_image_features.shape[0] != B:
            raise ValueError(
                f"clip_image_features batch mismatch: expected {B}, "
                f"got {clip_image_features.shape[0]}"
            )
        if clip_image_features.shape[1] != self.clip_dim:
            raise ValueError(
                f"clip_image_features channel mismatch: expected {self.clip_dim}, "
                f"got {clip_image_features.shape[1]}"
            )
        if tuple(clip_score_embed.shape) != (B, C, self.score_embed_dim, H, W):
            raise ValueError(
                f"clip_score_embed must be "
                f"[{B}, {C}, {self.score_embed_dim}, {H}, {W}], "
                f"got {tuple(clip_score_embed.shape)}"
            )

        bc = B * C

        clip_image_up = self._prepare_clip_image_features(
            clip_image_features=clip_image_features,
            target_hw=(H, W),
            dtype=encoder_features.dtype,
        )

        e_flat = encoder_features.reshape(bc, D, H, W)
        score_flat = clip_score_embed.reshape(bc, self.score_embed_dim, H, W)

        sam_flat = (
            sam_image_features[:, None]
            .expand(B, C, D, H, W)
            .reshape(bc, D, H, W)
        )

        clip_flat = (
            clip_image_up[:, None]
            .expand(B, C, D, H, W)
            .reshape(bc, D, H, W)
        )

        e_flat, orig_h, orig_w = self._pad_to_window(e_flat, self.window_size)
        score_flat, _, _ = self._pad_to_window(score_flat, self.window_size)
        sam_flat, _, _ = self._pad_to_window(sam_flat, self.window_size)
        clip_flat, _, _ = self._pad_to_window(clip_flat, self.window_size)

        pad_h, pad_w = e_flat.shape[-2], e_flat.shape[-1]

        if self.shift_size > 0:
            shift = self.shift_size
            e_flat = torch.roll(e_flat, shifts=(-shift, -shift), dims=(-2, -1))
            score_flat = torch.roll(score_flat, shifts=(-shift, -shift), dims=(-2, -1))
            sam_flat = torch.roll(sam_flat, shifts=(-shift, -shift), dims=(-2, -1))
            clip_flat = torch.roll(clip_flat, shifts=(-shift, -shift), dims=(-2, -1))

        e_windows = self._window_partition(e_flat)
        score_windows = self._window_partition(score_flat)
        sam_windows = self._window_partition(sam_flat)
        clip_windows = self._window_partition(clip_flat)

        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=encoder_features.device,
            dtype=encoder_features.dtype,
        )

        qk_input = torch.cat(
            [e_windows, sam_windows, clip_windows, score_windows], dim=-1,
        )

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

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(-2, -1))

        out = out[:, :, :orig_h, :orig_w]
        return out.reshape(B, C, D, H, W).contiguous()


# ---------------------------------------------------------------------------
# MultiScaleImageScoreWindowAttention
# ---------------------------------------------------------------------------


class MultiScaleImageScoreWindowAttention(nn.Module):
    """
    Multi-scale intra-class window attention.

    Downsamples 72×72 features to 36×36 and 18×18,
    runs regular + shifted window attention at each scale,
    takes the update delta (after - before) and upsamples back to 72×72.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.down_72_to_36 = ConvDownsample2d(self.hidden_dim)
        self.down_36_to_18 = ConvDownsample2d(self.hidden_dim)

        self.attn_36_regular = ImageScoreWindowAttention(
            hidden_dim=hidden_dim,
            clip_dim=clip_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            dropout=dropout,
        )
        self.attn_36_shifted = ImageScoreWindowAttention(
            hidden_dim=hidden_dim,
            clip_dim=clip_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
        )

        self.attn_18_regular = ImageScoreWindowAttention(
            hidden_dim=hidden_dim,
            clip_dim=clip_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            dropout=dropout,
        )
        self.attn_18_shifted = ImageScoreWindowAttention(
            hidden_dim=hidden_dim,
            clip_dim=clip_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
        )

        self.up_36_to_72 = BilinearConvUpsample2d(self.hidden_dim)
        self.up_18_to_36 = BilinearConvUpsample2d(self.hidden_dim)
        self.up_36_from_18_to_72 = BilinearConvUpsample2d(self.hidden_dim)

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

    def _upsample_update_36_to_72(
        self,
        update_36: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        update_36_flat, batch_size, num_classes = flatten_batch_class(update_36)
        update_72_flat = self.up_36_to_72(update_36_flat, target_hw=target_hw)
        return unflatten_batch_class(update_72_flat, batch_size, num_classes)

    def _upsample_update_18_to_72(
        self,
        update_18: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        update_18_flat, batch_size, num_classes = flatten_batch_class(update_18)
        update_36_flat = self.up_18_to_36(update_18_flat, target_hw=(36, 36))
        update_72_flat = self.up_36_from_18_to_72(update_36_flat, target_hw=target_hw)
        return unflatten_batch_class(update_72_flat, batch_size, num_classes)

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        sam_image_last_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        clip_score_embeds: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            encoder_features_72: [B, C, D, 72, 72]
            sam_image_last_72:   [B, D, 72, 72]
            clip_image_feat_map: [B, D_clip, Hc, Wc]
            clip_score_embeds:   {"scale_18": ..., "scale_36": ..., "scale_72": ...}

        Returns:
            spatial_update_72: [B, C, D, 72, 72]
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

        features_36_before_attn = features_36
        features_18_before_attn = features_18

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

        clip_image_feat_36 = F.interpolate(
            clip_image_feat_map,
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )
        clip_image_feat_18 = F.interpolate(
            clip_image_feat_map,
            size=(18, 18),
            mode="bilinear",
            align_corners=False,
        )

        clip_score_embed_36 = clip_score_embeds["scale_36"]
        clip_score_embed_18 = clip_score_embeds["scale_18"]

        features_36 = self.attn_36_regular(
            encoder_features=features_36,
            sam_image_features=sam_image_last_36,
            clip_image_features=clip_image_feat_36,
            clip_score_embed=clip_score_embed_36,
        )
        features_36 = self.attn_36_shifted(
            encoder_features=features_36,
            sam_image_features=sam_image_last_36,
            clip_image_features=clip_image_feat_36,
            clip_score_embed=clip_score_embed_36,
        )

        features_18 = self.attn_18_regular(
            encoder_features=features_18,
            sam_image_features=sam_image_last_18,
            clip_image_features=clip_image_feat_18,
            clip_score_embed=clip_score_embed_18,
        )
        features_18 = self.attn_18_shifted(
            encoder_features=features_18,
            sam_image_features=sam_image_last_18,
            clip_image_features=clip_image_feat_18,
            clip_score_embed=clip_score_embed_18,
        )

        update_36 = features_36 - features_36_before_attn
        update_18 = features_18 - features_18_before_attn

        update_36_to_72 = self._upsample_update_36_to_72(
            update_36,
            target_hw=(72, 72),
        )
        update_18_to_72 = self._upsample_update_18_to_72(
            update_18,
            target_hw=(72, 72),
        )

        return update_36_to_72 + update_18_to_72


# ---------------------------------------------------------------------------
# EncoderRefinerLayer
# ---------------------------------------------------------------------------


class EncoderRefinerLayer(nn.Module):
    """
    One refiner layer:

        1. ClassTokenScoreClassAttention at 72×72
        2. MultiScaleImageScoreWindowAttention (36×36 + 18×18)
        3. FFN

    Layer-level residual: output = ffn_output + layer_identity
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        clip_dim: int = 768,
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
            clip_dim=clip_dim,
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
        class_query_tokens: torch.Tensor,
        sam_image_last_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        clip_score_embed_18: torch.Tensor,
        clip_score_embed_36: torch.Tensor,
        clip_score_embed_72: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features_72:  [B, C, D, 72, 72]
            class_query_tokens:   [B, C, Q, D]
            sam_image_last_72:    [B, D, 72, 72]
            clip_image_feat_map:  [B, D_clip, Hc, Wc]
            clip_score_embed_18:  [B, C, D_score, 18, 18]
            clip_score_embed_36:  [B, C, D_score, 36, 36]
            clip_score_embed_72:  [B, C, D_score, 72, 72]

        Returns:
            encoder_features_72: [B, C, D, 72, 72]
        """
        layer_identity = encoder_features_72

        class_attended_features_72 = self.class_attn(
            encoder_features=encoder_features_72,
            class_query_tokens=class_query_tokens,
            clip_score_embed=clip_score_embed_72,
        )

        clip_score_embeds = {
            "scale_18": clip_score_embed_18,
            "scale_36": clip_score_embed_36,
            "scale_72": clip_score_embed_72,
        }

        spatial_update_72 = self.multiscale_spatial_attn(
            encoder_features_72=class_attended_features_72,
            sam_image_last_72=sam_image_last_72,
            clip_image_feat_map=clip_image_feat_map,
            clip_score_embeds=clip_score_embeds,
        )

        spatial_fused_features_72 = class_attended_features_72 + spatial_update_72
        output_features_72 = self._ffn(spatial_fused_features_72)

        return output_features_72 + layer_identity
