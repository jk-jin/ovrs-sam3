from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


# ---------------------------------------------------------------------------
# LowResScoreClassAttention
# ---------------------------------------------------------------------------


class LowResScoreClassAttention(nn.Module):
    """
    Inter-class attention on low-res score embeddings.

    q/k = concat(norm(score_embed), norm(sam_text_mean broadcast))
    v   = score_embed

    Attention is across C classes at each spatial position.
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

        # q/k use score_embed + sam_text_mean
        qk_in_dim = self.hidden_dim * 2

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm_qk = nn.LayerNorm(self.hidden_dim)
        self.norm_out = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        score_embed: torch.Tensor,
        sam_text_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            score_embed:  [B, C, D, H, W]
            sam_text_mean: [B, C, D]

        Returns:
            score_embed: [B, C, D, H, W]
        """
        B, C, D, H, W = score_embed.shape

        if tuple(sam_text_mean.shape) != (B, C, D):
            raise ValueError(
                f"sam_text_mean must be [{B}, {C}, {D}], "
                f"got {tuple(sam_text_mean.shape)}"
            )

        N = H * W

        # Flatten spatial dims for class attention.
        s_flat = score_embed.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)
        s_flat_norm = self.norm_qk(s_flat)

        sam_text_broadcast = (
            sam_text_mean.to(device=s_flat.device, dtype=s_flat.dtype)
            .unsqueeze(1)
            .expand(B, N, C, D)
            .reshape(B * N, C, D)
        )
        sam_text_broadcast_norm = self.norm_qk(sam_text_broadcast)

        qk_input = torch.cat([s_flat_norm, sam_text_broadcast_norm], dim=-1)

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(s_flat)

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
        out = self.norm_out(s_flat + self.dropout(out))

        return out.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# LowResScoreWindowAttention
# ---------------------------------------------------------------------------


class LowResScoreWindowAttention(nn.Module):
    """
    Intra-class window attention on low-res score embeddings.

    q/k = concat(norm(score_embed), norm(clip_final_feat), norm(sam_fpn_feat))
    v   = score_embed
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 9,
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

        # q/k use score_embed + clip_final_feat + sam_fpn_feat
        qk_in_dim = self.hidden_dim * 3

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm_input = nn.LayerNorm(self.hidden_dim)
        self.norm_out = nn.LayerNorm(self.hidden_dim)
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
        score_embed: torch.Tensor,
        clip_final_feat: torch.Tensor,
        sam_fpn_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            score_embed:    [B, C, D, H, W]
            clip_final_feat: [B, D, H, W]
            sam_fpn_feat:   [B, D, H, W]

        Returns:
            score_embed: [B, C, D, H, W]
        """
        B, C, D, H, W = score_embed.shape

        if tuple(clip_final_feat.shape) != (B, D, H, W):
            raise ValueError(
                f"clip_final_feat must be [{B}, {D}, {H}, {W}], "
                f"got {tuple(clip_final_feat.shape)}"
            )
        if tuple(sam_fpn_feat.shape) != (B, D, H, W):
            raise ValueError(
                f"sam_fpn_feat must be [{B}, {D}, {H}, {W}], "
                f"got {tuple(sam_fpn_feat.shape)}"
            )

        bc = B * C

        # Flatten batch + class.
        s_flat = score_embed.reshape(bc, D, H, W)
        clip_flat = clip_final_feat.unsqueeze(1).expand(B, C, D, H, W).reshape(bc, D, H, W)
        sam_flat = sam_fpn_feat.unsqueeze(1).expand(B, C, D, H, W).reshape(bc, D, H, W)

        # Pad to window size.
        s_flat, orig_h, orig_w = self._pad_to_window(s_flat, self.window_size)
        clip_flat, _, _ = self._pad_to_window(clip_flat, self.window_size)
        sam_flat, _, _ = self._pad_to_window(sam_flat, self.window_size)

        pad_h, pad_w = s_flat.shape[-2], s_flat.shape[-1]
        shift = self.shift_size

        if shift > 0:
            s_flat = torch.roll(s_flat, shifts=(-shift, -shift), dims=(-2, -1))
            clip_flat = torch.roll(clip_flat, shifts=(-shift, -shift), dims=(-2, -1))
            sam_flat = torch.roll(sam_flat, shifts=(-shift, -shift), dims=(-2, -1))

        # Window partition.
        s_windows = self._window_partition(s_flat)
        clip_windows = self._window_partition(clip_flat)
        sam_windows = self._window_partition(sam_flat)

        # Build attention mask for shifted windows.
        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=score_embed.device,
            dtype=score_embed.dtype,
        )

        # Normalize each source.
        s_windows_norm = self.norm_input(s_windows)
        clip_windows_norm = self.norm_input(clip_windows)
        sam_windows_norm = self.norm_input(sam_windows)

        qk_input = torch.cat(
            [s_windows_norm, clip_windows_norm, sam_windows_norm], dim=-1,
        )

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(s_windows)

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
        out = self.norm_out(s_windows + self.dropout(out))

        out = self._window_reverse(out, bc, pad_h, pad_w)

        if shift > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(-2, -1))

        out = out[:, :, :orig_h, :orig_w]
        return out.reshape(B, C, D, H, W).contiguous()


# ---------------------------------------------------------------------------
# ScoreConvFFN
# ---------------------------------------------------------------------------


class ScoreConvFFN(nn.Module):
    """Lightweight depthwise conv FFN for score embeddings."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.norm = nn.LayerNorm(self.hidden_dim)

        self.dw_conv = nn.Conv2d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=3, padding=1, groups=self.hidden_dim,
        )
        self.fc1 = nn.Conv2d(self.hidden_dim, self.hidden_dim * 4, kernel_size=1)
        self.fc2 = nn.Conv2d(self.hidden_dim * 4, self.hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, score_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_embed: [B, C, D, H, W]

        Returns:
            score_embed: [B, C, D, H, W]
        """
        B, C, D, H, W = score_embed.shape
        residual = score_embed

        # Pre-norm via LayerNorm on D dimension.
        x = score_embed.permute(0, 1, 3, 4, 2)  # [B, C, H, W, D]
        x = self.norm(x)
        x = x.permute(0, 1, 4, 2, 3).reshape(B * C, D, H, W)  # [B*C, D, H, W]

        x = self.dw_conv(x)
        x = F.gelu(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        x = x.reshape(B, C, D, H, W)
        return residual + x


# ---------------------------------------------------------------------------
# LowResScoreRefinerLayer
# ---------------------------------------------------------------------------


class LowResScoreRefinerLayer(nn.Module):
    """
    One low-res refiner layer:

        1. LowResScoreClassAttention
        2. LowResScoreWindowAttention (regular)
        3. LowResScoreWindowAttention (shifted)
        4. ScoreConvFFN
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.class_attn = LowResScoreClassAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.win_attn_regular = LowResScoreWindowAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            dropout=dropout,
        )

        self.win_attn_shifted = LowResScoreWindowAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
        )

        self.ffn = ScoreConvFFN(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def _output_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()

    def forward(
        self,
        score_embed: torch.Tensor,
        sam_text_mean: torch.Tensor,
        clip_final_feat: torch.Tensor,
        sam_fpn_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            score_embed:     [B, C, D, H, W]
            sam_text_mean:   [B, C, D]
            clip_final_feat: [B, D, H, W]
            sam_fpn_feat:    [B, D, H, W]

        Returns:
            score_embed: [B, C, D, H, W]
        """
        score_embed = self.class_attn(score_embed, sam_text_mean)
        score_embed = self.win_attn_regular(score_embed, clip_final_feat, sam_fpn_feat)
        score_embed = self.win_attn_shifted(score_embed, clip_final_feat, sam_fpn_feat)
        score_embed = self.ffn(score_embed)
        return self._output_layer_norm(score_embed)


# ---------------------------------------------------------------------------
# LowResScoreRefiner (top-level)
# ---------------------------------------------------------------------------


class LowResScoreRefiner(nn.Module):
    """
    Multi-layer low-resolution score refiner.

    Refines the 18x18 score embeddings with inter-class and
    intra-class window attention across 4 layers.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        lowres_layers: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.layers = nn.ModuleList([
            LowResScoreRefinerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                dropout=dropout,
            )
            for _ in range(int(lowres_layers))
        ])

    def forward(
        self,
        lowres_score_embed: torch.Tensor,
        sam_text_mean: torch.Tensor,
        clip_final_feat_18: torch.Tensor,
        sam_fpn_feat_18: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lowres_score_embed: [B, C, D, 18, 18]
            sam_text_mean:      [B, C, D]
            clip_final_feat_18: [B, D, 18, 18]
            sam_fpn_feat_18:    [B, D, 18, 18]

        Returns:
            refined_score_embed_18: [B, C, D, 18, 18]
        """
        score_embed = lowres_score_embed

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                score_embed = checkpoint(
                    layer,
                    score_embed,
                    sam_text_mean,
                    clip_final_feat_18,
                    sam_fpn_feat_18,
                    use_reentrant=False,
                )
            else:
                score_embed = layer(
                    score_embed=score_embed,
                    sam_text_mean=sam_text_mean,
                    clip_final_feat=clip_final_feat_18,
                    sam_fpn_feat=sam_fpn_feat_18,
                )

        return score_embed
