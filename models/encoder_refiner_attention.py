from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .refiner_spatial_attention import GlobalFeatureAttention, LocalScoreAttention


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
    """
    Inter-class attention at each spatial position with dual value updates.

    feature, score_embed and sam_text_mean are pre-normalized by the outer layer.
    q, k and both value paths are produced from the normalized inputs.

    q/k = concat(feature, sam_text_mean, score_embed)  → 768 dims
    v_feature = feature
    v_score   = score_embed

    Attention happens across C classes at every spatial position.
    Returns feature_update and score_update (no residual, no LayerNorm).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
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

        qk_in_dim = self.hidden_dim * 2 + self.score_embed_dim  # 256+256+256=768

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)

        self.v_feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_score_proj = nn.Linear(self.score_embed_dim, self.hidden_dim)

        self.out_feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_score_proj = nn.Linear(self.hidden_dim, self.score_embed_dim)

        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        feature: torch.Tensor,
        score_embed: torch.Tensor,
        sam_text_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature:       [B, C, D, H, W]  pre-normalized
            score_embed:   [B, C, D_score, H, W]  pre-normalized
            sam_text_mean: [B, C, D]  pre-normalized

        Returns:
            feature_update: [B, C, D, H, W]
            score_update:   [B, C, D_score, H, W]
        """
        B, C, D, H, W = feature.shape
        D_score = self.score_embed_dim

        if tuple(score_embed.shape) != (B, C, D_score, H, W):
            raise ValueError(
                f"score_embed must be [{B}, {C}, {D_score}, {H}, {W}], "
                f"got {tuple(score_embed.shape)}"
            )
        if tuple(sam_text_mean.shape) != (B, C, D):
            raise ValueError(
                f"sam_text_mean must be [{B}, {C}, {D}], "
                f"got {tuple(sam_text_mean.shape)}"
            )

        N = H * W

        # Flatten spatial dims into batch for per-position attention.
        # feature: [B, C, D, H, W] → [B*N, C, D]
        f_flat = feature.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)

        # score_embed: [B, C, D_score, H, W] → [B*N, C, D_score]
        s_flat = score_embed.permute(0, 3, 4, 1, 2).reshape(B * N, C, D_score)

        # Broadcast sam_text_mean to each spatial position.
        text_broadcast = (
            sam_text_mean.to(device=f_flat.device, dtype=f_flat.dtype)[:, None]
            .expand(B, N, C, D)
            .reshape(B * N, C, D)
        )

        # q/k from concat of pre-normalized inputs.
        qk_input = torch.cat([f_flat, text_broadcast, s_flat], dim=-1)  # [B*N, C, 768]

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v_feat = self.v_feature_proj(f_flat)
        v_score = self.v_score_proj(s_flat)

        head_dim = D // self.num_heads
        q = q.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v_feat = v_feat.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v_score = v_score.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out_feat = torch.matmul(attn, v_feat)
        out_feat = out_feat.permute(0, 2, 1, 3).reshape(B * N, C, D)
        out_feat = self.out_feature_proj(out_feat)
        out_feat = self.dropout(out_feat)

        out_score = torch.matmul(attn, v_score)
        out_score = out_score.permute(0, 2, 1, 3).reshape(B * N, C, D)
        out_score = self.out_score_proj(out_score)
        out_score = self.dropout(out_score)

        feature_update = out_feat.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()
        score_update = out_score.reshape(B, H, W, C, D_score).permute(0, 3, 4, 1, 2).contiguous()

        return feature_update, score_update


class EncoderRefinerLayer(nn.Module):
    """Class attention → global context → local 3×3 updates → dual FFN.

    Each attention/FFN updates the current stream by a direct residual.
    Global context is computed once per layer and reused without detach.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        local_attn_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.class_attn = ClassScoreAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        if local_attn_steps <= 0:
            raise ValueError("local_attn_steps must be positive")
        self.global_attn = GlobalFeatureAttention(hidden_dim, num_heads, dropout)
        self.local_attns = nn.ModuleList([
            LocalScoreAttention(hidden_dim, score_embed_dim, num_heads, dropout)
            for _ in range(local_attn_steps)
        ])

        # Pre-norm for class attention.
        self.class_norm_feature = nn.LayerNorm(hidden_dim)
        self.class_norm_score = nn.LayerNorm(score_embed_dim)
        self.class_norm_text = nn.LayerNorm(hidden_dim)

        # Pre-norm for FFN.
        self.ffn_norm_feature = nn.LayerNorm(hidden_dim)
        self.ffn_norm_score = nn.LayerNorm(score_embed_dim)

        # Per-token FFN for feature.
        self.ffn_fc1_feature = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn_fc2_feature = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_dropout_feature = nn.Dropout(float(dropout))

        # Per-token FFN for score.
        self.ffn_fc1_score = nn.Linear(score_embed_dim, score_embed_dim * 4)
        self.ffn_fc2_score = nn.Linear(score_embed_dim * 4, score_embed_dim)
        self.ffn_dropout_score = nn.Dropout(float(dropout))

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
        sam_text_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_36:      [B, C, 256, 36, 36]
            score_embed_36:  [B, C, 256, 36, 36]
            sam_text_mean:   [B, C, 256]

        Returns:
            feature_36:      [B, C, 256, 36, 36]
            score_embed_36:  [B, C, 256, 36, 36]
        """
        # Class attention: pre-norm → attention → direct residual.
        class_feature = apply_layer_norm_bcdhw(
            feature_36,
            self.class_norm_feature,
        )
        class_score = apply_layer_norm_bcdhw(
            score_embed_36,
            self.class_norm_score,
        )
        class_text = self.class_norm_text(sam_text_mean)

        feature_update, score_update = self.class_attn(
            feature=class_feature,
            score_embed=class_score,
            sam_text_mean=class_text,
        )

        feature_36 = feature_36 + feature_update
        score_embed_36 = score_embed_36 + score_update

        # Global context comes from the post-class-attention feature.
        context_36 = self.global_attn(feature_36)
        for local_attn in self.local_attns:
            feature_update, score_update = local_attn(feature_36, score_embed_36, context_36)
            feature_36 = feature_36 + feature_update
            score_embed_36 = score_embed_36 + score_update

        # Feature FFN.
        ffn_feature = apply_layer_norm_bcdhw(
            feature_36,
            self.ffn_norm_feature,
        )
        feature_update = self._ffn_feature_update(ffn_feature)
        feature_36 = feature_36 + feature_update

        # Score FFN.
        ffn_score = apply_layer_norm_bcdhw(
            score_embed_36,
            self.ffn_norm_score,
        )
        score_update = self._ffn_score_update(ffn_score)
        score_embed_36 = score_embed_36 + score_update

        return feature_36, score_embed_36

