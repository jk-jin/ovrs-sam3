from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# HighResEncoderClassAttention
# ---------------------------------------------------------------------------


class HighResEncoderClassAttention(nn.Module):
    """
    Inter-class attention on high-res encoder features, conditioned
    by clip guidance.

    q/k = concat(norm(encoder_features), norm(clip_guidance))
    v   = encoder_features

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
        encoder_features: torch.Tensor,
        clip_guidance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: [B, C, D, H, W]
            clip_guidance:    [B, C, D, H, W]

        Returns:
            encoder_features: [B, C, D, H, W]
        """
        B, C, D, H, W = encoder_features.shape

        if tuple(clip_guidance.shape) != (B, C, D, H, W):
            raise ValueError(
                f"clip_guidance must be [{B}, {C}, {D}, {H}, {W}], "
                f"got {tuple(clip_guidance.shape)}"
            )

        N = H * W

        e_flat = encoder_features.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)
        e_flat_norm = self.norm_qk(e_flat)

        g_flat = clip_guidance.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)
        g_flat_norm = self.norm_qk(g_flat)

        qk_input = torch.cat([e_flat_norm, g_flat_norm], dim=-1)

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
        out = self.norm_out(e_flat + self.dropout(out))

        return out.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# EncoderConvFFN
# ---------------------------------------------------------------------------


class EncoderConvFFN(nn.Module):
    """Depthwise conv FFN for high-res encoder features.

    Adds local spatial correction to complement inter-class attention.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]

        Returns:
            x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        residual = x

        x_norm = self.norm(x.permute(0, 1, 3, 4, 2))  # [B, C, H, W, D]
        x_norm = x_norm.permute(0, 1, 4, 2, 3).reshape(B * C, D, H, W)

        out = self.dw_conv(x_norm)
        out = F.gelu(out)
        out = self.fc1(out)
        out = F.gelu(out)
        out = self.fc2(out)
        out = self.dropout(out)

        out = out.reshape(B, C, D, H, W)
        return residual + out


# ---------------------------------------------------------------------------
# HighResEncoderRefinerLayer
# ---------------------------------------------------------------------------


class HighResEncoderRefinerLayer(nn.Module):
    """One high-res refiner layer: class attention + conv FFN."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.class_attn = HighResEncoderClassAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = EncoderConvFFN(
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
        encoder_features: torch.Tensor,
        clip_guidance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: [B, C, D, H, W]
            clip_guidance:    [B, C, D, H, W]

        Returns:
            encoder_features: [B, C, D, H, W]
        """
        x = self.class_attn(encoder_features, clip_guidance)
        x = self.ffn(x)
        return self._output_layer_norm(x)


# ---------------------------------------------------------------------------
# HighResEncoderRefiner (top-level)
# ---------------------------------------------------------------------------


class HighResEncoderRefiner(nn.Module):
    """
    Multi-layer high-resolution encoder refiner.

    Refines SAM3 encoder features at 72x72 using clip guidance
    as conditioning. Only inter-class attention + conv FFN,
    no window attention at this stage.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        highres_layers: int = 2,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.layers = nn.ModuleList([
            HighResEncoderRefinerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(int(highres_layers))
        ])

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        clip_guidance_72: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features_72: [B, C, D, 72, 72]
            clip_guidance_72:    [B, C, D, 72, 72]

        Returns:
            refined_encoder_features_72: [B, C, D, 72, 72]
        """
        x = encoder_features_72

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(
                    layer, x, clip_guidance_72, use_reentrant=False,
                )
            else:
                x = layer(
                    encoder_features=x,
                    clip_guidance=clip_guidance_72,
                )

        return x
