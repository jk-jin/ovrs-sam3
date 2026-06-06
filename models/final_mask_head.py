from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGnGelu(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        num_groups = min(8, out_ch)
        if out_ch % num_groups != 0:
            num_groups = 1

        self.block = nn.Sequential(
            nn.Conv2d(
                int(in_ch),
                int(out_ch),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups, int(out_ch)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FinalScoreFusionHead(nn.Module):
    """
    Fuse SAM3 semantic score embedding with upsampled features.

    Input:
        x_up:            [B, C, D_last, H, W]
        semantic_logits: [B, C, H, W]

    Flow:
        semantic_logits → sigmoid
          → ConvGnGelu(1→16) → ConvGnGelu(16→32)
          → final_score_embed [B*C, 32, H, W]

        concat(x_up, final_score_embed) → [B*C, 64, H, W]
          → 3 × ConvGnGelu(64→64)
          → Conv2d(64→1)
          → final_logits [B, C, H, W]
    """

    def __init__(
        self,
        in_ch: int,
        class_chunk_size: int = 4,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.class_chunk_size = int(class_chunk_size)

        if self.class_chunk_size <= 0:
            raise ValueError(
                f"class_chunk_size must be positive, got {class_chunk_size}."
            )

        # Score embedder: 1 → 16 → 32, stride 1 (full resolution)
        self.score_embed = nn.Sequential(
            ConvGnGelu(1, 16, kernel_size=3, stride=1),
            ConvGnGelu(16, 32, kernel_size=3, stride=1),
        )
        self.score_embed_dim = 32

        fused_ch = self.in_ch + self.score_embed_dim  # 32 + 32 = 64

        self.fusion = nn.Sequential(
            ConvGnGelu(fused_ch, fused_ch, kernel_size=3, stride=1),
            ConvGnGelu(fused_ch, fused_ch, kernel_size=3, stride=1),
            ConvGnGelu(fused_ch, fused_ch, kernel_size=3, stride=1),
            nn.Conv2d(fused_ch, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_up: torch.Tensor,
        semantic_logits: torch.Tensor,
    ) -> torch.Tensor:
        if x_up.dim() != 5:
            raise ValueError(
                f"FinalScoreFusionHead expects x_up as [B, C, D, H, W], "
                f"got {tuple(x_up.shape)}."
            )

        if semantic_logits.dim() != 4:
            raise ValueError(
                f"FinalScoreFusionHead expects semantic_logits as [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )

        B, C, D, H, W = x_up.shape

        if tuple(semantic_logits.shape) != (B, C, H, W):
            raise ValueError(
                f"semantic_logits shape mismatch: expected {(B, C, H, W)}, "
                f"got {tuple(semantic_logits.shape)}."
            )

        if D != self.in_ch:
            raise ValueError(
                f"x_up channel mismatch: expected {self.in_ch}, got {D}."
            )

        # Build score embedding: [B, C, H, W] → [B*C, 1, H, W] → [B*C, 32, H, W]
        score = torch.sigmoid(semantic_logits.detach())
        score_flat = score.reshape(B * C, 1, H, W)
        score_embed = self.score_embed(score_flat)  # [B*C, 32, H, W]

        # Fuse per class chunk
        outputs: list[torch.Tensor] = []

        for start in range(0, C, self.class_chunk_size):
            end = min(start + self.class_chunk_size, C)
            chunk_c = end - start

            x_chunk = x_up[:, start:end]  # [B, chunk_c, D, H, W]
            x_flat = x_chunk.reshape(B * chunk_c, D, H, W)

            score_chunk = score_embed[start * B:(start + chunk_c) * B]
            # Actually score_embed is [B*C, 32, H, W], so chunk it properly:
            score_chunk = score_embed.reshape(B, C, self.score_embed_dim, H, W)
            score_chunk = score_chunk[:, start:end]
            score_chunk = score_chunk.reshape(B * chunk_c, self.score_embed_dim, H, W)

            fused = torch.cat([x_flat, score_chunk], dim=1)  # [B*chunk_c, 64, H, W]

            logits_chunk = self.fusion(fused).reshape(B, chunk_c, H, W)
            outputs.append(logits_chunk)

        return torch.cat(outputs, dim=1).contiguous()
