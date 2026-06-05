from __future__ import annotations

import torch
import torch.nn as nn


class FinalMaskConvHead(nn.Module):
    """
    Shared final mask head.

    Input:
        x: [B, C, D, H, W]

    Output:
        final_logits: [B, C, H, W]
    """

    def __init__(
        self,
        in_ch: int,
        class_chunk_size: int = 4,
    ):
        super().__init__()
        self.class_chunk_size = int(class_chunk_size)
        self.head = nn.Conv2d(
            int(in_ch),
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"FinalMaskConvHead expects [B, C, D, H, W], got {tuple(x.shape)}."
            )

        B, C, D, H, W = x.shape
        outputs = []

        for start in range(0, C, self.class_chunk_size):
            end = min(start + self.class_chunk_size, C)
            x_chunk = x[:, start:end]
            chunk_size = end - start

            logits = self.head(
                x_chunk.reshape(B * chunk_size, D, H, W)
            ).reshape(B, chunk_size, H, W)

            outputs.append(logits)

        return torch.cat(outputs, dim=1).contiguous()