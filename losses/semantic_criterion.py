from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


@dataclass
class SemanticLossWeights:
    loss_bce: float = 1.0
    loss_dice: float = 1.0


class SemanticCriterion(nn.Module):
    """Binary text-conditioned semantic segmentation criterion.

    Expected outputs:
        semantic_logits: [B, 1, H, W]

    Expected targets:
        semantic_masks: [B, H, W] or [B, 1, H, W] or [B, N, H, W]
    """

    def __init__(self, weights: SemanticLossWeights | None = None):
        super().__init__()
        self.weights = weights or SemanticLossWeights()

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid().flatten(1)
        targets = targets.float().flatten(1)
        numerator = 2 * (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        return (1.0 - (numerator + 1.0) / (denominator + 1.0)).mean()

    @staticmethod
    def _prepare_target(targets: TensorDict, out_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
        semantic_masks = targets["semantic_masks"]
        if semantic_masks is None:
            raise ValueError("semantic_masks is required for semantic training")

        if semantic_masks.dim() == 2:
            semantic_masks = semantic_masks.unsqueeze(0).unsqueeze(0)
        elif semantic_masks.dim() == 3:
            semantic_masks = semantic_masks.unsqueeze(1)
        elif semantic_masks.dim() == 4:
            if semantic_masks.shape[1] != 1:
                semantic_masks = semantic_masks.any(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported semantic mask shape: {tuple(semantic_masks.shape)}")

        semantic_masks = semantic_masks.float().to(device)
        if semantic_masks.shape[-2:] != out_hw:
            semantic_masks = F.interpolate(semantic_masks, size=out_hw, mode="nearest")
        return semantic_masks

    def forward(self, outputs: TensorDict, targets: TensorDict) -> TensorDict:
        semantic_logits = outputs["semantic_logits"]
        target_mask = self._prepare_target(
            targets=targets,
            out_hw=semantic_logits.shape[-2:],
            device=semantic_logits.device,
        )

        loss_bce = F.binary_cross_entropy_with_logits(semantic_logits, target_mask)
        loss_dice = self._dice_loss(semantic_logits, target_mask)
        total = self.weights.loss_bce * loss_bce + self.weights.loss_dice * loss_dice

        return {
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "total_loss": total,
        }
