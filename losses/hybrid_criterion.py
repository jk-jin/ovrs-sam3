from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .instance_criterion import InstanceCriterion
from .semantic_criterion import SemanticCriterion


TensorDict = Dict[str, Dict[str, torch.Tensor]]


@dataclass
class HybridLossWeights:
    instance: float = 1.0
    semantic: float = 1.0


class HybridCriterion(nn.Module):
    def __init__(
        self,
        instance_criterion: InstanceCriterion,
        semantic_criterion: SemanticCriterion,
        weights: HybridLossWeights | None = None,
    ):
        super().__init__()
        self.instance_criterion = instance_criterion
        self.semantic_criterion = semantic_criterion
        self.weights = weights or HybridLossWeights()

    def forward(self, outputs: TensorDict, instance_targets: Dict[str, torch.Tensor], semantic_targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inst_losses = self.instance_criterion(outputs["instance_outputs"], instance_targets)
        sem_losses = self.semantic_criterion(outputs["semantic_outputs"], semantic_targets)
        total = self.weights.instance * inst_losses["total_loss"] + self.weights.semantic * sem_losses["total_loss"]

        merged: Dict[str, torch.Tensor] = {"total_loss": total}
        merged.update({f"instance.{k}": v for k, v in inst_losses.items()})
        merged.update({f"semantic.{k}": v for k, v in sem_losses.items()})
        return merged
