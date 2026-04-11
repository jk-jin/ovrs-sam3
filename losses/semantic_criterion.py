from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


@dataclass
class SemanticLossWeights:
    loss_ce: float = 1.0
    loss_dice: float = 0.0


class SemanticCriterion(nn.Module):
    """
    Multi-class semantic segmentation criterion.

    Expected outputs:
        fused_score_map: [B, C, H, W]

    Expected targets:
        label_map: [B, H, W]

    Supports two modes:
    1. Full-class mode:
       - chunk_class_ids is None
       - outputs channels correspond to the full class space

    2. Chunk mode:
       - chunk_class_ids is a list of global class ids covered by the current chunk
       - outputs channels correspond only to these chunk-local classes
       - target labels outside this chunk are mapped to ignore_index
    """

    def __init__(
        self,
        weights: SemanticLossWeights | None = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.weights = weights or SemanticLossWeights()
        self.ignore_index = int(ignore_index)

    def _prepare_target(
        self,
        targets: TensorDict,
        out_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if "label_map" not in targets:
            raise ValueError("label_map is required for multi-class semantic segmentation.")

        label_map = targets["label_map"]
        if label_map is None:
            raise ValueError("label_map is None.")

        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    f"Expected label_map as [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}"
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                f"Expected label_map as [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}"
            )

        label_map = label_map.long().to(device)

        if tuple(label_map.shape[-2:]) != tuple(out_hw):
            label_map = F.interpolate(
                label_map[:, None].float(),
                size=out_hw,
                mode="nearest",
            )[:, 0].long()

        return label_map

    def _remap_target_to_chunk(
        self,
        target: torch.Tensor,
        chunk_class_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(chunk_class_ids) == 0:
            raise ValueError("chunk_class_ids is empty.")

        local_target = torch.full_like(target, fill_value=self.ignore_index)

        seen = set()
        for local_id, global_id in enumerate(chunk_class_ids):
            global_id = int(global_id)
            if global_id in seen:
                raise ValueError(f"Duplicate class id found in chunk_class_ids: {global_id}")
            seen.add(global_id)

            local_target[target == global_id] = int(local_id)

        valid_mask = local_target != self.ignore_index
        return local_target, valid_mask

    def _multiclass_dice_loss(
        self,
        logits: torch.Tensor,   # [B, C, H, W]
        target: torch.Tensor,   # [B, H, W]
    ) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

        valid_mask = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)

        valid_classes = one_hot.sum(dim=dims) > 0
        if valid_classes.any():
            dice = (2.0 * intersection[valid_classes] + 1.0) / (denominator[valid_classes] + 1.0)
            return 1.0 - dice.mean()

        return logits.sum() * 0.0

    def forward(
        self,
        outputs: TensorDict,
        targets: TensorDict,
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
    ) -> TensorDict:
        if "fused_score_map" not in outputs:
            raise ValueError("fused_score_map is required in semantic outputs.")

        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction={reduction}. Expected 'mean' or 'sum'.")

        semantic_logits = outputs["fused_score_map"]  # [B, C, H, W]
        if semantic_logits.dim() != 4:
            raise ValueError(
                f"Expected fused_score_map as [B, C, H, W], got {tuple(semantic_logits.shape)}"
            )

        target = self._prepare_target(
            targets=targets,
            out_hw=semantic_logits.shape[-2:],
            device=semantic_logits.device,
        )

        if chunk_class_ids is not None:
            if semantic_logits.shape[1] != len(chunk_class_ids):
                raise ValueError(
                    f"Chunk class count mismatch: logits has {semantic_logits.shape[1]} channels, "
                    f"but chunk_class_ids has length {len(chunk_class_ids)}."
                )
            target, valid_mask = self._remap_target_to_chunk(target, chunk_class_ids)
        else:
            valid_mask = target != self.ignore_index

        num_valid = valid_mask.sum()
        if int(num_valid.item()) == 0:
            zero = semantic_logits.sum() * 0.0
            return {
                "loss_ce": zero,
                "loss_dice": zero,
                "total_loss": zero,
                "num_valid": num_valid,
            }

        ce_reduction = "sum" if reduction == "sum" else "mean"
        loss_ce = F.cross_entropy(
            semantic_logits,
            target,
            ignore_index=self.ignore_index,
            reduction=ce_reduction,
        )

        if self.weights.loss_dice > 0:
            dice_base = self._multiclass_dice_loss(semantic_logits, target)
            if reduction == "sum":
                loss_dice = dice_base * num_valid.to(dtype=semantic_logits.dtype)
            else:
                loss_dice = dice_base
        else:
            loss_dice = semantic_logits.sum() * 0.0

        total = self.weights.loss_ce * loss_ce + self.weights.loss_dice * loss_dice

        return {
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
            "total_loss": total,
            "num_valid": num_valid,
        }