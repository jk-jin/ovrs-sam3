from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    eps: float = 1e-6


class SemanticCriterion(nn.Module):
    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

    def _extract_logits(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "semantic_logits" not in outputs:
            raise ValueError("SemanticCriterion expects outputs['semantic_logits'].")

        logits = outputs["semantic_logits"]
        if logits.dim() != 4:
            raise ValueError(
                f"Expected semantic_logits as [B, C, H, W], got {tuple(logits.shape)}"
            )
        return logits

    def _extract_label_map(
        self,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "label_map" not in targets:
            raise ValueError("SemanticCriterion expects targets['label_map'].")

        label_map = targets["label_map"]
        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    f"Expected label_map as [B,1,H,W] or [B,H,W], got {tuple(label_map.shape)}"
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                f"Expected label_map as [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}"
            )

        return label_map.long()

    def _resize_label_map_to_logits(
        self,
        label_map: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tuple(label_map.shape[-2:]) == tuple(target_hw):
            return label_map

        resized = F.interpolate(
            label_map[:, None].float(),
            size=target_hw,
            mode="nearest",
        )[:, 0]
        return resized.long()

    def _build_chunk_targets(
        self,
        label_map: torch.Tensor,
        chunk_class_ids: Sequence[int],
        num_channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(chunk_class_ids) != num_channels:
            raise ValueError(
                f"chunk_class_ids length mismatch: expected {num_channels}, got {len(chunk_class_ids)}"
            )

        bsz, h, w = label_map.shape
        device = label_map.device

        valid_mask = label_map != int(self.cfg.ignore_index)
        target = torch.zeros((bsz, num_channels, h, w), dtype=torch.float32, device=device)

        for ch, class_id in enumerate(chunk_class_ids):
            target[:, ch] = (label_map == int(class_id)).to(torch.float32)

        valid_mask_4d = valid_mask[:, None].expand(bsz, num_channels, h, w)
        return target, valid_mask_4d

    def _binary_cross_entropy_sum(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_pixel = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        per_pixel = per_pixel * valid_mask.to(per_pixel.dtype)
        return per_pixel.sum()

    def _dice_loss_sum(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        prob = logits.sigmoid()
        prob = prob * valid_mask.to(prob.dtype)
        target = target * valid_mask.to(target.dtype)

        prob = prob.flatten(2)
        target = target.flatten(2)

        intersection = (prob * target).sum(dim=2)
        denominator = prob.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.cfg.eps) / (denominator + self.cfg.eps)
        dice_loss = 1.0 - dice
        return dice_loss.sum()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "sum",
    ) -> Dict[str, torch.Tensor]:
        if reduction != "sum":
            raise ValueError(
                f"SemanticCriterion only supports reduction='sum', got {reduction!r}"
            )

        logits = self._extract_logits(outputs)
        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_logits(label_map, target_hw=tuple(logits.shape[-2:]))

        num_channels = int(logits.shape[1])
        if chunk_class_ids is None:
            raise ValueError(
                "SemanticCriterion requires chunk_class_ids for chunk-wise semantic training."
            )

        target, valid_mask = self._build_chunk_targets(
            label_map=label_map,
            chunk_class_ids=chunk_class_ids,
            num_channels=num_channels,
        )

        num_valid_pixels = int((label_map != int(self.cfg.ignore_index)).sum().item())
        if num_valid_pixels <= 0:
            zero = logits.sum() * 0.0
            return {
                "loss_semantic_bce": zero,
                "loss_semantic_dice": zero,
                "total_loss": zero,
                "num_valid": torch.tensor(0, device=logits.device, dtype=torch.long),
            }

        loss_semantic_bce = self._binary_cross_entropy_sum(
            logits=logits,
            target=target,
            valid_mask=valid_mask,
        )

        loss_semantic_dice = self._dice_loss_sum(
            logits=logits,
            target=target,
            valid_mask=valid_mask,
        )

        total_loss = (
            float(self.cfg.bce_weight) * loss_semantic_bce
            + float(self.cfg.dice_weight) * loss_semantic_dice
        )

        return {
            "loss_semantic_bce": loss_semantic_bce,
            "loss_semantic_dice": loss_semantic_dice,
            "total_loss": total_loss,
            "num_valid": torch.tensor(
                num_valid_pixels,
                device=logits.device,
                dtype=torch.long,
            ),
        }


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "sum",
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("HybridCriterion is not implemented yet.")