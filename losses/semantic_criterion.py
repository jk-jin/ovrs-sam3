from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


class SemanticCriterion(nn.Module):
    """Semantic criterion: final BCE + present-only Dice."""

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

    def forward(
        self,
        outputs: TensorDict,
        targets: TensorDict,
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
    ) -> TensorDict:
        if reduction != "mean":
            raise ValueError(
                f"SemanticCriterion only supports reduction='mean', got {reduction!r}."
            )

        if OUTPUT_KEYS.final_logits not in outputs:
            raise ValueError(
                "SemanticCriterion requires outputs['final_logits']."
            )

        return self._forward_final(outputs=outputs, targets=targets)

    def _forward_final(
        self,
        outputs: TensorDict,
        targets: TensorDict,
    ) -> TensorDict:
        final_logits = self._extract_4d_tensor(
            outputs, OUTPUT_KEYS.final_logits, "[B, C, H, W]"
        )

        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_hw(
            label_map=label_map,
            target_hw=tuple(final_logits.shape[-2:]),
        )

        num_channels = int(final_logits.shape[1])
        class_ids = list(range(num_channels))

        target, valid_mask = self._build_binary_targets(
            label_map=label_map,
            class_ids=class_ids,
            num_channels=num_channels,
        )

        present_pair_mask = self._build_present_pair_mask(
            target=target,
            valid_mask=valid_mask,
        )

        num_loss_pixels = int(label_map.numel())
        zero = self._zero_loss(final_logits)

        if num_loss_pixels <= 0:
            return {
                "loss_final_bce": zero,
                "loss_final_dice": zero,
                "total_loss": zero,
                "num_valid": torch.tensor(
                    0,
                    device=final_logits.device,
                    dtype=torch.long,
                ),
            }

        loss_final_bce = self._binary_cross_entropy_present_absent_mean(
            logits=final_logits,
            target=target,
            present_pair_mask=present_pair_mask,
            absent_weight=float(self.cfg.bce_absent_class_weight),
        )

        if present_pair_mask.any():
            loss_final_dice = self._dice_loss_present_mean_from_logits(
                logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
        else:
            loss_final_dice = zero

        total_loss = (
            float(self.cfg.final_bce_weight) * loss_final_bce
            + float(self.cfg.final_dice_weight) * loss_final_dice
        )

        return {
            "loss_final_bce": loss_final_bce,
            "loss_final_dice": loss_final_dice,
            "total_loss": total_loss,
            "num_valid": torch.tensor(
                num_loss_pixels,
                device=final_logits.device,
                dtype=torch.long,
            ),
        }

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _extract_4d_tensor(outputs: TensorDict, key: str, shape_name: str) -> torch.Tensor:
        tensor = outputs.get(key, None)
        if tensor is None:
            raise ValueError(f"SemanticCriterion expects outputs['{key}'].")
        if tensor.dim() != 4:
            raise ValueError(f"Expected {key} as {shape_name}, got {tuple(tensor.shape)}.")
        return tensor

    def _extract_label_map(self, targets: TensorDict) -> torch.Tensor:
        if "label_map" not in targets:
            raise ValueError("SemanticCriterion expects targets['label_map'].")
        label_map = targets["label_map"]
        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    "Expected label_map as [B, 1, H, W] or [B, H, W]"
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                "Expected label_map as [B, H, W] or [B, 1, H, W]"
            )
        return label_map.long()

    @staticmethod
    def _resize_label_map_to_hw(label_map: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        if tuple(label_map.shape[-2:]) == tuple(target_hw):
            return label_map
        return F.interpolate(
            label_map[:, None].float(),
            size=target_hw,
            mode="nearest",
        )[:, 0].long()

    def _build_binary_targets(
        self, label_map: torch.Tensor, class_ids: Sequence[int], num_channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, W = label_map.shape
        valid_mask = label_map != int(self.cfg.ignore_index)
        target = torch.zeros((B, num_channels, H, W), dtype=torch.float32, device=label_map.device)
        for channel_idx, class_id in enumerate(class_ids):
            target[:, channel_idx] = (label_map == int(class_id)).float()
        valid_mask_4d = valid_mask[:, None].expand(B, num_channels, H, W)
        return target, valid_mask_4d

    @staticmethod
    def _build_present_pair_mask(target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        target_valid = target * valid_mask.to(dtype=target.dtype)
        fg_pixels_per_pair = target_valid.flatten(2).sum(dim=2)
        return fg_pixels_per_pair > 0

    @staticmethod
    def _binary_cross_entropy_present_absent_mean(
        logits: torch.Tensor,
        target: torch.Tensor,
        present_pair_mask: torch.Tensor,
        absent_weight: float,
    ) -> torch.Tensor:
        """
        BCE with fixed present/absent image-class pair weights.

        Present classes:
            weight = 1.0

        Absent classes:
            weight = absent_weight

        No dynamic class balancing.
        No CE-style class competition.
        """
        per_elem = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )  # [B, C, H, W]

        per_pair_loss = per_elem.flatten(2).mean(dim=2)  # [B, C]

        pair_weight = torch.full_like(
            per_pair_loss,
            fill_value=max(float(absent_weight), 0.0),
        )
        pair_weight[present_pair_mask] = 1.0

        weighted_loss = per_pair_loss * pair_weight
        denom = pair_weight.sum().clamp_min(1.0)

        return weighted_loss.sum() / denom

    def _dice_loss_present_mean_from_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dice for present classes only.

        Ignore pixels are treated as negative pixels because target is already 0
        at ignore locations.

        This penalizes present-class masks leaking into ignore regions.
        """
        del valid_mask  # intentionally unused

        prob = logits.sigmoid()

        prob = prob.flatten(2)  # [B, C, H*W]
        target = target.flatten(2)  # [B, C, H*W]

        intersection = (prob * target).sum(dim=2)
        denominator = prob.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.cfg.eps) / (
                denominator + self.cfg.eps
        )
        dice_loss = 1.0 - dice  # [B, C]

        pair_weight = present_pair_mask.to(dtype=dice_loss.dtype)
        return (dice_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")