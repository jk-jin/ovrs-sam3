from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


class SemanticCriterion(nn.Module):
    """
    Semantic segmentation criterion.

    Losses:
        1. mask BCE on final_logits
           - present class pairs use valid and ignore pixels
           - absent class pairs use valid pixels only
           - absent class pairs are weighted by bce_absent_class_weight
           - ignore pixels suppress leakage only for present classes

        2. present-only Dice on final_logits
           - optional
           - controlled by final_dice_weight
    """

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
        del chunk_class_ids

        if reduction != "mean":
            raise ValueError(
                f"SemanticCriterion only supports reduction='mean', got {reduction!r}."
            )

        if OUTPUT_KEYS.final_logits not in outputs:
            raise ValueError(
                f"SemanticCriterion requires outputs[{OUTPUT_KEYS.final_logits!r}]."
            )

        return self._forward_final(outputs=outputs, targets=targets)

    def _forward_final(
        self,
        outputs: TensorDict,
        targets: TensorDict,
    ) -> TensorDict:
        final_logits = self._extract_4d_tensor(
            outputs,
            OUTPUT_KEYS.final_logits,
            "[B, C, H, W]",
        )

        B, C, H, W = final_logits.shape

        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_hw(
            label_map=label_map,
            target_hw=(H, W),
        )

        class_ids = list(range(C))

        target, valid_mask = self._build_binary_targets(
            label_map=label_map,
            class_ids=class_ids,
            num_channels=C,
            dtype=final_logits.dtype,
        )

        presence_target = self._build_presence_target(
            label_map=label_map,
            valid_mask=valid_mask,
            class_ids=class_ids,
            dtype=final_logits.dtype,
        )

        num_valid_pixels = int(valid_mask.sum().item())
        zero = self._zero_loss(final_logits)

        if num_valid_pixels <= 0:
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

        loss_final_bce = self._binary_cross_entropy_pair_weighted_all_pixels(
            logits=final_logits,
            target=target,
            valid_mask=valid_mask,
            presence_target=presence_target,
            absent_weight=float(self.cfg.bce_absent_class_weight),
            valid_pixel_weight=float(self.cfg.bce_valid_pixel_weight),
            ignore_pixel_weight=float(self.cfg.bce_ignore_pixel_weight),
            eps=float(self.cfg.eps),
        )

        if float(self.cfg.final_dice_weight) > 0.0 and bool(
            presence_target.bool().any().item()
        ):
            loss_final_dice = self._dice_loss_present_mean_from_logits(
                logits=final_logits,
                target=target,
                presence_target=presence_target,
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
                num_valid_pixels,
                device=final_logits.device,
                dtype=torch.long,
            ),
        }

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _extract_4d_tensor(
        outputs: TensorDict,
        key: str,
        shape_name: str,
    ) -> torch.Tensor:
        tensor = outputs.get(key, None)
        if tensor is None:
            raise ValueError(f"SemanticCriterion expects outputs[{key!r}].")
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected {key} as {shape_name}, got {tuple(tensor.shape)}."
            )
        return tensor

    def _extract_label_map(self, targets: TensorDict) -> torch.Tensor:
        if "label_map" not in targets:
            raise ValueError("SemanticCriterion expects targets['label_map'].")

        label_map = targets["label_map"]

        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    "Expected label_map as [B, 1, H, W] or [B, H, W]."
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                "Expected label_map as [B, H, W] or [B, 1, H, W]."
            )

        return label_map.long()

    @staticmethod
    def _resize_label_map_to_hw(
        label_map: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tuple(label_map.shape[-2:]) == tuple(target_hw):
            return label_map

        return F.interpolate(
            label_map[:, None].float(),
            size=target_hw,
            mode="nearest",
        )[:, 0].long()

    def _build_binary_targets(
        self,
        label_map: torch.Tensor,
        class_ids: Sequence[int],
        num_channels: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build binary mask targets.

        Important:
            ignore_index pixels are NOT removed from mask BCE.
            They simply stay 0 for every class.
        """
        B, H, W = label_map.shape
        valid_mask = label_map != int(self.cfg.ignore_index)

        target = torch.zeros(
            (B, int(num_channels), H, W),
            dtype=dtype,
            device=label_map.device,
        )

        for channel_idx, class_id in enumerate(class_ids):
            target[:, channel_idx] = (label_map == int(class_id)).to(dtype=dtype)

        return target, valid_mask

    @staticmethod
    def _build_presence_target(
        label_map: torch.Tensor,
        valid_mask: torch.Tensor,
        class_ids: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build image-level class presence target.

        presence_target[b, c] = 1
            if class c appears in non-ignore region of image b.

        presence_target[b, c] = 0
            otherwise.
        """
        B = int(label_map.shape[0])
        C = len(class_ids)

        presence_target = torch.zeros(
            (B, C),
            dtype=dtype,
            device=label_map.device,
        )

        for channel_idx, class_id in enumerate(class_ids):
            appears = ((label_map == int(class_id)) & valid_mask).flatten(1).any(dim=1)
            presence_target[:, channel_idx] = appears.to(dtype=dtype)

        return presence_target

    @staticmethod
    def _binary_cross_entropy_pair_weighted_all_pixels(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        presence_target: torch.Tensor,
        absent_weight: float,
        valid_pixel_weight: float,
        ignore_pixel_weight: float,
        eps: float,
    ) -> torch.Tensor:
        """
        BCE over all image-class pairs with pair-dependent pixel masking.

        For present image-class pairs:
            valid pixels use valid_pixel_weight.
            ignore pixels use ignore_pixel_weight.

        For absent image-class pairs:
            valid pixels use valid_pixel_weight.
            ignore pixels are ignored with weight 0.

        Pair-level weights:
            present image-class pairs: 1.0
            absent image-class pairs:  absent_weight
        """
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        valid_pixel_weight = max(float(valid_pixel_weight), 0.0)
        ignore_pixel_weight = max(float(ignore_pixel_weight), 0.0)
        absent_weight = max(float(absent_weight), 0.0)

        if valid_mask.dim() != 3:
            raise ValueError(
                f"valid_mask must be [B, H, W], got {tuple(valid_mask.shape)}."
            )

        if tuple(valid_mask.shape) != tuple(logits.shape[0:1] + logits.shape[-2:]):
            raise ValueError(
                "valid_mask shape mismatch: "
                f"expected {(logits.shape[0], logits.shape[-2], logits.shape[-1])}, "
                f"got {tuple(valid_mask.shape)}."
            )

        if presence_target.dim() != 2:
            raise ValueError(
                f"presence_target must be [B, C], got {tuple(presence_target.shape)}."
            )

        if tuple(presence_target.shape) != tuple(logits.shape[:2]):
            raise ValueError(
                "presence_target shape mismatch: "
                f"expected {(logits.shape[0], logits.shape[1])}, "
                f"got {tuple(presence_target.shape)}."
            )

        present_pair = presence_target > 0.5          # [B, C]
        valid = valid_mask[:, None]                   # [B, 1, H, W]

        valid_weight_map = torch.full_like(pixel_loss, valid_pixel_weight)
        ignore_weight_map = torch.full_like(pixel_loss, ignore_pixel_weight)
        zero_weight_map = torch.zeros_like(pixel_loss)

        # present pair: valid -> valid_pixel_weight, ignore -> ignore_pixel_weight
        present_pixel_weight = torch.where(
            valid,
            valid_weight_map,
            ignore_weight_map,
        )

        # absent pair: valid -> valid_pixel_weight, ignore -> 0
        absent_pixel_weight = torch.where(
            valid,
            valid_weight_map,
            zero_weight_map,
        )

        pixel_weight = torch.where(
            present_pair[:, :, None, None],
            present_pixel_weight,
            absent_pixel_weight,
        )

        weighted_pixel_loss = pixel_loss * pixel_weight

        pixel_weight_sum = pixel_weight.flatten(2).sum(dim=2).clamp_min(float(eps))
        pair_loss = weighted_pixel_loss.flatten(2).sum(dim=2) / pixel_weight_sum

        pair_weight = torch.where(
            present_pair,
            torch.ones_like(presence_target),
            torch.full_like(presence_target, absent_weight),
        )

        weight_sum = pair_weight.sum()

        if bool(weight_sum.detach().le(0).item()):
            return logits.sum() * 0.0

        return (pair_loss * pair_weight).sum() / weight_sum.clamp_min(float(eps))

    def _dice_loss_present_mean_from_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        presence_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dice loss for present image-class pairs only.

        ignore_index pixels are not removed.
        They remain target=0 and therefore penalize leakage.
        """
        prob = logits.sigmoid()

        prob = prob.flatten(2)
        target = target.flatten(2)

        intersection = (prob * target).sum(dim=2)
        denominator = prob.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + float(self.cfg.eps)) / (
            denominator + float(self.cfg.eps)
        )
        dice_loss = 1.0 - dice

        pair_weight = presence_target.to(dtype=dice_loss.dtype)

        weight_sum = pair_weight.sum()
        if bool(weight_sum.detach().le(0).item()):
            return logits.sum() * 0.0

        return (dice_loss * pair_weight).sum() / weight_sum.clamp_min(float(self.cfg.eps))


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")