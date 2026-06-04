from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


class SemanticCriterion(nn.Module):
    """Simplified criterion: only final BCE + Dice + optional CE."""

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

        num_valid_pixels = int((label_map != int(self.cfg.ignore_index)).sum().item())
        zero = self._zero_loss(final_logits)

        if num_valid_pixels <= 0:
            return {
                "loss_final_bce": zero,
                "loss_final_dice": zero,
                "loss_final_ce": zero,
                "total_loss": zero,
                "num_valid": torch.tensor(0, device=final_logits.device, dtype=torch.long),
            }

        bce_class_weights = self._build_dynamic_pair_weights(
            target=target,
            valid_mask=valid_mask,
            present_pair_mask=present_pair_mask,
            clamp_min=float(self.cfg.bce_class_balance_clamp_min),
            clamp_max=float(self.cfg.bce_class_balance_clamp_max),
        )

        loss_final_bce, loss_final_dice, loss_final_ce, num_ce_valid = (
            self._basic_mask_losses(
                logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                bce_class_weights=bce_class_weights,
                ce_label_map=label_map,
                ce_class_weights=bce_class_weights,
            )
        )

        total_loss = (
            float(self.cfg.final_bce_weight) * loss_final_bce
            + float(self.cfg.final_dice_weight) * loss_final_dice
            + float(self.cfg.final_ce_weight) * loss_final_ce
        )

        return {
            "loss_final_bce": loss_final_bce,
            "loss_final_dice": loss_final_dice,
            "loss_final_ce": loss_final_ce,
            "total_loss": total_loss,
            "num_valid": torch.tensor(
                max(num_valid_pixels, num_ce_valid),
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
    def _build_dynamic_pair_weights(
        target: torch.Tensor, valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor, clamp_min: float, clamp_max: float,
    ) -> torch.Tensor:
        target_valid = target * valid_mask.to(dtype=target.dtype)
        fg_pixels = target_valid.flatten(2).sum(dim=2)
        class_weights = torch.zeros_like(fg_pixels, dtype=target.dtype)
        if present_pair_mask.any():
            present_fg = fg_pixels[present_pair_mask]
            mean_fg = present_fg.mean().clamp_min(1.0)
            class_weights[present_pair_mask] = (
                mean_fg / fg_pixels[present_pair_mask].clamp_min(1.0)
            )
        return class_weights.clamp(min=float(clamp_min), max=float(clamp_max))

    def _basic_mask_losses(
        self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor, bce_class_weights: torch.Tensor,
        ce_label_map: torch.Tensor, ce_class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if present_pair_mask.any():
            loss_bce = self._binary_cross_entropy_present_balanced_mean(
                logits=logits, target=target, valid_mask=valid_mask,
                present_pair_mask=present_pair_mask, class_weights=bce_class_weights,
            )
            loss_dice = self._dice_loss_present_mean_from_logits(
                logits=logits, target=target, valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
        else:
            loss_bce = self._zero_loss(logits)
            loss_dice = self._zero_loss(logits)

        if float(self.cfg.final_ce_weight) > 0:
            loss_ce, num_ce_valid = self._cross_entropy_loss(
                logits=logits, ce_label_map=ce_label_map,
                present_pair_mask=present_pair_mask, ce_class_weights=ce_class_weights,
            )
        else:
            loss_ce = self._zero_loss(logits)
            num_ce_valid = 0

        return loss_bce, loss_dice, loss_ce, num_ce_valid

    @staticmethod
    def _binary_cross_entropy_present_balanced_mean(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        BCE for present classes only.

        Ignore pixels are treated as negative pixels because target is already 0
        at ignore locations.

        Important:
            - present_pair_mask selects only classes that appear in labeled pixels.
            - ignore pixels are NOT removed.
            - absent classes are NOT supervised.
            - loss is averaged over H*W pixels for each present class.
        """
        del valid_mask  # intentionally unused

        per_elem = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )  # [B, C, H, W]

        # Average over all pixels, including ignore pixels as negative target=0.
        per_pair_loss = per_elem.flatten(2).mean(dim=2)  # [B, C]

        pair_weight = class_weights.to(
            device=per_pair_loss.device,
            dtype=per_pair_loss.dtype,
        )

        present_float = present_pair_mask.to(dtype=per_pair_loss.dtype)

        weighted_loss = per_pair_loss * pair_weight * present_float
        denom = (pair_weight * present_float).sum().clamp_min(1.0)

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

    def _cross_entropy_loss(
        self, logits: torch.Tensor, ce_label_map: torch.Tensor,
        present_pair_mask: torch.Tensor, ce_class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        B = int(logits.shape[0])
        ignore_index = int(self.cfg.ignore_index)
        total_loss = self._zero_loss(logits)
        total_weight = logits.new_tensor(0.0)
        total_valid = 0

        for batch_idx in range(B):
            present_ids = torch.nonzero(present_pair_mask[batch_idx], as_tuple=False).flatten()
            if int(present_ids.numel()) <= 1:
                continue

            label_b = ce_label_map[batch_idx]
            valid_pixel_mask = label_b != ignore_index
            if int(valid_pixel_mask.sum().item()) <= 0:
                continue

            logits_b = logits[batch_idx:batch_idx + 1, present_ids]
            local_label = torch.full_like(label_b, fill_value=ignore_index)
            for local_idx, class_idx in enumerate(present_ids.tolist()):
                local_label[label_b == int(class_idx)] = int(local_idx)
            local_valid = local_label != ignore_index
            num_local_valid = int(local_valid.sum().item())
            if num_local_valid <= 0:
                continue

            ce_weight = ce_class_weights[batch_idx, present_ids].to(
                device=logits_b.device, dtype=logits_b.dtype,
            )
            per_pixel_loss = F.cross_entropy(
                logits_b, local_label[None], weight=ce_weight,
                ignore_index=ignore_index, reduction="none",
            )
            valid_float = local_valid[None].to(dtype=per_pixel_loss.dtype)
            denom_b = valid_float.sum().clamp_min(1.0)
            loss_b = (per_pixel_loss * valid_float).sum() / denom_b
            image_weight = valid_float.sum().detach()
            total_loss = total_loss + loss_b * image_weight
            total_weight = total_weight + image_weight
            total_valid += num_local_valid

        if total_valid <= 0:
            return self._zero_loss(logits), 0
        return total_loss / total_weight.clamp_min(1.0), total_valid


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")