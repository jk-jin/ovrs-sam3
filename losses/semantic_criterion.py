from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.task_modes import OUTPUT_KEYS


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255

    final_bce_weight: float = 0.4
    final_dice_weight: float = 0.5
    final_ce_weight: float = 1.0
    final_ignore_bce_weight: float = 0.15

    presence_loss_weight: float = 0.1
    presence_layer_loss_weights: Optional[Sequence[float]] = None

    bce_class_balance_clamp_min: float = 0.2
    bce_class_balance_clamp_max: float = 5.0
    eps: float = 1e-6


class SemanticCriterion(nn.Module):
    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

    @staticmethod
    def _extract_required_logits(
        outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        logits = outputs.get(key, None)
        if logits is None:
            raise ValueError(f"SemanticCriterion expects outputs['{key}'].")
        if logits.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], got {tuple(logits.shape)}."
            )
        return logits

    @staticmethod
    def _extract_required_class_logits(
        outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        logits = outputs.get(key, None)
        if logits is None:
            raise ValueError(f"SemanticCriterion expects outputs['{key}'].")
        if logits.dim() != 2:
            raise ValueError(
                f"Expected {key} as [B, C], got {tuple(logits.shape)}."
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
                    f"Expected label_map as [B, 1, H, W] or [B, H, W], "
                    f"got {tuple(label_map.shape)}."
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                f"Expected label_map as [B, H, W] or [B, 1, H, W], "
                f"got {tuple(label_map.shape)}."
            )

        return label_map.long()

    @staticmethod
    def _resize_label_map_to_logits(
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

    def _make_zero_losses(
        self,
        ref: torch.Tensor,
        include_final: bool = True,
    ) -> Dict[str, torch.Tensor]:
        zero = ref.sum() * 0.0

        losses = {
            "loss_semantic_bce": zero,
            "loss_semantic_dice": zero,
        }

        if include_final:
            losses.update(
                {
                    "loss_final_bce": zero,
                    "loss_final_dice": zero,
                    "loss_final_ce": zero,
                    "loss_final_ignore_bce": zero,
                    "loss_presence_bce": zero,
                }
            )

        return losses

    def _build_class_ids(
        self,
        num_channels: int,
        chunk_class_ids: Optional[Sequence[int]],
    ) -> list[int]:
        if chunk_class_ids is None:
            return list(range(num_channels))

        if len(chunk_class_ids) != num_channels:
            raise ValueError(
                f"chunk_class_ids length mismatch: expected {num_channels}, "
                f"got {len(chunk_class_ids)}."
            )

        return [int(x) for x in chunk_class_ids]

    def _build_targets(
        self,
        label_map: torch.Tensor,
        class_ids: Sequence[int],
        num_channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(class_ids) != num_channels:
            raise ValueError(
                f"class_ids length mismatch: expected {num_channels}, "
                f"got {len(class_ids)}."
            )

        batch_size, height, width = label_map.shape
        valid_mask = label_map != int(self.cfg.ignore_index)

        target = torch.zeros(
            (batch_size, num_channels, height, width),
            dtype=torch.float32,
            device=label_map.device,
        )

        for channel_idx, class_id in enumerate(class_ids):
            target[:, channel_idx] = (label_map == int(class_id)).float()

        valid_mask_4d = valid_mask[:, None].expand(
            batch_size,
            num_channels,
            height,
            width,
        )
        return target, valid_mask_4d

    def _build_present_pair_mask(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_valid = target * valid_mask.to(dtype=target.dtype)
        fg_pixels_per_pair = target_valid.flatten(2).sum(dim=2)
        return fg_pixels_per_pair > 0

    def _build_presence_target(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target.dim() != 4:
            raise ValueError(
                f"target must be [B, C, H, W], got {tuple(target.shape)}."
            )
        if valid_mask.shape != target.shape:
            raise ValueError(
                "valid_mask shape mismatch for presence target: "
                f"valid_mask={tuple(valid_mask.shape)}, target={tuple(target.shape)}."
            )

        target_valid = target * valid_mask.to(dtype=target.dtype)
        presence_target = target_valid.flatten(2).sum(dim=2) > 0

        valid_per_image = valid_mask[:, 0].flatten(1).any(dim=1)
        presence_valid_mask = valid_per_image[:, None].expand_as(presence_target)

        return (
            presence_target.to(dtype=target.dtype),
            presence_valid_mask,
        )

    def _presence_bce_loss(
        self,
        presence_logits: torch.Tensor,
        presence_target: torch.Tensor,
        presence_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if presence_logits.shape != presence_target.shape:
            raise ValueError(
                "presence_logits and presence_target must have the same shape, "
                f"got {tuple(presence_logits.shape)} and "
                f"{tuple(presence_target.shape)}."
            )
        if presence_valid_mask.shape != presence_target.shape:
            raise ValueError(
                "presence_valid_mask and presence_target must have the same shape, "
                f"got {tuple(presence_valid_mask.shape)} and "
                f"{tuple(presence_target.shape)}."
            )

        per_elem = F.binary_cross_entropy_with_logits(
            presence_logits,
            presence_target,
            reduction="none",
        )
        per_elem = per_elem * presence_valid_mask.to(dtype=per_elem.dtype)

        denom = presence_valid_mask.to(dtype=per_elem.dtype).sum().clamp_min(1.0)
        return per_elem.sum() / denom

    def _presence_loss_from_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        presence_logits: torch.Tensor,
        presence_target: torch.Tensor,
        presence_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        layers_key = getattr(
            OUTPUT_KEYS,
            "presence_logits_layers",
            "presence_logits_layers",
        )
        presence_logits_layers = outputs.get(layers_key, None)

        if presence_logits_layers is None:
            return self._presence_bce_loss(
                presence_logits=presence_logits,
                presence_target=presence_target,
                presence_valid_mask=presence_valid_mask,
            )

        if presence_logits_layers.dim() != 3:
            raise ValueError(
                "presence_logits_layers must be [L, B, C], got "
                f"{tuple(presence_logits_layers.shape)}."
            )

        if tuple(presence_logits_layers.shape[1:]) != tuple(presence_target.shape):
            raise ValueError(
                "presence_logits_layers shape mismatch: expected [L, B, C] "
                f"with B,C={tuple(presence_target.shape)}, got "
                f"{tuple(presence_logits_layers.shape)}."
            )

        num_layers = int(presence_logits_layers.shape[0])
        weights = self.cfg.presence_layer_loss_weights

        if weights is None:
            weights_tensor = presence_logits_layers.new_full(
                (num_layers,),
                1.0 / max(num_layers, 1),
            )
        else:
            if len(weights) != num_layers:
                raise ValueError(
                    "presence_layer_loss_weights length must match presence "
                    f"layers: got {len(weights)} weights for {num_layers} layers."
                )
            weights_tensor = presence_logits_layers.new_tensor(list(weights))

        loss = presence_logits_layers.sum() * 0.0
        for layer_idx in range(num_layers):
            layer_loss = self._presence_bce_loss(
                presence_logits=presence_logits_layers[layer_idx],
                presence_target=presence_target,
                presence_valid_mask=presence_valid_mask,
            )
            loss = loss + weights_tensor[layer_idx] * layer_loss

        return loss

    def _final_ignore_bce_loss(
        self,
        final_logits: torch.Tensor,
        semantic_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if final_logits.shape != semantic_logits.shape:
            raise ValueError(
                "final_logits and semantic_logits must have the same shape for "
                "ignore-region consistency loss, "
                f"got {tuple(final_logits.shape)} and {tuple(semantic_logits.shape)}."
            )

        if valid_mask.shape != final_logits.shape:
            raise ValueError(
                "valid_mask must have the same shape as final_logits, "
                f"got {tuple(valid_mask.shape)} and {tuple(final_logits.shape)}."
            )

        ignore_mask = ~valid_mask

        if not ignore_mask.any():
            return final_logits.sum() * 0.0

        teacher_prob = semantic_logits.detach().sigmoid()

        per_elem = F.binary_cross_entropy_with_logits(
            final_logits,
            teacher_prob,
            reduction="none",
        )

        ignore_mask = ignore_mask.to(dtype=per_elem.dtype)
        per_elem = per_elem * ignore_mask

        denom = ignore_mask.sum().clamp_min(1.0)
        return per_elem.sum() / denom

    def _build_dynamic_class_weights(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
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

        return class_weights.clamp(
            min=float(self.cfg.bce_class_balance_clamp_min),
            max=float(self.cfg.bce_class_balance_clamp_max),
        )

    def _binary_cross_entropy_present_balanced_mean(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask_4d = present_pair_mask[:, :, None, None]
        effective_mask = valid_mask & pair_mask_4d

        per_elem = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        weight_4d = class_weights[:, :, None, None]
        per_elem = per_elem * weight_4d
        per_elem = per_elem * effective_mask.to(dtype=per_elem.dtype)

        denom = (
            effective_mask.to(dtype=per_elem.dtype) * weight_4d
        ).sum().clamp_min(1.0)
        return per_elem.sum() / denom

    def _dice_loss_present_mean_from_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        prob = logits.sigmoid()
        prob = prob * valid_mask.to(dtype=prob.dtype)
        target = target * valid_mask.to(dtype=target.dtype)

        prob = prob.flatten(2)
        target = target.flatten(2)

        intersection = (prob * target).sum(dim=2)
        denominator = prob.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.cfg.eps) / (
            denominator + self.cfg.eps
        )
        dice_loss = 1.0 - dice

        pair_weight = present_pair_mask.to(dtype=dice_loss.dtype)
        return (dice_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)

    def _build_ce_label_map(
        self,
        label_map: torch.Tensor,
        class_ids: Sequence[int],
    ) -> torch.Tensor:
        ce_label_map = torch.full_like(
            label_map,
            fill_value=int(self.cfg.ignore_index),
        )

        for local_idx, class_id in enumerate(class_ids):
            ce_label_map[label_map == int(class_id)] = int(local_idx)

        return ce_label_map.long()

    def _cross_entropy_loss(
        self,
        logits: torch.Tensor,
        ce_label_map: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if logits.shape[0] != ce_label_map.shape[0]:
            raise ValueError(
                f"Batch mismatch between logits and ce_label_map: "
                f"{tuple(logits.shape)} vs {tuple(ce_label_map.shape)}."
            )

        if logits.shape[-2:] != ce_label_map.shape[-2:]:
            raise ValueError(
                f"Spatial mismatch between logits and ce_label_map: "
                f"{tuple(logits.shape)} vs {tuple(ce_label_map.shape)}."
            )

        num_valid = int((ce_label_map != int(self.cfg.ignore_index)).sum().item())
        if num_valid <= 0:
            return logits.sum() * 0.0, 0

        loss = F.cross_entropy(
            logits,
            ce_label_map,
            ignore_index=int(self.cfg.ignore_index),
            reduction="mean",
        )
        return loss, num_valid

    def _forward_chunk(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        chunk_class_ids: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_required_logits(
            outputs,
            OUTPUT_KEYS.semantic_logits,
        )

        zero_losses = self._make_zero_losses(semantic_logits)
        zero = semantic_logits.sum() * 0.0

        return {
            "loss_semantic_bce": zero_losses["loss_semantic_bce"],
            "loss_semantic_dice": zero_losses["loss_semantic_dice"],
            "loss_final_bce": zero_losses["loss_final_bce"],
            "loss_final_dice": zero_losses["loss_final_dice"],
            "loss_final_ce": zero_losses["loss_final_ce"],
            "loss_final_ignore_bce": zero_losses["loss_final_ignore_bce"],
            "loss_presence_bce": zero_losses["loss_presence_bce"],
            "total_loss": zero,
            "num_valid": torch.tensor(
                0,
                device=semantic_logits.device,
                dtype=torch.long,
            ),
        }

    def _forward_final(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_required_logits(
            outputs,
            OUTPUT_KEYS.semantic_logits,
        )
        final_logits = self._extract_required_logits(
            outputs,
            OUTPUT_KEYS.final_logits,
        )
        presence_logits = self._extract_required_class_logits(
            outputs,
            OUTPUT_KEYS.presence_logits,
        )

        if final_logits.shape != semantic_logits.shape:
            raise ValueError(
                "final_logits and semantic_logits must have the same shape, "
                f"got {tuple(final_logits.shape)} and {tuple(semantic_logits.shape)}."
            )
        if presence_logits.shape != semantic_logits.shape[:2]:
            raise ValueError(
                "presence_logits must be [B, C] matching semantic_logits, "
                f"got {tuple(presence_logits.shape)} and "
                f"semantic_logits.shape[:2]={tuple(semantic_logits.shape[:2])}."
            )

        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_logits(
            label_map=label_map,
            target_hw=tuple(final_logits.shape[-2:]),
        )

        num_channels = int(final_logits.shape[1])
        class_ids = list(range(num_channels))

        target, valid_mask = self._build_targets(
            label_map=label_map,
            class_ids=class_ids,
            num_channels=num_channels,
        )
        present_pair_mask = self._build_present_pair_mask(
            target=target,
            valid_mask=valid_mask,
        )
        presence_target, presence_valid_mask = self._build_presence_target(
            target=target,
            valid_mask=valid_mask,
        )

        num_valid_pixels = int((label_map != int(self.cfg.ignore_index)).sum().item())
        zero_losses = self._make_zero_losses(final_logits)

        loss_final_ignore_bce = self._final_ignore_bce_loss(
            final_logits=final_logits,
            semantic_logits=semantic_logits,
            valid_mask=valid_mask,
        )

        loss_presence_bce = self._presence_loss_from_outputs(
            outputs=outputs,
            presence_logits=presence_logits,
            presence_target=presence_target,
            presence_valid_mask=presence_valid_mask,
        )

        if num_valid_pixels <= 0:
            total_loss = (
                float(self.cfg.final_ignore_bce_weight) * loss_final_ignore_bce
                + float(self.cfg.presence_loss_weight) * loss_presence_bce
            )

            return {
                "loss_semantic_bce": zero_losses["loss_semantic_bce"],
                "loss_semantic_dice": zero_losses["loss_semantic_dice"],
                "loss_final_bce": zero_losses["loss_final_bce"],
                "loss_final_dice": zero_losses["loss_final_dice"],
                "loss_final_ce": zero_losses["loss_final_ce"],
                "loss_final_ignore_bce": loss_final_ignore_bce,
                "loss_presence_bce": loss_presence_bce,
                "total_loss": total_loss,
                "num_valid": torch.tensor(
                    0,
                    device=final_logits.device,
                    dtype=torch.long,
                ),
            }

        if present_pair_mask.any():
            class_weights = self._build_dynamic_class_weights(
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
            loss_final_bce = self._binary_cross_entropy_present_balanced_mean(
                logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                class_weights=class_weights,
            )
            loss_final_dice = self._dice_loss_present_mean_from_logits(
                logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
        else:
            loss_final_bce = final_logits.sum() * 0.0
            loss_final_dice = final_logits.sum() * 0.0

        ce_label_map = self._build_ce_label_map(
            label_map=label_map,
            class_ids=class_ids,
        )
        loss_final_ce, num_ce_valid = self._cross_entropy_loss(
            logits=final_logits,
            ce_label_map=ce_label_map,
        )

        total_loss = (
            float(self.cfg.final_bce_weight) * loss_final_bce
            + float(self.cfg.final_dice_weight) * loss_final_dice
            + float(self.cfg.final_ce_weight) * loss_final_ce
            + float(self.cfg.final_ignore_bce_weight) * loss_final_ignore_bce
            + float(self.cfg.presence_loss_weight) * loss_presence_bce
        )

        return {
            "loss_semantic_bce": zero_losses["loss_semantic_bce"],
            "loss_semantic_dice": zero_losses["loss_semantic_dice"],
            "loss_final_bce": loss_final_bce,
            "loss_final_dice": loss_final_dice,
            "loss_final_ce": loss_final_ce,
            "loss_final_ignore_bce": loss_final_ignore_bce,
            "loss_presence_bce": loss_presence_bce,
            "total_loss": total_loss,
            "num_valid": torch.tensor(
                max(num_valid_pixels, num_ce_valid),
                device=final_logits.device,
                dtype=torch.long,
            ),
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        if reduction != "mean":
            raise ValueError(
                f"SemanticCriterion only supports reduction='mean', got {reduction!r}."
            )

        is_final_stage = OUTPUT_KEYS.final_logits in outputs

        if is_final_stage:
            return self._forward_final(
                outputs=outputs,
                targets=targets,
            )

        if chunk_class_ids is None:
            raise ValueError(
                "Chunk-stage SemanticCriterion requires chunk_class_ids."
            )

        return self._forward_chunk(
            outputs=outputs,
            targets=targets,
            chunk_class_ids=chunk_class_ids,
        )


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")