from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


class SemanticCriterion(nn.Module):
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
                "SemanticCriterion only supports final-stage outputs. "
                "Chunk-stage semantic loss has been removed."
            )

        return self._forward_final(outputs=outputs, targets=targets)

    def _forward_final(
        self,
        outputs: TensorDict,
        targets: TensorDict,
    ) -> TensorDict:
        semantic_logits = self._extract_required_tensor(
            outputs=outputs,
            key=OUTPUT_KEYS.semantic_logits,
            ndim=4,
            shape_name="[B, C, H, W]",
        )
        final_logits = self._extract_required_tensor(
            outputs=outputs,
            key=OUTPUT_KEYS.final_logits,
            ndim=4,
            shape_name="[B, C, H, W]",
        )
        presence_logits = self._extract_required_tensor(
            outputs=outputs,
            key=OUTPUT_KEYS.presence_logits,
            ndim=2,
            shape_name="[B, C]",
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
        presence_target, presence_valid_mask = self._build_presence_target(
            target=target,
            valid_mask=valid_mask,
        )

        num_valid_pixels = int((label_map != int(self.cfg.ignore_index)).sum().item())
        zero = self._zero_loss(final_logits)

        loss_final_ignore_bce = self._final_ignore_bce_loss(
            final_logits=final_logits,
            semantic_logits=semantic_logits,
            valid_mask=valid_mask,
        )
        loss_presence_bce, presence_loss_log_items = self._presence_loss_from_outputs(
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

            loss_dict = {
                "loss_final_bce": zero,
                "loss_final_dice": zero,
                "loss_final_ce": zero,
                "loss_final_ignore_bce": loss_final_ignore_bce,
                "loss_presence_bce": loss_presence_bce,
                "loss_mask_layers_aux": zero,
                "loss_clip_sam_layers_aux": zero,
                "total_loss": total_loss,
                "num_valid": torch.tensor(
                    0,
                    device=final_logits.device,
                    dtype=torch.long,
                ),
            }
            loss_dict.update(presence_loss_log_items)
            return loss_dict

        bce_class_weights = self._build_dynamic_pair_weights(
            target=target,
            valid_mask=valid_mask,
            present_pair_mask=present_pair_mask,
            clamp_min=float(self.cfg.bce_class_balance_clamp_min),
            clamp_max=float(self.cfg.bce_class_balance_clamp_max),
        )
        ce_class_weights = self._build_dynamic_pair_weights(
            target=target,
            valid_mask=valid_mask,
            present_pair_mask=present_pair_mask,
            clamp_min=float(self.cfg.ce_class_balance_clamp_min),
            clamp_max=float(self.cfg.ce_class_balance_clamp_max),
        )
        ce_label_map = self._build_ce_label_map(
            label_map=label_map,
            class_ids=class_ids,
        )

        loss_final_bce, loss_final_dice, loss_final_ce, num_ce_valid = (
            self._basic_mask_losses(
                logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                bce_class_weights=bce_class_weights,
                ce_label_map=ce_label_map,
                ce_class_weights=ce_class_weights,
            )
        )

        loss_aux_mask_layers, aux_loss_dict = self._aux_mask_layer_losses(
            outputs=outputs,
            final_logits=final_logits,
            target=target,
            valid_mask=valid_mask,
            present_pair_mask=present_pair_mask,
            bce_class_weights=bce_class_weights,
            ce_label_map=ce_label_map,
            ce_class_weights=ce_class_weights,
        )

        loss_aux_clip_sam_layers, clip_sam_aux_loss_dict = (
            self._clip_sam_layer_losses(
                outputs=outputs,
                final_logits=final_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                bce_class_weights=bce_class_weights,
                ce_label_map=ce_label_map,
                ce_class_weights=ce_class_weights,
            )
        )

        total_loss = (
            float(self.cfg.final_bce_weight) * loss_final_bce
            + float(self.cfg.final_dice_weight) * loss_final_dice
            + float(self.cfg.final_ce_weight) * loss_final_ce
            + float(self.cfg.final_ignore_bce_weight) * loss_final_ignore_bce
            + float(self.cfg.presence_loss_weight) * loss_presence_bce
            + loss_aux_mask_layers
            + loss_aux_clip_sam_layers
        )

        loss_dict = {
            "loss_final_bce": loss_final_bce,
            "loss_final_dice": loss_final_dice,
            "loss_final_ce": loss_final_ce,
            "loss_final_ignore_bce": loss_final_ignore_bce,
            "loss_presence_bce": loss_presence_bce,
            "loss_mask_layers_aux": loss_aux_mask_layers,
            "loss_clip_sam_layers_aux": loss_aux_clip_sam_layers,
            "total_loss": total_loss,
            "num_valid": torch.tensor(
                max(num_valid_pixels, num_ce_valid),
                device=final_logits.device,
                dtype=torch.long,
            ),
        }
        loss_dict.update(presence_loss_log_items)
        loss_dict.update(aux_loss_dict)
        loss_dict.update(clip_sam_aux_loss_dict)
        return loss_dict

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _extract_required_tensor(
        outputs: TensorDict,
        key: str,
        ndim: int,
        shape_name: str,
    ) -> torch.Tensor:
        tensor = outputs.get(key, None)
        if tensor is None:
            raise ValueError(f"SemanticCriterion expects outputs['{key}'].")
        if tensor.dim() != int(ndim):
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
                    "Expected label_map as [B, 1, H, W] or [B, H, W], "
                    f"got {tuple(label_map.shape)}."
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                "Expected label_map as [B, H, W] or [B, 1, H, W], "
                f"got {tuple(label_map.shape)}."
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

    @staticmethod
    def _build_present_pair_mask(
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

        return presence_target.to(dtype=target.dtype), presence_valid_mask

    def _presence_loss_from_outputs(
        self,
        outputs: TensorDict,
        presence_logits: torch.Tensor,
        presence_target: torch.Tensor,
        presence_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, TensorDict]:
        # outputs is kept in the signature for call-site stability.
        # The new design predicts presence only once before the fusion layers.
        _ = outputs

        loss = self._presence_bce_loss(
            presence_logits=presence_logits,
            presence_target=presence_target,
            presence_valid_mask=presence_valid_mask,
        )
        return loss, {"loss_presence_final_bce": loss.detach()}

    @staticmethod
    def _presence_bce_loss(
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
        valid = presence_valid_mask.to(dtype=per_elem.dtype)
        return (per_elem * valid).sum() / valid.sum().clamp_min(1.0)

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
            return self._zero_loss(final_logits)

        teacher_prob = semantic_logits.detach().sigmoid()
        per_elem = F.binary_cross_entropy_with_logits(
            final_logits,
            teacher_prob,
            reduction="none",
        )

        ignore_weight = ignore_mask.to(dtype=per_elem.dtype)
        return (per_elem * ignore_weight).sum() / ignore_weight.sum().clamp_min(1.0)

    def _basic_mask_losses(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
        bce_class_weights: torch.Tensor,
        ce_label_map: torch.Tensor,
        ce_class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if present_pair_mask.any():
            loss_bce = self._binary_cross_entropy_present_balanced_mean(
                logits=logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                class_weights=bce_class_weights,
            )
            loss_dice = self._dice_loss_present_mean_from_logits(
                logits=logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
            )
        else:
            loss_bce = self._zero_loss(logits)
            loss_dice = self._zero_loss(logits)

        loss_ce, num_ce_valid = self._cross_entropy_loss(
            logits=logits,
            ce_label_map=ce_label_map,
            present_pair_mask=present_pair_mask,
            ce_class_weights=ce_class_weights,
        )
        return loss_bce, loss_dice, loss_ce, num_ce_valid

    @staticmethod
    def _build_dynamic_pair_weights(
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
        clamp_min: float,
        clamp_max: float,
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

    @staticmethod
    def _binary_cross_entropy_present_balanced_mean(
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
        present_pair_mask: torch.Tensor,
        ce_class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if logits.shape[0] != ce_label_map.shape[0]:
            raise ValueError(
                "Batch mismatch between logits and ce_label_map: "
                f"{tuple(logits.shape)} vs {tuple(ce_label_map.shape)}."
            )

        if logits.shape[-2:] != ce_label_map.shape[-2:]:
            raise ValueError(
                "Spatial mismatch between logits and ce_label_map: "
                f"{tuple(logits.shape)} vs {tuple(ce_label_map.shape)}."
            )

        if present_pair_mask.shape != logits.shape[:2]:
            raise ValueError(
                "present_pair_mask must be [B, C] matching logits, "
                f"got {tuple(present_pair_mask.shape)} vs "
                f"logits[:2]={tuple(logits.shape[:2])}."
            )

        if ce_class_weights.shape != logits.shape[:2]:
            raise ValueError(
                "ce_class_weights must be [B, C] matching logits, "
                f"got {tuple(ce_class_weights.shape)} vs "
                f"logits[:2]={tuple(logits.shape[:2])}."
            )

        batch_size = int(logits.shape[0])
        ignore_index = int(self.cfg.ignore_index)

        total_loss = self._zero_loss(logits)
        total_weight = logits.new_tensor(0.0)
        total_valid = 0

        for batch_idx in range(batch_size):
            loss_b, image_weight, num_valid = self._cross_entropy_loss_one_image(
                logits=logits,
                ce_label_map=ce_label_map,
                present_pair_mask=present_pair_mask,
                ce_class_weights=ce_class_weights,
                batch_idx=batch_idx,
                ignore_index=ignore_index,
            )
            if num_valid <= 0:
                continue

            total_loss = total_loss + loss_b * image_weight
            total_weight = total_weight + image_weight
            total_valid += num_valid

        if total_valid <= 0:
            return self._zero_loss(logits), 0

        return total_loss / total_weight.clamp_min(1.0), total_valid

    def _cross_entropy_loss_one_image(
        self,
        logits: torch.Tensor,
        ce_label_map: torch.Tensor,
        present_pair_mask: torch.Tensor,
        ce_class_weights: torch.Tensor,
        batch_idx: int,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        present_ids = torch.nonzero(
            present_pair_mask[batch_idx],
            as_tuple=False,
        ).flatten()

        if int(present_ids.numel()) <= 1:
            return self._zero_loss(logits), logits.new_tensor(0.0), 0

        label_b = ce_label_map[batch_idx]
        valid_pixel_mask = label_b != ignore_index
        if int(valid_pixel_mask.sum().item()) <= 0:
            return self._zero_loss(logits), logits.new_tensor(0.0), 0

        logits_b = logits[batch_idx:batch_idx + 1, present_ids]

        local_label = torch.full_like(label_b, fill_value=ignore_index)
        for local_idx, class_idx in enumerate(present_ids.tolist()):
            local_label[label_b == int(class_idx)] = int(local_idx)

        local_valid = local_label != ignore_index
        num_local_valid = int(local_valid.sum().item())
        if num_local_valid <= 0:
            return self._zero_loss(logits), logits.new_tensor(0.0), 0

        ce_weight = ce_class_weights[batch_idx, present_ids].to(
            device=logits_b.device,
            dtype=logits_b.dtype,
        )

        per_pixel_loss = F.cross_entropy(
            logits_b,
            local_label[None],
            weight=ce_weight,
            ignore_index=ignore_index,
            reduction="none",
        )

        safe_local_label = local_label.clamp(min=0)
        safe_local_label = safe_local_label.clamp(max=int(present_ids.numel()) - 1)

        pixel_weight = ce_weight[safe_local_label].to(
            device=per_pixel_loss.device,
            dtype=per_pixel_loss.dtype,
        )
        pixel_weight = pixel_weight[None]

        valid_float = local_valid[None].to(dtype=per_pixel_loss.dtype)
        valid_weight = pixel_weight * valid_float

        denom_b = valid_weight.sum().clamp_min(1.0)
        loss_b = (per_pixel_loss * valid_float).sum() / denom_b
        image_weight = valid_weight.sum().detach()

        return loss_b, image_weight, num_local_valid

    def _aux_mask_layer_losses(
        self,
        outputs: TensorDict,
        final_logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
        bce_class_weights: torch.Tensor,
        ce_label_map: torch.Tensor,
        ce_class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, TensorDict]:
        mask_logits_layers = outputs.get(OUTPUT_KEYS.mask_logits_layers, None)

        if mask_logits_layers is None:
            return self._zero_loss(final_logits), {}

        if mask_logits_layers.dim() != 5:
            raise ValueError(
                "mask_logits_layers must be [L, B, C, H, W], "
                f"got {tuple(mask_logits_layers.shape)}."
            )

        if tuple(mask_logits_layers.shape[1:]) != tuple(final_logits.shape):
            raise ValueError(
                "mask_logits_layers shape mismatch: expected [L, B, C, H, W] "
                f"with B,C,H,W={tuple(final_logits.shape)}, got "
                f"{tuple(mask_logits_layers.shape)}."
            )

        num_layers = int(mask_logits_layers.shape[0])
        num_aux_layers = max(num_layers - 1, 0)
        if num_aux_layers == 0:
            return self._zero_loss(final_logits), {}

        weights_tensor = self._build_aux_mask_layer_weights(
            mask_logits_layers=mask_logits_layers,
            num_aux_layers=num_aux_layers,
        )

        aux_total = self._zero_loss(final_logits)
        loss_dict: TensorDict = {}

        for layer_idx in range(num_aux_layers):
            layer_logits = mask_logits_layers[layer_idx]
            loss_bce, loss_dice, loss_ce, _ = self._basic_mask_losses(
                logits=layer_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                bce_class_weights=bce_class_weights,
                ce_label_map=ce_label_map,
                ce_class_weights=ce_class_weights,
            )

            layer_total = (
                float(self.cfg.final_bce_weight) * loss_bce
                + float(self.cfg.final_dice_weight) * loss_dice
                + float(self.cfg.final_ce_weight) * loss_ce
            )
            weighted_layer_total = weights_tensor[layer_idx] * layer_total
            aux_total = aux_total + weighted_layer_total

            loss_dict[f"loss_mask_layer_{layer_idx}_bce"] = loss_bce
            loss_dict[f"loss_mask_layer_{layer_idx}_dice"] = loss_dice
            loss_dict[f"loss_mask_layer_{layer_idx}_ce"] = loss_ce
            loss_dict[f"loss_mask_layer_{layer_idx}_total"] = weighted_layer_total

        aux_total = float(self.cfg.mask_layer_loss_weight) * aux_total
        loss_dict["loss_mask_layers_aux"] = aux_total

        return aux_total, loss_dict

    def _clip_sam_layer_losses(
        self,
        outputs: TensorDict,
        final_logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        present_pair_mask: torch.Tensor,
        bce_class_weights: torch.Tensor,
        ce_label_map: torch.Tensor,
        ce_class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, TensorDict]:
        clip_sam_logits_layers = outputs.get(
            OUTPUT_KEYS.clip_sam_logits_layers,
            None,
        )

        if clip_sam_logits_layers is None:
            return self._zero_loss(final_logits), {}

        if clip_sam_logits_layers.dim() != 5:
            raise ValueError(
                "clip_sam_logits_layers must be [L, B, C, H, W], "
                f"got {tuple(clip_sam_logits_layers.shape)}."
            )

        if tuple(clip_sam_logits_layers.shape[1:]) != tuple(final_logits.shape):
            raise ValueError(
                "clip_sam_logits_layers shape mismatch: expected [L, B, C, H, W] "
                f"with B,C,H,W={tuple(final_logits.shape)}, got "
                f"{tuple(clip_sam_logits_layers.shape)}."
            )

        num_layers = int(clip_sam_logits_layers.shape[0])
        if num_layers <= 0:
            return self._zero_loss(final_logits), {}

        weights_tensor = self._build_clip_sam_layer_weights(
            clip_sam_logits_layers=clip_sam_logits_layers,
            num_layers=num_layers,
        )

        aux_total = self._zero_loss(final_logits)
        loss_dict: TensorDict = {}

        for layer_idx in range(num_layers):
            layer_logits = clip_sam_logits_layers[layer_idx]

            loss_bce, loss_dice, loss_ce, _ = self._basic_mask_losses(
                logits=layer_logits,
                target=target,
                valid_mask=valid_mask,
                present_pair_mask=present_pair_mask,
                bce_class_weights=bce_class_weights,
                ce_label_map=ce_label_map,
                ce_class_weights=ce_class_weights,
            )

            layer_total = (
                float(self.cfg.final_bce_weight) * loss_bce
                + float(self.cfg.final_dice_weight) * loss_dice
                + float(self.cfg.final_ce_weight) * loss_ce
            )

            weighted_layer_total = weights_tensor[layer_idx] * layer_total
            aux_total = aux_total + weighted_layer_total

            loss_dict[f"loss_clip_sam_layer_{layer_idx}_bce"] = loss_bce.detach()
            loss_dict[f"loss_clip_sam_layer_{layer_idx}_dice"] = loss_dice.detach()
            loss_dict[f"loss_clip_sam_layer_{layer_idx}_ce"] = loss_ce.detach()
            loss_dict[f"loss_clip_sam_layer_{layer_idx}_total"] = (
                weighted_layer_total.detach()
            )

        aux_total = float(self.cfg.clip_sam_layer_loss_weight) * aux_total
        loss_dict["loss_clip_sam_layers_aux"] = aux_total

        return aux_total, loss_dict

    def _build_clip_sam_layer_weights(
        self,
        clip_sam_logits_layers: torch.Tensor,
        num_layers: int,
    ) -> torch.Tensor:
        weights = self.cfg.clip_sam_layer_weights
        if weights is None:
            return clip_sam_logits_layers.new_ones(num_layers)

        if len(weights) != num_layers:
            raise ValueError(
                "clip_sam_layer_weights length must equal len(clip_sam_logits_layers). "
                f"Got {len(weights)} weights for {num_layers} layers."
            )

        return clip_sam_logits_layers.new_tensor(list(weights))

    def _build_aux_mask_layer_weights(
        self,
        mask_logits_layers: torch.Tensor,
        num_aux_layers: int,
    ) -> torch.Tensor:
        weights = self.cfg.mask_layer_weights
        if weights is None:
            return mask_logits_layers.new_ones(num_aux_layers)

        if len(weights) != num_aux_layers:
            raise ValueError(
                "mask_layer_weights length must equal len(mask_logits_layers) - 1. "
                f"Got {len(weights)} weights for {num_aux_layers} aux layers."
            )
        return mask_logits_layers.new_tensor(list(weights))


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")
