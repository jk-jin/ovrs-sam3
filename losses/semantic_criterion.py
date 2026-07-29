from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


@dataclass
class SemanticStreamingContext:
    """Pre-computed global statistics for streaming per-chunk loss.

    All denominators are global (full batch, all classes) so that summing
    per-chunk contributions yields the exact same result as a single
    all-class forward.
    """

    label_map: torch.Tensor
    valid_mask: torch.Tensor
    presence_target: torch.Tensor          # [B, C] bool
    present_negative_target: torch.Tensor  # [B, C] bool
    num_classes: int
    num_present_pairs: int
    num_present_negative_pairs: int
    num_absent_pairs: int
    distill_pixel_denom: int
    positive_weight: float
    present_negative_weight: float
    absent_negative_weight: float


class SemanticCriterion(nn.Module):
    """Semantic segmentation criterion with streaming per-chunk support.

    Losses:
        1. Positive-negative balanced BCE on final_logits
           - Positive pixels and negative pixels are averaged separately
             within each image-class pair, then averaged across pairs.
           - 0.5 * positive_bce + 0.25 * present_negative_bce
             + 0.25 * absent_negative_bce
           - Label 255 is negative for all classes in main BCE.

        2. Optional Dice on final_logits (default weight 0.0)

        3. SAM3 teacher mask distillation BCE (default weight 0.05)
           - Frozen SAM3 semantic head logits as soft targets
           - Only present pairs and non-255 pixels
    """

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: TensorDict,
        targets: TensorDict,
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
    ) -> TensorDict:
        """Full-batch forward (inference / eval / legacy)."""
        del chunk_class_ids

        if reduction != "mean":
            raise ValueError(
                f"SemanticCriterion only supports reduction='mean', got {reduction!r}."
            )

        if OUTPUT_KEYS.final_logits not in outputs:
            raise ValueError(
                f"SemanticCriterion requires outputs[{OUTPUT_KEYS.final_logits!r}]."
            )

        final_logits = self._extract_4d_tensor(
            outputs, OUTPUT_KEYS.final_logits, "[B, C, H, W]"
        )
        B, C = final_logits.shape[:2]

        context = self.prepare_streaming_context(
            targets=targets,
            num_classes=C,
            target_hw=final_logits.shape[-2:],
        )

        # Build full target for Dice.
        target_full, _ = self._build_binary_targets(
            label_map=context.label_map,
            num_channels=C,
            dtype=final_logits.dtype,
        )

        loss_dict = self.forward_chunk(
            outputs=outputs,
            context=context,
            class_start=0,
            class_end=C,
            target_full=target_full,
        )

        return loss_dict

    def prepare_streaming_context(
        self,
        targets: TensorDict,
        num_classes: int,
        target_hw: tuple[int, int] = (288, 288),
    ) -> SemanticStreamingContext:
        """Build global statistics for streaming per-chunk loss.

        Called once per batch before the chunk loop.
        """
        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_hw(
            label_map=label_map,
            target_hw=target_hw,
        )

        B = int(label_map.shape[0])
        C = int(num_classes)

        # Validate non-255 labels are in [0, C-1].
        valid_mask = label_map != int(self.cfg.ignore_index)
        valid_labels = label_map[valid_mask]
        if valid_labels.numel() > 0:
            if valid_labels.min() < 0 or valid_labels.max() >= C:
                raise ValueError(
                    f"Labels must be in [0, {C - 1}] or {self.cfg.ignore_index}, "
                    f"got min={valid_labels.min().item()}, max={valid_labels.max().item()}."
                )

        # Compute presence and positive pixel counts per class in one pass.
        num_pixels = int(label_map.shape[-2] * label_map.shape[-1])
        positive_pixel_counts = torch.zeros(
            (B, C), dtype=torch.long, device=label_map.device
        )
        presence_target = torch.zeros(
            (B, C), dtype=torch.bool, device=label_map.device
        )

        for class_id in range(C):
            class_pixel_count = (
                label_map == class_id
            ).flatten(1).sum(dim=1)
            positive_pixel_counts[:, class_id] = class_pixel_count
            presence_target[:, class_id] = class_pixel_count > 0

        # Present-negative: present AND has at least one pixel that is NOT
        # this class (255 counts as negative here, matching main BCE behaviour).
        present_negative_target = (
            presence_target
            & (positive_pixel_counts < num_pixels)
        )

        num_present_pairs = int(presence_target.sum().item())
        num_present_negative_pairs = int(present_negative_target.sum().item())
        num_absent_pairs = int((~presence_target).sum().item())

        # Distillation denominator: Σ presence[b,c] × valid_pixel_count[b].
        valid_pixel_count = valid_mask.flatten(1).sum(dim=1)  # [B]
        distill_pixel_denom = int(
            (
                presence_target.to(torch.long)
                * valid_pixel_count[:, None]
            ).sum().item()
        )

        # Determine weights based on which groups are non-empty.
        has_present = num_present_pairs > 0
        has_present_neg = num_present_negative_pairs > 0
        has_absent = num_absent_pairs > 0

        if has_present and has_present_neg and has_absent:
            pos_w, pn_w, an_w = 0.5, 0.25, 0.25
        elif has_present and has_present_neg and not has_absent:
            pos_w, pn_w, an_w = 0.5, 0.5, 0.0
        elif has_present and not has_present_neg and has_absent:
            pos_w, pn_w, an_w = 0.5, 0.0, 0.5
        elif not has_present and has_absent:
            pos_w, pn_w, an_w = 0.0, 0.0, 1.0
        elif has_present and not has_present_neg and not has_absent:
            pos_w, pn_w, an_w = 1.0, 0.0, 0.0
        else:
            # All 255 — treat as absent.
            pos_w, pn_w, an_w = 0.0, 0.0, 1.0

        return SemanticStreamingContext(
            label_map=label_map,
            valid_mask=valid_mask,
            presence_target=presence_target,
            present_negative_target=present_negative_target,
            num_classes=C,
            num_present_pairs=num_present_pairs,
            num_present_negative_pairs=num_present_negative_pairs,
            num_absent_pairs=num_absent_pairs,
            distill_pixel_denom=distill_pixel_denom,
            positive_weight=pos_w,
            present_negative_weight=pn_w,
            absent_negative_weight=an_w,
        )

    def forward_chunk(
        self,
        outputs: TensorDict,
        context: SemanticStreamingContext,
        class_start: int,
        class_end: int,
        target_full: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        """Compute per-chunk loss contributions using global denominators.

        The returned values are *contributions* to the global mean. Summing
        across all chunks yields the exact same result as a single full
        forward.
        """
        final_logits_chunk = outputs[OUTPUT_KEYS.final_logits]
        if final_logits_chunk.ndim != 4:
            raise ValueError(
                f"final_logits must be [B, C_chunk, H, W], "
                f"got {tuple(final_logits_chunk.shape)}."
            )

        B, C_chunk = final_logits_chunk.shape[:2]
        C = context.num_classes
        class_ids = list(range(class_start, class_end))

        if len(class_ids) != C_chunk:
            raise ValueError(
                f"Chunk class count mismatch: class_start={class_start}, "
                f"class_end={class_end}, but logits have {C_chunk} channels."
            )

        label_map = context.label_map
        valid_mask = context.valid_mask
        device = final_logits_chunk.device
        dtype = final_logits_chunk.dtype
        class_ids_t = torch.tensor(
            class_ids, device=device, dtype=torch.long
        )

        # Build boolean and float chunk targets.
        target_bool = (
            label_map[:, None].to(device=device)
            == class_ids_t[None, :, None, None]
        )  # [B, C_chunk, H, W]
        target_chunk = target_bool.to(dtype=dtype)

        zero = self._zero_loss(final_logits_chunk)

        # ---- Balanced BCE (per image-class pair, pos/neg separated) ----
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            final_logits_chunk,
            target_chunk,
            reduction="none",
        )  # [B, C_chunk, H, W]

        # Count positive and negative pixels per pair.
        positive_pixel_count = target_bool.flatten(2).sum(dim=2)  # [B, C_chunk]
        negative_pixel_count = (~target_bool).flatten(2).sum(dim=2)

        # Positive-pixel BCE per pair.
        positive_pair_bce = (
            bce_per_pixel * target_chunk
        ).flatten(2).sum(dim=2)
        positive_pair_bce = positive_pair_bce / positive_pixel_count.to(
            dtype=bce_per_pixel.dtype
        ).clamp_min(1.0)

        # Negative-pixel BCE per pair.
        negative_target = (~target_bool).to(dtype=bce_per_pixel.dtype)
        negative_pair_bce = (
            bce_per_pixel * negative_target
        ).flatten(2).sum(dim=2)
        negative_pair_bce = negative_pair_bce / negative_pixel_count.to(
            dtype=bce_per_pixel.dtype
        ).clamp_min(1.0)

        # Slice masks from context.
        present_mask = context.presence_target[
            :, class_start:class_end
        ].to(device=device)

        present_negative_mask = context.present_negative_target[
            :, class_start:class_end
        ].to(device=device)

        absent_mask = ~present_mask

        # Positive contribution: only present pairs, using positive-pixel BCE.
        positive_contribution = (
            positive_pair_bce * present_mask.to(dtype=positive_pair_bce.dtype)
        ).sum() / max(context.num_present_pairs, 1)

        # Present-negative contribution: present pairs that have negative
        # pixels, using negative-pixel BCE.
        present_negative_contribution = (
            negative_pair_bce
            * present_negative_mask.to(dtype=negative_pair_bce.dtype)
        ).sum() / max(context.num_present_negative_pairs, 1)

        # Absent contribution: absent pairs, using negative-pixel BCE.
        absent_negative_contribution = (
            negative_pair_bce * absent_mask.to(dtype=negative_pair_bce.dtype)
        ).sum() / max(context.num_absent_pairs, 1)

        balanced_bce_contribution = (
            context.positive_weight * positive_contribution
            + context.present_negative_weight * present_negative_contribution
            + context.absent_negative_weight * absent_negative_contribution
        )

        # ---- Dice (optional) ----
        dice_weight = float(self.cfg.final_dice_weight)
        if dice_weight > 0.0 and context.num_present_pairs > 0:
            if target_full is not None:
                target_chunk_full = target_full[:, class_start:class_end]
            else:
                target_chunk_full = target_chunk
            # presence_target from context is bool; convert for Dice.
            chunk_presence_float = present_mask.to(dtype=torch.float32)
            dice_contribution = self._dice_contribution_chunk(
                logits=final_logits_chunk,
                target=target_chunk_full,
                presence_target_chunk=chunk_presence_float,
                global_n_present=context.num_present_pairs,
            )
        else:
            dice_contribution = zero

        # ---- SAM3 teacher distillation ----
        distill_weight = float(self.cfg.sam3_mask_distill_weight)
        if distill_weight > 0.0 and context.distill_pixel_denom > 0:
            if OUTPUT_KEYS.sam3_teacher_logits not in outputs:
                raise ValueError(
                    "sam3_mask_distill_weight > 0 but "
                    f"{OUTPUT_KEYS.sam3_teacher_logits!r} is missing."
                )
            chunk_presence_float = present_mask.to(dtype=torch.float32)
            distill_contribution = self._distill_contribution_chunk(
                final_logits=final_logits_chunk,
                teacher_logits=outputs[OUTPUT_KEYS.sam3_teacher_logits],
                valid_mask=valid_mask,
                presence_target_chunk=chunk_presence_float,
                global_denom=context.distill_pixel_denom,
            )
        else:
            distill_contribution = zero

        # ---- Total ----
        chunk_total = (
            float(self.cfg.final_balanced_bce_weight) * balanced_bce_contribution
            + dice_weight * dice_contribution
            + distill_weight * distill_contribution
        )

        return {
            "loss_positive_bce": positive_contribution,
            "loss_present_negative_bce": present_negative_contribution,
            "loss_absent_negative_bce": absent_negative_contribution,
            "loss_final_balanced_bce": balanced_bce_contribution,
            "loss_final_dice": dice_contribution,
            "loss_sam3_mask_distill_bce": distill_contribution,
            "total_loss": chunk_total,
        }

    # ------------------------------------------------------------------
    # Distillation (per-chunk, global denominator)
    # ------------------------------------------------------------------

    def _distill_contribution_chunk(
        self,
        final_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        presence_target_chunk: torch.Tensor,
        global_denom: int,
    ) -> torch.Tensor:
        """Distillation contribution for one chunk using global denominator."""
        teacher_prob = teacher_logits.detach().sigmoid()

        pixel_ce = F.binary_cross_entropy_with_logits(
            final_logits,
            teacher_prob,
            reduction="none",
        )

        present = presence_target_chunk > 0.5
        if not present.any().item():
            return final_logits.sum() * 0.0

        pair_pixel_mask = (
            present.to(device=pixel_ce.device, dtype=torch.bool)[:, :, None, None]
            & valid_mask.to(device=pixel_ce.device, dtype=torch.bool)[:, None, :, :]
        )
        pixel_weight = pair_pixel_mask.to(dtype=pixel_ce.dtype)

        denom = max(global_denom, 1)
        return (pixel_ce * pixel_weight).sum() / denom

    # ------------------------------------------------------------------
    # Dice (per-chunk, global denominator)
    # ------------------------------------------------------------------

    def _dice_contribution_chunk(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        presence_target_chunk: torch.Tensor,
        global_n_present: int,
    ) -> torch.Tensor:
        prob = logits.sigmoid()

        prob_flat = prob.flatten(2)
        target_flat = target.flatten(2)

        intersection = (prob_flat * target_flat).sum(dim=2)
        denominator = prob_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + float(self.cfg.eps)) / (
            denominator + float(self.cfg.eps)
        )
        dice_loss = 1.0 - dice

        pair_weight = presence_target_chunk.to(dtype=dice_loss.dtype)

        weight_sum = pair_weight.sum()
        if bool(weight_sum.detach().le(0).item()):
            return logits.sum() * 0.0

        pair_mean = (dice_loss * pair_weight).sum() / weight_sum.clamp_min(
            float(self.cfg.eps)
        )
        return pair_mean * (weight_sum / max(global_n_present, 1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        num_channels: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, W = label_map.shape
        ignore_index = int(self.cfg.ignore_index)
        valid_mask = label_map != ignore_index

        target = torch.zeros(
            (B, H, W, int(num_channels)),
            dtype=dtype,
            device=label_map.device,
        )

        valid_labels = label_map[valid_mask]
        if valid_labels.numel() > 0:
            one_hot = F.one_hot(
                valid_labels,
                num_classes=int(num_channels),
            ).to(dtype=dtype)
            target[valid_mask] = one_hot

        target = target.permute(0, 3, 1, 2).contiguous()
        return target, valid_mask


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")
