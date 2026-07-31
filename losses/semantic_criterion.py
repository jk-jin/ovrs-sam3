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
    presence_target: torch.Tensor          # [B, C] bool — kept for Dice / distillation
    num_classes: int
    num_present_pairs: int                 # kept for Dice
    distill_pixel_denom: int               # kept for distillation
    total_valid_pixels: int                # Σ valid pixels × C, denominator for plain BCE
    sam3_mask_distill_weight: float


class SemanticCriterion(nn.Module):
    """Semantic segmentation criterion with streaming per-chunk support.

    Losses:
        1. Plain BCE on final_logits — every valid pixel (label ≠ 255)
           contributes equally regardless of class presence or pixel sign.
           Global mean over all B × C × H × W valid pixel positions.

        2. Optional Dice on final_logits (default weight 0.0)

        3. SAM3 teacher mask distillation BCE (cosine-decayed weight)
           - Frozen SAM3 semantic head logits as soft targets
           - Only present pairs and non-255 pixels
    """

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

        # Validate distillation schedule config.
        initial_weight = float(self.cfg.sam3_mask_distill_weight)
        decay_start_iter = int(
            self.cfg.sam3_mask_distill_decay_start_iter
        )
        decay_end_iter = int(
            self.cfg.sam3_mask_distill_decay_end_iter
        )

        if initial_weight < 0.0:
            raise ValueError(
                "sam3_mask_distill_weight must be >= 0, "
                f"got {initial_weight}."
            )
        if decay_start_iter < 0:
            raise ValueError(
                "sam3_mask_distill_decay_start_iter must be >= 0, "
                f"got {decay_start_iter}."
            )
        if decay_end_iter <= decay_start_iter:
            raise ValueError(
                "sam3_mask_distill_decay_end_iter "
                f"({decay_end_iter}) must be > "
                "sam3_mask_distill_decay_start_iter "
                f"({decay_start_iter})."
            )

    def get_sam3_mask_distill_weight(
        self,
        global_iter: int,
    ) -> float:
        """Compute effective distillation weight for the given training step.

        Uses cosine decay from decay_start_iter to decay_end_iter.
        Returns exact 0.0 when global_iter >= decay_end_iter.
        """
        global_iter = int(global_iter)

        if global_iter < 0:
            raise ValueError(
                f"global_iter must be >= 0, got {global_iter}."
            )

        initial_weight = float(
            self.cfg.sam3_mask_distill_weight
        )
        decay_start_iter = int(
            self.cfg.sam3_mask_distill_decay_start_iter
        )
        decay_end_iter = int(
            self.cfg.sam3_mask_distill_decay_end_iter
        )

        if initial_weight == 0.0:
            return 0.0

        if global_iter <= decay_start_iter:
            return initial_weight

        if global_iter >= decay_end_iter:
            return 0.0

        progress = (
            global_iter - decay_start_iter
        ) / (
            decay_end_iter - decay_start_iter
        )

        return 0.5 * initial_weight * (
            1.0 + math.cos(math.pi * progress)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: TensorDict,
        targets: TensorDict,
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
        global_iter: int = 0,
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
            global_iter=global_iter,
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
        global_iter: int = 0,
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

        # Global BCE denominator: every valid pixel × every class.
        total_valid_pixels = int(valid_mask.sum().item()) * C

        # Presence: which image-class pairs actually appear (for Dice / distillation).
        presence_target = torch.zeros(
            (B, C), dtype=torch.bool, device=label_map.device
        )
        for class_id in range(C):
            presence_target[:, class_id] = (
                (label_map == class_id).flatten(1).sum(dim=1) > 0
            )

        num_present_pairs = int(presence_target.sum().item())

        # Distillation denominator: Σ presence[b,c] × valid_pixel_count[b].
        valid_pixel_count = valid_mask.flatten(1).sum(dim=1)  # [B]
        distill_pixel_denom = int(
            (
                presence_target.to(torch.long)
                * valid_pixel_count[:, None]
            ).sum().item()
        )

        distill_weight = self.get_sam3_mask_distill_weight(
            global_iter
        )

        return SemanticStreamingContext(
            label_map=label_map,
            valid_mask=valid_mask,
            presence_target=presence_target,
            num_classes=C,
            num_present_pairs=num_present_pairs,
            distill_pixel_denom=distill_pixel_denom,
            total_valid_pixels=total_valid_pixels,
            sam3_mask_distill_weight=distill_weight,
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

        # ---- Plain BCE (every valid pixel equal, global mean) ----
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            final_logits_chunk,
            target_chunk,
            reduction="none",
        )  # [B, C_chunk, H, W]

        valid_float = valid_mask.to(
            device=device, dtype=bce_per_pixel.dtype
        )[:, None, :, :]  # [B, 1, H, W]

        plain_bce_contribution = (
            bce_per_pixel * valid_float
        ).sum() / max(context.total_valid_pixels, 1)

        # ---- Dice (optional) ----
        present_mask = context.presence_target[
            :, class_start:class_end
        ].to(device=device)

        dice_weight = float(self.cfg.final_dice_weight)
        if dice_weight > 0.0 and context.num_present_pairs > 0:
            if target_full is not None:
                target_chunk_full = target_full[:, class_start:class_end]
            else:
                target_chunk_full = target_chunk
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
        distill_weight = float(context.sam3_mask_distill_weight)
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

        weighted_distill_contribution = (
            distill_weight * distill_contribution
        )

        # ---- Total ----
        chunk_total = (
            float(self.cfg.final_balanced_bce_weight) * plain_bce_contribution
            + dice_weight * dice_contribution
            + weighted_distill_contribution
        )

        return {
            "loss_final_bce": plain_bce_contribution,
            "loss_final_dice": dice_contribution,
            "loss_sam3_mask_distill_bce": distill_contribution,
            "loss_sam3_mask_distill_weighted": weighted_distill_contribution,
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
