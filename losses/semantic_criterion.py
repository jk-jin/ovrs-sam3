from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


@dataclass
class SemanticStreamingContext:
    """Global statistics shared by all prompt chunks."""

    label_map: torch.Tensor

    # label != ignore_index
    valid_mask: torch.Tensor

    # [P], each prompt channel maps to one original forward class id.
    prompt_to_class_id: torch.Tensor

    # [B, P], prompt-level presence inherited from original classes.
    presence_target: torch.Tensor

    # [B, P] bool, selected by the distillation class scope.
    distill_pair_mask: torch.Tensor

    num_prompt_channels: int
    num_label_classes: int

    # Number of present image-prompt pairs.
    num_present_pairs: int

    # Global number of selected image-prompt pairs (all chunks).
    num_distill_pairs: int

    # Number of valid pixels multiplied by P.
    total_valid_pixels: int

    sam3_mask_distill_weight: float


class SemanticCriterion(nn.Module):
    """Semantic segmentation criterion with streaming per-chunk support.

    Losses:
        1. Plain BCE on final_logits — every valid pixel (label ≠ 255)
           contributes equally regardless of class presence or pixel sign.
           Global mean over all B × P × H × W valid pixel positions.

        2. Optional Dice on final_logits (default weight 0.0)

        3. SAM3 teacher mask distillation BCE (fixed weight)
           - Frozen SAM3 semantic head logits as soft targets
           - All or only present image-prompt pairs
           - Full image, including GT ignore pixels
           - Separate means above/below the teacher probability threshold,
             weighted by a and 1-a, then averaged over selected pairs
    """

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

        distill_weight = float(self.cfg.sam3_mask_distill_weight)
        if not math.isfinite(distill_weight) or distill_weight < 0.0:
            raise ValueError(
                "sam3_mask_distill_weight must be finite and >= 0, "
                f"got {distill_weight}."
            )

        for name in (
            "sam3_mask_distill_teacher_threshold",
            "sam3_mask_distill_high_weight",
        ):
            value = float(getattr(self.cfg, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        if self.cfg.sam3_mask_distill_class_scope not in ("present", "all"):
            raise ValueError(
                "sam3_mask_distill_class_scope must be 'present' or 'all', "
                f"got {self.cfg.sam3_mask_distill_class_scope!r}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: TensorDict,
        targets: TensorDict,
        prompt_to_class_id: torch.Tensor,
        num_label_classes: int,
        reduction: str = "mean",
    ) -> TensorDict:
        """Full prompt-space loss forward."""
        if reduction != "mean":
            raise ValueError(
                "SemanticCriterion only supports reduction='mean', "
                f"got {reduction!r}."
            )

        if OUTPUT_KEYS.final_logits not in outputs:
            raise ValueError(
                "SemanticCriterion requires "
                f"outputs[{OUTPUT_KEYS.final_logits!r}]."
            )

        final_logits = self._extract_4d_tensor(
            outputs,
            OUTPUT_KEYS.final_logits,
            "[B, P, H, W]",
        )

        num_prompt_channels = int(final_logits.shape[1])

        context = self.prepare_streaming_context(
            targets=targets,
            num_prompt_channels=num_prompt_channels,
            prompt_to_class_id=prompt_to_class_id,
            num_label_classes=int(num_label_classes),
            target_hw=final_logits.shape[-2:],
        )

        target_full, _ = self._build_binary_targets(
            label_map=context.label_map,
            prompt_to_class_id=context.prompt_to_class_id,
            dtype=final_logits.dtype,
        )

        return self.forward_chunk(
            outputs=outputs,
            context=context,
            class_start=0,
            class_end=num_prompt_channels,
            target_full=target_full,
        )

    def prepare_streaming_context(
        self,
        targets: TensorDict,
        num_prompt_channels: int,
        prompt_to_class_id: torch.Tensor,
        num_label_classes: int,
        target_hw: tuple[int, int] = (288, 288),
    ) -> SemanticStreamingContext:
        """Build full-batch statistics before the prompt chunk loop."""
        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_hw(
            label_map=label_map,
            target_hw=target_hw,
        )

        num_prompt_channels = int(num_prompt_channels)
        num_label_classes = int(num_label_classes)

        if num_prompt_channels <= 0:
            raise ValueError(
                "num_prompt_channels must be positive."
            )

        if num_label_classes <= 0:
            raise ValueError(
                "num_label_classes must be positive."
            )

        prompt_to_class_id = torch.as_tensor(
            prompt_to_class_id,
            dtype=torch.long,
            device=label_map.device,
        )

        if prompt_to_class_id.ndim != 1:
            raise ValueError(
                "prompt_to_class_id must be a 1D tensor, "
                f"got shape={tuple(prompt_to_class_id.shape)}."
            )

        if int(prompt_to_class_id.numel()) != num_prompt_channels:
            raise ValueError(
                "prompt_to_class_id length must equal the number of "
                f"prompt channels. Got {prompt_to_class_id.numel()} "
                f"and {num_prompt_channels}."
            )

        if prompt_to_class_id.numel() > 0:
            min_class_id = int(prompt_to_class_id.min().item())
            max_class_id = int(prompt_to_class_id.max().item())

            if min_class_id < 0 or max_class_id >= num_label_classes:
                raise ValueError(
                    "prompt_to_class_id values must be in "
                    f"[0, {num_label_classes - 1}], got "
                    f"min={min_class_id}, max={max_class_id}."
                )

        prompt_counts = torch.bincount(
            prompt_to_class_id,
            minlength=num_label_classes,
        )

        if bool((prompt_counts == 0).any().item()):
            missing = torch.nonzero(
                prompt_counts == 0,
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                "Every original forward class must own at least one "
                f"prompt. Missing class ids: {missing}."
            )

        ignore_index = int(self.cfg.ignore_index)
        valid_mask = label_map != ignore_index

        valid_labels = label_map[valid_mask]
        if valid_labels.numel() > 0:
            min_label = int(valid_labels.min().item())
            max_label = int(valid_labels.max().item())

            if min_label < 0 or max_label >= num_label_classes:
                raise ValueError(
                    "Labels must be in "
                    f"[0, {num_label_classes - 1}] or "
                    f"{ignore_index}, got min={min_label}, "
                    f"max={max_label}."
                )

        total_valid_pixels = (
            int(valid_mask.sum().item())
            * num_prompt_channels
        )

        class_ids = torch.arange(
            num_label_classes,
            device=label_map.device,
            dtype=torch.long,
        )

        # [B, M, H, W]
        class_mask_by_class = (
            label_map[:, None, :, :]
            == class_ids[None, :, None, None]
        )

        # [B, M]
        original_presence = (
            class_mask_by_class
            .flatten(2)
            .any(dim=2)
        )

        # [B, P]
        presence_target = original_presence.index_select(
            dim=1,
            index=prompt_to_class_id,
        )

        num_present_pairs = int(
            presence_target.sum().item()
        )

        # GT determines presence only; it never restricts distillation pixels.
        distill_pair_mask = (
            presence_target
            if self.cfg.sam3_mask_distill_class_scope == "present"
            else torch.ones_like(presence_target)
        )
        num_distill_pairs = int(distill_pair_mask.sum().item())

        return SemanticStreamingContext(
            label_map=label_map,
            valid_mask=valid_mask,
            prompt_to_class_id=prompt_to_class_id,
            presence_target=presence_target,
            distill_pair_mask=distill_pair_mask,
            num_prompt_channels=num_prompt_channels,
            num_label_classes=num_label_classes,
            num_present_pairs=num_present_pairs,
            num_distill_pairs=num_distill_pairs,
            total_valid_pixels=total_valid_pixels,
            sam3_mask_distill_weight=float(
                self.cfg.sam3_mask_distill_weight
            ),
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
        num_prompt_channels = context.num_prompt_channels

        if not (
            0
            <= int(class_start)
            < int(class_end)
            <= num_prompt_channels
        ):
            raise ValueError(
                "Prompt chunk indices are out of range: "
                f"class_start={class_start}, "
                f"class_end={class_end}, "
                f"num_prompt_channels={num_prompt_channels}."
            )

        prompt_class_ids = context.prompt_to_class_id[
            class_start:class_end
        ]

        if int(prompt_class_ids.numel()) != C_chunk:
            raise ValueError(
                "Prompt chunk count mismatch: "
                f"class_start={class_start}, "
                f"class_end={class_end}, "
                f"logit_channels={C_chunk}."
            )

        label_map = context.label_map
        valid_mask = context.valid_mask
        device = final_logits_chunk.device
        dtype = final_logits_chunk.dtype

        prompt_class_ids = prompt_class_ids.to(
            device=device,
            dtype=torch.long,
        )

        # Build boolean and float chunk targets.
        target_bool = (
            label_map[:, None].to(device=device)
            == prompt_class_ids[None, :, None, None]
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
        if distill_weight > 0.0 and context.num_distill_pairs > 0:
            if OUTPUT_KEYS.sam3_teacher_logits not in outputs:
                raise ValueError(
                    "sam3_mask_distill_weight > 0 but "
                    f"{OUTPUT_KEYS.sam3_teacher_logits!r} is missing."
                )
            distill_contribution = self._distill_contribution_chunk(
                final_logits=final_logits_chunk,
                teacher_logits=outputs[OUTPUT_KEYS.sam3_teacher_logits],
                distill_pair_mask=context.distill_pair_mask[
                    :, class_start:class_end
                ],
                global_num_pairs=context.num_distill_pairs,
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
    # Distillation (per-pair region means, global pair denominator)
    # ------------------------------------------------------------------

    def _distill_contribution_chunk(
        self,
        final_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        distill_pair_mask: torch.Tensor,
        global_num_pairs: int,
    ) -> torch.Tensor:
        """Full-image soft BCE, balanced by teacher-defined regions per pair."""
        if teacher_logits.shape != final_logits.shape:
            raise ValueError("Teacher and student logits must have the same shape.")

        # FP32 reductions avoid overflow on full 288x288 masks under AMP.
        teacher_prob = teacher_logits.detach().float().sigmoid()
        pixel_ce = F.binary_cross_entropy_with_logits(
            final_logits.float(), teacher_prob, reduction="none"
        )
        high_mask = teacher_prob > float(
            self.cfg.sam3_mask_distill_teacher_threshold
        )
        low_mask = ~high_mask

        # Empty regions contribute zero; the other coefficient is unchanged.
        high_mean = (pixel_ce * high_mask).sum(dim=(-2, -1)) / (
            high_mask.sum(dim=(-2, -1)).clamp_min(1)
        )
        low_mean = (pixel_ce * low_mask).sum(dim=(-2, -1)) / (
            low_mask.sum(dim=(-2, -1)).clamp_min(1)
        )
        high_weight = float(self.cfg.sam3_mask_distill_high_weight)
        pair_loss = high_weight * high_mean + (1.0 - high_weight) * low_mean
        selected = distill_pair_mask.to(device=pair_loss.device)
        return (pair_loss * selected).sum() / max(global_num_pairs, 1)

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
        prompt_to_class_id: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ignore_index = int(self.cfg.ignore_index)
        valid_mask = label_map != ignore_index

        prompt_to_class_id = torch.as_tensor(
            prompt_to_class_id,
            dtype=torch.long,
            device=label_map.device,
        )

        if prompt_to_class_id.ndim != 1:
            raise ValueError(
                "prompt_to_class_id must be 1D."
            )

        target = (
            label_map[:, None]
            == prompt_to_class_id[None, :, None, None]
        ).to(dtype=dtype)

        target = target * valid_mask[:, None].to(
            dtype=dtype
        )

        return target.contiguous(), valid_mask


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")
