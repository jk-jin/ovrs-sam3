from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


@dataclass
class SemanticStreamingContext:
    """Global image/class statistics shared by every prompt chunk."""

    label_map: torch.Tensor
    valid_mask: torch.Tensor
    prompt_to_class_id: torch.Tensor
    presence_target: torch.Tensor  # [B, P], inherited from original classes
    prompt_weights: torch.Tensor  # [P], reciprocal prompts per original class
    distill_outer_boundary_by_class: torch.Tensor  # [B, M, H, W]
    distill_active_prompts: list[bool]  # [P], CPU flags avoid per-chunk GPU sync
    num_prompt_channels: int
    num_label_classes: int
    num_present_pairs: int  # prompt pairs, used only by optional Dice
    num_present_classes: int  # image/original-class pairs across the full batch
    num_absent_classes: int
    sam3_mask_distill_weight: float

    def needs_teacher_logits(self, class_start: int, class_end: int) -> bool:
        return self.sam3_mask_distill_weight > 0.0 and any(
            self.distill_active_prompts[class_start:class_end]
        )


class SemanticCriterion(nn.Module):
    """Region means, then original-class means, then presence-group weights.

    Main BCE uses GT binary targets (ignore pixels are zero targets for
    present classes). Distillation always uses detached teacher probabilities.
    Chunk contributions sum to the full-batch loss without retaining logits.
    """

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

        for name in (
            "final_bce_present_valid_weight",
            "final_bce_present_class_weight",
            "sam3_mask_distill_present_region_weight",
            "sam3_mask_distill_present_class_weight",
        ):
            value = float(getattr(self.cfg, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        distill_weight = float(self.cfg.sam3_mask_distill_weight)
        if distill_weight < 0.0:
            raise ValueError(
                "sam3_mask_distill_weight must be >= 0, "
                f"got {distill_weight}."
            )

        boundary_width = (
            self.cfg.sam3_mask_distill_boundary_width
        )

        if (
            isinstance(boundary_width, bool)
            or not isinstance(boundary_width, int)
        ):
            raise TypeError(
                "sam3_mask_distill_boundary_width must be "
                "a non-negative integer, "
                f"got {boundary_width!r}."
            )

        if boundary_width < 0:
            raise ValueError(
                "sam3_mask_distill_boundary_width must be >= 0, "
                f"got {boundary_width}."
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
        ignore_mask = label_map == ignore_index

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

        boundary_width = int(
            self.cfg.sam3_mask_distill_boundary_width
        )

        # [B, M, H, W]
        distill_outer_boundary_by_class = (
            self._build_outer_distill_boundary_by_class(
                class_mask_by_class=class_mask_by_class,
                ignore_mask=ignore_mask,
                boundary_width=boundary_width,
            )
        )

        # Group denominators count original classes, never expanded prompts.
        num_present_classes = int(original_presence.sum().item())
        num_absent_classes = original_presence.numel() - num_present_classes
        prompt_weights = (
            prompt_counts.index_select(0, prompt_to_class_id).float().reciprocal()
        )

        primary_region = valid_mask[:, None] | distill_outer_boundary_by_class
        region_weight = float(self.cfg.sam3_mask_distill_present_region_weight)
        class_weight = float(self.cfg.sam3_mask_distill_present_class_weight)
        present_active = (
            (region_weight > 0.0) & primary_region.flatten(2).any(2)
        ) | (
            (region_weight < 1.0) & (~primary_region).flatten(2).any(2)
        )
        absent_active = valid_mask.flatten(1).any(1)[:, None]
        active_by_class = (
            original_presence & present_active & (class_weight > 0.0)
        ) | (
            ~original_presence & absent_active & (class_weight < 1.0)
        )

        return SemanticStreamingContext(
            label_map=label_map,
            valid_mask=valid_mask,
            prompt_to_class_id=prompt_to_class_id,
            presence_target=presence_target,
            prompt_weights=prompt_weights,
            distill_outer_boundary_by_class=distill_outer_boundary_by_class,
            distill_active_prompts=(
                active_by_class.index_select(1, prompt_to_class_id).any(dim=0).tolist()
            ),
            num_prompt_channels=num_prompt_channels,
            num_label_classes=num_label_classes,
            num_present_pairs=num_present_pairs,
            num_present_classes=num_present_classes,
            num_absent_classes=num_absent_classes,
            sam3_mask_distill_weight=float(self.cfg.sam3_mask_distill_weight),
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
        if (B, *final_logits_chunk.shape[-2:]) != tuple(context.label_map.shape):
            raise ValueError(
                "Chunk logits must match the streaming label batch and spatial shape."
            )
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

        present_mask = context.presence_target[:, class_start:class_end].to(device)
        prompt_weights = context.prompt_weights[class_start:class_end].to(device)
        valid_region = valid_mask.to(device)[:, None]

        bce_per_pixel = F.binary_cross_entropy_with_logits(
            final_logits_chunk, target_chunk, reduction="none",
        )
        valid_bce = self._region_mean(bce_per_pixel, valid_region)
        ignore_bce = self._region_mean(bce_per_pixel, ~valid_region)
        valid_weight = float(self.cfg.final_bce_present_valid_weight)
        present_bce = valid_weight * valid_bce + (1.0 - valid_weight) * ignore_bce
        pair_bce = torch.where(present_mask, present_bce, valid_bce)
        grouped_bce_contribution = self._grouped_contribution(
            pair_bce, present_mask, prompt_weights, context,
            float(self.cfg.final_bce_present_class_weight),
        )

        # ---- Dice (optional, unchanged prompt-level reduction) ----
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
        if context.needs_teacher_logits(class_start, class_end):
            teacher_logits = self._extract_4d_tensor(
                outputs, OUTPUT_KEYS.sam3_teacher_logits, "[B, P_chunk, H, W]",
            )
            if teacher_logits.shape != final_logits_chunk.shape:
                raise ValueError("Teacher and student logits must have identical shapes.")
            pixel_distill = F.binary_cross_entropy_with_logits(
                final_logits_chunk, teacher_logits.detach().sigmoid(), reduction="none",
            )
            outer_boundary = context.distill_outer_boundary_by_class.index_select(
                1, prompt_class_ids,
            ).to(device)
            primary_region = valid_region | outer_boundary
            primary_mean = self._region_mean(pixel_distill, primary_region)
            other_mean = self._region_mean(pixel_distill, ~primary_region)
            region_weight = float(self.cfg.sam3_mask_distill_present_region_weight)
            present_distill = (
                region_weight * primary_mean + (1.0 - region_weight) * other_mean
            )
            absent_distill = self._region_mean(pixel_distill, valid_region)
            pair_distill = torch.where(present_mask, present_distill, absent_distill)
            distill_contribution = self._grouped_contribution(
                pair_distill, present_mask, prompt_weights, context,
                float(self.cfg.sam3_mask_distill_present_class_weight),
            )
        else:
            distill_contribution = zero

        weighted_distill_contribution = (
            distill_weight * distill_contribution
        )

        # ---- Total ----
        chunk_total = (
            float(self.cfg.final_balanced_bce_weight) * grouped_bce_contribution
            + dice_weight * dice_contribution
            + weighted_distill_contribution
        )

        return {
            "loss_final_bce": grouped_bce_contribution,
            "loss_final_dice": dice_contribution,
            "loss_sam3_mask_distill_bce": distill_contribution,
            "loss_sam3_mask_distill_weighted": weighted_distill_contribution,
            "total_loss": chunk_total,
        }

    # ------------------------------------------------------------------
    # Region construction and grouped reduction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_outer_distill_boundary_by_class(
        class_mask_by_class: torch.Tensor,
        ignore_mask: torch.Tensor,
        boundary_width: int,
    ) -> torch.Tensor:
        """Build class-specific outer GT boundary bands.

        Args:
            class_mask_by_class:
                Boolean tensor [B, M, H, W]. Each channel is
                the GT mask of one original label class.
            ignore_mask:
                Boolean tensor [B, H, W], label == ignore_index.
            boundary_width:
                Width of the outer boundary at loss resolution.
                Zero disables the outer boundary.

        Returns:
            Boolean tensor [B, M, H, W]. Only ignore pixels
            within the configured distance from each class mask
            are True.
        """
        if boundary_width == 0:
            return torch.zeros_like(
                class_mask_by_class,
                dtype=torch.bool,
            )

        kernel_size = 2 * boundary_width + 1

        dilated = (
            F.max_pool2d(
                class_mask_by_class.to(dtype=torch.float32),
                kernel_size=kernel_size,
                stride=1,
                padding=boundary_width,
            )
            > 0.5
        )

        outer_boundary = (
            dilated
            & ~class_mask_by_class
            & ignore_mask[:, None, :, :]
        )

        return outer_boundary.contiguous()

    @staticmethod
    def _region_mean(pixel_loss: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """Per-image/prompt means; an empty region contributes zero.

        Accumulate in FP32 for AMP: a 288x288 pixel sum can overflow FP16.
        Region may broadcast across prompts, so no full-size float mask is kept.
        """
        numerator = pixel_loss.masked_fill(~region, 0).sum(
            dim=(-2, -1), dtype=torch.float32,
        )
        denominator = region.sum(dim=(-2, -1)).clamp_min(1)
        return numerator / denominator

    @staticmethod
    def _grouped_contribution(
        pair_loss: torch.Tensor,
        present: torch.Tensor,
        prompt_weights: torch.Tensor,
        context: SemanticStreamingContext,
        present_weight: float,
    ) -> torch.Tensor:
        # Dividing each prompt by its class prompt count first ensures that
        # synonyms count as one original class, even across chunk boundaries.
        group_weights = torch.where(
            present,
            present_weight / max(context.num_present_classes, 1),
            (1.0 - present_weight) / max(context.num_absent_classes, 1),
        )
        return (pair_loss * prompt_weights[None] * group_weights).sum()

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
        return reference.sum(dtype=torch.float32) * 0.0

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
