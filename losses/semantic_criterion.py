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
    """Global statistics shared by all prompt chunks."""

    label_map: torch.Tensor

    # label != ignore_index
    valid_mask: torch.Tensor

    # [P], each prompt channel maps to one original forward class id.
    prompt_to_class_id: torch.Tensor

    # [B, P], prompt-level presence inherited from original classes.
    presence_target: torch.Tensor

    # [B, M, H, W] bool.
    # M is the number of original label classes.
    # Only contains class-specific outer boundaries in ignore pixels.
    distill_outer_boundary_by_class: torch.Tensor

    num_prompt_channels: int
    num_label_classes: int

    # Number of present image-prompt pairs.
    num_present_pairs: int

    # Total selected distillation pixels across all present prompts.
    distill_pixel_denom: int

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
           - Only present image-prompt pairs
           - All valid GT pixels plus a configurable class-specific
             outer boundary restricted to ignore pixels
    """

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

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

        # Every present prompt distills all valid pixels.
        # [B]
        valid_pixel_count = (
            valid_mask
            .to(dtype=torch.long)
            .flatten(1)
            .sum(dim=1)
        )

        # Each original class additionally owns its own outer boundary.
        # [B, M]
        outer_boundary_pixel_count_by_class = (
            distill_outer_boundary_by_class
            .to(dtype=torch.long)
            .flatten(2)
            .sum(dim=2)
        )

        # Valid pixels and outer boundaries are disjoint because the
        # outer boundary is restricted to ignore_mask.
        # [B, M]
        distill_pixel_count_by_class = (
            valid_pixel_count[:, None]
            + outer_boundary_pixel_count_by_class
        )

        # Multiple prompts belonging to the same original class reuse
        # and independently count the same class boundary.
        # [B, P]
        distill_pixel_count_by_prompt = (
            distill_pixel_count_by_class.index_select(
                dim=1,
                index=prompt_to_class_id,
            )
        )

        # Global denominator across the complete prompt space.
        distill_pixel_denom = int(
            (
                distill_pixel_count_by_prompt
                * presence_target.to(dtype=torch.long)
            )
            .sum()
            .item()
        )

        return SemanticStreamingContext(
            label_map=label_map,
            valid_mask=valid_mask,
            prompt_to_class_id=prompt_to_class_id,
            presence_target=presence_target,
            distill_outer_boundary_by_class=(
                distill_outer_boundary_by_class
            ),
            num_prompt_channels=num_prompt_channels,
            num_label_classes=num_label_classes,
            num_present_pairs=num_present_pairs,
            distill_pixel_denom=distill_pixel_denom,
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
        if distill_weight > 0.0 and context.distill_pixel_denom > 0:
            if OUTPUT_KEYS.sam3_teacher_logits not in outputs:
                raise ValueError(
                    "sam3_mask_distill_weight > 0 but "
                    f"{OUTPUT_KEYS.sam3_teacher_logits!r} is missing."
                )
            chunk_presence_float = present_mask.to(dtype=torch.float32)

            outer_boundary_chunk = (
                context.distill_outer_boundary_by_class.index_select(
                    dim=1,
                    index=prompt_class_ids,
                )
            )

            # Fixed design:
            # all valid pixels + class-specific outer ignore boundary.
            distill_mask_chunk = (
                context.valid_mask[:, None, :, :]
                | outer_boundary_chunk
            )

            distill_contribution = self._distill_contribution_chunk(
                final_logits=final_logits_chunk,
                teacher_logits=outputs[
                    OUTPUT_KEYS.sam3_teacher_logits
                ],
                distill_mask_chunk=distill_mask_chunk,
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

    def _distill_contribution_chunk(
        self,
        final_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        distill_mask_chunk: torch.Tensor,
        presence_target_chunk: torch.Tensor,
        global_denom: int,
    ) -> torch.Tensor:
        """Distill present prompts on valid pixels and outer GT boundaries."""
        teacher_prob = teacher_logits.detach().sigmoid()

        pixel_ce = F.binary_cross_entropy_with_logits(
            final_logits,
            teacher_prob,
            reduction="none",
        )

        present = presence_target_chunk > 0.5

        if not bool(present.any().item()):
            return final_logits.sum() * 0.0

        pair_pixel_mask = (
            present.to(
                device=pixel_ce.device,
                dtype=torch.bool,
            )[:, :, None, None]
            & distill_mask_chunk.to(
                device=pixel_ce.device,
                dtype=torch.bool,
            )
        )

        pixel_weight = pair_pixel_mask.to(
            dtype=pixel_ce.dtype
        )

        return (
            pixel_ce * pixel_weight
        ).sum() / max(int(global_denom), 1)

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
