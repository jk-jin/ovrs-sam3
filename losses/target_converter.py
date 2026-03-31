from __future__ import annotations

from typing import Dict

import torch

from ..models.box_ops import box_cxcywh_to_xyxy
from ..models.data_misc import BatchedDatapoint, BatchedFindTarget


class TargetConverter:
    """Convert your dataclass targets into criterion-friendly dictionaries."""

    @staticmethod
    def _positive_map_from_boxes(targets: BatchedFindTarget) -> torch.Tensor:
        # Current simplified setup uses one prompt / one binary label channel.
        num_instances = int(len(targets.boxes))
        return targets.boxes.new_ones((num_instances, 1))

    @classmethod
    def for_instance(cls, targets: BatchedFindTarget) -> Dict[str, torch.Tensor]:
        boxes = targets.boxes.view(-1, 4)
        return {
            "boxes": boxes,
            "boxes_xyxy": box_cxcywh_to_xyxy(boxes),
            "boxes_padded": targets.boxes_padded,
            "positive_map": cls._positive_map_from_boxes(targets),
            "num_boxes": targets.num_boxes,
            "masks": targets.segments,
            "semantic_masks": targets.semantic_segments,
            "is_valid_mask": targets.is_valid_segment,
            "is_exhaustive": targets.is_exhaustive,
            "object_ids_packed": targets.object_ids,
            "object_ids_padded": targets.object_ids_padded,
        }

    @staticmethod
    def for_semantic(targets: BatchedFindTarget) -> Dict[str, torch.Tensor]:
        return {
            "semantic_masks": targets.semantic_segments,
            "is_valid_mask": targets.is_valid_segment,
            "is_exhaustive": targets.is_exhaustive,
            # Keeping instance masks available is often useful for debugging.
            "instance_masks": targets.segments,
            "num_boxes": targets.num_boxes,
        }

    @classmethod
    def from_batch(cls, batch: BatchedDatapoint, task: str) -> Dict[str, torch.Tensor]:
        assert len(batch.find_targets) == 1, (
            "Current simplified trainer assumes exactly one target stage per batch."
        )
        target = batch.find_targets[0]
        if task == "instance":
            return cls.for_instance(target)
        if task == "semantic":
            return cls.for_semantic(target)
        raise ValueError(f"Unsupported task: {task}")
