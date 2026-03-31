from __future__ import annotations

import math
from dataclasses import is_dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..models.box_ops import box_xyxy_to_cxcywh

try:
    from ..models.data_misc import (
        BatchedDatapoint,
        BatchedFindTarget,
        BatchedInferenceMetadata,
        FindStage,
    )
except Exception as e:  # pragma: no cover - project integration fallback
    raise ImportError(
        'SAM3BatchCollator expects your extracted official `models/data_misc.py` to be present.'
    ) from e


Sample = MutableMapping[str, Any]


def _pad_tensor_hw(x: torch.Tensor, out_h: int, out_w: int, value: float = 0.0) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=value)


def _round_up(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)


def _normalize_boxes_cxcywh(boxes_cxcywh: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh.reshape(0, 4)
    scale = boxes_cxcywh.new_tensor([w, h, w, h])
    return boxes_cxcywh / scale.clamp(min=1.0)


def _boxes_to_cxcywh(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).float()
    if box_format == 'cxcywh':
        return boxes.float()
    if box_format == 'xyxy':
        return box_xyxy_to_cxcywh(boxes.float())
    raise ValueError(f'Unsupported box_format: {box_format}')


class SAM3BatchCollator:
    """Convert per-sample dicts into the official SAM3 batch dataclasses.

    Expected per-sample keys from the dataset:
    - image: Tensor[C, H, W]
    - text: str
    - boxes: Tensor[N, 4]
    - instance_masks: Tensor[N, H, W] or empty tensor
    - semantic_mask: Tensor[H, W] or None
    - object_ids: Tensor[N]
    - image_id: int
    - original_size: tuple(H, W)
    - bbox_format: 'xyxy' or 'cxcywh'
    - is_exhaustive: bool
    """

    def __init__(
        self,
        image_pad_value: float = 0.0,
        pad_size_divisor: int = 1,
        normalize_boxes: bool = True,
        box_format: str = 'xyxy',
    ):
        self.image_pad_value = float(image_pad_value)
        self.pad_size_divisor = int(pad_size_divisor)
        self.normalize_boxes = bool(normalize_boxes)
        self.box_format = box_format

    def _collate_images(self, samples: Sequence[Sample]) -> tuple[torch.Tensor, list[tuple[int, int]], tuple[int, int]]:
        sizes = [(int(s['image'].shape[-2]), int(s['image'].shape[-1])) for s in samples]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)
        if self.pad_size_divisor > 1:
            max_h = _round_up(max_h, self.pad_size_divisor)
            max_w = _round_up(max_w, self.pad_size_divisor)
        imgs = [_pad_tensor_hw(s['image'], max_h, max_w, self.image_pad_value) for s in samples]
        return torch.stack(imgs, dim=0), sizes, (max_h, max_w)

    def __call__(self, samples: Sequence[Sample]) -> BatchedDatapoint:
        samples = list(samples)
        if len(samples) == 0:
            raise ValueError('Empty batch.')

        img_batch, image_sizes, padded_hw = self._collate_images(samples)
        batch_size = len(samples)
        batch_texts = [str(s['text']) for s in samples]

        input_boxes = torch.zeros((0, batch_size, 4), dtype=torch.float32)
        input_boxes_mask = torch.zeros((batch_size, 0), dtype=torch.bool)
        input_boxes_label = torch.zeros((0, batch_size), dtype=torch.long)
        input_points = torch.zeros((0, batch_size, 2), dtype=torch.float32)
        input_points_mask = torch.zeros((batch_size, 0), dtype=torch.bool)
        find_stage = FindStage(
            img_ids=torch.arange(batch_size, dtype=torch.long),
            text_ids=torch.arange(batch_size, dtype=torch.long),
            input_boxes=input_boxes,
            input_boxes_mask=input_boxes_mask,
            input_boxes_label=input_boxes_label,
            input_points=input_points,
            input_points_mask=input_points_mask,
            object_ids=None,
        )

        packed_boxes: List[torch.Tensor] = []
        packed_object_ids: List[torch.Tensor] = []
        num_boxes_list: List[int] = []
        is_exhaustive_list: List[bool] = []
        semantic_list: List[torch.Tensor] = []
        valid_semantic_list: List[bool] = []
        original_size_list: List[torch.Tensor] = []
        image_id_list: List[int] = []

        n_max = max(int(s.get('boxes', torch.zeros((0, 4))).shape[0]) for s in samples)
        padded_boxes = torch.zeros((batch_size, n_max, 4), dtype=torch.float32)
        padded_masks = torch.zeros((batch_size, n_max, padded_hw[0], padded_hw[1]), dtype=torch.bool)
        object_ids_padded = torch.zeros((batch_size, n_max), dtype=torch.long)

        for b, sample in enumerate(samples):
            h, w = image_sizes[b]
            boxes = sample.get('boxes')
            if boxes is None:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            boxes = boxes.float().reshape(-1, 4)
            cur_box_format = str(sample.get('bbox_format', self.box_format))
            boxes_cxcywh = _boxes_to_cxcywh(boxes, cur_box_format)
            if self.normalize_boxes:
                boxes_cxcywh = _normalize_boxes_cxcywh(boxes_cxcywh, h=h, w=w)

            num_boxes = int(boxes_cxcywh.shape[0])
            num_boxes_list.append(num_boxes)
            packed_boxes.append(boxes_cxcywh)
            if num_boxes > 0:
                padded_boxes[b, :num_boxes] = boxes_cxcywh

            instance_masks = sample.get('instance_masks')
            if instance_masks is None:
                instance_masks = torch.zeros((0, h, w), dtype=torch.bool)
            if instance_masks.ndim == 2:
                instance_masks = instance_masks[None]
            instance_masks = instance_masks.bool()
            if instance_masks.shape[-2:] != (padded_hw[0], padded_hw[1]):
                instance_masks = _pad_tensor_hw(instance_masks, padded_hw[0], padded_hw[1], 0).bool()
            if num_boxes > 0:
                padded_masks[b, :num_boxes] = instance_masks[:num_boxes]

            semantic_mask = sample.get('semantic_mask')
            if semantic_mask is None:
                semantic_list.append(torch.zeros(padded_hw, dtype=torch.bool))
                valid_semantic_list.append(False)
            else:
                semantic_mask = semantic_mask.bool()
                if semantic_mask.shape[-2:] != (padded_hw[0], padded_hw[1]):
                    semantic_mask = _pad_tensor_hw(semantic_mask, padded_hw[0], padded_hw[1], 0).bool()
                semantic_list.append(semantic_mask)
                valid_semantic_list.append(True)

            object_ids = sample.get('object_ids')
            if object_ids is None:
                object_ids = torch.arange(1, num_boxes + 1, dtype=torch.long)
            object_ids = object_ids.long().reshape(-1)
            if num_boxes > 0:
                object_ids_padded[b, :num_boxes] = object_ids[:num_boxes]
                packed_object_ids.append(object_ids[:num_boxes])
            else:
                packed_object_ids.append(torch.zeros((0,), dtype=torch.long))

            image_id_list.append(int(sample.get('image_id', b)))
            orig_h, orig_w = sample.get('original_size', (h, w))
            original_size_list.append(torch.tensor([orig_h, orig_w], dtype=torch.long))
            is_exhaustive_list.append(bool(sample.get('is_exhaustive', True)))

        boxes_packed = torch.cat(packed_boxes, dim=0) if packed_boxes else torch.zeros((0, 4), dtype=torch.float32)
        object_ids_packed = torch.cat(packed_object_ids, dim=0) if packed_object_ids else torch.zeros((0,), dtype=torch.long)
        semantic_segments = torch.stack(semantic_list, dim=0)

        find_target = BatchedFindTarget(
            num_boxes=torch.tensor(num_boxes_list, dtype=torch.long),
            boxes=boxes_packed,
            boxes_padded=padded_boxes,
            repeated_boxes=boxes_packed.clone(),
            segments=padded_masks,
            semantic_segments=semantic_segments,
            is_valid_segment=torch.tensor(valid_semantic_list, dtype=torch.bool),
            is_exhaustive=torch.tensor(is_exhaustive_list, dtype=torch.bool),
            object_ids=object_ids_packed,
            object_ids_padded=object_ids_padded,
        )

        metadata = BatchedInferenceMetadata(
            coco_image_id=torch.tensor(image_id_list, dtype=torch.long),
            original_image_id=torch.tensor(image_id_list, dtype=torch.long),
            original_category_id=torch.zeros((batch_size,), dtype=torch.int),
            original_size=torch.stack(original_size_list, dim=0),
            object_id=torch.zeros((batch_size,), dtype=torch.long),
            frame_index=torch.zeros((batch_size,), dtype=torch.long),
            is_conditioning_only=[False for _ in range(batch_size)],
        )

        raw_images = [s.get('raw_image') for s in samples] if any('raw_image' in s for s in samples) else None
        return BatchedDatapoint(
            img_batch=img_batch,
            find_text_batch=batch_texts,
            find_inputs=[find_stage],
            find_targets=[find_target],
            find_metadatas=[metadata],
            raw_images=raw_images,
        )
