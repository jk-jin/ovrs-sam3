from __future__ import annotations

import math
from typing import Any, MutableMapping, Sequence

import torch
import torch.nn.functional as F

from ..models.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)

Sample = MutableMapping[str, Any]


def _pad_tensor_hw(
    x: torch.Tensor,
    out_h: int,
    out_w: int,
    value: float = 0.0,
) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=value)


def _round_up(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)


def _normalize_bg_mapping(bg_mapping):
    return {
        "enabled": bool(bg_mapping.get("enabled", False)),
        "background_id": bg_mapping.get("background_id", None),
        "default_background_id": int(bg_mapping.get("default_background_id", 255)),
    }


class OVSemanticCollator:
    def __init__(
        self,
        image_pad_value: float = 0.0,
        pad_size_divisor: int = 1,
        label_pad_value: int = 255,
    ):
        self.image_pad_value = float(image_pad_value)
        self.pad_size_divisor = int(pad_size_divisor)
        self.label_pad_value = int(label_pad_value)

    def _collate_images(self, samples: Sequence[Sample]):
        sizes = [(int(s["image"].shape[-2]), int(s["image"].shape[-1])) for s in samples]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        if self.pad_size_divisor > 1:
            max_h = _round_up(max_h, self.pad_size_divisor)
            max_w = _round_up(max_w, self.pad_size_divisor)

        imgs = [
            _pad_tensor_hw(s["image"], max_h, max_w, self.image_pad_value)
            for s in samples
        ]
        return torch.stack(imgs, dim=0), sizes, (max_h, max_w)

    def _collect_optional_images(
        self,
        samples: Sequence[Sample],
        key: str,
    ) -> list[Any] | None:
        if not any(key in s and s[key] is not None for s in samples):
            return None
        return [s.get(key, None) for s in samples]

    def __call__(self, samples: Sequence[Sample]) -> BatchedDatapoint:
        samples = list(samples)
        if len(samples) == 0:
            raise ValueError("Empty batch.")

        img_batch, image_sizes, padded_hw = self._collate_images(samples)
        batch_size = len(samples)

        label_maps = []
        image_id_list = []
        original_size_list = []

        shared_full_class_texts = None
        shared_active_class_texts = None
        shared_active_class_ids = None
        shared_background_mapping = None

        for b, sample in enumerate(samples):
            full_texts = [str(x) for x in sample["class_texts"]]
            active_texts = [str(x) for x in sample.get("active_class_texts", full_texts)]
            active_ids = [int(x) for x in sample.get("active_class_ids", list(range(len(full_texts))))]
            bg_mapping = _normalize_bg_mapping(sample.get("background_mapping", {"enabled": False, "background_id": None, "default_background_id": 255}))

            if shared_full_class_texts is None:
                shared_full_class_texts = full_texts
                shared_active_class_texts = active_texts
                shared_active_class_ids = active_ids
                shared_background_mapping = bg_mapping
            else:
                if full_texts != shared_full_class_texts:
                    raise ValueError(
                        "All samples in one batch must share the same class_texts order. "
                        f"Got mismatch at sample index {b}."
                    )
                if active_texts != shared_active_class_texts:
                    raise ValueError(
                        "All samples in one batch must share the same active_class_texts. "
                        f"Got mismatch at sample index {b}."
                    )
                if active_ids != shared_active_class_ids:
                    raise ValueError(
                        "All samples in one batch must share the same active_class_ids. "
                        f"Got mismatch at sample index {b}."
                    )
                if bg_mapping != shared_background_mapping:
                    raise ValueError(
                        "All samples in one batch must share the same background_mapping. "
                        f"Got mismatch at sample index {b}."
                    )

            label_map = sample["label_map"].long()
            if tuple(label_map.shape[-2:]) != tuple(padded_hw):
                label_map = _pad_tensor_hw(
                    label_map,
                    padded_hw[0],
                    padded_hw[1],
                    self.label_pad_value,
                ).long()
            label_maps.append(label_map)

            image_id_list.append(int(sample.get("image_id", b)))
            orig_h, orig_w = sample.get("original_size", image_sizes[b])
            original_size_list.append(torch.tensor([orig_h, orig_w], dtype=torch.long))

        if shared_full_class_texts is None:
            raise ValueError("shared_full_class_texts is None.")
        if shared_active_class_texts is None:
            raise ValueError("shared_active_class_texts is None.")

        find_stage = FindStage(
            img_ids=None,
            text_ids=None,
            input_boxes=torch.zeros((0, 0, 4), dtype=torch.float32),
            input_boxes_mask=torch.zeros((0, 0), dtype=torch.bool),
            input_boxes_label=torch.zeros((0, 0), dtype=torch.long),
            input_points=torch.zeros((0, 0, 2), dtype=torch.float32),
            input_points_mask=torch.zeros((0, 0), dtype=torch.bool),
        )

        find_target = BatchedFindTarget(
            semantic_label_map=torch.stack(label_maps, dim=0),  # [B, H, W]
        )

        metadata = BatchedInferenceMetadata(
            original_image_id=torch.tensor(image_id_list, dtype=torch.long),
            original_size=torch.stack(original_size_list, dim=0),
            num_classes=len(shared_full_class_texts),
            class_names=shared_full_class_texts,
            active_class_ids=list(shared_active_class_ids),
            active_class_names=list(shared_active_class_texts),
            background_mapping_enabled=bool(shared_background_mapping["enabled"]),
            background_id=shared_background_mapping["background_id"],
            default_background_id=int(shared_background_mapping["default_background_id"]),
        )

        raw_images = self._collect_optional_images(samples, "raw_image")
        raw_images_original = self._collect_optional_images(samples, "raw_image_original")

        return BatchedDatapoint(
            img_batch=img_batch,
            find_text_batch=shared_active_class_texts,
            find_inputs=[find_stage],
            find_targets=[find_target],
            find_metadatas=[metadata],
            raw_images=raw_images,
            raw_images_original=raw_images_original,
        )