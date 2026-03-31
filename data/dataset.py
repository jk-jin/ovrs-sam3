from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import transforms as T

Sample = MutableMapping[str, Any]


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def _resolve_data_root(data_root: str | Path, ann_file: str | Path) -> Path:
    data_root = Path(data_root)
    ann_file = Path(ann_file)
    if ann_file.is_absolute() or data_root.exists():
        return data_root
    return ann_file.parent / data_root


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def _load_mask(path: Path) -> torch.Tensor:
    mask = Image.open(path)
    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr > 0)


def _build_transform_from_cfg(cfg: Any) -> Optional[Callable[[Sample], Sample]]:
    if cfg is None:
        return None
    if callable(cfg):
        return cfg
    if isinstance(cfg, list):
        transforms = [_build_transform_from_cfg(x) for x in cfg]
        transforms = [x for x in transforms if x is not None]
        return T.Compose(transforms)
    if not isinstance(cfg, dict):
        raise TypeError(f'Unsupported transform cfg type: {type(cfg)}')

    cfg = dict(cfg)
    t_type = cfg.pop('type')
    if '.' in t_type:
        module_name, class_name = t_type.rsplit('.', 1)
        mod = __import__(module_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
    else:
        cls = getattr(T, t_type)
    if 'transforms' in cfg:
        cfg['transforms'] = [_build_transform_from_cfg(x) for x in cfg['transforms']]
    return cls(**cfg)


class JsonPromptSegDataset(Dataset):
    """A simple prompt-conditioned segmentation dataset template.

    Expected annotation schema (JSON or JSONL):

    {
      "samples": [
        {
          "image": "images/0001.jpg",
          "text": "car",
          "image_id": 1,
          "semantic_mask": "semantic/0001.png",
          "instances": [
            {
              "bbox": [x1, y1, x2, y2],
              "mask": "instances/0001_obj1.png",
              "object_id": 1
            }
          ],
          "is_exhaustive": true
        }
      ]
    }

    Notes:
    - `bbox` may be xyxy or cxcywh depending on `box_format`.
    - `semantic_mask` is optional for instance-only training.
    - `instances` is optional for semantic-only training.
    - The dataset returns a plain dict; the collator converts it into BatchedDatapoint.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        transforms: Optional[Any] = None,
        image_key: str = 'image',
        text_key: str = 'text',
        instances_key: str = 'instances',
        semantic_mask_key: str = 'semantic_mask',
        image_id_key: str = 'image_id',
        object_id_key: str = 'object_id',
        box_key: str = 'bbox',
        mask_key: str = 'mask',
        box_format: str = 'xyxy',
        filter_empty: bool = False,
        return_raw_image: bool = False,
    ):
        super().__init__()
        self.ann_file = Path(ann_file)
        self.data_root = _resolve_data_root(data_root, ann_file)
        self.image_key = image_key
        self.text_key = text_key
        self.instances_key = instances_key
        self.semantic_mask_key = semantic_mask_key
        self.image_id_key = image_id_key
        self.object_id_key = object_id_key
        self.box_key = box_key
        self.mask_key = mask_key
        self.box_format = box_format
        self.filter_empty = bool(filter_empty)
        self.return_raw_image = bool(return_raw_image)
        self.transforms = _build_transform_from_cfg(transforms)

        raw = _read_json(self.ann_file)
        if isinstance(raw, dict) and 'samples' in raw:
            samples = raw['samples']
        elif isinstance(raw, list):
            samples = raw
        else:
            raise ValueError('Expected ann_file to contain a list or a dict with key `samples`.')
        self.samples = [s for s in samples if (not self.filter_empty or self._is_valid_sample(s))]

    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        has_inst = len(sample.get(self.instances_key, []) or []) > 0
        has_sem = sample.get(self.semantic_mask_key) is not None
        return has_inst or has_sem

    def __len__(self) -> int:
        return len(self.samples)

    def _load_instances(self, sample: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instances = sample.get(self.instances_key, []) or []
        if len(instances) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0, 1, 1), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.long),
            )

        boxes = []
        masks = []
        object_ids = []
        for idx, inst in enumerate(instances):
            boxes.append(inst[self.box_key])
            mask_path = inst.get(self.mask_key)
            if mask_path is None:
                raise ValueError('Each instance must provide a mask path in this template dataset.')
            masks.append(_load_mask(self.data_root / mask_path))
            object_ids.append(int(inst.get(self.object_id_key, idx + 1)))

        masks = torch.stack(masks, dim=0).bool()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        object_ids = torch.as_tensor(object_ids, dtype=torch.long)
        return boxes, masks, object_ids

    def __getitem__(self, index: int) -> Sample:
        ann = self.samples[index]
        img_path = self.data_root / ann[self.image_key]
        image = _load_image(img_path)
        raw_image = image.copy() if self.return_raw_image else None

        boxes, instance_masks, object_ids = self._load_instances(ann)
        semantic_mask_path = ann.get(self.semantic_mask_key)
        semantic_mask = _load_mask(self.data_root / semantic_mask_path) if semantic_mask_path else None

        sample: Sample = {
            'image': image,
            'text': ann[self.text_key],
            'image_id': int(ann.get(self.image_id_key, index)),
            'original_size': image.size[::-1],  # (H, W)
            'boxes': boxes,
            'instance_masks': instance_masks,
            'semantic_mask': semantic_mask,
            'object_ids': object_ids,
            'is_exhaustive': bool(ann.get('is_exhaustive', True)),
            'bbox_format': ann.get('bbox_format', self.box_format),
        }
        if raw_image is not None:
            sample['raw_image'] = raw_image

        if self.transforms is not None:
            sample = self.transforms(sample)

        if not isinstance(sample['image'], torch.Tensor):
            raise TypeError('Dataset expects transforms to convert `image` to torch.Tensor. Add ToTensor().')
        if sample['image'].dtype != torch.float32:
            sample['image'] = sample['image'].float()

        if sample.get('semantic_mask') is not None and sample['semantic_mask'].dtype != torch.bool:
            sample['semantic_mask'] = sample['semantic_mask'] > 0
        if sample.get('instance_masks') is not None and sample['instance_masks'].dtype != torch.bool:
            sample['instance_masks'] = sample['instance_masks'] > 0
        return sample
