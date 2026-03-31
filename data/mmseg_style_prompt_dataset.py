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

_BACKGROUND_NAMES = {
    'background', 'bg', '__background__', '_background_', 'ignore', 'void'
}


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


def _resolve_path(root: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return root / path


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def _load_seg_map(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _normalize_classes(
    classes: Optional[Sequence[str]],
    reduce_zero_label: bool,
) -> Optional[List[str]]:
    if classes is None:
        return None
    classes = list(classes)
    if reduce_zero_label and classes:
        first = str(classes[0]).strip().lower()
        if first in _BACKGROUND_NAMES:
            classes = classes[1:]
    return classes


class MMSegStylePromptDataset(Dataset):
    """Generic mmseg-style semantic dataset for prompt-conditioned SAM3 training.

    This dataset reads image / annotation directories directly, like a classic mmseg
    semantic segmentation dataset, but expands each image into one or more
    prompt-conditioned binary-mask samples:

        (image, class_text_prompt, binary_mask_for_that_class)

    Example directory layout::

        data_root/
        ├── img_dir/
        │   ├── train/
        │   │   ├── 0001.jpg
        │   │   └── ...
        │   └── val/
        └── ann_dir/
            ├── train/
            │   ├── 0001.png
            │   └── ...
            └── val/

    Recommended config usage::

        dataset=dict(
            type='your_pkg.data.mmseg_style_prompt_dataset.MMSegStylePromptDataset',
            data_root='ld50k',
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            classes=[...],
            ignore_index=255,
            reduce_zero_label=True,
            transforms=[...],
        )

    Args:
        data_root: Common dataset root.
        img_dir: Image directory, relative to data_root or absolute.
        ann_dir: Annotation directory, relative to data_root or absolute.
        classes: Class names used as text prompts. If ``reduce_zero_label=True``
            and the first class looks like a background label, it will be dropped
            automatically so prompt ids stay aligned with shifted labels.
        img_suffix: Image file suffix.
        seg_map_suffix: Segmentation label suffix.
        split_file: Optional text file listing sample ids (without suffix by default).
        ignore_index: Ignore label id in segmentation maps.
        reduce_zero_label: mmseg-compatible behavior. Original label 0 is converted
            to ignore_index, and all valid labels > 0 are shifted down by 1.
            This is the recommended setting when you want to ignore background.
        class_id_to_prompt: Optional explicit mapping from normalized class id to prompt.
        filter_empty_gt: Skip images that have no valid foreground class after ignore /
            reduce_zero_label processing.
        max_classes_per_image: Optional cap on how many prompt samples one image can
            expand into.
        transforms: Same transform style as the earlier template datasets.
        return_raw_image: Whether to preserve the PIL image for visualization.
        cache_index_file: Optional JSON cache file to store the expanded prompt index.
    """

    def __init__(
        self,
        data_root: str,
        img_dir: str,
        ann_dir: str,
        classes: Sequence[str],
        img_suffix: str = '.jpg',
        seg_map_suffix: str = '.png',
        split_file: Optional[str] = None,
        ignore_index: int = 255,
        reduce_zero_label: bool = False,
        class_id_to_prompt: Optional[Dict[int, str]] = None,
        filter_empty_gt: bool = True,
        max_classes_per_image: Optional[int] = None,
        transforms: Optional[Any] = None,
        return_raw_image: bool = False,
        cache_index_file: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.img_dir = _resolve_path(self.data_root, img_dir)
        self.ann_dir = _resolve_path(self.data_root, ann_dir)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split_file = _resolve_path(self.data_root, split_file) if split_file else None
        self.ignore_index = int(ignore_index)
        self.reduce_zero_label = bool(reduce_zero_label)
        self.filter_empty_gt = bool(filter_empty_gt)
        self.max_classes_per_image = max_classes_per_image
        self.return_raw_image = bool(return_raw_image)
        self.transforms = _build_transform_from_cfg(transforms)
        self.class_names = _normalize_classes(classes, self.reduce_zero_label)
        self.class_id_to_prompt = {int(k): str(v) for k, v in (class_id_to_prompt or {}).items()}
        self.cache_index_file = _resolve_path(self.data_root, cache_index_file) if cache_index_file else None

        if self.class_names is None or len(self.class_names) == 0:
            raise ValueError('`classes` must be provided and contain at least one foreground class.')
        if not self.img_dir.exists():
            raise FileNotFoundError(f'img_dir does not exist: {self.img_dir}')
        if not self.ann_dir.exists():
            raise FileNotFoundError(f'ann_dir does not exist: {self.ann_dir}')

        self.samples = self._build_or_load_index()
        if len(self.samples) == 0:
            raise RuntimeError('No valid samples found. Check classes / reduce_zero_label / ignore_index settings.')

    def _load_ids_from_split(self) -> List[str]:
        if self.split_file is None:
            return []
        ids: List[str] = []
        for line in self.split_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            ids.append(Path(line).stem)
        return ids

    def _collect_file_pairs(self) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        sample_ids = self._load_ids_from_split()
        if sample_ids:
            basenames = sample_ids
        else:
            basenames = sorted(p.stem for p in self.ann_dir.glob(f'*{self.seg_map_suffix}'))

        for image_id, stem in enumerate(basenames):
            img_path = self.img_dir / f'{stem}{self.img_suffix}'
            ann_path = self.ann_dir / f'{stem}{self.seg_map_suffix}'
            if not img_path.exists():
                raise FileNotFoundError(f'Image not found for stem `{stem}`: {img_path}')
            if not ann_path.exists():
                raise FileNotFoundError(f'Annotation not found for stem `{stem}`: {ann_path}')
            pairs.append({
                'stem': stem,
                'image_id': image_id,
                'img_path': str(img_path),
                'ann_path': str(ann_path),
            })
        return pairs

    def _apply_reduce_zero_label(self, seg_map: np.ndarray) -> np.ndarray:
        seg_map = seg_map.copy()
        if not self.reduce_zero_label:
            return seg_map
        zero_mask = seg_map == 0
        ignore_mask = seg_map == self.ignore_index
        seg_map[zero_mask] = self.ignore_index
        valid = ~ignore_mask & ~zero_mask
        seg_map[valid] = seg_map[valid] - 1
        return seg_map

    def _present_class_ids(self, seg_map: np.ndarray) -> List[int]:
        seg_map = self._apply_reduce_zero_label(seg_map)
        class_ids = np.unique(seg_map)
        class_ids = [int(x) for x in class_ids.tolist() if int(x) != self.ignore_index]
        class_ids = [x for x in class_ids if 0 <= x < len(self.class_names)]
        class_ids.sort()
        return class_ids

    def _class_id_to_text(self, class_id: int) -> str:
        if class_id in self.class_id_to_prompt:
            return self.class_id_to_prompt[class_id]
        return str(self.class_names[class_id])

    def _build_index(self) -> List[Dict[str, Any]]:
        index: List[Dict[str, Any]] = []
        for pair in self._collect_file_pairs():
            seg_map = _load_seg_map(Path(pair['ann_path']))
            present_ids = self._present_class_ids(seg_map)
            if self.filter_empty_gt and len(present_ids) == 0:
                continue
            if self.max_classes_per_image is not None:
                present_ids = present_ids[: int(self.max_classes_per_image)]
            for class_id in present_ids:
                index.append({
                    'stem': pair['stem'],
                    'image_id': pair['image_id'],
                    'img_path': pair['img_path'],
                    'ann_path': pair['ann_path'],
                    'class_id': class_id,
                    'text': self._class_id_to_text(class_id),
                })
        return index

    def _build_or_load_index(self) -> List[Dict[str, Any]]:
        if self.cache_index_file and self.cache_index_file.exists():
            return json.loads(self.cache_index_file.read_text())

        index = self._build_index()
        if self.cache_index_file:
            self.cache_index_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2))
        return index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        meta = self.samples[index]
        img_path = Path(meta['img_path'])
        ann_path = Path(meta['ann_path'])
        class_id = int(meta['class_id'])

        image = _load_image(img_path)
        raw_image = image.copy() if self.return_raw_image else None
        original_h, original_w = image.size[1], image.size[0]

        seg_map = _load_seg_map(ann_path)
        seg_map = self._apply_reduce_zero_label(seg_map)
        semantic_mask = torch.from_numpy(seg_map == class_id).bool()

        sample: Sample = {
            'image': image,
            'text': str(meta['text']),
            'image_id': int(meta.get('image_id', index)),
            'original_size': (original_h, original_w),
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'instance_masks': torch.zeros((0, original_h, original_w), dtype=torch.bool),
            'semantic_mask': semantic_mask,
            'object_ids': torch.zeros((0,), dtype=torch.long),
            'is_exhaustive': True,
            'bbox_format': 'xyxy',
            'class_id': class_id,
            'stem': meta.get('stem', img_path.stem),
        }
        if raw_image is not None:
            sample['raw_image'] = raw_image

        if self.transforms is not None:
            sample = self.transforms(sample)

        if not isinstance(sample['image'], torch.Tensor):
            raise TypeError('Dataset expects transforms to convert `image` to torch.Tensor. Add ToTensor().')
        if sample['image'].dtype != torch.float32:
            sample['image'] = sample['image'].float()
        if sample['semantic_mask'].dtype != torch.bool:
            sample['semantic_mask'] = sample['semantic_mask'] > 0
        if sample['instance_masks'].dtype != torch.bool:
            sample['instance_masks'] = sample['instance_masks'] > 0
        return sample
