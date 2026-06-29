from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import transforms as T


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_label_map(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr).long()


def _build_transform_from_cfg(cfg: Any) -> Optional[Callable]:
    if cfg is None:
        return None
    if callable(cfg):
        return cfg
    if isinstance(cfg, list):
        transforms = [_build_transform_from_cfg(x) for x in cfg]
        transforms = [x for x in transforms if x is not None]
        return T.Compose(transforms)
    if not isinstance(cfg, dict):
        raise TypeError(f"Unsupported transform cfg type: {type(cfg)}")

    cfg = dict(cfg)
    t_type = cfg.pop("type")
    if "." in t_type:
        module_name, class_name = t_type.rsplit(".", 1)
        mod = __import__(module_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
    else:
        cls = getattr(T, t_type)

    if "transforms" in cfg:
        cfg["transforms"] = [_build_transform_from_cfg(x) for x in cfg["transforms"]]
    return cls(**cfg)


class OVSemanticSegDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        classes: list[str],
        transforms: Optional[Any] = None,
        img_suffix: str = ".png",
        seg_suffix: str = ".png",
        ignore_index: int = 255,
        reduce_zero_label: bool = False,
        return_raw_image: bool = False,
        background_mapping: Optional[dict] = None,
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.classes = list(classes)
        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix
        self.ignore_index = int(ignore_index)
        self.reduce_zero_label = bool(reduce_zero_label)
        self.return_raw_image = bool(return_raw_image)
        self.transforms = _build_transform_from_cfg(transforms)

        self.background_mapping = self._normalize_background_mapping(background_mapping)

        self.full_class_ids = list(range(len(self.classes)))
        self.full_class_names = list(self.classes)

        if self.background_mapping["enabled"]:
            bg_id = int(self.background_mapping["background_id"])
            self.active_class_ids = [i for i in self.full_class_ids if i != bg_id]
        else:
            self.active_class_ids = list(self.full_class_ids)

        self.active_class_names = [self.classes[i] for i in self.active_class_ids]

        if not self.img_dir.exists():
            raise FileNotFoundError(f"img_dir not found: {self.img_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"ann_dir not found: {self.ann_dir}")

        self.img_paths = sorted(self.img_dir.glob(f"*{self.img_suffix}"))
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {self.img_dir} with suffix {self.img_suffix}")

        self.seg_paths = [self.ann_dir / f"{p.stem}{self.seg_suffix}" for p in self.img_paths]
        missing = [str(p) for p in self.seg_paths if not p.exists()]
        if missing:
            preview = "\n".join(missing[:20])
            raise FileNotFoundError(f"Some segmentation labels are missing:\n{preview}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def _normalize_background_mapping(self, cfg):
        if cfg is None:
            return {
                "enabled": False,
                "background_id": None,
                "default_background_id": self.ignore_index,
            }

        cfg = dict(cfg)
        enabled = bool(cfg.get("enabled", False))
        background_id = cfg.get("background_id", None)
        default_background_id = int(cfg.get("default_background_id", self.ignore_index))

        if enabled:
            if background_id is None:
                raise ValueError("background_mapping.enabled=True requires background_id.")
            background_id = int(background_id)
            if not 0 <= background_id < len(self.classes):
                raise ValueError(
                    f"background_id={background_id} out of range for "
                    f"{len(self.classes)} classes."
                )

        return {
            "enabled": enabled,
            "background_id": None if background_id is None else int(background_id),
            "default_background_id": default_background_id,
        }

    def _process_label_map(self, label_map: torch.Tensor) -> torch.Tensor:
        label_map = label_map.long()

        if self.reduce_zero_label:
            bg_mask = label_map == 0
            valid_mask = label_map != self.ignore_index

            label_map = label_map.clone()
            label_map[valid_mask] -= 1
            label_map[bg_mask] = self.ignore_index

        return label_map

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        seg_path = self.seg_paths[index]

        image = _load_image(img_path)

        raw_image_original = image.copy() if self.return_raw_image else None
        raw_image = image.copy() if self.return_raw_image else None

        label_map = _load_label_map(seg_path)
        label_map = self._process_label_map(label_map)

        sample = {
            "image": image,
            "label_map": label_map,
            "class_texts": self.full_class_names,
            "active_class_texts": self.active_class_names,
            "active_class_ids": self.active_class_ids,
            "background_mapping": self.background_mapping,
            "image_id": index,
            "original_size": image.size[::-1],
            "img_path": str(img_path),
            "seg_path": str(seg_path),
        }

        if raw_image is not None:
            sample["raw_image"] = raw_image
        if raw_image_original is not None:
            sample["raw_image_original"] = raw_image_original

        if self.transforms is not None:
            sample = self.transforms(sample)

        if not isinstance(sample["image"], torch.Tensor):
            raise TypeError("Dataset expects transforms to convert `image` to torch.Tensor.")

        if sample["image"].dtype != torch.float32:
            sample["image"] = sample["image"].float()

        if not isinstance(sample["label_map"], torch.Tensor):
            raise TypeError("`label_map` must be torch.Tensor after transforms.")

        sample["label_map"] = sample["label_map"].long()
        return sample