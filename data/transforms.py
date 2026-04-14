from __future__ import annotations

import random
from typing import Any, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


Sample = MutableMapping[str, Any]


def _to_tensor_image(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            return image.float()
        raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")

    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = image[..., None]
        if image.ndim != 3:
            raise ValueError(f"Unsupported image array shape: {image.shape}")
        tensor = torch.from_numpy(image)
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor.float()

    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_tensor_mask(mask: Any) -> torch.Tensor:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if isinstance(mask, np.ndarray):
        return torch.from_numpy(mask)
    raise TypeError(f"Unsupported mask type: {type(mask)}")


def _resize_tensor_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")
    image = image[None]
    out = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    return out[0]


def _resize_label_map(label_map: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if label_map is None:
        return None
    if label_map.ndim != 2:
        raise ValueError(f"Unsupported label_map shape: {tuple(label_map.shape)}")
    label_map = label_map[None, None].float()
    label_map = F.interpolate(label_map, size=size, mode="nearest")[0, 0]
    return label_map.long()


def _compute_keep_ratio_size(
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> Tuple[int, int]:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scale = min(dst_h / max(src_h, 1), dst_w / max(src_w, 1))
    out_h = max(1, int(round(src_h * scale)))
    out_w = max(1, int(round(src_w * scale)))
    return out_h, out_w


def _crop_last_two_dims(
    x: torch.Tensor,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
) -> torch.Tensor:
    return x[..., top:top + crop_h, left:left + crop_w]


def _pad_last_two_dims(
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


class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)

        sample["image"] = _to_tensor_image(sample["image"])

        if "raw_image" in sample and sample["raw_image"] is not None:
            sample["raw_image"] = _to_tensor_image(sample["raw_image"])

        if "label_map" in sample and sample["label_map"] is not None:
            sample["label_map"] = _to_tensor_mask(sample["label_map"]).long()

        return sample


class ConvertImageDtype:
    def __init__(self, dtype: torch.dtype | str = torch.float32, scale: bool = True):
        self.dtype = self._parse_dtype(dtype)
        self.scale = bool(scale)

    @staticmethod
    def _parse_dtype(dtype: torch.dtype | str) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype

        if isinstance(dtype, str):
            key = dtype.strip().lower()
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
                "float64": torch.float64,
                "fp64": torch.float64,
                "double": torch.float64,
                "uint8": torch.uint8,
                "int8": torch.int8,
                "int16": torch.int16,
                "short": torch.int16,
                "int32": torch.int32,
                "int": torch.int32,
                "int64": torch.int64,
                "long": torch.int64,
                "bool": torch.bool,
            }
            if key not in mapping:
                raise ValueError(
                    f"Unsupported dtype string: {dtype}. "
                    f"Supported keys are: {sorted(mapping.keys())}"
                )
            return mapping[key]

        raise TypeError(f"Unsupported dtype type: {type(dtype)}")

    def _convert_image_like(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        if self.scale and x.max() > 1.0:
            x = x / 255.0
        return x

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)

        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("ConvertImageDtype expects image to be a torch.Tensor")
        sample["image"] = self._convert_image_like(image)

        if "raw_image" in sample and sample["raw_image"] is not None:
            raw_image = sample["raw_image"]
            if not isinstance(raw_image, torch.Tensor):
                raise TypeError("ConvertImageDtype expects raw_image to be a torch.Tensor")
            sample["raw_image"] = self._convert_image_like(raw_image)

        return sample


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample["image"]
        sample["image"] = (
            image - self.mean.to(image.device, image.dtype)
        ) / self.std.to(image.device, image.dtype)
        # 不对 raw_image 做 Normalize
        return sample


class Resize:
    def __init__(self, size: Tuple[int, int], keep_ratio: bool = False):
        self.size = tuple(size)
        self.keep_ratio = bool(keep_ratio)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample["image"]
        h, w = image.shape[-2:]

        if self.keep_ratio:
            out_h, out_w = _compute_keep_ratio_size((h, w), self.size)
        else:
            out_h, out_w = self.size

        sample["image"] = _resize_tensor_image(image, (out_h, out_w))

        if "raw_image" in sample and sample["raw_image"] is not None:
            sample["raw_image"] = _resize_tensor_image(sample["raw_image"], (out_h, out_w))

        if "label_map" in sample and sample["label_map"] is not None:
            sample["label_map"] = _resize_label_map(sample["label_map"], (out_h, out_w))

        sample["img_shape"] = (out_h, out_w)
        sample["scale_factor"] = (out_w / w, out_h / h)
        return sample


class ResizeLongestSide:
    def __init__(self, long_side: int):
        self.long_side = int(long_side)

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        h, w = image.shape[-2:]
        scale = self.long_side / max(h, w)
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        return Resize((out_h, out_w))(sample)


class RandomResizeByRatio:
    def __init__(
        self,
        base_scale: Tuple[int, int],
        ratio_range: Tuple[float, float] = (0.5, 2.0),
        keep_ratio: bool = True,
    ):
        self.base_scale = tuple(base_scale)
        self.ratio_range = tuple(ratio_range)
        self.keep_ratio = bool(keep_ratio)

    def __call__(self, sample: Sample) -> Sample:
        min_ratio, max_ratio = self.ratio_range
        ratio = random.uniform(min_ratio, max_ratio)

        target_h = max(1, int(round(self.base_scale[0] * ratio)))
        target_w = max(1, int(round(self.base_scale[1] * ratio)))

        return Resize((target_h, target_w), keep_ratio=self.keep_ratio)(sample)


class RandomCrop:
    def __init__(
        self,
        crop_size: Tuple[int, int],
        cat_max_ratio: float = 0.75,
        ignore_index: int = 255,
        pad_if_needed: bool = True,
        image_pad_value: float = 0.0,
        num_retry: int = 10,
    ):
        self.crop_size = tuple(crop_size)
        self.cat_max_ratio = float(cat_max_ratio)
        self.ignore_index = int(ignore_index)
        self.pad_if_needed = bool(pad_if_needed)
        self.image_pad_value = float(image_pad_value)
        self.num_retry = int(num_retry)

    def _is_valid_crop(self, label_map) -> bool:
        if label_map is None:
            return True
        if self.cat_max_ratio >= 1.0:
            return True

        valid = label_map != self.ignore_index
        if not valid.any():
            return True

        _, counts = torch.unique(label_map[valid], return_counts=True)
        if counts.numel() == 0:
            return True

        max_ratio = counts.max().float() / counts.sum().float().clamp(min=1.0)
        return float(max_ratio.item()) <= self.cat_max_ratio

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        crop_h, crop_w = self.crop_size

        image = sample["image"]
        raw_image = sample.get("raw_image", None)
        label_map = sample.get("label_map", None)

        if self.pad_if_needed:
            image = _pad_last_two_dims(image, crop_h, crop_w, self.image_pad_value)

            if raw_image is not None:
                raw_image = _pad_last_two_dims(raw_image, crop_h, crop_w, self.image_pad_value)

            if label_map is not None:
                label_map = _pad_last_two_dims(
                    label_map,
                    crop_h,
                    crop_w,
                    self.ignore_index,
                ).long()

        h, w = image.shape[-2:]
        if h < crop_h or w < crop_w:
            raise ValueError(
                f"Image size {(h, w)} is smaller than crop size {(crop_h, crop_w)} "
                "and pad_if_needed=False."
            )

        chosen_top = 0
        chosen_left = 0

        for _ in range(self.num_retry):
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)

            crop_label = None
            if label_map is not None:
                crop_label = _crop_last_two_dims(label_map, top, left, crop_h, crop_w)

            if self._is_valid_crop(crop_label):
                chosen_top = top
                chosen_left = left
                break
        else:
            chosen_top = random.randint(0, h - crop_h)
            chosen_left = random.randint(0, w - crop_w)

        sample["image"] = _crop_last_two_dims(image, chosen_top, chosen_left, crop_h, crop_w)

        if raw_image is not None:
            sample["raw_image"] = _crop_last_two_dims(
                raw_image,
                chosen_top,
                chosen_left,
                crop_h,
                crop_w,
            )

        if label_map is not None:
            sample["label_map"] = _crop_last_two_dims(
                label_map,
                chosen_top,
                chosen_left,
                crop_h,
                crop_w,
            ).long()

        sample["img_shape"] = (crop_h, crop_w)
        return sample


class RandomVerticalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample

        sample = dict(sample)
        sample["image"] = torch.flip(sample["image"], dims=[-2])

        if "raw_image" in sample and sample["raw_image"] is not None:
            sample["raw_image"] = torch.flip(sample["raw_image"], dims=[-2])

        if "label_map" in sample and sample["label_map"] is not None:
            sample["label_map"] = torch.flip(sample["label_map"], dims=[-2])

        return sample


class RandomRotate90:
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample

        sample = dict(sample)
        k = random.randint(1, 3)

        sample["image"] = torch.rot90(sample["image"], k=k, dims=(-2, -1))

        if "raw_image" in sample and sample["raw_image"] is not None:
            sample["raw_image"] = torch.rot90(sample["raw_image"], k=k, dims=(-2, -1))

        if "label_map" in sample and sample["label_map"] is not None:
            sample["label_map"] = torch.rot90(sample["label_map"], k=k, dims=(-2, -1)).long()

        return sample


class RandomResize:
    def __init__(self, scales: Sequence[Tuple[int, int]]):
        self.scales = list(scales)

    def __call__(self, sample: Sample) -> Sample:
        size = random.choice(self.scales)
        return Resize(size)(sample)


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample

        sample = dict(sample)
        sample["image"] = torch.flip(sample["image"], dims=[-1])

        if "raw_image" in sample and sample["raw_image"] is not None:
            sample["raw_image"] = torch.flip(sample["raw_image"], dims=[-1])

        if "label_map" in sample and sample["label_map"] is not None:
            sample["label_map"] = torch.flip(sample["label_map"], dims=[-1])

        return sample


class PadToSize:
    def __init__(
        self,
        size: Tuple[int, int],
        image_pad_value: float = 0.0,
        label_pad_value: int = 255,
    ):
        self.size = tuple(size)
        self.image_pad_value = float(image_pad_value)
        self.label_pad_value = int(label_pad_value)

    def _pad_last_two_dims(self, x: torch.Tensor, pad_value: float) -> torch.Tensor:
        out_h, out_w = self.size
        h, w = x.shape[-2:]
        pad_h = max(0, out_h - h)
        pad_w = max(0, out_w - w)
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)

        sample["image"] = self._pad_last_two_dims(sample["image"], self.image_pad_value)

        if "raw_image" in sample and sample["raw_image"] is not None:
            sample["raw_image"] = self._pad_last_two_dims(sample["raw_image"], self.image_pad_value)

        if "label_map" in sample and sample["label_map"] is not None:
            sample["label_map"] = self._pad_last_two_dims(
                sample["label_map"],
                self.label_pad_value,
            ).long()

        sample["pad_shape"] = self.size
        return sample