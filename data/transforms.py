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


_LABEL_KEYS = ("label_map", "eval_label_map")


def _apply_to_label_keys(sample: dict, fn):
    """Apply *fn* to every label key present in the sample, returning a new dict."""
    sample = dict(sample)
    for key in _LABEL_KEYS:
        if key in sample and sample[key] is not None:
            sample[key] = fn(sample[key]).long()
    return sample


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

        sample = _apply_to_label_keys(
            sample,
            lambda m: _to_tensor_mask(m).long(),
        )

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

        sample = _apply_to_label_keys(
            sample,
            lambda m: _resize_label_map(m, (out_h, out_w)),
        )

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


class ResizeShortestEdge:
    """将图像短边确定性地缩放到指定长度，长边按相同比例缩放。"""

    def __init__(self, short_edge: int):
        self.short_edge = int(short_edge)

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        h, w = image.shape[-2:]

        short_side = min(h, w)
        scale = self.short_edge / max(short_side, 1)
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))

        return Resize((out_h, out_w))(sample)


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

            sample = _apply_to_label_keys(
                sample,
                lambda m: _pad_last_two_dims(
                    m, crop_h, crop_w, self.ignore_index,
                ).long(),
            )
            label_map = sample.get("label_map", None)

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
                # Use label_map (forward) for cat_max_ratio check
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

        sample = _apply_to_label_keys(
            sample,
            lambda m: _crop_last_two_dims(
                m, chosen_top, chosen_left, crop_h, crop_w,
            ).long(),
        )

        sample["img_shape"] = (crop_h, crop_w)
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

        sample = _apply_to_label_keys(
            sample,
            lambda m: torch.flip(m, dims=[-1]),
        )

        return sample


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    """RGB→HSV，image 为 [3, H, W]，值域 [0, 1]"""
    r, g, b = image[0], image[1], image[2]
    max_val, max_idx = torch.max(image, dim=0)
    min_val, _ = torch.min(image, dim=0)
    delta = max_val - min_val

    h = torch.zeros_like(max_val)
    s = torch.zeros_like(max_val)
    v = max_val

    denom = delta.clone().clamp_min(1e-8)

    r_max = (max_idx == 0) & (delta > 0)
    h[r_max] = ((g[r_max] - b[r_max]) / denom[r_max]) % 6

    g_max = (max_idx == 1) & (delta > 0)
    h[g_max] = (b[g_max] - r[g_max]) / denom[g_max] + 2

    b_max = (max_idx == 2) & (delta > 0)
    h[b_max] = (r[b_max] - g[b_max]) / denom[b_max] + 4

    h = h / 6.0

    mask = max_val > 0
    s[mask] = delta[mask] / max_val[mask]

    return torch.stack([h, s, v], dim=0)


def _hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """HSV→RGB，image 为 [3, H, W]，值域 [0, 1]"""
    h, s, v = image[0], image[1], image[2]
    h = h * 6.0
    sector = h.long().clamp(0, 5)
    f = h - sector.float()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    m0 = sector == 0
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    m1 = sector == 1
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    m2 = sector == 2
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    m3 = sector == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    m4 = sector == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    m5 = sector == 5
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]

    return torch.stack([r, g, b], dim=0)


class ColorAugSSD:
    """SSD 风格颜色增强，与 Detectron2 PointRend 保持一致。

    每种扰动以 0.5 概率独立启用。
    图像值域 [0, 1]，所有参数内部换算到此范围。
    image 和 raw_image 共享同一次采样参数。
    """

    def __init__(
        self,
        brightness_delta: float = 32.0,
        contrast_low: float = 0.5,
        contrast_high: float = 1.5,
        saturation_low: float = 0.5,
        saturation_high: float = 1.5,
        hue_delta: float = 18.0,
    ):
        self.brightness_delta = float(brightness_delta)
        self.contrast_low = float(contrast_low)
        self.contrast_high = float(contrast_high)
        self.saturation_low = float(saturation_low)
        self.saturation_high = float(saturation_high)
        self.hue_delta = float(hue_delta)

    def _sample_params(self) -> dict:
        return {
            "contrast_first": random.random() < 0.5,
            "do_brightness": random.random() < 0.5,
            "do_contrast": random.random() < 0.5,
            "do_saturation": random.random() < 0.5,
            "do_hue": random.random() < 0.5,
            "brightness_delta": random.uniform(
                -self.brightness_delta, self.brightness_delta
            ) / 255.0,
            "contrast_alpha": random.uniform(self.contrast_low, self.contrast_high),
            "saturation_factor": random.uniform(
                self.saturation_low, self.saturation_high
            ),
            "hue_shift": random.randint(
                -int(self.hue_delta), int(self.hue_delta)
            ) / 180.0,
        }

    def _apply(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        # 亮度
        if params["do_brightness"]:
            image = image + params["brightness_delta"]

        # 对比度（在 HSV 之前）
        if params["contrast_first"] and params["do_contrast"]:
            image = image * params["contrast_alpha"]

        if params["do_saturation"] or params["do_hue"]:
            image = image.clamp(0.0, 1.0)
            hsv = _rgb_to_hsv(image)

            if params["do_saturation"]:
                hsv[1] = (hsv[1] * params["saturation_factor"]).clamp(0.0, 1.0)

            if params["do_hue"]:
                hsv[0] = (hsv[0] + params["hue_shift"]) % 1.0

            image = _hsv_to_rgb(hsv)

        # 对比度（在 HSV 之后）
        if not params["contrast_first"] and params["do_contrast"]:
            image = image * params["contrast_alpha"]

        return image.clamp(0.0, 1.0)

    def __call__(self, sample: Sample) -> Sample:
        params = self._sample_params()
        sample = dict(sample)

        sample["image"] = self._apply(sample["image"], params)

        if "raw_image" in sample and sample["raw_image"] is not None:
            sample["raw_image"] = self._apply(sample["raw_image"], params)

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

        sample = _apply_to_label_keys(
            sample,
            lambda m: self._pad_last_two_dims(
                m, self.label_pad_value,
            ).long(),
        )

        sample["pad_shape"] = self.size
        return sample