from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


Sample = MutableMapping[str, Any]


def _to_tensor_image(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            return image.float()
        raise ValueError(f'Unsupported image tensor shape: {tuple(image.shape)}')
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = image[..., None]
        if image.ndim != 3:
            raise ValueError(f'Unsupported image array shape: {image.shape}')
        tensor = torch.from_numpy(image)
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor.float()
    raise TypeError(f'Unsupported image type: {type(image)}')


def _to_tensor_mask(mask: Any) -> torch.Tensor:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if isinstance(mask, np.ndarray):
        return torch.from_numpy(mask)
    raise TypeError(f'Unsupported mask type: {type(mask)}')


def _resize_tensor_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if image.ndim == 3:
        image = image[None]
        out = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        return out[0]
    raise ValueError(f'Unsupported image shape: {tuple(image.shape)}')


def _resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask[None, None].float()
        return F.interpolate(mask, size=size, mode='nearest')[0, 0]
    if mask.ndim == 3:
        mask = mask[:, None].float()
        return F.interpolate(mask, size=size, mode='nearest')[:, 0]
    if mask.ndim == 4:
        return F.interpolate(mask.float(), size=size, mode='nearest')
    raise ValueError(f'Unsupported mask shape: {tuple(mask.shape)}')


def _flip_boxes_horizontally(boxes: torch.Tensor, image_width: int, box_format: str) -> torch.Tensor:
    if boxes is None:
        return None
    boxes = boxes.clone()
    if box_format == 'xyxy':
        x1 = boxes[..., 0].clone()
        x2 = boxes[..., 2].clone()
        boxes[..., 0] = image_width - x2
        boxes[..., 2] = image_width - x1
        return boxes
    if box_format == 'cxcywh':
        boxes[..., 0] = image_width - boxes[..., 0]
        return boxes
    raise ValueError(f'Unsupported box_format: {box_format}')


def _scale_boxes(boxes: torch.Tensor, scale_x: float, scale_y: float, box_format: str) -> torch.Tensor:
    if boxes is None:
        return None
    boxes = boxes.clone().float()
    if box_format == 'xyxy':
        boxes[..., 0] *= scale_x
        boxes[..., 2] *= scale_x
        boxes[..., 1] *= scale_y
        boxes[..., 3] *= scale_y
        return boxes
    if box_format == 'cxcywh':
        boxes[..., 0] *= scale_x
        boxes[..., 2] *= scale_x
        boxes[..., 1] *= scale_y
        boxes[..., 3] *= scale_y
        return boxes
    raise ValueError(f'Unsupported box_format: {box_format}')


class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """Convert common image / mask containers into torch tensors.

    Expected keys (all optional except ``image``):
    - image
    - semantic_mask
    - instance_masks
    - boxes
    """

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        sample['image'] = _to_tensor_image(sample['image'])
        if 'semantic_mask' in sample:
            sample['semantic_mask'] = _to_tensor_mask(sample.get('semantic_mask'))
        if 'instance_masks' in sample:
            sample['instance_masks'] = _to_tensor_mask(sample.get('instance_masks'))
        if 'boxes' in sample and sample['boxes'] is not None and not isinstance(sample['boxes'], torch.Tensor):
            sample['boxes'] = torch.as_tensor(sample['boxes'])
        return sample


class ConvertImageDtype:
    def __init__(self, dtype: torch.dtype = torch.float32, scale: bool = True):
        self.dtype = dtype
        self.scale = bool(scale)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample['image']
        if not isinstance(image, torch.Tensor):
            raise TypeError('ConvertImageDtype expects image to be a torch.Tensor')
        image = image.to(self.dtype)
        if self.scale and image.max() > 1.0:
            image = image / 255.0
        sample['image'] = image
        return sample


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample['image']
        sample['image'] = (image - self.mean.to(image.device, image.dtype)) / self.std.to(image.device, image.dtype)
        return sample


class Resize:
    def __init__(self, size: Tuple[int, int], box_format: str = 'xyxy'):
        self.size = tuple(size)
        self.box_format = box_format

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        image = sample['image']
        h, w = image.shape[-2:]
        out_h, out_w = self.size
        sample['image'] = _resize_tensor_image(image, (out_h, out_w))

        if 'semantic_mask' in sample and sample['semantic_mask'] is not None:
            sample['semantic_mask'] = _resize_mask(sample['semantic_mask'], (out_h, out_w))
        if 'instance_masks' in sample and sample['instance_masks'] is not None:
            sample['instance_masks'] = _resize_mask(sample['instance_masks'], (out_h, out_w))
        if 'boxes' in sample and sample['boxes'] is not None:
            sample['boxes'] = _scale_boxes(sample['boxes'], out_w / w, out_h / h, self.box_format)
        sample['img_shape'] = (out_h, out_w)
        sample['scale_factor'] = (out_w / w, out_h / h)
        return sample


class ResizeLongestSide:
    def __init__(self, long_side: int, box_format: str = 'xyxy'):
        self.long_side = int(long_side)
        self.box_format = box_format

    def __call__(self, sample: Sample) -> Sample:
        image = sample['image']
        h, w = image.shape[-2:]
        scale = self.long_side / max(h, w)
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        return Resize((out_h, out_w), box_format=self.box_format)(sample)


class RandomResize:
    def __init__(self, scales: Sequence[Tuple[int, int]], box_format: str = 'xyxy'):
        self.scales = list(scales)
        self.box_format = box_format

    def __call__(self, sample: Sample) -> Sample:
        size = random.choice(self.scales)
        return Resize(size, box_format=self.box_format)(sample)


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5, box_format: str = 'xyxy'):
        self.prob = float(prob)
        self.box_format = box_format

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        sample = dict(sample)
        image = sample['image']
        image_width = int(image.shape[-1])
        sample['image'] = torch.flip(image, dims=[-1])

        if 'semantic_mask' in sample and sample['semantic_mask'] is not None:
            sample['semantic_mask'] = torch.flip(sample['semantic_mask'], dims=[-1])
        if 'instance_masks' in sample and sample['instance_masks'] is not None:
            sample['instance_masks'] = torch.flip(sample['instance_masks'], dims=[-1])
        if 'boxes' in sample and sample['boxes'] is not None:
            sample['boxes'] = _flip_boxes_horizontally(sample['boxes'], image_width, self.box_format)
        return sample


class PadToSize:
    def __init__(self, size: Tuple[int, int], image_pad_value: float = 0.0, mask_pad_value: int = 0):
        self.size = tuple(size)
        self.image_pad_value = float(image_pad_value)
        self.mask_pad_value = int(mask_pad_value)

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
        sample['image'] = self._pad_last_two_dims(sample['image'], self.image_pad_value)
        if 'semantic_mask' in sample and sample['semantic_mask'] is not None:
            sample['semantic_mask'] = self._pad_last_two_dims(sample['semantic_mask'], self.mask_pad_value)
        if 'instance_masks' in sample and sample['instance_masks'] is not None:
            sample['instance_masks'] = self._pad_last_two_dims(sample['instance_masks'], self.mask_pad_value)
        sample['pad_shape'] = self.size
        return sample


class ClampBoxes:
    def __init__(self, image_key: str = 'image', box_format: str = 'xyxy'):
        self.image_key = image_key
        self.box_format = box_format

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        boxes = sample.get('boxes')
        if boxes is None:
            return sample
        h, w = sample[self.image_key].shape[-2:]
        boxes = boxes.clone()
        if self.box_format == 'xyxy':
            boxes[..., 0::2] = boxes[..., 0::2].clamp(0, w)
            boxes[..., 1::2] = boxes[..., 1::2].clamp(0, h)
        elif self.box_format == 'cxcywh':
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(min=0)
            boxes[..., 3] = boxes[..., 3].clamp(min=0)
        else:
            raise ValueError(f'Unsupported box_format: {self.box_format}')
        sample['boxes'] = boxes
        return sample
