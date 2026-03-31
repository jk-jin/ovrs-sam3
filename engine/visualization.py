from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


Color = Tuple[int, int, int]
WriterFn = Callable[["VisualizationManager", "VisualizationContext"], None]


@dataclass
class VisualizerConfig:
    enabled: bool = False
    save_dir: str = './visualizations'
    save_stage: str = 'val'  # val | train | all
    alpha: float = 0.45
    threshold: float = 0.5
    save_original: bool = True
    save_prediction: bool = True
    save_ground_truth: bool = True
    pred_color: Color = (255, 0, 0)
    gt_color: Color = (0, 255, 0)
    max_samples: Optional[int] = None
    image_folder_pattern: str = 'image_{image_id:06d}'
    prompt_in_filename: bool = True


@dataclass
class VisualizationContext:
    sample_dir: Path
    image_id: int
    prompt_text: str
    epoch: Optional[int]
    stage: str
    task: str
    original_image: Image.Image
    pred_mask: Optional[torch.Tensor] = None
    gt_mask: Optional[torch.Tensor] = None
    batch: Any = None
    outputs: Any = None
    targets: Any = None
    batch_index: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


class VisualizationManager:
    """Reusable visualization manager for validation / debug saving.

    Design goals:
    - configurable on/off and save directory
    - one folder per image
    - default saves: original, pred overlay, gt overlay
    - easy to extend: register additional writer callbacks later
    """

    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._num_saved = 0
        self._writers: List[Tuple[str, WriterFn]] = []
        self._register_default_writers()

    @classmethod
    def from_cfg(cls, cfg_dict: Optional[Dict[str, Any]], work_dir: Optional[str] = None) -> Optional['VisualizationManager']:
        if cfg_dict is None:
            return None
        cfg = VisualizerConfig(**cfg_dict)
        if not cfg.enabled:
            return None
        save_dir = Path(cfg.save_dir)
        if not save_dir.is_absolute() and work_dir is not None:
            cfg.save_dir = str(Path(work_dir) / save_dir)
        return cls(cfg)

    def register_writer(self, name: str, fn: WriterFn) -> None:
        self._writers.append((name, fn))

    def should_save(self, stage: str) -> bool:
        if not self.cfg.enabled:
            return False
        if self.cfg.save_stage == 'all':
            return True
        return self.cfg.save_stage == stage

    def _register_default_writers(self) -> None:
        if self.cfg.save_original:
            self.register_writer('original', self._write_original)
        if self.cfg.save_prediction:
            self.register_writer('prediction', self._write_prediction)
        if self.cfg.save_ground_truth:
            self.register_writer('ground_truth', self._write_ground_truth)

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r'[^a-z0-9\-_]+', '_', text)
        text = re.sub(r'_+', '_', text).strip('_')
        return text or 'prompt'

    @staticmethod
    def _to_uint8_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        if isinstance(image, torch.Tensor):
            x = image.detach().cpu()
            if x.dim() == 4:
                x = x[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            x = x.float()
            x = x.clamp(0, 1)
            arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            return Image.fromarray(arr, mode='RGB')
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode='RGB')
        raise TypeError(f'Unsupported image type: {type(image)}')

    @staticmethod
    def _to_bool_mask(mask: Any, out_hw: Tuple[int, int]) -> np.ndarray:
        if mask is None:
            return np.zeros(out_hw, dtype=bool)
        if isinstance(mask, torch.Tensor):
            m = mask.detach().float().cpu()
            if m.dim() == 4:
                m = m[0, 0]
            elif m.dim() == 3:
                if m.shape[0] == 1:
                    m = m[0]
                else:
                    m = m.any(dim=0).float()
            if tuple(m.shape[-2:]) != tuple(out_hw):
                m = F.interpolate(m[None, None], size=out_hw, mode='nearest')[0, 0]
            return (m >= 0.5).numpy().astype(bool)
        if isinstance(mask, np.ndarray):
            if mask.shape != out_hw:
                pil = Image.fromarray(mask.astype(np.uint8) * 255)
                pil = pil.resize((out_hw[1], out_hw[0]), Image.NEAREST)
                mask = np.array(pil) > 0
            return mask.astype(bool)
        raise TypeError(f'Unsupported mask type: {type(mask)}')

    @staticmethod
    def _overlay_mask(image: Image.Image, mask: Any, color: Color, alpha: float) -> Image.Image:
        base = np.asarray(image.convert('RGB')).copy().astype(np.float32)
        h, w = base.shape[:2]
        mask_bool = VisualizationManager._to_bool_mask(mask, (h, w))
        color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        out = base.copy()
        out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr.reshape(3)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode='RGB')

    def _resolve_sample_dir(self, image_id: int, epoch: Optional[int], stage: str) -> Path:
        parts = [self.save_dir, stage]
        if epoch is not None:
            parts.append(Path(f'epoch_{epoch:03d}'))
        parts.append(Path(self.cfg.image_folder_pattern.format(image_id=image_id)))
        sample_dir = Path(*parts)
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    def _filename(self, prefix: str, prompt_text: str) -> str:
        if not self.cfg.prompt_in_filename:
            return f'{prefix}.png'
        slug = self._slugify(prompt_text)
        return f'{prefix}_{slug}.png'

    def _write_original(self, _manager: 'VisualizationManager', ctx: VisualizationContext) -> None:
        path = ctx.sample_dir / 'original.png'
        if not path.exists():
            ctx.original_image.save(path)

    def _write_prediction(self, _manager: 'VisualizationManager', ctx: VisualizationContext) -> None:
        if ctx.pred_mask is None:
            return
        overlay = self._overlay_mask(ctx.original_image, ctx.pred_mask, self.cfg.pred_color, self.cfg.alpha)
        overlay.save(ctx.sample_dir / self._filename('pred', ctx.prompt_text))

    def _write_ground_truth(self, _manager: 'VisualizationManager', ctx: VisualizationContext) -> None:
        if ctx.gt_mask is None:
            return
        overlay = self._overlay_mask(ctx.original_image, ctx.gt_mask, self.cfg.gt_color, self.cfg.alpha)
        overlay.save(ctx.sample_dir / self._filename('gt', ctx.prompt_text))

    def save_named_image(self, sample_dir: Path, name: str, image: Any) -> None:
        """Generic helper for future extension: save more intermediate results."""
        self._to_uint8_image(image).save(sample_dir / f'{name}.png')

    def save_named_mask_overlay(self, sample_dir: Path, original_image: Any, name: str, mask: Any, color: Color = (0, 0, 255)) -> None:
        base = self._to_uint8_image(original_image)
        overlay = self._overlay_mask(base, mask, color=color, alpha=self.cfg.alpha)
        overlay.save(sample_dir / f'{name}.png')

    def _extract_original_image(self, batch: Any, batch_index: int) -> Image.Image:
        raw_images = getattr(batch, 'raw_images', None)
        if raw_images is not None and batch_index < len(raw_images) and raw_images[batch_index] is not None:
            return self._to_uint8_image(raw_images[batch_index])
        return self._to_uint8_image(batch.img_batch[batch_index])

    @staticmethod
    def _extract_image_id(batch: Any, batch_index: int) -> int:
        try:
            meta = batch.find_metadatas[0]
            return int(meta.original_image_id[batch_index].item())
        except Exception:
            return int(batch_index)

    @staticmethod
    def _extract_prompt_text(batch: Any, batch_index: int) -> str:
        try:
            return str(batch.find_text_batch[batch_index])
        except Exception:
            return f'prompt_{batch_index}'

    def save_semantic_batch(self, batch: Any, semantic_outputs: Dict[str, torch.Tensor], semantic_targets: Dict[str, torch.Tensor], *, epoch: Optional[int], stage: str = 'val') -> None:
        if not self.should_save(stage):
            return
        logits = semantic_outputs['semantic_logits']
        pred = (logits.sigmoid() >= self.cfg.threshold)
        gt = semantic_targets['semantic_masks']
        bsz = int(logits.shape[0])

        for b in range(bsz):
            if self.cfg.max_samples is not None and self._num_saved >= self.cfg.max_samples:
                return
            image_id = self._extract_image_id(batch, b)
            prompt_text = self._extract_prompt_text(batch, b)
            sample_dir = self._resolve_sample_dir(image_id=image_id, epoch=epoch, stage=stage)
            ctx = VisualizationContext(
                sample_dir=sample_dir,
                image_id=image_id,
                prompt_text=prompt_text,
                epoch=epoch,
                stage=stage,
                task='semantic',
                original_image=self._extract_original_image(batch, b),
                pred_mask=pred[b],
                gt_mask=gt[b],
                batch=batch,
                outputs=semantic_outputs,
                targets=semantic_targets,
                batch_index=b,
            )
            for _, fn in self._writers:
                fn(self, ctx)
            self._num_saved += 1

    def save_instance_batch(self, batch: Any, instance_outputs: Dict[str, torch.Tensor], instance_targets: Dict[str, torch.Tensor], *, epoch: Optional[int], stage: str = 'val') -> None:
        if not self.should_save(stage):
            return
        pred_masks = instance_outputs.get('pred_masks')
        pred_logits = instance_outputs.get('pred_logits')
        if pred_masks is None or pred_logits is None:
            return
        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        if pred_logits.dim() == 4:
            pred_logits = pred_logits[-1]
        pred_scores = pred_logits.squeeze(-1).sigmoid()
        bsz = int(pred_masks.shape[0])
        gt_masks = instance_targets.get('masks')

        for b in range(bsz):
            if self.cfg.max_samples is not None and self._num_saved >= self.cfg.max_samples:
                return
            image_id = self._extract_image_id(batch, b)
            prompt_text = self._extract_prompt_text(batch, b)
            sample_dir = self._resolve_sample_dir(image_id=image_id, epoch=epoch, stage=stage)
            keep = pred_scores[b] >= self.cfg.threshold
            pred_union = pred_masks[b][keep].sigmoid().ge(self.cfg.threshold).any(dim=0) if keep.any() else torch.zeros_like(pred_masks[b, 0], dtype=torch.bool)
            gt_union = gt_masks[b].any(dim=0) if gt_masks is not None and gt_masks.ndim >= 4 else None
            ctx = VisualizationContext(
                sample_dir=sample_dir,
                image_id=image_id,
                prompt_text=prompt_text,
                epoch=epoch,
                stage=stage,
                task='instance',
                original_image=self._extract_original_image(batch, b),
                pred_mask=pred_union,
                gt_mask=gt_union,
                batch=batch,
                outputs=instance_outputs,
                targets=instance_targets,
                batch_index=b,
            )
            for _, fn in self._writers:
                fn(self, ctx)
            self._num_saved += 1
