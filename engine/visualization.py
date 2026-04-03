from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

@dataclass
class VisualizerConfig:
    enabled: bool = False
    save_dir: str = './visualizations'
    save_stage: str = 'val'
    alpha: float = 0.45

    save_original: bool = True
    save_prediction: bool = True
    save_ground_truth: bool = True
    save_dense_prediction: bool = True

    max_samples: Optional[int] = None
    image_folder_pattern: str = 'image_{image_id:06d}'
    ignore_index: int = 255


class VisualizationManager:
    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._num_saved = 0

    @classmethod
    def from_cfg(
        cls,
        cfg_dict: Optional[Dict[str, Any]],
        work_dir: Optional[str] = None,
    ) -> Optional['VisualizationManager']:
        if cfg_dict is None:
            return None
        cfg = VisualizerConfig(**cfg_dict)
        if not cfg.enabled:
            return None

        save_dir = Path(cfg.save_dir)
        if not save_dir.is_absolute() and work_dir is not None:
            cfg.save_dir = str(Path(work_dir) / save_dir)
        return cls(cfg)

    def should_save(self, stage: str) -> bool:
        if not self.cfg.enabled:
            return False
        if self.cfg.save_stage == 'all':
            return True
        return self.cfg.save_stage == stage

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
            x = x.float().clamp(0, 1)
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
    def _extract_original_image(batch: Any, batch_index: int) -> Image.Image:
        raw_images = getattr(batch, 'raw_images', None)
        if raw_images is not None and batch_index < len(raw_images) and raw_images[batch_index] is not None:
            return VisualizationManager._to_uint8_image(raw_images[batch_index])
        return VisualizationManager._to_uint8_image(batch.img_batch[batch_index])

    @staticmethod
    def _extract_image_id(batch: Any, batch_index: int) -> int:
        try:
            meta = batch.find_metadatas[0]
            return int(meta.original_image_id[batch_index].item())
        except Exception:
            return int(batch_index)

    def _resolve_sample_dir(self, image_id: int, epoch: Optional[int], stage: str) -> Path:
        parts = [self.save_dir, stage]
        if epoch is not None:
            parts.append(Path(f'epoch_{epoch:03d}'))
        parts.append(Path(self.cfg.image_folder_pattern.format(image_id=image_id)))
        sample_dir = Path(*parts)
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    @staticmethod
    def _prepare_label_map(label_map: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = label_map.detach().cpu()
        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError(f'Expected [1,H,W] or [H,W], got {tuple(x.shape)}')
            x = x[0]
        if x.dim() != 2:
            raise ValueError(f'Expected [H,W], got {tuple(x.shape)}')

        if tuple(x.shape[-2:]) != tuple(out_hw):
            x = F.interpolate(x[None, None].float(), size=out_hw, mode='nearest')[0, 0].long()
        else:
            x = x.long()
        return x

    @staticmethod
    def _build_palette(num_classes: int) -> np.ndarray:
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for i in range(num_classes):
            lab = i
            r = g = b = 0
            for j in range(8):
                r |= ((lab >> 0) & 1) << (7 - j)
                g |= ((lab >> 1) & 1) << (7 - j)
                b |= ((lab >> 2) & 1) << (7 - j)
                lab >>= 3
            palette[i] = np.array([r, g, b], dtype=np.uint8)
        return palette

    def _colorize_label_map(self, label_map: torch.Tensor, num_classes: int) -> Image.Image:
        label_map_np = label_map.cpu().numpy().astype(np.int64)
        h, w = label_map_np.shape
        palette = self._build_palette(num_classes)

        color = np.zeros((h, w, 3), dtype=np.uint8)
        valid = label_map_np != self.cfg.ignore_index

        safe_label = label_map_np.copy()
        safe_label[~valid] = 0
        color[valid] = palette[safe_label[valid]]
        return Image.fromarray(color, mode='RGB')

    def _overlay_label_map(
        self,
        image: Image.Image,
        label_map: torch.Tensor,
        num_classes: int,
    ) -> Image.Image:
        base = np.asarray(image.convert('RGB')).astype(np.float32)
        color = np.asarray(self._colorize_label_map(label_map, num_classes)).astype(np.float32)

        valid = (label_map.cpu().numpy() != self.cfg.ignore_index)[..., None]
        out = base.copy()
        out[valid[..., 0]] = (
            (1.0 - self.cfg.alpha) * base[valid[..., 0]]
            + self.cfg.alpha * color[valid[..., 0]]
        )
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode='RGB')

    @staticmethod
    def _extract_pred_from_logits(
        logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        输入 logits: [B, C, H, W]
        输出 pred:   [B, H, W]
        """
        if logits is None:
            return None
        if logits.dim() != 4:
            raise ValueError(f'Expected logits [B,C,H,W], got {tuple(logits.shape)}')
        return logits.argmax(dim=1)

    def save_semantic_batch(
        self,
        batch: Any,
        semantic_outputs: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
        *,
        epoch: Optional[int],
        stage: str = 'val',
    ) -> None:
        if not self.should_save(stage):
            return

        if 'semantic_logits' not in semantic_outputs:
            raise ValueError("semantic_outputs must contain 'semantic_logits'.")

        fused_logits = semantic_outputs['semantic_logits']             # [B,C,H,W]
        fused_pred = self._extract_pred_from_logits(fused_logits)     # [B,H,W]

        dense_logits = semantic_outputs.get('semantic_logits_dense', None)
        dense_pred = self._extract_pred_from_logits(dense_logits) if dense_logits is not None else None

        gt = semantic_targets['label_map']                            # [B,H,W]

        if gt.dim() == 4:
            if gt.shape[1] != 1:
                raise ValueError(f'Expected gt [B,H,W] or [B,1,H,W], got {tuple(gt.shape)}')
            gt = gt[:, 0]
        if gt.dim() != 3:
            raise ValueError(f'Expected gt [B,H,W], got {tuple(gt.shape)}')

        num_classes = int(fused_logits.shape[1])
        bsz = int(fused_logits.shape[0])

        for b in range(bsz):
            if self.cfg.max_samples is not None and self._num_saved >= self.cfg.max_samples:
                return

            image_id = self._extract_image_id(batch, b)
            sample_dir = self._resolve_sample_dir(image_id=image_id, epoch=epoch, stage=stage)

            image = self._extract_original_image(batch, b)
            out_hw = image.size[::-1]

            fused_pred_label = self._prepare_label_map(fused_pred[b], out_hw)
            gt_label = self._prepare_label_map(gt[b], out_hw)

            dense_pred_label = None
            if dense_pred is not None:
                dense_pred_label = self._prepare_label_map(dense_pred[b], out_hw)

            if self.cfg.save_original:
                image.save(sample_dir / 'original.png')

            if self.cfg.save_prediction:
                self._overlay_label_map(image, fused_pred_label, num_classes).save(
                    sample_dir / 'pred_overlay.png'
                )

            if self.cfg.save_dense_prediction and dense_pred_label is not None:
                self._overlay_label_map(image, dense_pred_label, num_classes).save(
                    sample_dir / 'pred_semantic_overlay.png'
                )

            if self.cfg.save_ground_truth:
                self._overlay_label_map(image, gt_label, num_classes).save(
                    sample_dir / 'gt_overlay.png'
                )

            try:
                class_names: List[str] = batch.find_metadatas[0].class_names
                with open(sample_dir / 'classes.txt', 'w', encoding='utf-8') as f:
                    for i, name in enumerate(class_names):
                        f.write(f'{i}\t{name}\n')
            except Exception:
                pass

            self._num_saved += 1