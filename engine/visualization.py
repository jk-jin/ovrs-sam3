from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.task_modes import OUTPUT_KEYS


@dataclass
class VisualizerConfig:
    enabled: bool = False
    save_dir: str = "./visualizations"
    save_stage: str = "val"
    alpha: float = 0.45

    save_original: bool = True
    save_prediction: bool = True
    save_ground_truth: bool = True
    save_semantic_prediction: bool = True

    vis_prob: float = 0.05
    max_samples_per_epoch: Optional[int] = 50
    vis_seed: int = 42

    image_folder_pattern: str = "image_{image_id:06d}"
    ignore_index: int = 255


@dataclass
class VisualizationContext:
    model: torch.nn.Module
    batch: Any
    semantic_outputs: Dict[str, torch.Tensor]
    semantic_targets: Dict[str, torch.Tensor]
    epoch: Optional[int]
    stage: str
    selected_indices: List[int]


class VisualizationTask:
    name = "base"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        raise NotImplementedError


class BaseSemanticOverlayTask(VisualizationTask):
    name = "base_semantic_overlay"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        semantic_outputs = ctx.semantic_outputs
        semantic_targets = ctx.semantic_targets
        batch = ctx.batch

        if OUTPUT_KEYS.final_score_map not in semantic_outputs:
            raise ValueError(f"outputs must contain '{OUTPUT_KEYS.final_score_map}'.")

        final_score_map = semantic_outputs[OUTPUT_KEYS.final_score_map]
        final_pred = manager._extract_pred_from_logits(final_score_map)

        semantic_score_map = semantic_outputs.get(OUTPUT_KEYS.semantic_score_map, None)
        semantic_pred = (
            manager._extract_pred_from_logits(semantic_score_map)
            if semantic_score_map is not None else None
        )

        gt = semantic_targets["label_map"]
        if gt.dim() == 4:
            if gt.shape[1] != 1:
                raise ValueError(f"Expected gt as [B,1,H,W] or [B,H,W], got {tuple(gt.shape)}")
            gt = gt[:, 0]

        pred_num_classes = int(final_score_map.shape[1])

        try:
            class_names: List[str] = batch.find_metadatas[0].class_names
            gt_num_classes = len(class_names)
        except Exception:
            gt_num_classes = manager._infer_num_classes_from_label_map(
                gt,
                ignore_index=manager.cfg.ignore_index,
                fallback=pred_num_classes,
            )
            class_names = None

        semantic_num_classes = pred_num_classes
        if semantic_score_map is not None:
            semantic_num_classes = int(semantic_score_map.shape[1])

        for b in ctx.selected_indices:
            image_id = manager._extract_image_id(batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(batch, b)
            original_image = manager._extract_original_reference_image(batch, b)
            out_hw = overlay_image.size[::-1]

            final_pred_label = manager._prepare_label_map(final_pred[b], out_hw)
            gt_label = manager._prepare_label_map(gt[b], out_hw)

            if manager.cfg.save_original:
                original_image.save(sample_dir / "original.png")

            if manager.cfg.save_prediction:
                manager._overlay_label_map(
                    overlay_image,
                    final_pred_label,
                    pred_num_classes,
                ).save(sample_dir / "pred_overlay.png")

            if manager.cfg.save_ground_truth:
                manager._overlay_label_map(
                    overlay_image,
                    gt_label,
                    gt_num_classes,
                ).save(sample_dir / "gt_overlay.png")

            if semantic_pred is not None and manager.cfg.save_semantic_prediction:
                semantic_pred_label = manager._prepare_label_map(semantic_pred[b], out_hw)
                manager._overlay_label_map(
                    overlay_image,
                    semantic_pred_label,
                    semantic_num_classes,
                ).save(sample_dir / "pred_semantic_overlay.png")

            if class_names is not None:
                with open(sample_dir / "classes.txt", "w", encoding="utf-8") as f:
                    for i, name in enumerate(class_names):
                        f.write(f"{i}\t{name}\n")


class VisualizationManager:
    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._saved_counts: Dict[Tuple[str, int], int] = {}
        self.tasks = self._build_tasks()

    def _build_tasks(self):
        return [
            BaseSemanticOverlayTask(),
        ]

    @classmethod
    def from_cfg(
        cls,
        cfg_dict: Optional[Dict[str, Any]],
        work_dir: Optional[str] = None,
    ) -> Optional["VisualizationManager"]:
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
        if self.cfg.save_stage == "all":
            return True
        return self.cfg.save_stage == stage

    @staticmethod
    def _to_uint8_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

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
            return Image.fromarray(arr, mode="RGB")

        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _extract_overlay_image(batch: Any, batch_index: int) -> Image.Image:
        raw_images = getattr(batch, "raw_images", None)
        if raw_images is not None and batch_index < len(raw_images) and raw_images[batch_index] is not None:
            return VisualizationManager._to_uint8_image(raw_images[batch_index])
        return VisualizationManager._to_uint8_image(batch.img_batch[batch_index])

    @staticmethod
    def _extract_original_reference_image(batch: Any, batch_index: int) -> Image.Image:
        raw_images_original = getattr(batch, "raw_images_original", None)
        if (
            raw_images_original is not None
            and batch_index < len(raw_images_original)
            and raw_images_original[batch_index] is not None
        ):
            return VisualizationManager._to_uint8_image(raw_images_original[batch_index])

        raw_images = getattr(batch, "raw_images", None)
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
            parts.append(Path(f"epoch_{epoch:03d}"))
        parts.append(Path(self.cfg.image_folder_pattern.format(image_id=image_id)))
        sample_dir = Path(*parts)
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    @staticmethod
    def _prepare_label_map(label_map: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = label_map.detach().cpu()
        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError(f"Expected [1,H,W] or [H,W], got {tuple(x.shape)}")
            x = x[0]
        if x.dim() != 2:
            raise ValueError(f"Expected [H,W], got {tuple(x.shape)}")

        if tuple(x.shape[-2:]) != tuple(out_hw):
            x = F.interpolate(x[None, None].float(), size=out_hw, mode="nearest")[0, 0].long()
        else:
            x = x.long()
        return x

    @staticmethod
    def _infer_num_classes_from_label_map(
        label_map: torch.Tensor,
        ignore_index: int,
        fallback: int,
    ) -> int:
        x = label_map.detach().cpu().long()
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}")
            x = x[:, 0]
        elif x.dim() != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(x.shape)}")

        valid = x != int(ignore_index)
        if not valid.any():
            return int(fallback)

        max_label = int(x[valid].max().item())
        return max(int(fallback), max_label + 1)

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

        valid = label_map_np != self.cfg.ignore_index
        if valid.any():
            max_label = int(label_map_np[valid].max())
            num_classes = max(int(num_classes), max_label + 1)
        else:
            num_classes = int(num_classes)

        palette = self._build_palette(num_classes)

        color = np.zeros((h, w, 3), dtype=np.uint8)
        safe_label = label_map_np.copy()
        safe_label[~valid] = 0
        color[valid] = palette[safe_label[valid]]
        return Image.fromarray(color, mode="RGB")

    def _overlay_label_map(
        self,
        image: Image.Image,
        label_map: torch.Tensor,
        num_classes: int,
    ) -> Image.Image:
        base = np.asarray(image.convert("RGB")).astype(np.float32)
        color = np.asarray(self._colorize_label_map(label_map, num_classes)).astype(np.float32)

        valid = (label_map.cpu().numpy() != self.cfg.ignore_index)[..., None]
        out = base.copy()
        out[valid[..., 0]] = (
            (1.0 - self.cfg.alpha) * base[valid[..., 0]]
            + self.cfg.alpha * color[valid[..., 0]]
        )
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    @staticmethod
    def _extract_pred_from_logits(
        logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if logits is None:
            return None
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")
        return logits.argmax(dim=1)

    def _get_epoch_key(self, stage: str, epoch: Optional[int]) -> Tuple[str, int]:
        return stage, (-1 if epoch is None else int(epoch))

    def _get_saved_count(self, stage: str, epoch: Optional[int]) -> int:
        key = self._get_epoch_key(stage, epoch)
        return int(self._saved_counts.get(key, 0))

    def _increase_saved_count(self, stage: str, epoch: Optional[int]) -> None:
        key = self._get_epoch_key(stage, epoch)
        self._saved_counts[key] = self._get_saved_count(stage, epoch) + 1

    def _should_save_sample(
        self,
        image_id: int,
        stage: str,
        epoch: Optional[int],
    ) -> bool:
        if not self.should_save(stage):
            return False

        if self.cfg.vis_prob <= 0:
            return False

        saved_count = self._get_saved_count(stage, epoch)
        if self.cfg.max_samples_per_epoch is not None:
            if saved_count >= int(self.cfg.max_samples_per_epoch):
                return False

        if self.cfg.vis_prob >= 1.0:
            return True

        epoch_value = -1 if epoch is None else int(epoch)
        token = f"{self.cfg.vis_seed}:{stage}:{epoch_value}:{int(image_id)}"
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / float(16 ** 8 - 1)

        return value < float(self.cfg.vis_prob)

    def run(
        self,
        model: torch.nn.Module,
        batch: Any,
        semantic_outputs: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
        *,
        epoch: Optional[int],
        stage: str = "val",
    ) -> None:
        if not self.should_save(stage):
            return

        bsz = int(batch.img_batch.shape[0])
        selected_indices = []

        for b in range(bsz):
            image_id = self._extract_image_id(batch, b)
            if self._should_save_sample(image_id=image_id, stage=stage, epoch=epoch):
                selected_indices.append(b)
                self._increase_saved_count(stage, epoch)

        if len(selected_indices) == 0:
            return

        ctx = VisualizationContext(
            model=model,
            batch=batch,
            semantic_outputs=semantic_outputs,
            semantic_targets=semantic_targets,
            epoch=epoch,
            stage=stage,
            selected_indices=selected_indices,
        )

        for task in self.tasks:
            task.run(self, ctx)