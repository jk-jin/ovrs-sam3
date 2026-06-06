from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config_dataclasses import VisualizerConfig
from ..models.task_modes import OUTPUT_KEYS


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
        outputs = ctx.semantic_outputs
        targets = ctx.semantic_targets
        batch = ctx.batch

        (
            final_logits,
            raw_final_score_map,
            final_score_map,
            raw_final_source,
            final_source,
        ) = manager._extract_final_logits_raw_and_gated_score_maps(outputs)

        raw_final_pred = manager._build_eval_style_final_pred(
            outputs=outputs,
            final_score_map=raw_final_score_map,
        )

        final_pred = manager._build_eval_style_final_pred(
            outputs=outputs,
            final_score_map=final_score_map,
        )

        semantic_logits, semantic_score_map, semantic_source = (
            manager._extract_semantic_logits_and_score_map(outputs)
        )
        semantic_pred = (
            manager._extract_pred_from_logits(semantic_logits)
            if semantic_logits is not None
            else None
        )

        gt = targets["label_map"]
        if gt.dim() == 4:
            if gt.shape[1] != 1:
                raise ValueError(
                    f"Expected gt as [B, 1, H, W] or [B, H, W], "
                    f"got {tuple(gt.shape)}."
                )
            gt = gt[:, 0]
        elif gt.dim() != 3:
            raise ValueError(
                f"Expected gt as [B, H, W] or [B, 1, H, W], "
                f"got {tuple(gt.shape)}."
            )

        final_num_classes = int(final_score_map.shape[1])

        try:
            class_names: Optional[List[str]] = [
                str(x) for x in batch.find_metadatas[0].class_names
            ]
            gt_num_classes = len(class_names)
        except Exception:
            class_names = None
            gt_num_classes = manager._infer_num_classes_from_label_map(
                gt,
                ignore_index=manager.cfg.ignore_index,
                fallback=final_num_classes,
            )

        semantic_num_classes = final_num_classes
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

            if manager.cfg.save_raw_final_prediction:
                raw_final_pred_label = manager._prepare_label_map(
                    raw_final_pred[b],
                    out_hw,
                )

                manager._colorize_label_map(
                    raw_final_pred_label,
                    final_num_classes,
                ).save(sample_dir / "pred_raw_final.png")

                manager._overlay_label_map(
                    overlay_image,
                    raw_final_pred_label,
                    final_num_classes,
                ).save(sample_dir / "pred_raw_final_overlay.png")

            if manager.cfg.save_prediction:
                manager._colorize_label_map(
                    final_pred_label,
                    final_num_classes,
                ).save(sample_dir / "pred.png")

                manager._overlay_label_map(
                    overlay_image,
                    final_pred_label,
                    final_num_classes,
                ).save(sample_dir / "pred_overlay.png")

            if manager.cfg.save_ground_truth:
                manager._colorize_label_map(
                    gt_label,
                    gt_num_classes,
                ).save(sample_dir / "gt.png")

                manager._overlay_label_map(
                    overlay_image,
                    gt_label,
                    gt_num_classes,
                ).save(sample_dir / "gt_overlay.png")

            if semantic_pred is not None and manager.cfg.save_semantic_prediction:
                semantic_pred_label = manager._prepare_label_map(
                    semantic_pred[b],
                    out_hw,
                )

                manager._colorize_label_map(
                    semantic_pred_label,
                    semantic_num_classes,
                ).save(sample_dir / "pred_semantic.png")

                manager._overlay_label_map(
                    overlay_image,
                    semantic_pred_label,
                    semantic_num_classes,
                ).save(sample_dir / "pred_semantic_overlay.png")

            with open(sample_dir / "visualization_sources.txt", "w", encoding="utf-8") as f:
                f.write("item\tsource\n")
                if manager.cfg.save_raw_final_prediction:
                    f.write(
                        "pred_raw_final.png\t"
                        f"{raw_final_source}; eval_style_pred="
                        f"use_score_map={manager.eval_use_score_map}, "
                        f"prob_thd={manager.eval_prob_thd}, "
                        f"bg_idx={manager.eval_bg_idx}\n"
                    )
                    f.write(
                        "pred_raw_final_overlay.png\t"
                        f"{raw_final_source}; eval_style_pred="
                        f"use_score_map={manager.eval_use_score_map}, "
                        f"prob_thd={manager.eval_prob_thd}, "
                        f"bg_idx={manager.eval_bg_idx}\n"
                    )
                f.write(
                    "pred.png\t"
                    f"{final_source}; eval_style_pred="
                    f"use_score_map={manager.eval_use_score_map}, "
                    f"prob_thd={manager.eval_prob_thd}, "
                    f"bg_idx={manager.eval_bg_idx}\n"
                )
                f.write(
                    "pred_overlay.png\t"
                    f"{final_source}; eval_style_pred="
                    f"use_score_map={manager.eval_use_score_map}, "
                    f"prob_thd={manager.eval_prob_thd}, "
                    f"bg_idx={manager.eval_bg_idx}\n"
                )
                f.write(f"final_score_heatmaps\t{final_source}\n")
                if semantic_source is not None:
                    f.write(f"pred_semantic.png\t{semantic_source}\n")
                    f.write(f"pred_semantic_overlay.png\t{semantic_source}\n")

            if class_names is not None:
                with open(sample_dir / "classes.txt", "w", encoding="utf-8") as f:
                    for i, name in enumerate(class_names):
                        f.write(f"{i}\t{name}\n")


class ScoreAnalysisTask(VisualizationTask):
    name = "score_analysis"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        outputs = ctx.semantic_outputs
        batch = ctx.batch

        _, semantic_score_map, _ = manager._extract_semantic_logits_and_score_map(outputs)

        (
            _,
            raw_final_score_map,
            final_score_map,
            _,
            _,
        ) = manager._extract_final_logits_raw_and_gated_score_maps(outputs)

        if semantic_score_map is None:
            return

        if semantic_score_map.dim() != 4:
            raise ValueError(
                f"Expected semantic_score_map as [B, C, H, W], "
                f"got {tuple(semantic_score_map.shape)}."
            )
        if raw_final_score_map.dim() != 4:
            raise ValueError(
                f"Expected raw_final_score_map as [B, C, H, W], "
                f"got {tuple(raw_final_score_map.shape)}."
            )
        if final_score_map.dim() != 4:
            raise ValueError(
                f"Expected final_score_map as [B, C, H, W], "
                f"got {tuple(final_score_map.shape)}."
            )
        if semantic_score_map.shape != raw_final_score_map.shape:
            raise ValueError(
                "semantic_score_map and raw_final_score_map shape mismatch: "
                f"{tuple(semantic_score_map.shape)} vs {tuple(raw_final_score_map.shape)}."
            )
        if semantic_score_map.shape != final_score_map.shape:
            raise ValueError(
                "semantic_score_map and final_score_map shape mismatch: "
                f"{tuple(semantic_score_map.shape)} vs {tuple(final_score_map.shape)}."
            )

        try:
            class_names: Optional[List[str]] = [
                str(x) for x in batch.find_metadatas[0].class_names
            ]
        except Exception:
            class_names = None

        for b in ctx.selected_indices:
            image_id = manager._extract_image_id(batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(batch, b)
            out_hw = overlay_image.size[::-1]

            if manager.cfg.save_score_summary:
                manager._save_score_summary(
                    sample_dir=sample_dir,
                    semantic_scores=semantic_score_map[b],
                    raw_final_scores=raw_final_score_map[b],
                    final_scores=final_score_map[b],
                    class_names=class_names,
                )

            if manager.cfg.save_score_heatmaps:
                manager._save_all_score_heatmaps(
                    sample_dir=sample_dir,
                    semantic_scores=semantic_score_map[b],
                    raw_final_scores=raw_final_score_map[b],
                    final_scores=final_score_map[b],
                    out_hw=out_hw,
                    class_names=class_names,
                )


class Sam3DirectSegmentationTask(VisualizationTask):
    name = "sam3_direct_segmentation"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        if not manager.cfg.save_sam3_direct_segmentation:
            return

        direct_logits = manager._build_sam3_direct_segmentation_for_visualization(
            model=ctx.model,
            batch=ctx.batch,
        )
        if direct_logits is None:
            return

        if direct_logits.dim() != 4:
            raise ValueError(
                "Expected sam3 direct logits as [B, 1, H, W] or [B, C, H, W], "
                f"got {tuple(direct_logits.shape)}."
            )

        for b in ctx.selected_indices:
            image_id = manager._extract_image_id(ctx.batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(ctx.batch, b)
            out_hw = overlay_image.size[::-1]
            logits_b = direct_logits[b]

            if logits_b.shape[0] == 1:
                score_map = logits_b[0].sigmoid()
                pred_mask = (
                    score_map >= float(manager.cfg.sam3_direct_seg_threshold)
                ).long()
            else:
                score_map = logits_b.softmax(dim=0).max(dim=0).values
                pred_mask = logits_b.argmax(dim=0).long()

            manager._to_heatmap_image(
                score_map,
                out_hw=out_hw,
                normalize="prob",
            ).save(sample_dir / "sam3_direct_score_heatmap.png")

            pred_mask_out = manager._prepare_label_map(pred_mask, out_hw)

            manager._colorize_label_map(
                pred_mask_out,
                num_classes=max(2, int(logits_b.shape[0])),
            ).save(sample_dir / "sam3_direct_pred.png")

            manager._overlay_label_map(
                overlay_image,
                pred_mask_out,
                num_classes=max(2, int(logits_b.shape[0])),
            ).save(sample_dir / "sam3_direct_overlay.png")

class VisualizationManager:
    def __init__(
        self,
        cfg: VisualizerConfig,
        eval_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.eval_cfg = dict(eval_cfg or {})

        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._saved_counts: Dict[Tuple[str, int], int] = {}
        self.tasks = self._build_tasks()

    @property
    def eval_prob_thd(self) -> Optional[float]:
        value = self.eval_cfg.get("prob_thd", None)
        if value is None:
            return None
        return float(value)

    @property
    def eval_bg_idx(self) -> int:
        return int(self.eval_cfg.get("bg_idx", 0))

    @property
    def eval_use_score_map(self) -> bool:
        return bool(self.eval_cfg.get("use_score_map", True))

    def _build_tasks(self) -> List[VisualizationTask]:
        return [
            BaseSemanticOverlayTask(),
            ScoreAnalysisTask(),
            Sam3DirectSegmentationTask(),
        ]

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return getattr(model, "module", model)

    @classmethod
    def _extract_core_model(cls, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        model = cls._unwrap_model(model)
        return getattr(model, "core", None)

    def _build_eval_style_final_pred(
        self,
        outputs: Dict[str, torch.Tensor],
        final_score_map: torch.Tensor,
    ) -> torch.Tensor:
        if final_score_map.dim() != 4:
            raise ValueError(
                f"Expected final_score_map as [B, C, H, W], "
                f"got {tuple(final_score_map.shape)}."
            )

        if self.eval_use_score_map:
            num_classes = int(final_score_map.shape[1])
            bg_idx = self.eval_bg_idx

            if not (0 <= bg_idx < num_classes):
                raise ValueError(
                    f"bg_idx={bg_idx} is out of range for num_classes={num_classes}."
                )

            max_score, pred = final_score_map.max(dim=1)
            prob_thd = self.eval_prob_thd

            if prob_thd is not None:
                pred = pred.clone()
                pred[max_score < prob_thd] = bg_idx

            return pred.long()

        final_pred = outputs.get(OUTPUT_KEYS.final_pred, None)
        if final_pred is not None:
            if final_pred.dim() != 3:
                raise ValueError(
                    f"Expected final_pred as [B, H, W], got {tuple(final_pred.shape)}."
                )
            return final_pred.long()

        return final_score_map.argmax(dim=1).long()

    def _build_sam3_direct_segmentation_for_visualization(
        self,
        model: torch.nn.Module,
        batch: Any,
    ) -> Optional[torch.Tensor]:
        core = self._extract_core_model(model)
        if core is None:
            return None

        backbone = getattr(core, "backbone", None)
        segmentation_head = getattr(core, "segmentation_head", None)
        if backbone is None or segmentation_head is None:
            return None

        if not hasattr(backbone, "forward_image"):
            return None

        pixel_decoder = getattr(segmentation_head, "pixel_decoder", None)
        semantic_seg_head = getattr(segmentation_head, "semantic_seg_head", None)

        if pixel_decoder is None or semantic_seg_head is None:
            return None

        img_batch = getattr(batch, "img_batch", None)
        if not isinstance(img_batch, torch.Tensor):
            return None

        seg_device = next(segmentation_head.parameters()).device

        with torch.no_grad():
            backbone_out = backbone.forward_image(img_batch)

            if not isinstance(backbone_out, dict):
                raise ValueError(
                    "backbone.forward_image must return a dict for visualization."
                )
            if "backbone_fpn" not in backbone_out:
                raise ValueError(
                    "backbone.forward_image output must contain 'backbone_fpn'."
                )

            backbone_feats = backbone_out["backbone_fpn"]
            if not isinstance(backbone_feats, (list, tuple)) or len(backbone_feats) == 0:
                raise ValueError(
                    "backbone_out['backbone_fpn'] must be a non-empty list/tuple."
                )

            backbone_feats = [feat.to(device=seg_device) for feat in backbone_feats]
            pixel_embed = pixel_decoder(backbone_feats)
            direct_logits = semantic_seg_head(pixel_embed)

        return direct_logits.detach()

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
            if x.shape[0] != 3:
                raise ValueError(
                    f"Expected tensor image with 1 or 3 channels, got {tuple(x.shape)}."
                )
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
        if (
            raw_images is not None
            and batch_index < len(raw_images)
            and raw_images[batch_index] is not None
        ):
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
        if (
            raw_images is not None
            and batch_index < len(raw_images)
            and raw_images[batch_index] is not None
        ):
            return VisualizationManager._to_uint8_image(raw_images[batch_index])

        return VisualizationManager._to_uint8_image(batch.img_batch[batch_index])

    @staticmethod
    def _extract_image_id(batch: Any, batch_index: int) -> int:
        try:
            meta = batch.find_metadatas[0]
            return int(meta.original_image_id[batch_index].item())
        except Exception:
            return int(batch_index)

    def _resolve_sample_dir(
        self,
        image_id: int,
        epoch: Optional[int],
        stage: str,
    ) -> Path:
        parts = [self.save_dir, stage]
        if epoch is not None:
            parts.append(Path(f"epoch_{epoch:03d}"))
        parts.append(Path(self.cfg.image_folder_pattern.format(image_id=image_id)))
        sample_dir = Path(*parts)
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir

    @staticmethod
    def _prepare_label_map(
        label_map: torch.Tensor,
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        x = label_map.detach().cpu()
        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError(f"Expected [1, H, W] or [H, W], got {tuple(x.shape)}.")
            x = x[0]
        if x.dim() != 2:
            raise ValueError(f"Expected [H, W], got {tuple(x.shape)}.")

        if tuple(x.shape[-2:]) != tuple(out_hw):
            x = F.interpolate(
                x[None, None].float(),
                size=out_hw,
                mode="nearest",
            )[0, 0].long()
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
                raise ValueError(
                    f"Expected [B, 1, H, W] or [B, H, W], got {tuple(x.shape)}."
                )
            x = x[:, 0]
        elif x.dim() != 3:
            raise ValueError(f"Expected [B, H, W], got {tuple(x.shape)}.")

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
    def _get_output(
        outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> Optional[torch.Tensor]:
        value = outputs.get(key, None)
        if value is None:
            value = outputs.get(str(key), None)
        return value

    def _extract_final_logits_raw_and_gated_score_maps(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        raw_final_score_map = self._get_output(outputs, OUTPUT_KEYS.raw_final_score_map)
        final_score_map = self._get_output(outputs, OUTPUT_KEYS.final_score_map)
        final_logits = self._get_output(outputs, OUTPUT_KEYS.final_logits)

        if raw_final_score_map is None:
            if final_logits is None:
                raise ValueError(
                    f"outputs must contain '{OUTPUT_KEYS.raw_final_score_map}' "
                    f"or '{OUTPUT_KEYS.final_logits}'."
                )
            if final_logits.dim() != 4:
                raise ValueError(
                    f"Expected final_logits as [B, C, H, W], "
                    f"got {tuple(final_logits.shape)}."
                )
            raw_final_score_map = final_logits.sigmoid()
            raw_source = f"sigmoid({OUTPUT_KEYS.final_logits})"
        else:
            if raw_final_score_map.dim() != 4:
                raise ValueError(
                    f"Expected raw_final_score_map as [B, C, H, W], "
                    f"got {tuple(raw_final_score_map.shape)}."
                )
            raw_source = OUTPUT_KEYS.raw_final_score_map

            if final_logits is None:
                eps = 1e-6
                final_logits = torch.logit(raw_final_score_map.clamp(eps, 1.0 - eps))

        if final_logits is None:
            raise ValueError(
                f"Cannot infer {OUTPUT_KEYS.final_logits} for visualization."
            )

        if final_score_map is None:
            final_score_map = raw_final_score_map
        else:
            if final_score_map.dim() != 4:
                raise ValueError(
                    f"Expected final_score_map as [B, C, H, W], "
                    f"got {tuple(final_score_map.shape)}."
                )

        if tuple(final_score_map.shape) != tuple(raw_final_score_map.shape):
            raise ValueError(
                "raw_final_score_map and final_score_map shape mismatch: "
                f"{tuple(raw_final_score_map.shape)} vs {tuple(final_score_map.shape)}."
            )

        if tuple(final_logits.shape) != tuple(raw_final_score_map.shape):
            raise ValueError(
                "final_logits and raw_final_score_map shape mismatch: "
                f"{tuple(final_logits.shape)} vs {tuple(raw_final_score_map.shape)}."
            )

        gated_source = raw_source
        return (
            final_logits,
            raw_final_score_map,
            final_score_map,
            raw_source,
            gated_source,
        )

    def _extract_final_logits_and_score_map(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        final_logits, _, final_score_map, _, gated_source = (
            self._extract_final_logits_raw_and_gated_score_maps(outputs)
        )
        return final_logits, final_score_map, gated_source

    def _extract_semantic_logits_and_score_map(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
        semantic_score_map = self._get_output(outputs, OUTPUT_KEYS.semantic_score_map)
        semantic_logits = self._get_output(outputs, OUTPUT_KEYS.semantic_logits)

        if semantic_score_map is not None:
            if semantic_score_map.dim() != 4:
                raise ValueError(
                    f"Expected semantic_score_map as [B, C, H, W], "
                    f"got {tuple(semantic_score_map.shape)}."
                )

            if semantic_logits is not None:
                if semantic_logits.dim() != 4:
                    raise ValueError(
                        f"Expected semantic_logits as [B, C, H, W], "
                        f"got {tuple(semantic_logits.shape)}."
                    )
                if tuple(semantic_logits.shape) != tuple(semantic_score_map.shape):
                    raise ValueError(
                        "semantic_logits and semantic_score_map shape mismatch: "
                        f"{tuple(semantic_logits.shape)} vs {tuple(semantic_score_map.shape)}."
                    )
            else:
                semantic_logits = semantic_score_map.clamp_min(1e-12).log()

            return semantic_logits, semantic_score_map, OUTPUT_KEYS.semantic_score_map

        if semantic_logits is not None:
            if semantic_logits.dim() != 4:
                raise ValueError(
                    f"Expected semantic_logits as [B, C, H, W], "
                    f"got {tuple(semantic_logits.shape)}."
                )
            return (
                semantic_logits,
                semantic_logits.softmax(dim=1),
                f"softmax({OUTPUT_KEYS.semantic_logits})",
            )

        return None, None, None

    @staticmethod
    def _extract_pred_from_logits(
        logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if logits is None:
            return None
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [B, C, H, W], got {tuple(logits.shape)}.")
        return logits.argmax(dim=1).long()

    @staticmethod
    def _apply_turbo_colormap(x: np.ndarray) -> np.ndarray:
        x = np.clip(x.astype(np.float32), 0.0, 1.0)

        coeffs = np.array(
            [
                [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
                [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
                [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
            ],
            dtype=np.float32,
        )

        powers = np.stack(
            [
                np.ones_like(x),
                x,
                x ** 2,
                x ** 3,
                x ** 4,
                x ** 5,
            ],
            axis=-1,
        )

        rgb = powers @ coeffs.T
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    @staticmethod
    def _apply_gray_colormap(x: np.ndarray) -> np.ndarray:
        x = np.clip(x.astype(np.float32), 0.0, 1.0)
        arr = (x * 255.0).astype(np.uint8)
        return np.stack([arr, arr, arr], axis=-1)

    def _apply_colormap(self, x: np.ndarray) -> np.ndarray:
        name = str(getattr(self.cfg, "heatmap_colormap", "turbo")).lower()

        if name == "turbo":
            return self._apply_turbo_colormap(x)

        if name in {"gray", "grey", "grayscale"}:
            return self._apply_gray_colormap(x)

        raise ValueError(
            f"Unsupported heatmap_colormap={name!r}. "
            "Supported values are: 'turbo', 'gray'."
        )

    @staticmethod
    def _normalize_heatmap_values(
        x: torch.Tensor,
        normalize: str,
    ) -> torch.Tensor:
        normalize = str(normalize)

        if normalize == "prob":
            return x.clamp(0.0, 1.0)

        if normalize == "sigmoid":
            return x.sigmoid()

        if normalize == "minmax":
            x_min = x.min()
            x_max = x.max()
            return (x - x_min) / (x_max - x_min).clamp_min(1e-6)

        if normalize == "auto":
            x_min = x.min()
            x_max = x.max()

            if float(x_min.item()) >= 0.0 and float(x_max.item()) <= 1.0:
                return x.clamp(0.0, 1.0)

            return (x - x_min) / (x_max - x_min).clamp_min(1e-6)

        raise ValueError(
            f"Unknown heatmap normalize mode={normalize!r}. "
            "Supported modes are: 'auto', 'prob', 'sigmoid', 'minmax'."
        )

    def _to_heatmap_image(
        self,
        score_map: torch.Tensor,
        out_hw: Tuple[int, int],
        normalize: str = "auto",
    ) -> Image.Image:
        x = score_map.detach().cpu().float()

        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError(f"Expected [1, H, W] or [H, W], got {tuple(x.shape)}.")
            x = x[0]

        if x.dim() != 2:
            raise ValueError(f"Expected [H, W], got {tuple(x.shape)}.")

        if tuple(x.shape[-2:]) != tuple(out_hw):
            x = F.interpolate(
                x[None, None],
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        x = self._normalize_heatmap_values(x, normalize=normalize)

        arr = x.numpy()
        heat = self._apply_colormap(arr)
        return Image.fromarray(heat, mode="RGB")

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        safe = []
        for ch in str(text):
            if ch.isalnum() or ch in ("-", "_"):
                safe.append(ch)
            elif ch in (" ", "/", "\\", "."):
                safe.append("_")
        value = "".join(safe).strip("_")
        return value or "class"

    @staticmethod
    def _per_class_max_mean(
        score_map: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if score_map is None:
            return None, None

        if score_map.dim() != 3:
            raise ValueError(
                f"Expected score_map as [C, H, W], got {tuple(score_map.shape)}."
            )

        flat = score_map.flatten(1)
        return flat.max(dim=1).values, flat.mean(dim=1)

    def _save_score_summary(
        self,
        sample_dir: Path,
        semantic_scores: torch.Tensor,
        raw_final_scores: torch.Tensor,
        final_scores: torch.Tensor,
        class_names: Optional[List[str]],
    ) -> None:
        if semantic_scores.dim() != 3:
            raise ValueError(
                f"Expected semantic_scores as [C, H, W], got {tuple(semantic_scores.shape)}."
            )
        if raw_final_scores.dim() != 3:
            raise ValueError(
                f"Expected raw_final_scores as [C, H, W], got {tuple(raw_final_scores.shape)}."
            )
        if final_scores.dim() != 3:
            raise ValueError(
                f"Expected final_scores as [C, H, W], got {tuple(final_scores.shape)}."
            )

        num_classes = int(semantic_scores.shape[0])
        if raw_final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch: semantic={num_classes}, "
                f"raw_final={raw_final_scores.shape[0]}."
            )
        if final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch: semantic={num_classes}, final={final_scores.shape[0]}."
            )

        semantic_max, semantic_mean = self._per_class_max_mean(semantic_scores)
        raw_final_max, raw_final_mean = self._per_class_max_mean(raw_final_scores)
        final_max, final_mean = self._per_class_max_mean(final_scores)

        order = torch.argsort(final_max, descending=True)

        with open(sample_dir / "branch_score_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                "rank\tclass_id\tclass_name\t"
                "semantic_max\tsemantic_mean\t"
                "raw_final_max\traw_final_mean\t"
                "gated_final_max\tgated_final_mean\n"
            )

            for rank, cls_idx in enumerate(order.tolist()):
                class_name = (
                    class_names[cls_idx]
                    if class_names is not None and cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )

                f.write(
                    f"{rank}\t{cls_idx}\t{class_name}\t"
                    f"{float(semantic_max[cls_idx].item()):.6f}\t"
                    f"{float(semantic_mean[cls_idx].item()):.6f}\t"
                    f"{float(raw_final_max[cls_idx].item()):.6f}\t"
                    f"{float(raw_final_mean[cls_idx].item()):.6f}\t"
                    f"{float(final_max[cls_idx].item()):.6f}\t"
                    f"{float(final_mean[cls_idx].item()):.6f}\n"
                )

    def _save_all_score_heatmaps(
        self,
        sample_dir: Path,
        semantic_scores: torch.Tensor,
        raw_final_scores: torch.Tensor,
        final_scores: torch.Tensor,
        out_hw: Tuple[int, int],
        class_names: Optional[List[str]],
    ) -> None:
        if semantic_scores.dim() != 3:
            raise ValueError(
                f"Expected semantic_scores as [C, H, W], got {tuple(semantic_scores.shape)}."
            )
        if raw_final_scores.dim() != 3:
            raise ValueError(
                f"Expected raw_final_scores as [C, H, W], got {tuple(raw_final_scores.shape)}."
            )
        if final_scores.dim() != 3:
            raise ValueError(
                f"Expected final_scores as [C, H, W], got {tuple(final_scores.shape)}."
            )

        num_classes = int(semantic_scores.shape[0])
        if raw_final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch in score heatmaps: "
                f"semantic={num_classes}, raw_final={raw_final_scores.shape[0]}."
            )
        if final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch in score heatmaps: "
                f"semantic={num_classes}, final={final_scores.shape[0]}."
            )

        heatmap_root = sample_dir / "score_heatmaps"
        semantic_dir = heatmap_root / "semantic"
        raw_final_dir = heatmap_root / "final_raw"
        gated_final_dir = heatmap_root / "final_presence_gated"

        semantic_dir.mkdir(parents=True, exist_ok=True)
        raw_final_dir.mkdir(parents=True, exist_ok=True)
        gated_final_dir.mkdir(parents=True, exist_ok=True)

        for cls_idx in range(num_classes):
            class_name = (
                class_names[cls_idx]
                if class_names is not None and cls_idx < len(class_names)
                else f"class_{cls_idx}"
            )
            filename = f"{cls_idx:03d}_{self._sanitize_filename(class_name)}.png"

            self._to_heatmap_image(
                semantic_scores[cls_idx],
                out_hw=out_hw,
                normalize="prob",
            ).save(semantic_dir / filename)

            self._to_heatmap_image(
                raw_final_scores[cls_idx],
                out_hw=out_hw,
                normalize="prob",
            ).save(raw_final_dir / filename)

            self._to_heatmap_image(
                final_scores[cls_idx],
                out_hw=out_hw,
                normalize="prob",
            ).save(gated_final_dir / filename)

    def _infer_batch_size(
        self,
        semantic_outputs: Dict[str, torch.Tensor],
        batch: Any,
    ) -> int:
        for key in (
            OUTPUT_KEYS.final_score_map,
            OUTPUT_KEYS.final_logits,
            OUTPUT_KEYS.semantic_score_map,
            OUTPUT_KEYS.semantic_logits,
        ):
            value = semantic_outputs.get(key, None)
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                return int(value.shape[0])

        img_batch = getattr(batch, "img_batch", None)
        if isinstance(img_batch, torch.Tensor) and img_batch.dim() >= 1:
            return int(img_batch.shape[0])

        raise ValueError("Cannot infer batch size for visualization.")

    def _sample_key(
        self,
        stage: str,
        epoch: Optional[int],
    ) -> Tuple[str, int]:
        epoch_key = -1 if epoch is None else int(epoch)
        return str(stage), epoch_key

    def _sample_score(
        self,
        stage: str,
        epoch: Optional[int],
        image_id: int,
        batch_index: int,
    ) -> float:
        text = (
            f"{int(self.cfg.vis_seed)}|{stage}|"
            f"{-1 if epoch is None else int(epoch)}|"
            f"{int(image_id)}|{int(batch_index)}"
        )
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16)
        return value / float(0xFFFFFFFF)

    def _select_indices(
        self,
        batch: Any,
        semantic_outputs: Dict[str, torch.Tensor],
        stage: str,
        epoch: Optional[int],
    ) -> List[int]:
        batch_size = self._infer_batch_size(
            semantic_outputs=semantic_outputs,
            batch=batch,
        )

        key = self._sample_key(stage=stage, epoch=epoch)
        current_count = int(self._saved_counts.get(key, 0))

        max_samples = self.cfg.max_samples_per_epoch
        if max_samples is not None and current_count >= int(max_samples):
            return []

        vis_prob = float(self.cfg.vis_prob)
        if vis_prob <= 0.0:
            return []

        selected: List[int] = []

        for b in range(batch_size):
            if max_samples is not None and current_count + len(selected) >= int(max_samples):
                break

            image_id = self._extract_image_id(batch, b)
            score = self._sample_score(
                stage=stage,
                epoch=epoch,
                image_id=image_id,
                batch_index=b,
            )

            if score <= vis_prob:
                selected.append(b)

        if len(selected) > 0:
            self._saved_counts[key] = current_count + len(selected)

        return selected

    def run(
        self,
        model: torch.nn.Module,
        batch: Any,
        semantic_outputs: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
        epoch: Optional[int],
        stage: str,
    ) -> None:
        if not self.should_save(stage):
            return

        selected_indices = self._select_indices(
            batch=batch,
            semantic_outputs=semantic_outputs,
            stage=stage,
            epoch=epoch,
        )
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