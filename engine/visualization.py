from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from ..models.task_modes import OUTPUT_KEYS
from ..config_dataclasses import VisualizerConfig

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

        final_score_map = outputs.get(OUTPUT_KEYS.final_score_map, None)
        final_logits = outputs.get(OUTPUT_KEYS.final_logits, None)
        if final_score_map is None and final_logits is not None:
            final_score_map = final_logits.sigmoid()

        if final_score_map is None:
            raise ValueError(
                f"outputs must contain '{OUTPUT_KEYS.final_score_map}' "
                f"or '{OUTPUT_KEYS.final_logits}'."
            )
        if final_score_map.dim() != 4:
            raise ValueError(
                f"Expected final_score_map as [B, C, H, W], "
                f"got {tuple(final_score_map.shape)}."
            )

        final_pred = manager._extract_pred_from_logits(final_score_map)

        semantic_score_map = outputs.get(OUTPUT_KEYS.semantic_score_map, None)
        semantic_logits = outputs.get(OUTPUT_KEYS.semantic_logits, None)
        if semantic_score_map is None and semantic_logits is not None:
            semantic_score_map = semantic_logits.sigmoid()

        semantic_pred = (
            manager._extract_pred_from_logits(semantic_score_map)
            if semantic_score_map is not None
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

            if manager.cfg.save_prediction:
                manager._overlay_label_map(
                    overlay_image,
                    final_pred_label,
                    final_num_classes,
                ).save(sample_dir / "pred_overlay.png")

            if manager.cfg.save_ground_truth:
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
                manager._overlay_label_map(
                    overlay_image,
                    semantic_pred_label,
                    semantic_num_classes,
                ).save(sample_dir / "pred_semantic_overlay.png")

            if class_names is not None:
                with open(sample_dir / "classes.txt", "w", encoding="utf-8") as f:
                    for i, name in enumerate(class_names):
                        f.write(f"{i}\t{name}\n")


class ScoreAnalysisTask(VisualizationTask):
    name = "score_analysis"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        outputs = ctx.semantic_outputs
        batch = ctx.batch

        semantic_score_map = outputs.get(OUTPUT_KEYS.semantic_score_map, None)
        semantic_logits = outputs.get(OUTPUT_KEYS.semantic_logits, None)
        if semantic_score_map is None and semantic_logits is not None:
            semantic_score_map = semantic_logits.sigmoid()

        final_score_map = outputs.get(OUTPUT_KEYS.final_score_map, None)
        final_logits = outputs.get(OUTPUT_KEYS.final_logits, None)
        if final_score_map is None and final_logits is not None:
            final_score_map = final_logits.sigmoid()

        if semantic_score_map is None:
            raise ValueError(
                f"outputs must contain '{OUTPUT_KEYS.semantic_score_map}' "
                f"or '{OUTPUT_KEYS.semantic_logits}'."
            )
        if final_score_map is None:
            raise ValueError(
                f"outputs must contain '{OUTPUT_KEYS.final_score_map}' "
                f"or '{OUTPUT_KEYS.final_logits}'."
            )

        if semantic_score_map.dim() != 4:
            raise ValueError(
                f"Expected semantic_score_map as [B, C, H, W], "
                f"got {tuple(semantic_score_map.shape)}."
            )
        if final_score_map.dim() != 4:
            raise ValueError(
                f"Expected final_score_map as [B, C, H, W], "
                f"got {tuple(final_score_map.shape)}."
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

            semantic_scores_b = semantic_score_map[b]
            final_scores_b = final_score_map[b]

            if manager.cfg.save_score_summary:
                manager._save_score_summary(
                    sample_dir=sample_dir,
                    semantic_scores=semantic_scores_b,
                    final_scores=final_scores_b,
                    class_names=class_names,
                )

            if manager.cfg.save_score_heatmaps:
                manager._save_all_score_heatmaps(
                    sample_dir=sample_dir,
                    semantic_scores=semantic_scores_b,
                    final_scores=final_scores_b,
                    out_hw=out_hw,
                    class_names=class_names,
                )


class PresenceScoreTask(VisualizationTask):
    name = "presence_score"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        if not manager.cfg.save_presence_scores:
            return

        outputs = ctx.semantic_outputs
        batch = ctx.batch

        presence_score = outputs.get(OUTPUT_KEYS.presence_score, None)
        presence_logits = outputs.get(OUTPUT_KEYS.presence_logits, None)

        if presence_score is None and presence_logits is not None:
            presence_score = presence_logits.sigmoid()

        if presence_score is None:
            return

        if presence_score.dim() != 2:
            raise ValueError(
                f"Expected presence_score as [B, C], got {tuple(presence_score.shape)}."
            )

        if presence_logits is not None:
            if presence_logits.dim() != 2:
                raise ValueError(
                    f"Expected presence_logits as [B, C], "
                    f"got {tuple(presence_logits.shape)}."
                )
            if presence_logits.shape != presence_score.shape:
                raise ValueError(
                    "presence_logits and presence_score shape mismatch: "
                    f"{tuple(presence_logits.shape)} vs {tuple(presence_score.shape)}."
                )

        presence_logits_layers = outputs.get(
            getattr(OUTPUT_KEYS, "presence_logits_layers", "presence_logits_layers"),
            None,
        )
        if presence_logits_layers is not None:
            if presence_logits_layers.dim() != 3:
                raise ValueError(
                    "presence_logits_layers must be [L, B, C], "
                    f"got {tuple(presence_logits_layers.shape)}."
                )
            if tuple(presence_logits_layers.shape[1:]) != tuple(presence_score.shape):
                raise ValueError(
                    "presence_logits_layers shape mismatch: expected [L, B, C] "
                    f"with B,C={tuple(presence_score.shape)}, got "
                    f"{tuple(presence_logits_layers.shape)}."
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

            presence_logits_b = (
                presence_logits[b]
                if presence_logits is not None
                else None
            )

            manager._save_presence_scores(
                sample_dir=sample_dir,
                presence_score=presence_score[b],
                presence_logits=presence_logits_b,
                class_names=class_names,
            )

            if (
                manager.cfg.save_presence_layers
                and presence_logits_layers is not None
            ):
                manager._save_presence_layer_scores(
                    sample_dir=sample_dir,
                    presence_logits_layers=presence_logits_layers[:, b],
                    class_names=class_names,
                )



class FinalMixerMaskLayerTask(VisualizationTask):
    name = "final_mixer_mask_layers"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        if not manager.cfg.save_final_mixer_mask_layers:
            return

        outputs = ctx.semantic_outputs
        batch = ctx.batch

        mask_layers_key = getattr(
            OUTPUT_KEYS,
            "mask_logits_layers",
            "mask_logits_layers",
        )
        mask_logits_layers = outputs.get(mask_layers_key, None)

        if mask_logits_layers is None:
            return

        if mask_logits_layers.dim() != 5:
            raise ValueError(
                "mask_logits_layers must be [L, B, C, H, W], "
                f"got {tuple(mask_logits_layers.shape)}."
            )

        num_layers = int(mask_logits_layers.shape[0])
        batch_size = int(mask_logits_layers.shape[1])
        num_classes = int(mask_logits_layers.shape[2])

        if num_layers <= 0:
            return

        try:
            class_names: Optional[List[str]] = [
                str(x) for x in batch.find_metadatas[0].class_names
            ]
        except Exception:
            class_names = None

        if class_names is not None and len(class_names) != num_classes:
            class_names = None

        for b in ctx.selected_indices:
            if b < 0 or b >= batch_size:
                raise ValueError(
                    f"Selected batch index {b} is out of range for "
                    f"mask_logits_layers batch size {batch_size}."
                )

            image_id = manager._extract_image_id(batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(batch, b)
            out_hw = overlay_image.size[::-1]

            manager._save_final_mixer_mask_layer_outputs(
                sample_dir=sample_dir,
                overlay_image=overlay_image,
                mask_logits_layers=mask_logits_layers[:, b],
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


class ClipImageTextScoreTask(VisualizationTask):
    name = "clip_image_text_score"

    def run(self, manager: "VisualizationManager", ctx: VisualizationContext) -> None:
        if not manager.cfg.save_clip_argmax_prediction:
            return

        clip_score_map = manager._build_clip_image_text_score_map_for_visualization(
            model=ctx.model,
            batch=ctx.batch,
        )
        if clip_score_map is None:
            return

        if clip_score_map.dim() != 4:
            raise ValueError(
                f"Expected clip_score_map as [B, C, Hc, Wc], "
                f"got {tuple(clip_score_map.shape)}."
            )

        num_classes = int(clip_score_map.shape[1])

        for b in ctx.selected_indices:
            image_id = manager._extract_image_id(ctx.batch, b)
            sample_dir = manager._resolve_sample_dir(
                image_id=image_id,
                epoch=ctx.epoch,
                stage=ctx.stage,
            )

            overlay_image = manager._extract_overlay_image(ctx.batch, b)
            out_hw = overlay_image.size[::-1]

            clip_score_up = F.interpolate(
                clip_score_map[b:b + 1],
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )[0]

            clip_pred = clip_score_up.argmax(dim=0).long()

            manager._colorize_label_map(
                clip_pred.detach().cpu(),
                num_classes=num_classes,
            ).save(sample_dir / "clip_argmax_pred.png")

            manager._overlay_label_map(
                overlay_image,
                clip_pred.detach().cpu(),
                num_classes=num_classes,
            ).save(sample_dir / "clip_argmax_overlay.png")


class VisualizationManager:
    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._saved_counts: Dict[Tuple[str, int], int] = {}
        self.tasks = self._build_tasks()

    def _build_tasks(self) -> List[VisualizationTask]:
        return [
            BaseSemanticOverlayTask(),
            ScoreAnalysisTask(),
            PresenceScoreTask(),
            FinalMixerMaskLayerTask(),
            Sam3DirectSegmentationTask(),
            ClipImageTextScoreTask(),
        ]

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return getattr(model, "module", model)

    @classmethod
    def _extract_core_model(cls, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        model = cls._unwrap_model(model)
        return getattr(model, "core", None)

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

            backbone_feats = [
                feat.to(device=seg_device)
                for feat in backbone_feats
            ]

            pixel_embed = pixel_decoder(backbone_feats)
            direct_logits = semantic_seg_head(pixel_embed)

        return direct_logits.detach()

    def _build_clip_image_text_score_map_for_visualization(
        self,
        model: torch.nn.Module,
        batch: Any,
    ) -> Optional[torch.Tensor]:
        core = self._extract_core_model(model)
        if core is None:
            return None

        required_attrs = [
            "clip_image_encoder",
            "clip_text_encoder",
            "_prepare_openclip_image_batch",
        ]
        for name in required_attrs:
            if not hasattr(core, name):
                return None

        if core.clip_image_encoder is None or core.clip_text_encoder is None:
            return None

        class_texts = list(getattr(batch, "find_text_batch", []))
        if len(class_texts) == 0:
            return None

        raw_images = getattr(batch, "raw_images", None)
        if raw_images is None:
            return None

        templates = list(getattr(core, "clip_prompt_templates", []))
        if len(templates) == 0:
            return None

        normalize_label = bool(getattr(core, "normalize_label_for_clip", True))
        device = core.device

        with torch.no_grad():
            clip_img_batch = core._prepare_openclip_image_batch(
                raw_images=raw_images,
                device=device,
            )

            clip_image_feat_map = core.clip_image_encoder(clip_img_batch)
            if not isinstance(clip_image_feat_map, torch.Tensor):
                return None
            if clip_image_feat_map.dim() != 4:
                raise ValueError(
                    "clip_image_encoder must return [B, D_clip, Hc, Wc], "
                    f"got {tuple(clip_image_feat_map.shape)}."
                )

            clip_text_tokens = core.clip_text_encoder.encode_prompt_templates(
                class_names=class_texts,
                templates=templates,
                device=device,
                normalize_label=normalize_label,
                normalize=False,
            )

            if clip_text_tokens.dim() != 3:
                raise ValueError(
                    "Expected clip_text_tokens as [C, K, D_clip], "
                    f"got {tuple(clip_text_tokens.shape)}."
                )

            batch_size, image_dim, grid_h, grid_w = clip_image_feat_map.shape
            num_classes, _, text_dim = clip_text_tokens.shape

            if int(image_dim) != int(text_dim):
                raise ValueError(
                    f"CLIP image/text dim mismatch: image_dim={image_dim}, "
                    f"text_dim={text_dim}."
                )

            if int(num_classes) != len(class_texts):
                raise ValueError(
                    f"CLIP text class count mismatch: {num_classes} vs "
                    f"{len(class_texts)}."
                )

            image_features = clip_image_feat_map.flatten(2).transpose(1, 2)
            image_features = image_features.contiguous()
            image_features = F.normalize(image_features, dim=-1, eps=1e-6)

            text_features = F.normalize(clip_text_tokens, dim=-1, eps=1e-6)
            text_features = text_features.mean(dim=1)
            text_features = F.normalize(text_features, dim=-1, eps=1e-6)

            clip_score = torch.einsum(
                "bnd,cd->bcn",
                image_features,
                text_features,
            )

            clip_score_map = clip_score.reshape(
                batch_size,
                num_classes,
                grid_h,
                grid_w,
            ).contiguous()

        return clip_score_map.detach()

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
                raise ValueError(f"Expected [B, 1, H, W] or [B, H, W], got {tuple(x.shape)}.")
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
    def _extract_pred_from_logits(
        logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if logits is None:
            return None
        if logits.dim() != 4:
            raise ValueError(f"Expected logits [B, C, H, W], got {tuple(logits.shape)}.")
        return logits.argmax(dim=1)

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
        final_scores: torch.Tensor,
        class_names: Optional[List[str]],
    ) -> None:
        if semantic_scores.dim() != 3:
            raise ValueError(
                f"Expected semantic_scores as [C, H, W], got {tuple(semantic_scores.shape)}."
            )
        if final_scores.dim() != 3:
            raise ValueError(
                f"Expected final_scores as [C, H, W], got {tuple(final_scores.shape)}."
            )

        num_classes = int(semantic_scores.shape[0])
        if final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch: semantic={num_classes}, final={final_scores.shape[0]}."
            )

        semantic_max, semantic_mean = self._per_class_max_mean(semantic_scores)
        final_max, final_mean = self._per_class_max_mean(final_scores)

        order = torch.argsort(final_max, descending=True)

        with open(sample_dir / "branch_score_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                "rank\tclass_id\tclass_name\t"
                "semantic_max\tsemantic_mean\t"
                "final_max\tfinal_mean\n"
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
                    f"{float(final_max[cls_idx].item()):.6f}\t"
                    f"{float(final_mean[cls_idx].item()):.6f}\n"
                )

    def _save_all_score_heatmaps(
        self,
        sample_dir: Path,
        semantic_scores: torch.Tensor,
        final_scores: torch.Tensor,
        out_hw: Tuple[int, int],
        class_names: Optional[List[str]],
    ) -> None:
        if semantic_scores.dim() != 3:
            raise ValueError(
                f"Expected semantic_scores as [C, H, W], got {tuple(semantic_scores.shape)}."
            )
        if final_scores.dim() != 3:
            raise ValueError(
                f"Expected final_scores as [C, H, W], got {tuple(final_scores.shape)}."
            )

        num_classes = int(semantic_scores.shape[0])
        if final_scores.shape[0] != num_classes:
            raise ValueError(
                f"Class count mismatch in score heatmaps: "
                f"semantic={num_classes}, final={final_scores.shape[0]}."
            )

        heatmap_root = sample_dir / "score_heatmaps"
        semantic_dir = heatmap_root / "semantic"
        final_dir = heatmap_root / "final"

        semantic_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

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
                final_scores[cls_idx],
                out_hw=out_hw,
                normalize="prob",
            ).save(final_dir / filename)

    @staticmethod
    def _format_optional_float(value: Optional[float]) -> str:
        if value is None:
            return "nan"
        return f"{float(value):.6f}"

    def _select_heatmap_class_indices(
        self,
        score_map: torch.Tensor,
    ) -> List[int]:
        if score_map.dim() != 3:
            raise ValueError(
                f"Expected score_map as [C, H, W], got {tuple(score_map.shape)}."
            )

        num_classes = int(score_map.shape[0])
        max_classes = self.cfg.max_final_mixer_layer_heatmap_classes

        if max_classes is None:
            return list(range(num_classes))

        max_classes = int(max_classes)
        if max_classes <= 0:
            return []

        if max_classes >= num_classes:
            return list(range(num_classes))

        score_cpu = score_map.detach().float().sigmoid()
        rank_score = score_cpu.flatten(1).max(dim=1).values
        indices = torch.argsort(rank_score, descending=True)[:max_classes]
        return [int(x) for x in indices.tolist()]

    def _save_final_mixer_layer_summary(
        self,
        layer_dir: Path,
        layer_logits: torch.Tensor,
        class_names: Optional[List[str]],
    ) -> None:
        if layer_logits.dim() != 3:
            raise ValueError(
                "layer_logits must be [C, H, W], "
                f"got {tuple(layer_logits.shape)}."
            )

        logits_cpu = layer_logits.detach().cpu().float()
        scores_cpu = logits_cpu.sigmoid()
        num_classes = int(logits_cpu.shape[0])

        logit_max, logit_mean = self._per_class_max_mean(logits_cpu)
        score_max, score_mean = self._per_class_max_mean(scores_cpu)

        order = torch.argsort(score_max, descending=True)

        with open(layer_dir / "score_summary.txt", "w", encoding="utf-8") as f:
            f.write(
                "rank\tclass_id\tclass_name\t"
                "prob_max\tprob_mean\tlogit_max\tlogit_mean\n"
            )

            for rank, cls_idx in enumerate(order.tolist()):
                class_name = (
                    class_names[cls_idx]
                    if class_names is not None and cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )

                f.write(
                    f"{rank}\t{cls_idx}\t{class_name}\t"
                    f"{float(score_max[cls_idx].item()):.6f}\t"
                    f"{float(score_mean[cls_idx].item()):.6f}\t"
                    f"{float(logit_max[cls_idx].item()):.6f}\t"
                    f"{float(logit_mean[cls_idx].item()):.6f}\n"
                )

    def _save_final_mixer_mask_layer_outputs(
        self,
        sample_dir: Path,
        overlay_image: Image.Image,
        mask_logits_layers: torch.Tensor,
        out_hw: Tuple[int, int],
        class_names: Optional[List[str]],
    ) -> None:
        if mask_logits_layers.dim() != 4:
            raise ValueError(
                "mask_logits_layers for one sample must be [L, C, H, W], "
                f"got {tuple(mask_logits_layers.shape)}."
            )

        num_layers = int(mask_logits_layers.shape[0])
        num_classes = int(mask_logits_layers.shape[1])

        layer_root = sample_dir / "final_mixer_layers"
        layer_root.mkdir(parents=True, exist_ok=True)

        with open(layer_root / "README.txt", "w", encoding="utf-8") as f:
            f.write(
                "This folder contains per-layer final mixer mask outputs.\n"
                "Each layer folder contains:\n"
                "- pred.png: argmax segmentation map from this layer's mask logits\n"
                "- overlay.png: pred.png overlaid on the original image\n"
                "- mask_heatmaps/: per-class sigmoid(logit) heatmaps\n"
                "- score_summary.txt: per-class score statistics for this layer\n"
            )

        for layer_idx in range(num_layers):
            layer_logits = mask_logits_layers[layer_idx]
            if layer_logits.dim() != 3:
                raise ValueError(
                    "Each final mixer layer logits must be [C, H, W], "
                    f"got {tuple(layer_logits.shape)}."
                )

            layer_dir = layer_root / f"layer_{layer_idx:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            self._save_final_mixer_layer_summary(
                layer_dir=layer_dir,
                layer_logits=layer_logits,
                class_names=class_names,
            )

            pred_mask = layer_logits.argmax(dim=0).long()
            pred_mask_out = self._prepare_label_map(pred_mask, out_hw)

            if self.cfg.save_final_mixer_layer_predictions:
                self._colorize_label_map(
                    pred_mask_out,
                    num_classes=num_classes,
                ).save(layer_dir / "pred.png")

            if self.cfg.save_final_mixer_layer_overlays:
                self._overlay_label_map(
                    overlay_image,
                    pred_mask_out,
                    num_classes=num_classes,
                ).save(layer_dir / "overlay.png")

            if not self.cfg.save_final_mixer_layer_heatmaps:
                continue

            heatmap_dir = layer_dir / "mask_heatmaps"
            heatmap_dir.mkdir(parents=True, exist_ok=True)

            score_map = layer_logits.sigmoid()
            class_indices = self._select_heatmap_class_indices(layer_logits)

            for cls_idx in class_indices:
                class_name = (
                    class_names[cls_idx]
                    if class_names is not None and cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )
                filename = f"{cls_idx:03d}_{self._sanitize_filename(class_name)}.png"

                self._to_heatmap_image(
                    score_map[cls_idx],
                    out_hw=out_hw,
                    normalize="prob",
                ).save(heatmap_dir / filename)

    def _save_presence_scores(
        self,
        sample_dir: Path,
        presence_score: torch.Tensor,
        presence_logits: Optional[torch.Tensor],
        class_names: Optional[List[str]],
    ) -> None:
        if presence_score.dim() != 1:
            raise ValueError(
                f"Expected presence_score as [C], got {tuple(presence_score.shape)}."
            )

        num_classes = int(presence_score.shape[0])

        if presence_logits is not None:
            if presence_logits.dim() != 1:
                raise ValueError(
                    f"Expected presence_logits as [C], got {tuple(presence_logits.shape)}."
                )
            if presence_logits.shape[0] != num_classes:
                raise ValueError(
                    "presence_logits and presence_score class count mismatch: "
                    f"{presence_logits.shape[0]} vs {num_classes}."
                )

        score_cpu = presence_score.detach().cpu().float()
        logits_cpu = (
            presence_logits.detach().cpu().float()
            if presence_logits is not None
            else None
        )

        with open(sample_dir / "presence_scores.txt", "w", encoding="utf-8") as f:
            f.write("class_id\tclass_name\tpresence_score\tpresence_logit\n")
            for cls_idx in range(num_classes):
                class_name = (
                    class_names[cls_idx]
                    if class_names is not None and cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )
                logit_value = (
                    float(logits_cpu[cls_idx].item())
                    if logits_cpu is not None
                    else None
                )
                f.write(
                    f"{cls_idx}\t{class_name}\t"
                    f"{float(score_cpu[cls_idx].item()):.6f}\t"
                    f"{self._format_optional_float(logit_value)}\n"
                )

        self._save_presence_score_image(
            sample_dir=sample_dir,
            presence_score=score_cpu,
            presence_logits=logits_cpu,
            class_names=class_names,
        )

    def _save_presence_score_image(
        self,
        sample_dir: Path,
        presence_score: torch.Tensor,
        presence_logits: Optional[torch.Tensor],
        class_names: Optional[List[str]],
    ) -> None:
        num_classes = int(presence_score.shape[0])
        row_h = 26
        header_h = 34
        left_w = 260
        bar_w = 360
        right_w = 160
        width = left_w + bar_w + right_w
        height = header_h + max(1, num_classes) * row_h + 12

        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        draw.text((10, 10), "presence scores", fill=(0, 0, 0), font=font)

        for cls_idx in range(num_classes):
            y = header_h + cls_idx * row_h
            class_name = (
                class_names[cls_idx]
                if class_names is not None and cls_idx < len(class_names)
                else f"class_{cls_idx}"
            )
            class_name = str(class_name)
            if len(class_name) > 34:
                class_name = class_name[:31] + "..."

            score = float(presence_score[cls_idx].item())
            score = max(0.0, min(1.0, score))

            logit_text = ""
            if presence_logits is not None:
                logit_text = f" logit={float(presence_logits[cls_idx].item()):.3f}"

            draw.text(
                (10, y + 6),
                f"{cls_idx:03d} {class_name}",
                fill=(0, 0, 0),
                font=font,
            )

            x0 = left_w
            y0 = y + 6
            x1 = left_w + bar_w
            y1 = y + row_h - 6

            draw.rectangle((x0, y0, x1, y1), outline=(80, 80, 80), fill=(235, 235, 235))
            fill_x1 = x0 + int(round(score * bar_w))
            draw.rectangle((x0, y0, fill_x1, y1), outline=None, fill=(30, 144, 255))

            draw.text(
                (left_w + bar_w + 10, y + 6),
                f"{score:.4f}{logit_text}",
                fill=(0, 0, 0),
                font=font,
            )

        image.save(sample_dir / "presence_scores.png")

    def _save_presence_layer_scores(
        self,
        sample_dir: Path,
        presence_logits_layers: torch.Tensor,
        class_names: Optional[List[str]],
    ) -> None:
        if presence_logits_layers.dim() != 2:
            raise ValueError(
                "presence_logits_layers for one sample must be [L, C], "
                f"got {tuple(presence_logits_layers.shape)}."
            )

        logits_cpu = presence_logits_layers.detach().cpu().float()
        scores_cpu = logits_cpu.sigmoid()

        num_layers, num_classes = logits_cpu.shape

        with open(sample_dir / "presence_layer_scores.txt", "w", encoding="utf-8") as f:
            f.write("layer_id\tclass_id\tclass_name\tpresence_score\tpresence_logit\n")
            for layer_idx in range(num_layers):
                for cls_idx in range(num_classes):
                    class_name = (
                        class_names[cls_idx]
                        if class_names is not None and cls_idx < len(class_names)
                        else f"class_{cls_idx}"
                    )
                    f.write(
                        f"{layer_idx}\t{cls_idx}\t{class_name}\t"
                        f"{float(scores_cpu[layer_idx, cls_idx].item()):.6f}\t"
                        f"{float(logits_cpu[layer_idx, cls_idx].item()):.6f}\n"
                    )

        self._save_presence_layer_image(
            sample_dir=sample_dir,
            presence_scores_layers=scores_cpu,
            class_names=class_names,
        )

    def _save_presence_layer_image(
        self,
        sample_dir: Path,
        presence_scores_layers: torch.Tensor,
        class_names: Optional[List[str]],
    ) -> None:
        num_layers, num_classes = presence_scores_layers.shape

        row_h = 26
        header_h = 38
        left_w = 260
        cell_w = 90
        right_w = 20
        width = left_w + num_layers * cell_w + right_w
        height = header_h + max(1, num_classes) * row_h + 12

        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        draw.text((10, 10), "presence scores by mixer layer", fill=(0, 0, 0), font=font)

        for layer_idx in range(num_layers):
            x = left_w + layer_idx * cell_w
            draw.text((x + 8, 22), f"L{layer_idx}", fill=(0, 0, 0), font=font)

        for cls_idx in range(num_classes):
            y = header_h + cls_idx * row_h

            class_name = (
                class_names[cls_idx]
                if class_names is not None and cls_idx < len(class_names)
                else f"class_{cls_idx}"
            )
            class_name = str(class_name)
            if len(class_name) > 34:
                class_name = class_name[:31] + "..."

            draw.text(
                (10, y + 6),
                f"{cls_idx:03d} {class_name}",
                fill=(0, 0, 0),
                font=font,
            )

            for layer_idx in range(num_layers):
                score = float(presence_scores_layers[layer_idx, cls_idx].item())
                score = max(0.0, min(1.0, score))

                x0 = left_w + layer_idx * cell_w + 6
                y0 = y + 6
                x1 = x0 + cell_w - 14
                y1 = y + row_h - 6

                draw.rectangle(
                    (x0, y0, x1, y1),
                    outline=(80, 80, 80),
                    fill=(235, 235, 235),
                )
                fill_x1 = x0 + int(round(score * (x1 - x0)))
                draw.rectangle(
                    (x0, y0, fill_x1, y1),
                    outline=None,
                    fill=(30, 144, 255),
                )
                draw.text(
                    (x0 + 3, y0 + 2),
                    f"{score:.2f}",
                    fill=(0, 0, 0),
                    font=font,
                )

        image.save(sample_dir / "presence_layer_scores.png")

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