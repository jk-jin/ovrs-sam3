from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ...config_dataclasses import AdapterConfig
from ..data_misc import BatchedDatapoint
from ..task_modes import OUTPUT_KEYS


class SemanticSegAdapter(nn.Module):
    def __init__(self, cfg: Optional[AdapterConfig] = None):
        super().__init__()
        self.cfg = cfg or AdapterConfig()

    @staticmethod
    def _require(
        raw_outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        value = raw_outputs.get(key, None)
        if value is None:
            raise ValueError(f"Raw outputs must contain '{key}'.")
        return value

    @staticmethod
    def _as_4d_map(
        x: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(
                    f"Expected {key} as [B, C, 1, H, W] when 5D, "
                    f"got {tuple(x.shape)}."
                )
            x = x[:, :, 0]

        if x.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], got {tuple(x.shape)}."
            )

        return x

    @staticmethod
    def _check_same_shape(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        lhs_key: str,
        rhs_key: str,
    ) -> None:
        if tuple(lhs.shape) != tuple(rhs.shape):
            raise ValueError(
                f"Shape mismatch between {lhs_key} and {rhs_key}: "
                f"{tuple(lhs.shape)} vs {tuple(rhs.shape)}."
            )

    @staticmethod
    def _get_semantic_metadata(batch: BatchedDatapoint) -> dict:
        if len(batch.find_metadatas) == 0:
            raise ValueError("batch.find_metadatas is empty.")

        meta = batch.find_metadatas[0]

        original_num_classes = int(meta.num_classes)
        full_class_names = list(meta.class_names)

        active_class_ids = getattr(meta, "active_class_ids", None)
        if active_class_ids is None or len(active_class_ids) == 0:
            active_class_ids = list(range(original_num_classes))
        active_class_ids = [int(x) for x in active_class_ids]

        active_class_names = getattr(meta, "active_class_names", None)
        if active_class_names is None or len(active_class_names) == 0:
            active_class_names = [full_class_names[i] for i in active_class_ids]

        bg_enabled = bool(getattr(meta, "background_mapping_enabled", False))
        bg_id = getattr(meta, "background_id", None)
        bg_id = None if bg_id is None else int(bg_id)

        default_bg_id = int(getattr(meta, "default_background_id", 255))

        if len(active_class_ids) == 0:
            raise ValueError("active_class_ids is empty.")

        if len(set(active_class_ids)) != len(active_class_ids):
            raise ValueError("active_class_ids contains duplicates.")

        for cls_id in active_class_ids:
            if not 0 <= cls_id < original_num_classes:
                raise ValueError(
                    f"active_class_ids contains {cls_id} which is out of range "
                    f"[0, {original_num_classes})."
                )

        return {
            "original_num_classes": original_num_classes,
            "full_class_names": full_class_names,
            "active_class_ids": active_class_ids,
            "active_class_names": active_class_names,
            "background_mapping_enabled": bg_enabled,
            "background_id": bg_id,
            "default_background_id": default_bg_id,
        }

    def _build_full_score_tensors(
        self,
        active_tensor: torch.Tensor,
        active_class_ids_tensor: torch.Tensor,
        original_num_classes: int,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Expand an active-class-only tensor to full class order."""
        B = active_tensor.shape[0]
        H, W = active_tensor.shape[2], active_tensor.shape[3]
        full = torch.full(
            (B, original_num_classes, H, W),
            fill_value,
            device=active_tensor.device,
            dtype=active_tensor.dtype,
        )
        full[:, active_class_ids_tensor] = active_tensor
        return full

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "final",
    ) -> Dict[str, torch.Tensor]:
        output_mode = str(output_mode).lower()
        if output_mode not in {"final", "infer", "infer_raw"}:
            raise ValueError(
                f"Unknown output_mode={output_mode!r}. "
                "Supported modes are: 'final', 'infer', 'infer_raw'."
            )

        final_logits = self._as_4d_map(
            self._require(raw_outputs, OUTPUT_KEYS.final_logits),
            OUTPUT_KEYS.final_logits,
        )

        metadata = self._get_semantic_metadata(batch)
        expected_active = len(metadata["active_class_ids"])

        if final_logits.shape[1] != expected_active:
            raise ValueError(
                f"final_logits has {final_logits.shape[1]} channels, "
                f"but active_class_ids has {expected_active} ids."
            )

        active_class_ids_tensor = torch.tensor(
            metadata["active_class_ids"],
            device=final_logits.device,
            dtype=torch.long,
        )
        original_num_classes_tensor = torch.tensor(
            metadata["original_num_classes"],
            device=final_logits.device,
            dtype=torch.long,
        )

        if output_mode == "final":
            # Training: keep compact active logits. Loss needs compact logits
            # because it aligns channels via active_class_ids.
            outputs = {
                OUTPUT_KEYS.final_logits: final_logits,
                OUTPUT_KEYS.active_class_ids: active_class_ids_tensor,
                OUTPUT_KEYS.original_num_classes: original_num_classes_tensor,
            }

            for key in (
                OUTPUT_KEYS.class_thresholds,
                OUTPUT_KEYS.class_threshold_logits,
            ):
                if key in raw_outputs:
                    outputs[key] = raw_outputs[key]

            return outputs

        # --- Shared: build active score map and raw full score map ---
        active_score_map = final_logits.sigmoid()

        # Always build raw_full_score_map in full class order for consistency
        # with visualization and downstream consumers.
        raw_full_score_map = self._build_full_score_tensors(
            active_score_map,
            active_class_ids_tensor,
            metadata["original_num_classes"],
            fill_value=0.0,
        )

        # Build full final_logits for inference/debug so visualization
        # sees consistent shapes.
        eps = 1e-6
        full_final_logits = torch.logit(raw_full_score_map.clamp(eps, 1.0 - eps))

        # --- infer_raw: return pre-threshold outputs for TTA ---
        if output_mode == "infer_raw":
            outputs = dict(raw_outputs)
            outputs[OUTPUT_KEYS.final_logits] = full_final_logits
            outputs[OUTPUT_KEYS.raw_final_score_map] = raw_full_score_map
            outputs[OUTPUT_KEYS.active_class_ids] = active_class_ids_tensor
            outputs[OUTPUT_KEYS.original_num_classes] = original_num_classes_tensor

            for key in (
                OUTPUT_KEYS.class_thresholds,
                OUTPUT_KEYS.class_threshold_logits,
            ):
                if key in raw_outputs:
                    outputs[key] = raw_outputs[key]

            for key in (
                OUTPUT_KEYS.encoder_features,
                OUTPUT_KEYS.refined_encoder_features,
                OUTPUT_KEYS.refiner_features_36,
                OUTPUT_KEYS.score_embed_36,
                OUTPUT_KEYS.clip_score_embed_36,
                OUTPUT_KEYS.sam_score_embed_36,
                OUTPUT_KEYS.template_clip_text_features,
                OUTPUT_KEYS.clip_score_maps,
                OUTPUT_KEYS.clip_score_embed,
                OUTPUT_KEYS.clip_mid_features,
            ):
                if key in raw_outputs:
                    outputs[key] = raw_outputs[key]

            if (
                OUTPUT_KEYS.clip_score_embed_36 in outputs
                and OUTPUT_KEYS.clip_score_embed not in outputs
            ):
                outputs[OUTPUT_KEYS.clip_score_embed] = outputs[
                    OUTPUT_KEYS.clip_score_embed_36
                ]

            return outputs

        # --- infer mode: full postprocessing ---
        postprocessed = self.postprocess_infer_outputs(
            raw_outputs=raw_outputs,
            batch=batch,
        )

        # Ensure final_logits and score maps are full class order in the
        # returned outputs (postprocess_infer_outputs already handles
        # final_score_map, final_pred, raw_final_score_map).
        # Override raw_outputs' compact final_logits with the full version.
        postprocessed[OUTPUT_KEYS.final_logits] = full_final_logits
        if OUTPUT_KEYS.raw_final_score_map not in postprocessed:
            postprocessed[OUTPUT_KEYS.raw_final_score_map] = raw_full_score_map

        return postprocessed

    def postprocess_infer_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
    ) -> Dict[str, torch.Tensor]:
        """Unified inference postprocessing.

        Handles threshold filtering, background region detection,
        active→full id remapping, and full score map construction.

        Background id comes from dataset metadata.
        Base threshold comes from adapter_cfg.threshold.
        """
        metadata = self._get_semantic_metadata(batch)

        # Resolve active score map and raw full score map.
        raw_full_score_map = raw_outputs.get(OUTPUT_KEYS.raw_final_score_map, None)

        if raw_full_score_map is not None:
            active_class_ids_tensor = torch.tensor(
                metadata["active_class_ids"],
                device=raw_full_score_map.device,
                dtype=torch.long,
            )

            if raw_full_score_map.shape[1] == metadata["original_num_classes"]:
                # Already full order (e.g. from TTA merge).
                active_score_map = raw_full_score_map[:, active_class_ids_tensor]
            elif raw_full_score_map.shape[1] == len(metadata["active_class_ids"]):
                # Compact order — need to expand for consistent output.
                active_score_map = raw_full_score_map
                raw_full_score_map = self._build_full_score_tensors(
                    active_score_map,
                    active_class_ids_tensor,
                    metadata["original_num_classes"],
                    fill_value=0.0,
                )
            else:
                raise ValueError(
                    f"raw_final_score_map has {raw_full_score_map.shape[1]} channels; "
                    f"expected {metadata['original_num_classes']} (full) or "
                    f"{len(metadata['active_class_ids'])} (active)."
                )
        else:
            final_logits = self._as_4d_map(
                self._require(raw_outputs, OUTPUT_KEYS.final_logits),
                OUTPUT_KEYS.final_logits,
            )
            active_score_map = final_logits.sigmoid()
            active_class_ids_tensor = torch.tensor(
                metadata["active_class_ids"],
                device=final_logits.device,
                dtype=torch.long,
            )
            raw_full_score_map = self._build_full_score_tensors(
                active_score_map,
                active_class_ids_tensor,
                metadata["original_num_classes"],
                fill_value=0.0,
            )

        B, C_active, H, W = active_score_map.shape
        device = active_score_map.device
        dtype = active_score_map.dtype

        dynamic_thresholds = raw_outputs.get(OUTPUT_KEYS.class_thresholds, None)

        base_thd = float(self.cfg.threshold)

        if dynamic_thresholds is not None:
            effective_thresholds = torch.maximum(
                dynamic_thresholds.to(dtype=dtype, device=device),
                torch.full_like(dynamic_thresholds, base_thd, dtype=dtype, device=device),
            )
        else:
            effective_thresholds = torch.full(
                (B, C_active),
                base_thd,
                device=device,
                dtype=dtype,
            )

        # Per-class threshold mask.
        keep = active_score_map >= effective_thresholds[:, :, None, None]

        object_region = keep.any(dim=1)
        background_region = ~object_region

        # Mask out classes that fail the threshold before argmax.
        neg_inf = torch.finfo(dtype).min
        masked_active_score = active_score_map.masked_fill(~keep, neg_inf)
        active_pred = masked_active_score.argmax(dim=1)
        mapped_active_pred = active_class_ids_tensor[active_pred]

        default_bg_id = metadata["default_background_id"]

        pred = torch.full(
            (B, H, W),
            fill_value=default_bg_id,
            device=device,
            dtype=torch.long,
        )
        pred[object_region] = mapped_active_pred[object_region]

        if metadata["background_mapping_enabled"] and metadata["background_id"] is not None:
            pred[background_region] = int(metadata["background_id"])

        # Build filtered full score map.
        filtered_active_score = active_score_map.masked_fill(~keep, 0.0)
        full_score_map = self._build_full_score_tensors(
            filtered_active_score,
            active_class_ids_tensor,
            metadata["original_num_classes"],
            fill_value=0.0,
        )

        if metadata["background_mapping_enabled"] and metadata["background_id"] is not None:
            full_score_map[:, int(metadata["background_id"])] = background_region.to(dtype=dtype)

        original_num_classes_tensor = torch.tensor(
            metadata["original_num_classes"],
            device=device,
            dtype=torch.long,
        )

        outputs = dict(raw_outputs)
        outputs[OUTPUT_KEYS.raw_final_score_map] = raw_full_score_map
        outputs[OUTPUT_KEYS.final_score_map] = full_score_map
        outputs[OUTPUT_KEYS.final_pred] = pred
        outputs[OUTPUT_KEYS.active_class_ids] = active_class_ids_tensor
        outputs[OUTPUT_KEYS.original_num_classes] = original_num_classes_tensor
        outputs[OUTPUT_KEYS.background_region] = background_region
        outputs[OUTPUT_KEYS.object_region] = object_region

        for key in (
            OUTPUT_KEYS.class_thresholds,
            OUTPUT_KEYS.class_threshold_logits,
        ):
            if key in raw_outputs:
                outputs[key] = raw_outputs[key]

        for key in (
            OUTPUT_KEYS.encoder_features,
            OUTPUT_KEYS.refined_encoder_features,
            OUTPUT_KEYS.refiner_features_36,
            OUTPUT_KEYS.score_embed_36,
            OUTPUT_KEYS.clip_score_embed_36,
            OUTPUT_KEYS.sam_score_embed_36,
            OUTPUT_KEYS.template_clip_text_features,
            OUTPUT_KEYS.clip_score_maps,
            OUTPUT_KEYS.clip_score_embed,
            OUTPUT_KEYS.clip_mid_features,
        ):
            if key in raw_outputs:
                outputs[key] = raw_outputs[key]

        if (
            OUTPUT_KEYS.clip_score_embed_36 in outputs
            and OUTPUT_KEYS.clip_score_embed not in outputs
        ):
            outputs[OUTPUT_KEYS.clip_score_embed] = outputs[
                OUTPUT_KEYS.clip_score_embed_36
            ]

        return outputs


class HybridSegAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "final",
    ):
        raise NotImplementedError(
            "HybridSegAdapter is not implemented yet."
        )
