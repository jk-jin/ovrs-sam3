from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..data_misc import BatchedDatapoint
from ..task_modes import OUTPUT_KEYS


class SemanticSegAdapter(nn.Module):
    def __init__(
        self,
        class_relative_prob_thd: Optional[float] = None,
        class_relative_eps: float = 1e-6,
    ):
        super().__init__()

        if class_relative_prob_thd is not None:
            class_relative_prob_thd = float(class_relative_prob_thd)
            if not math.isfinite(class_relative_prob_thd) or not (
                0.0 <= class_relative_prob_thd <= 1.0
            ):
                raise ValueError(
                    "class_relative_prob_thd must be None or a finite value "
                    f"in [0.0, 1.0], got {class_relative_prob_thd}."
                )

        class_relative_eps = float(class_relative_eps)
        if not math.isfinite(class_relative_eps) or class_relative_eps <= 0.0:
            raise ValueError(
                "class_relative_eps must be a finite positive value, "
                f"got {class_relative_eps}."
            )

        self.class_relative_prob_thd = class_relative_prob_thd
        self.class_relative_eps = class_relative_eps

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
    def _infer_expected_num_classes(
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Optional[int]:
        if expected_num_classes is not None:
            return int(expected_num_classes)

        if len(batch.find_metadatas) == 0:
            return None

        try:
            return int(batch.find_metadatas[0].num_classes)
        except Exception:
            return None

    @staticmethod
    def _check_class_count(
        actual_num_classes: int,
        expected_num_classes: Optional[int],
    ) -> None:
        if expected_num_classes is None:
            return

        if actual_num_classes != int(expected_num_classes):
            raise ValueError(
                f"Class count mismatch: expected {expected_num_classes}, "
                f"but got {actual_num_classes} channels."
            )

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

    def _apply_class_relative_filter(
        self,
        raw_score_map: torch.Tensor,
    ) -> torch.Tensor:
        raw_score_map = self._as_4d_map(
            raw_score_map, OUTPUT_KEYS.raw_final_score_map
        )

        if self.class_relative_prob_thd is None:
            return raw_score_map

        spatial_min = raw_score_map.amin(dim=(-2, -1), keepdim=True)
        spatial_max = raw_score_map.amax(dim=(-2, -1), keepdim=True)
        span = spatial_max - spatial_min

        relative_score = (
            (raw_score_map - spatial_min)
            / span.clamp_min(self.class_relative_eps)
        )

        keep = relative_score >= self.class_relative_prob_thd
        keep = keep | (span <= self.class_relative_eps)
        return raw_score_map.masked_fill(~keep, 0.0)

    def build_infer_score_outputs(
        self,
        raw_final_score_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        final_score_map = self._apply_class_relative_filter(raw_final_score_map)
        return {
            OUTPUT_KEYS.raw_final_score_map: raw_final_score_map,
            OUTPUT_KEYS.final_score_map: final_score_map,
            OUTPUT_KEYS.final_pred: final_score_map.argmax(dim=1),
        }

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "final",
    ) -> Dict[str, torch.Tensor]:
        output_mode = str(output_mode).lower()
        if output_mode not in {"final", "infer"}:
            raise ValueError(
                f"Unknown output_mode={output_mode!r}. "
                "Supported modes are: 'final', 'infer'."
            )

        final_logits = self._as_4d_map(
            self._require(raw_outputs, OUTPUT_KEYS.final_logits),
            OUTPUT_KEYS.final_logits,
        )

        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        self._check_class_count(
            actual_num_classes=int(final_logits.shape[1]),
            expected_num_classes=expected_num_classes,
        )

        if output_mode == "final":
            return {OUTPUT_KEYS.final_logits: final_logits}

        # --- infer mode below ---

        outputs = dict(raw_outputs)
        outputs[OUTPUT_KEYS.final_logits] = final_logits

        raw_final_score_map = final_logits.sigmoid()
        outputs.update(self.build_infer_score_outputs(raw_final_score_map))

        for key in (
            OUTPUT_KEYS.encoder_features,
            OUTPUT_KEYS.refined_encoder_features,
            OUTPUT_KEYS.refiner_features_36,
            OUTPUT_KEYS.score_embed_36,
            OUTPUT_KEYS.clip_score_embed_36,
            OUTPUT_KEYS.template_clip_text_features,
            OUTPUT_KEYS.clip_score_maps,
            OUTPUT_KEYS.clip_score_embed,
            OUTPUT_KEYS.clip_mid_features,
        ):
            if key in raw_outputs:
                outputs[key] = raw_outputs[key]

        # Compatibility alias: old code may reference clip_score_embed.
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