from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..data_misc import BatchedDatapoint
from ..task_modes import OUTPUT_KEYS


class SemanticSegAdapter(nn.Module):
    def __init__(self):
        super().__init__()

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
    def _as_presence_score(
        x: torch.Tensor,
        batch_size: int,
        num_classes: int,
        key: str,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"Expected {key} as [B, C], got {tuple(x.shape)}."
            )

        if tuple(x.shape) != (int(batch_size), int(num_classes)):
            raise ValueError(
                f"Expected {key} shape {(int(batch_size), int(num_classes))}, "
                f"got {tuple(x.shape)}."
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

        semantic_logits = self._as_4d_map(
            self._require(raw_outputs, OUTPUT_KEYS.semantic_logits),
            OUTPUT_KEYS.semantic_logits,
        )
        final_logits = self._as_4d_map(
            self._require(raw_outputs, OUTPUT_KEYS.final_logits),
            OUTPUT_KEYS.final_logits,
        )

        self._check_same_shape(
            semantic_logits,
            final_logits,
            OUTPUT_KEYS.semantic_logits,
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

        B, C = int(final_logits.shape[0]), int(final_logits.shape[1])

        presence_logits = self._as_presence_score(
            self._require(raw_outputs, OUTPUT_KEYS.presence_logits),
            batch_size=B,
            num_classes=C,
            key=OUTPUT_KEYS.presence_logits,
        )
        presence_score = self._as_presence_score(
            self._require(raw_outputs, OUTPUT_KEYS.presence_score),
            batch_size=B,
            num_classes=C,
            key=OUTPUT_KEYS.presence_score,
        )

        outputs = dict(raw_outputs)

        outputs[OUTPUT_KEYS.semantic_logits] = semantic_logits
        outputs[OUTPUT_KEYS.final_logits] = final_logits
        outputs[OUTPUT_KEYS.presence_logits] = presence_logits
        outputs[OUTPUT_KEYS.presence_score] = presence_score

        outputs[OUTPUT_KEYS.semantic_score_map] = semantic_logits.sigmoid()

        raw_final_score_map = final_logits.sigmoid()
        final_score_map = raw_final_score_map * presence_score[:, :, None, None]

        outputs[OUTPUT_KEYS.raw_final_score_map] = raw_final_score_map
        outputs[OUTPUT_KEYS.final_score_map] = final_score_map
        outputs[OUTPUT_KEYS.final_pred] = final_score_map.argmax(dim=1)

        for key in (
            OUTPUT_KEYS.class_tokens,
            OUTPUT_KEYS.class_feature_low,
            OUTPUT_KEYS.clip_score_maps,
            OUTPUT_KEYS.sam3_score_low,
            OUTPUT_KEYS.sam3_fpn_features,
            OUTPUT_KEYS.clip_mid_features,
            OUTPUT_KEYS.clip_dense_low,
            OUTPUT_KEYS.class_text_guidance,
            OUTPUT_KEYS.stage_text_guidance_history,
        ):
            if key in raw_outputs:
                outputs[key] = raw_outputs[key]

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