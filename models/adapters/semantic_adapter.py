from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_misc import BatchedDatapoint
from ..task_modes import OUTPUT_KEYS


class SemanticSegAdapter(nn.Module):
    def __init__(
        self,
        presence_base: float = 0.5,
        init_presence_modulation_alpha: float = 1.0,
    ):
        super().__init__()
        self.presence_base = float(presence_base)
        self.presence_modulation_alpha = nn.Parameter(
            torch.tensor(float(init_presence_modulation_alpha))
        )

    @staticmethod
    def _extract_semantic_logits(
        raw_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        semantic_logits = raw_outputs.get(OUTPUT_KEYS.semantic_logits, None)
        if semantic_logits is None:
            raise ValueError(
                f"Raw outputs must contain '{OUTPUT_KEYS.semantic_logits}'."
            )

        if semantic_logits.dim() == 5:
            if semantic_logits.shape[2] != 1:
                raise ValueError(
                    f"Expected semantic_logits as [B, C, 1, H, W], got {tuple(semantic_logits.shape)}"
                )
            semantic_logits = semantic_logits[:, :, 0]
        elif semantic_logits.dim() != 4:
            raise ValueError(
                f"Expected semantic_logits as [B, C, H, W] or [B, C, 1, H, W], got {tuple(semantic_logits.shape)}"
            )

        return semantic_logits

    @staticmethod
    def _extract_presence_logits(
        raw_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        presence_logits = raw_outputs.get(OUTPUT_KEYS.presence_logits, None)
        if presence_logits is None:
            raise ValueError(
                f"Raw outputs must contain '{OUTPUT_KEYS.presence_logits}'."
            )

        if presence_logits.dim() == 3:
            if presence_logits.shape[-1] != 1:
                raise ValueError(
                    f"Expected presence_logits as [B, C, 1], got {tuple(presence_logits.shape)}"
                )
            presence_logits = presence_logits[..., 0]
        elif presence_logits.dim() != 2:
            raise ValueError(
                f"Expected presence_logits as [B, C] or [B, C, 1], got {tuple(presence_logits.shape)}"
            )

        return presence_logits

    @staticmethod
    def _resize_to_match(
        x: Optional[torch.Tensor],
        target_hw: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None

        if tuple(x.shape[-2:]) == tuple(target_hw):
            return x

        return F.interpolate(
            x,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

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
    def _validate_class_count(
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
    def _validate_presence_class_count(
        semantic_logits: torch.Tensor,
        presence_logits: torch.Tensor,
    ) -> None:
        if semantic_logits.shape[:2] != presence_logits.shape:
            raise ValueError(
                "Shape mismatch between semantic_logits and presence_logits: "
                f"semantic_logits.shape[:2]={tuple(semantic_logits.shape[:2])}, "
                f"presence_logits.shape={tuple(presence_logits.shape)}"
            )

    def _build_final_logits(
            self,
            semantic_logits: torch.Tensor,
            presence_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            semantic_logits: [B, C, H, W]
            presence_logits: [B, C]

        Returns:
            final_logits: [B, C, H, W]
            presence_score: [B, C]
        """
        if semantic_logits.dim() != 4:
            raise ValueError(
                f"Expected semantic_logits as [B, C, H, W], got {tuple(semantic_logits.shape)}"
            )
        if presence_logits.dim() != 2:
            raise ValueError(
                f"Expected presence_logits as [B, C], got {tuple(presence_logits.shape)}"
            )
        if semantic_logits.shape[:2] != presence_logits.shape:
            raise ValueError(
                "Shape mismatch between semantic_logits and presence_logits: "
                f"{tuple(semantic_logits.shape[:2])} vs {tuple(presence_logits.shape)}"
            )

        presence_score = presence_logits.sigmoid()  # [B, C]

        presence_modulation_map = torch.sigmoid(
            semantic_logits * self.presence_modulation_alpha
        )  # [B, C, H, W]

        spatial_presence = self.presence_base + (
                presence_score[:, :, None, None] * presence_modulation_map
        )  # [B, C, H, W]

        final_logits = semantic_logits * spatial_presence  # [B, C, H, W]
        return final_logits, presence_score

    def _build_train_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_semantic_logits(raw_outputs)
        presence_logits = self._extract_presence_logits(raw_outputs)

        actual_num_classes = int(semantic_logits.shape[1])
        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        self._validate_class_count(
            actual_num_classes=actual_num_classes,
            expected_num_classes=expected_num_classes,
        )
        self._validate_presence_class_count(
            semantic_logits=semantic_logits,
            presence_logits=presence_logits,
        )

        final_logits, presence_score = self._build_final_logits(
            semantic_logits=semantic_logits,
            presence_logits=presence_logits,
        )

        return {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.presence_logits: presence_logits,
            OUTPUT_KEYS.presence_score: presence_score,
            OUTPUT_KEYS.final_logits: final_logits,
        }

    def _build_inference_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._extract_semantic_logits(raw_outputs)
        presence_logits = self._extract_presence_logits(raw_outputs)

        actual_num_classes = int(semantic_logits.shape[1])
        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        self._validate_class_count(
            actual_num_classes=actual_num_classes,
            expected_num_classes=expected_num_classes,
        )
        self._validate_presence_class_count(
            semantic_logits=semantic_logits,
            presence_logits=presence_logits,
        )

        semantic_score_map = semantic_logits.sigmoid()

        final_logits, presence_score = self._build_final_logits(
            semantic_logits=semantic_logits,
            presence_logits=presence_logits,
        )

        final_score_map = final_logits.sigmoid()
        final_pred = final_score_map.argmax(dim=1)

        return {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.semantic_score_map: semantic_score_map,
            OUTPUT_KEYS.presence_logits: presence_logits,
            OUTPUT_KEYS.presence_score: presence_score,
            OUTPUT_KEYS.final_logits: final_logits,
            OUTPUT_KEYS.final_score_map: final_score_map,
            OUTPUT_KEYS.final_pred: final_pred,
        }

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "both",
    ) -> Dict[str, Dict[str, torch.Tensor]] | Dict[str, torch.Tensor]:
        output_mode = str(output_mode)

        if output_mode == "train":
            return self._build_train_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

        if output_mode == "infer":
            return self._build_inference_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

        if output_mode == "both":
            train_outputs = self._build_train_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

            inference_outputs = self._build_inference_outputs(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=expected_num_classes,
            )

            return {
                "train_outputs": train_outputs,
                "inference_outputs": inference_outputs,
            }

        raise ValueError(
            f"Unknown output_mode={output_mode}. "
            "Supported modes are: 'train', 'infer', 'both'."
        )


class HybridSegAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
        output_mode: str = "both",
    ):
        raise NotImplementedError(
            "HybridSegAdapter is not implemented yet."
        )