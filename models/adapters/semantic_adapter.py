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
        prediction_prob_threshold: Optional[float] = None,
    ):
        super().__init__()

        self.prediction_prob_threshold = self._validate_threshold(
            prediction_prob_threshold,
            "prediction_prob_threshold",
        )

    @staticmethod
    def _validate_threshold(
        value: Optional[float],
        name: str,
    ) -> Optional[float]:
        if value is None:
            return None

        value = float(value)
        if (
            not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"{name} must be None or a finite value "
                f"in [0.0, 1.0], got {value}."
            )
        return value

    @staticmethod
    def _require(
        raw_outputs: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        value = raw_outputs.get(key, None)
        if value is None:
            raise ValueError(
                f"Raw outputs must contain {key!r}."
            )
        return value

    @staticmethod
    def _as_4d_map(
        x: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(
                    f"Expected {key} as [B, C, 1, H, W] "
                    f"when 5D, got {tuple(x.shape)}."
                )
            x = x[:, :, 0]

        if x.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], "
                f"got {tuple(x.shape)}."
            )

        return x

    @staticmethod
    def _get_metadata(
        batch: BatchedDatapoint,
    ):
        if len(batch.find_metadatas) != 1:
            raise ValueError(
                "Semantic inference requires exactly one "
                "BatchedInferenceMetadata entry."
            )
        return batch.find_metadatas[0]

    @staticmethod
    def _resolve_prompt_layout(
        metadata,
        device: torch.device,
    ) -> tuple[int, int, torch.Tensor]:
        num_prompts = int(metadata.num_prompts)
        num_classes = int(metadata.num_classes)

        if num_prompts <= 0:
            raise ValueError(
                "metadata.num_prompts must be positive."
            )

        if num_classes <= 0:
            raise ValueError(
                "metadata.num_classes must be positive."
            )

        prompt_to_class_id = torch.as_tensor(
            metadata.prompt_to_class_id,
            dtype=torch.long,
            device=device,
        )

        if prompt_to_class_id.ndim != 1:
            raise ValueError(
                "metadata.prompt_to_class_id must be 1D, "
                f"got {tuple(prompt_to_class_id.shape)}."
            )

        if int(prompt_to_class_id.numel()) != num_prompts:
            raise ValueError(
                "metadata.prompt_to_class_id length must "
                "equal metadata.num_prompts."
            )

        if prompt_to_class_id.numel() > 0:
            min_id = int(prompt_to_class_id.min().item())
            max_id = int(prompt_to_class_id.max().item())

            if min_id < 0 or max_id >= num_classes:
                raise ValueError(
                    "metadata.prompt_to_class_id values must "
                    f"be in [0, {num_classes - 1}], got "
                    f"min={min_id}, max={max_id}."
                )

        prompt_counts = torch.bincount(
            prompt_to_class_id,
            minlength=num_classes,
        )

        if bool((prompt_counts == 0).any().item()):
            missing = torch.nonzero(
                prompt_counts == 0,
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                "Every original class must own at least one "
                f"prompt. Missing class ids: {missing}."
            )

        return (
            num_prompts,
            num_classes,
            prompt_to_class_id,
        )

    @staticmethod
    def _merge_prompt_channels(
        prompt_map: torch.Tensor,
        prompt_to_class_id: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        if prompt_map.dim() != 4:
            raise ValueError(
                "prompt_map must be [B, P, H, W]."
            )

        if int(prompt_map.shape[1]) != int(
            prompt_to_class_id.numel()
        ):
            raise ValueError(
                "Prompt map channel count does not match "
                "prompt_to_class_id length."
            )

        merged_classes: list[torch.Tensor] = []

        for class_id in range(int(num_classes)):
            prompt_indices = torch.nonzero(
                prompt_to_class_id == class_id,
                as_tuple=False,
            ).flatten()

            if prompt_indices.numel() == 0:
                raise ValueError(
                    f"Original class {class_id} has no prompts."
                )

            class_prompt_map = prompt_map.index_select(
                dim=1,
                index=prompt_indices,
            )

            merged_classes.append(
                class_prompt_map.amax(
                    dim=1,
                    keepdim=False,
                )
            )

        return torch.stack(
            merged_classes,
            dim=1,
        )

    def _apply_prediction_filter(
        self,
        student_score: torch.Tensor,
        background_enabled: bool,
    ) -> torch.Tensor:
        # Without background, keep all scores so argmax can choose a real class.
        if not background_enabled or self.prediction_prob_threshold is None:
            return student_score
        return student_score.masked_fill(
            student_score < self.prediction_prob_threshold, 0.0
        )

    def build_infer_score_outputs(
        self,
        raw_prompt_score_map: torch.Tensor,
        metadata,
    ) -> Dict[str, torch.Tensor]:
        raw_prompt_score_map = self._as_4d_map(
            raw_prompt_score_map,
            OUTPUT_KEYS.raw_prompt_score_map,
        )

        (
            num_prompts,
            num_classes,
            prompt_to_class_id,
        ) = self._resolve_prompt_layout(
            metadata=metadata,
            device=raw_prompt_score_map.device,
        )

        if int(raw_prompt_score_map.shape[1]) != num_prompts:
            raise ValueError(
                "raw_prompt_score_map channel count does not "
                f"match metadata.num_prompts: "
                f"{raw_prompt_score_map.shape[1]} vs "
                f"{num_prompts}."
            )

        raw_final_score_map = self._merge_prompt_channels(
            prompt_map=raw_prompt_score_map,
            prompt_to_class_id=prompt_to_class_id,
            num_classes=num_classes,
        )

        final_score_map = self._apply_prediction_filter(
            raw_final_score_map,
            background_enabled=bool(metadata.background_enabled),
        )

        outputs = {
            OUTPUT_KEYS.raw_prompt_score_map:
                raw_prompt_score_map,
            OUTPUT_KEYS.raw_final_score_map:
                raw_final_score_map,
            OUTPUT_KEYS.final_score_map:
                final_score_map,
            OUTPUT_KEYS.final_pred:
                final_score_map.argmax(dim=1),
        }
        return outputs

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        output_mode: str = "final",
    ) -> Dict[str, torch.Tensor]:
        output_mode = str(output_mode).lower()

        if output_mode not in {"final", "infer"}:
            raise ValueError(
                f"Unknown output_mode={output_mode!r}. "
                "Supported modes are: 'final', 'infer'."
            )

        prompt_logits = self._as_4d_map(
            self._require(
                raw_outputs,
                OUTPUT_KEYS.final_logits,
            ),
            OUTPUT_KEYS.final_logits,
        )

        metadata = self._get_metadata(batch)

        (
            num_prompts,
            num_classes,
            prompt_to_class_id,
        ) = self._resolve_prompt_layout(
            metadata=metadata,
            device=prompt_logits.device,
        )

        if int(prompt_logits.shape[1]) != num_prompts:
            raise ValueError(
                "Prompt logit channel count does not match "
                f"metadata.num_prompts: "
                f"{prompt_logits.shape[1]} vs "
                f"{num_prompts}."
            )

        if output_mode == "final":
            return {
                OUTPUT_KEYS.final_logits: prompt_logits,
            }

        class_logits = self._merge_prompt_channels(
            prompt_map=prompt_logits,
            prompt_to_class_id=prompt_to_class_id,
            num_classes=num_classes,
        )

        raw_prompt_score_map = prompt_logits.sigmoid()

        outputs = dict(raw_outputs)
        outputs[OUTPUT_KEYS.prompt_logits] = prompt_logits
        outputs[OUTPUT_KEYS.final_logits] = class_logits

        outputs.update(
            self.build_infer_score_outputs(
                raw_prompt_score_map=
                    raw_prompt_score_map,
                metadata=metadata,
            )
        )

        for key in (
            OUTPUT_KEYS.prompt_conditioned_features_72,
            OUTPUT_KEYS.aggregated_features_36,
            OUTPUT_KEYS.semantic_state_36,
            OUTPUT_KEYS.clip_guidance_36,
            OUTPUT_KEYS.remoteclip_template_text_features,
            OUTPUT_KEYS.clip_template_scores_36,
            OUTPUT_KEYS.clip_spatial_guidance_features,
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
