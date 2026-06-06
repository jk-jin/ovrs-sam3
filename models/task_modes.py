from __future__ import annotations

from dataclasses import dataclass


TASK_MODE_SEMANTIC = "semantic"
TASK_MODE_HYBRID = "hybrid"

VALID_TASK_MODES = {
    TASK_MODE_SEMANTIC,
    TASK_MODE_HYBRID,
}


@dataclass(frozen=True)
class ModelOutputKeys:
    semantic_logits: str = "semantic_logits"
    semantic_score_map: str = "semantic_score_map"

    class_tokens: str = "class_tokens"
    class_feature_low: str = "class_feature_low"

    final_logits: str = "final_logits"
    raw_final_score_map: str = "raw_final_score_map"
    final_score_map: str = "final_score_map"
    final_pred: str = "final_pred"

    clip_score_maps: str = "clip_score_maps"
    sam3_score_low: str = "sam3_score_low"

    sam3_fpn_features: str = "sam3_fpn_features"

    clip_mid_features: str = "clip_mid_features"
    clip_dense_low: str = "clip_dense_low"

    class_text_guidance: str = "class_text_guidance"


OUTPUT_KEYS = ModelOutputKeys()


def normalize_task_mode(task_mode: str) -> str:
    value = str(task_mode).strip().lower()
    if value not in VALID_TASK_MODES:
        raise ValueError(
            f"Unknown task_mode={task_mode!r}. "
            f"Supported modes are: {sorted(VALID_TASK_MODES)}"
        )
    return value


def is_semantic_mode(task_mode: str) -> bool:
    return normalize_task_mode(task_mode) == TASK_MODE_SEMANTIC


def is_hybrid_mode(task_mode: str) -> bool:
    return normalize_task_mode(task_mode) == TASK_MODE_HYBRID