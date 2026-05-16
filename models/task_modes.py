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

    final_logits: str = "final_logits"
    final_score_map: str = "final_score_map"
    final_pred: str = "final_pred"

    presence_logits: str = "presence_logits"
    presence_score: str = "presence_score"
    presence_logits_layers: str = "presence_logits_layers"

    sam3_pixel_feature: str = "sam3_pixel_feature"
    class_code_scales: str = "class_code_scales"


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