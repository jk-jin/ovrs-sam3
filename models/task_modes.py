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
    # Final outputs
    final_logits: str = "final_logits"
    raw_final_score_map: str = "raw_final_score_map"
    final_score_map: str = "final_score_map"
    final_pred: str = "final_pred"

    # Dynamic threshold outputs
    class_thresholds: str = "class_thresholds"
    class_threshold_logits: str = "class_threshold_logits"

    # Encoder feature refiner outputs
    encoder_features: str = "encoder_features"
    refined_encoder_features: str = "refined_encoder_features"
    refiner_features_36: str = "refiner_features_36"
    score_embed_36: str = "score_embed_36"
    clip_score_embed_36: str = "clip_score_embed_36"
    sam_score_embed_36: str = "sam_score_embed_36"
    template_clip_text_features: str = "template_clip_text_features"
    clip_score_maps: str = "clip_score_maps"
    clip_score_embed: str = "clip_score_embed"
    clip_mid_features: str = "clip_mid_features"


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