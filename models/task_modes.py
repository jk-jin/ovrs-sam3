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

    # Template score builder outputs
    template_score_maps_18: str = "template_score_maps_18"
    lowres_score_embed: str = "lowres_score_embed"

    # Low-res score refiner outputs
    refined_score_embed_18: str = "refined_score_embed_18"

    # Guidance upsampler outputs
    clip_guidance_36: str = "clip_guidance_36"
    clip_guidance_72: str = "clip_guidance_72"

    # Encoder feature refiner outputs
    encoder_features_72: str = "encoder_features_72"
    refined_encoder_features_72: str = "refined_encoder_features_72"

    # CLIP intermediate features
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
