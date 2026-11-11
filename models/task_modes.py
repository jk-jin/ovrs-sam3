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
    # Original-class inference outputs.
    final_logits: str = "final_logits"
    raw_final_score_map: str = "raw_final_score_map"
    final_score_map: str = "final_score_map"
    final_pred: str = "final_pred"

    # Expanded prompt-space outputs used for alias-aware inference and TTA.
    prompt_logits: str = "prompt_logits"
    raw_prompt_score_map: str = "raw_prompt_score_map"

    # Frozen SAM3 reference path, decoded only when inference requests it.
    # sam3_reference_logits is [B, P, H, W]; score maps merge prompts to
    # original classes by pixel-wise maximum.
    sam3_reference_logits: str = "sam3_reference_logits"
    raw_reference_prompt_score_map: str = "raw_reference_prompt_score_map"
    raw_reference_score_map: str = "raw_reference_score_map"

    # RemoteCLIP-guided aggregation outputs
    prompt_conditioned_features_72: str = "prompt_conditioned_features_72"
    aggregated_features_36: str = "aggregated_features_36"
    semantic_state_36: str = "semantic_state_36"
    clip_guidance_36: str = "clip_guidance_36"
    clip_template_scores_36: str = "clip_template_scores_36"
    remoteclip_template_text_features: str = (
        "remoteclip_template_text_features"
    )
    clip_spatial_guidance_features: str = "clip_spatial_guidance_features"

    # Frozen SAM3 teacher logits — per-chunk temporary output during training
    # for distillation only. Inference writes its optional full-prompt result
    # to sam3_reference_logits instead.
    sam3_teacher_logits: str = "sam3_teacher_logits"


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
