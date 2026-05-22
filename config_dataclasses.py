from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FreezeConfig:
    train_adapters_only: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)


@dataclass
class OpenCLIPConfig:
    enabled: bool = False
    model_name: str = "ViT-L-14"
    pretrained: Optional[str] = None
    default_output: str = "feat_map"

    image_encoder_mode: str = "maskclip"
    maskclip_skip_last_layers: int = 1

    prompt_templates: list[str] = field(
        default_factory=lambda: [
            "a remote sensing image of {}.",
            "an aerial image of {}.",
        ]
    )
    num_prompt_templates: int = 2
    num_clip_text_latents: int = 32

    normalize_label_for_clip: bool = True


@dataclass
class ClipSamUpsampleConfig:
    enabled: bool = True
    window_size: int = 8
    shift_size: int = 4
    dropout: float = 0.1
    gamma_init: float = 1.0
    gamma_max: float = 2.0


@dataclass
class DynamicCodeConfig:
    source: str = "class_token_to_sam3_text"


@dataclass
class MaskPriorConfig:
    type: str = "softmax"
    tau: float = 16.0
    multiply_presence: bool = True


@dataclass
class WindowAttentionConfig:
    window_size: int = 8
    shift_size: int = 4
    dropout: float = 0.1


@dataclass
class MaskHeadConfig:
    type: str = "attn_feature_dot_dynamic_code"
    direct_dot: bool = True
    class_feature_pool_stride: int = 4


@dataclass
class FinalMixerConfig:
    enabled: bool = True

    num_class_tokens: int = 32
    fusion_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    presence_enabled: bool = True

    clip_sam_upsample_cfg: ClipSamUpsampleConfig = field(
        default_factory=ClipSamUpsampleConfig
    )
    dynamic_code_cfg: DynamicCodeConfig = field(default_factory=DynamicCodeConfig)
    mask_prior_cfg: MaskPriorConfig = field(default_factory=MaskPriorConfig)
    window_attention_cfg: WindowAttentionConfig = field(
        default_factory=WindowAttentionConfig
    )
    mask_head_cfg: MaskHeadConfig = field(default_factory=MaskHeadConfig)


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255

    final_bce_weight: float = 0.4
    final_dice_weight: float = 1.0
    final_ce_weight: float = 0.4
    final_ignore_bce_weight: float = 0.1

    presence_loss_weight: float = 1.0
    presence_layer_loss_weights: Optional[list[float]] = field(
        default_factory=lambda: [0.02, 0.05, 0.1, 0.2]
    )

    mask_layer_loss_weight: float = 1.0
    mask_layer_weights: Optional[list[float]] = field(
        default_factory=lambda: [0.1, 0.2, 0.4]
    )

    bce_class_balance_clamp_min: float = 0.2
    bce_class_balance_clamp_max: float = 5.0

    ce_class_balance_clamp_min: float = 0.2
    ce_class_balance_clamp_max: float = 5.0

    eps: float = 1e-6


@dataclass
class AdapterConfig:
    pass


@dataclass
class SegmentorBuildConfig:
    task_mode: str = "semantic"

    bpe_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_from_hf: bool = True
    device: str = "cuda"
    eval_mode: bool = True
    compile: bool = False

    prompt_chunk_size: Optional[int] = None

    freeze_cfg: FreezeConfig = field(default_factory=FreezeConfig)
    openclip_cfg: OpenCLIPConfig = field(default_factory=OpenCLIPConfig)
    final_mixer_cfg: FinalMixerConfig = field(default_factory=FinalMixerConfig)
    criterion_cfg: SemanticCriterionConfig = field(
        default_factory=SemanticCriterionConfig
    )
    adapter_cfg: AdapterConfig = field(default_factory=AdapterConfig)


@dataclass
class TrainerConfig:
    max_iters: int = 10000
    log_window_size: int = 20
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 0.1

    save_dir: str = "./work_dirs/default"
    save_interval: int = 1000
    eval_interval: int = 1000

    monitor: str = "semantic.miou"
    monitor_mode: str = "max"
    max_keep_ckpts: int = 5

    device: str = "cuda"
    auto_resume: bool = False

    tta_cfg: Optional[Dict] = None
    eval_cfg: Optional[Dict] = None


@dataclass
class CheckpointManagerConfig:
    save_dir: str
    monitor: str = "total_loss"
    mode: str = "min"
    max_keep: int = 5
    save_latest: bool = True
    save_best: bool = True

@dataclass
class LoggerHookConfig:
    interval: int = 20
    val_interval: int = 50
    print_metric_tables: bool = True
    print_per_class_metrics: bool = True
    priority: int = 70

@dataclass
class VisualizerConfig:
    enabled: bool = False
    save_dir: str = "./visualizations"
    save_stage: str = "val"
    alpha: float = 0.45

    save_original: bool = True
    save_prediction: bool = True
    save_ground_truth: bool = True
    save_semantic_prediction: bool = True

    save_score_summary: bool = True
    save_score_heatmaps: bool = True
    heatmap_colormap: str = "turbo"

    save_clip_argmax_prediction: bool = True

    save_sam3_direct_segmentation: bool = True
    sam3_direct_seg_threshold: float = 0.5

    save_presence_scores: bool = True
    save_presence_layers: bool = True

    save_final_mixer_mask_layers: bool = True
    save_final_mixer_layer_heatmaps: bool = True
    save_final_mixer_layer_predictions: bool = True
    save_final_mixer_layer_overlays: bool = True
    max_final_mixer_layer_heatmap_classes: Optional[int] = None

    vis_prob: float = 0.05
    max_samples_per_epoch: Optional[int] = 50
    vis_seed: int = 42

    image_folder_pattern: str = "image_{image_id:06d}"
    ignore_index: int = 255

    save_modulated_semantic_prediction: bool = False
    save_delta_heatmaps: bool = False
    save_modulated_semantic_heatmaps: bool = False