from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FreezeConfig:
    train_adapters_only: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)

    # "frozen" | "attention" | "transformer" | "full"
    openclip_text_finetune: str = "frozen"

    # "frozen" | "attention" | "transformer" | "full"
    openclip_image_finetune: str = "frozen"


@dataclass
class OpenCLIPConfig:
    enabled: bool = False
    model_name: str = "ViT-L-14"
    pretrained: Optional[str] = None
    default_output: str = "feat_map"
    image_size: int = 504

    image_intermediate_layers: list[int] = field(default_factory=lambda: [7, 15])

    prompt_templates: list[str] = field(default_factory=lambda: [
        "a remote sensing image of {}.",
        "a satellite image of {}.",
        "an aerial image of {}.",
        "a high-resolution overhead image of {}.",
        "a top-down view of {}.",
        "a bird's-eye view image of {}.",
        "a remote sensing scene containing {}.",
        "a satellite scene containing {}.",
        "an aerial scene containing {}.",
        "a high-resolution remote sensing scene of {}.",
        "a land cover region of {} in a satellite image.",
        "a land use area of {} in an aerial image.",
        "a semantic segmentation region of {}.",
        "a labeled mask region corresponding to {}.",
        "a continuous area of {} in overhead imagery.",
        "a visible region of {} from above.",
        "the texture pattern of {} in a satellite image.",
        "the spatial pattern of {} in remote sensing imagery.",
        "the shape and boundary of {} in an aerial image.",
        "the object boundary of {} from an overhead view.",
        "a small-scale remote sensing object of {}.",
        "a large-scale remote sensing region of {}.",
        "multiple instances of {} in a satellite image.",
        "dense objects of {} in overhead imagery.",
        "sparse objects of {} in remote sensing imagery.",
        "urban remote sensing imagery showing {}.",
        "rural remote sensing imagery showing {}.",
        "natural land surface containing {}.",
        "man-made structures containing {}.",
        "a homogeneous area of {}.",
        "a complex background with {}.",
        "an object or region classified as {} in remote sensing imagery.",
    ])
    normalize_label_for_clip: bool = True


@dataclass
class EncoderRefinerConfig:
    enabled: bool = True

    fusion_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    hidden_dim: int = 256

    score_embed_dim: int = 256
    clip_score_embed_dim: int = 192
    sam_score_embed_dim: int = 64
    clip_score_conv_kernel: int = 7

    refiner_hw: int = 36
    encoder_hw: int = 72

    window_size: int = 12
    shift_size: int = 6

    use_checkpoint: bool = True
    early_prompt_attention: bool = False


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255

    final_bce_weight: float = 1.0
    final_dice_weight: float = 0.0

    # 0.0 = absent classes not supervised for mask BCE.
    # Set to 0.01 / 0.05 for mild absent-class suppression.
    bce_absent_class_weight: float = 0.0

    # Pixel-level BCE weights.
    # valid pixels: label_map != ignore_index
    # ignore pixels: label_map == ignore_index
    bce_valid_pixel_weight: float = 1.0
    bce_ignore_pixel_weight: float = 1.0

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
    encoder_refiner_cfg: EncoderRefinerConfig = field(default_factory=EncoderRefinerConfig)
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

    # Maximum number of validation batches per validation call.
    # None or <=0 means full validation.
    val_max_iters: Optional[int] = None

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
class MetricsJsonlHookConfig:
    enabled: bool = True
    filename: str = "metrics.jsonl"
    train_interval: int = 20
    val_interval: int = 1  # reserved; currently only after_val is recorded, not after_val_iter
    priority: int = 80


@dataclass
class WandbHookConfig:
    enabled: bool = False
    project: str = "ovrs-sam3"
    name: Optional[str] = None
    group: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    train_interval: int = 20
    log_val_iter: bool = False  # reserved; per-batch val logging is skipped when False
    priority: int = 90

    name_from_config_keys: list[str] = field(default_factory=list)
    name_prefix: Optional[str] = None

@dataclass
class VisualizerConfig:
    enabled: bool = False
    save_dir: str = "./visualizations"
    save_stage: str = "val"
    alpha: float = 0.45

    save_original: bool = True
    save_prediction: bool = True
    save_raw_final_prediction: bool = True
    save_ground_truth: bool = True
    save_semantic_prediction: bool = True

    save_score_summary: bool = True
    save_score_heatmaps: bool = True
    heatmap_colormap: str = "turbo"

    save_sam3_direct_segmentation: bool = False
    sam3_direct_seg_threshold: float = 0.5

    vis_prob: float = 0.05
    max_samples_per_epoch: Optional[int] = 50
    vis_seed: int = 42

    image_folder_pattern: str = "image_{image_id:06d}"
    ignore_index: int = 255