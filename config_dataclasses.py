from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FreezeConfig:
    train_adapters_only: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)

    # "frozen", "attention", "full"
    openclip_text_finetune: str = "frozen"

    # "frozen", "attention", "full"
    openclip_image_finetune: str = "frozen"


@dataclass
class OpenCLIPConfig:
    enabled: bool = False
    model_name: str = "ViT-L-14"
    pretrained: Optional[str] = None
    default_output: str = "feat_map"

    image_intermediate_layers: list[int] = field(default_factory=lambda: [7, 15])

    prompt_templates: list[str] = field(default_factory=lambda: [
        "a remote sensing image of {}.",
        "a satellite image of {}.",
        "an aerial image of {}.",
        "an overhead image of {}.",
        "a high resolution satellite image of {}.",
        "a high resolution aerial image of {}.",
        "a top down view of {}.",
        "a bird's eye view of {}.",
        "a remote sensing scene containing {}.",
        "a satellite scene containing {}.",
        "an aerial scene containing {}.",
        "an overhead scene containing {}.",
        "a remote sensing image showing {}.",
        "a satellite image showing {}.",
        "an aerial image showing {}.",
        "an overhead image showing {}.",
        "a segmented region of {} in a remote sensing image.",
        "a semantic segmentation mask of {} in a satellite image.",
        "a land cover region of {}.",
        "a land use region of {}.",
        "a large area of {} in an aerial image.",
        "a small area of {} in a satellite image.",
        "dense {} in a remote sensing image.",
        "sparse {} in a remote sensing image.",
        "the boundary of {} in a satellite image.",
        "the texture of {} in an aerial image.",
        "the shape of {} from an overhead view.",
        "the pattern of {} in a remote sensing scene.",
        "{} in urban remote sensing imagery.",
        "{} in rural remote sensing imagery.",
        "{} on the ground surface from above.",
        "{} visible from satellite imagery.",
    ])

    normalize_label_for_clip: bool = True


@dataclass
class TemplateGuidedRefinerConfig:
    enabled: bool = True

    hidden_dim: int = 256
    num_prompt_templates: int = 32

    lowres_hw: int = 18
    lowres_layers: int = 4

    highres_hw: int = 72
    highres_layers: int = 2

    num_heads: int = 8
    dropout: float = 0.1

    window_size: int = 9
    shift_size: int = 4

    use_checkpoint: bool = True
    early_prompt_attention: bool = False


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255

    final_bce_weight: float = 1.0
    final_dice_weight: float = 0.0

    bce_absent_class_weight: float = 0.0

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
    template_guided_refiner_cfg: TemplateGuidedRefinerConfig = field(
        default_factory=TemplateGuidedRefinerConfig
    )
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