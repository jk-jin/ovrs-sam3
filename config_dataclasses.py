from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .data.prompt_templates import REMOTECLIP_PROMPT_TEMPLATES


@dataclass
class FreezeConfig:
    train_adapters_only: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)

    # "frozen" | "attention" | "transformer" | "full"
    remoteclip_text_finetune: str = "frozen"

    # "frozen" | "attention" | "transformer" | "full"
    remoteclip_image_finetune: str = "frozen"


@dataclass
class RemoteCLIPConfig:
    enabled: bool = False
    model_name: str = "ViT-L-14"
    pretrained: Optional[str] = None
    default_output: str = "feat_map"
    image_size: int = 504

    image_intermediate_layers: list[int] = field(
        default_factory=lambda: [7, 15]
    )

    prompt_templates: list[str] = field(
        default_factory=lambda: list(REMOTECLIP_PROMPT_TEMPLATES)
    )
    normalize_label_for_clip: bool = True

    text_prompt_batch_size: int = 64
    text_prompt_use_checkpoint: bool = True


@dataclass
class AggregationConfig:
    enabled: bool = True

    num_blocks: int = 4  # Total blocks: cross → intra → cross → intra.
    num_heads: int = 8
    dropout: float = 0.1

    feature_dim: int = 256
    semantic_dim: int = 256

    aggregation_hw: int = 36
    encoder_hw: int = 72

    window_size: int = 12
    shift_size: int = 6

    use_checkpoint: bool = True


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255

    seg_bce_weight: float = 1.0
    seg_dice_weight: float = 0.0

    eps: float = 1e-6

    reference_distill_weight: float = 0.5
    reference_boundary_width: int = 3

    present_ignore_suppression_weight: float = 0.01


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

    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    remoteclip: RemoteCLIPConfig = field(default_factory=RemoteCLIPConfig)
    aggregation: AggregationConfig = field(
        default_factory=AggregationConfig
    )

    reference_prob_threshold: Optional[float] = 0.1
    prediction_prob_threshold: Optional[float] = 0.5


@dataclass
class TrainerConfig:
    max_iters: int = 10000
    log_window_size: int = 20
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 0.01

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

    tta_cfg: Optional[Dict] = None
    eval_cfg: Optional[Dict] = None


@dataclass
class CheckpointManagerConfig:
    save_dir: str
    monitor: str = "total_loss"
    mode: str = "min"
    max_keep: int = 5


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
    priority: int = 80


@dataclass
class WandbHookConfig:
    enabled: bool = True
    project: str = "reca-sam3"
    name: Optional[str] = None
    group: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    mode: str = "offline"
    train_interval: int = 20
    log_per_class: bool = False
    log_extra: bool = False
    priority: int = 90


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

    save_reference_segmentation: bool = False
    reference_seg_threshold: float = 0.5

    vis_prob: float = 0.05
    max_samples_per_epoch: Optional[int] = 50
    vis_seed: int = 42

    image_folder_pattern: str = "image_{image_id:06d}"
    ignore_index: int = 255


@dataclass
class ExperimentConfig:
    name: str = "reca_sam3"
    method_id: str = "reca_sam3"
    method_name: str = "ReCA-SAM3"
    variant_id: str = "full"
    implementation_id: str = "reca-sam3-paper-v1"
    kind: str = "main"
    factors: Dict = field(default_factory=dict)
    note: Optional[str] = None


@dataclass
class TrainingProtocolConfig:
    regime: str = "supervised_adaptation"
    dataset: Optional[str] = None
    split: Optional[str] = None
    protocol_id: Optional[str] = None


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    work_dir: str = "./work_dirs/reca_sam3"


@dataclass
class ConsoleLoggingConfig:
    enabled: bool = True
    interval: int = 20
    val_interval: int = 50
    print_metric_tables: bool = True
    print_per_class_metrics: bool = True


@dataclass
class LoggingConfig:
    console: ConsoleLoggingConfig = field(
        default_factory=ConsoleLoggingConfig
    )
    jsonl: MetricsJsonlHookConfig = field(
        default_factory=MetricsJsonlHookConfig
    )
    wandb: WandbHookConfig = field(default_factory=WandbHookConfig)


@dataclass
class EvaluationConfig:
    dataset: Optional[str] = None
    split: Optional[str] = None
    data_version: Optional[str] = None
    protocol_id: Optional[str] = None

    ignore_index: int = 255
    background_prob_threshold: Optional[float] = 0.0
    metric_groups: list[dict] = field(default_factory=list)

    geometry: Dict = field(
        default_factory=lambda: {
            "input_hw": [1008, 1008],
            "keep_ratio": False,
        }
    )
    tta: Dict = field(
        default_factory=lambda: {
            "enabled": False,
            "scales": [1.0],
            "flip_modes": ["none", "h", "v"],
            "size_divisor": 14,
        }
    )
    postprocess: Dict = field(
        default_factory=lambda: {
            "reference_prob_threshold": 0.1,
            "prediction_prob_threshold": 0.5,
        }
    )
