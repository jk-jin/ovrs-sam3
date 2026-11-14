from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import runpy
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ..config_dataclasses import (
    AggregationConfig,
    EvaluationConfig,
    ExperimentConfig,
    FreezeConfig,
    RuntimeConfig,
    RemoteCLIPConfig,
    SemanticCriterionConfig,
    TrainingProtocolConfig,
    VisualizerConfig,
)


ROOT_KEYS = {
    "experiment",
    "training",
    "model",
    "loss",
    "train_dataloader",
    "val_dataloader",
    "optimizer",
    "param_scheduler",
    "train_cfg",
    "evaluation",
    "runtime",
    "logging",
    "visualization",
}

TRAIN_CFG_KEYS = {
    "max_iters",
    "log_window_size",
    "grad_clip_norm",
    "save_interval",
    "eval_interval",
    "val_max_iters",
    "monitor",
    "monitor_mode",
    "max_keep_ckpts",
}

SEMANTIC_FINGERPRINT_EXCLUDE = {
    # Run/local fields.
    "runtime.work_dir",
    "runtime.device",
    "logging",
    "visualization",
    "model.compile",
    "experiment.name",
    "experiment.method_name",
    "experiment.kind",
    "experiment.note",
    # Operational training/validation cadence and worker settings.
    "train_cfg.save_interval",
    "train_cfg.eval_interval",
    "train_cfg.log_window_size",
    "train_cfg.max_keep_ckpts",
    "train_cfg.val_max_iters",
    "train_dataloader.num_workers",
    "train_dataloader.persistent_workers",
    "train_dataloader.pin_memory",
    "train_dataloader.prefetch_factor",
    "val_dataloader.num_workers",
    "val_dataloader.persistent_workers",
    "val_dataloader.pin_memory",
    "val_dataloader.prefetch_factor",
    # Data paths identify a machine-local location, not the protocol.
    # Training/evaluation protocol ids carry the dataset identity.
    "train_dataloader.dataset.img_dir",
    "train_dataloader.dataset.ann_dir",
    "val_dataloader.dataset.img_dir",
    "val_dataloader.dataset.ann_dir",
}

SUPPORTED_FINETUNE_MODES = {"frozen", "attention", "transformer", "full"}


def _parse_cfg_value(raw: str):
    text = str(raw).strip()

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def get_by_dot_path(cfg, key: str):
    parts = key.split(".")
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"Invalid dot path: {key!r}")

    obj = cfg
    for part in parts:
        if isinstance(obj, dict):
            if part not in obj:
                raise KeyError(f"Unknown config path {key!r}, missing {part!r}")
            obj = obj[part]
        else:
            if not hasattr(obj, part):
                raise KeyError(f"Unknown config path {key!r}, missing {part!r}")
            obj = getattr(obj, part)
    return obj


def set_by_dot_path(cfg, key: str, value):
    parts = key.split(".")
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"Invalid cfg option key: {key!r}")

    obj = cfg
    for part in parts[:-1]:
        if isinstance(obj, dict):
            if part not in obj:
                raise KeyError(f"Unknown config path: {key!r}, missing {part!r}")
            obj = obj[part]
        else:
            if not hasattr(obj, part):
                raise KeyError(f"Unknown config path: {key!r}, missing {part!r}")
            obj = getattr(obj, part)

    last = parts[-1]
    if isinstance(obj, dict):
        if last not in obj:
            raise KeyError(f"Unknown config key: {key!r}, missing {last!r}")
        obj[last] = value
    else:
        if not hasattr(obj, last):
            raise KeyError(f"Unknown config key: {key!r}, missing {last!r}")
        setattr(obj, last, value)


def apply_cfg_options(cfg, cfg_options):
    if not cfg_options:
        return cfg

    for item in cfg_options:
        if "=" not in item:
            raise ValueError(
                f"Invalid --cfg-options item {item!r}. Expected key=value."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty config key in item {item!r}.")
        set_by_dot_path(cfg, key, _parse_cfg_value(raw_value))

    return cfg


class Config(dict):
    """Small python-config loader with mmseg-like `_base_` support."""

    @classmethod
    def fromfile(cls, filename: str | Path) -> 'Config':
        filename = Path(filename).resolve()
        cfg_dict = cls._load_recursive(filename, stack=())
        return cls(cfg_dict)

    @classmethod
    def _load_recursive(
        cls,
        filename: Path,
        stack: tuple[Path, ...],
    ) -> Dict[str, Any]:
        filename = filename.resolve()
        if filename in stack:
            chain = " -> ".join(str(p) for p in (*stack, filename))
            raise ValueError(f"Config inheritance cycle detected: {chain}")

        namespace = runpy.run_path(str(filename))
        base_files = namespace.pop('_base_', [])
        if isinstance(base_files, str):
            base_files = [base_files]

        merged: Dict[str, Any] = {}
        for base_file in base_files:
            base_path = (filename.parent / base_file).resolve()
            base_cfg = cls._load_recursive(
                base_path, stack=(*stack, filename)
            )
            merged = cls._merge_dicts(merged, base_cfg)

        current_cfg = {
            k: v for k, v in namespace.items()
            if not k.startswith('_')
            and not isinstance(v, types.ModuleType)
        }

        merged = cls._merge_dicts(merged, current_cfg)
        return merged

    @classmethod
    def _merge_dicts(cls, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = cls._merge_dicts(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out


def _dataclass_defaults(cls, **overrides) -> dict:
    obj = cls(**overrides)
    return asdict(obj)


def _deep_fill(target: dict, defaults: dict) -> dict:
    out = copy.deepcopy(target)
    for key, value in defaults.items():
        if key not in out:
            out[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = _deep_fill(out[key], value)
    return out


def _require_dict(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict, got {type(value).__name__}.")
    return value


def _reject_unknown(obj: dict, allowed: set[str], name: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise KeyError(f"Unknown {name} keys: {unknown}")


def _coerce_dataclass(obj, cls, name: str):
    obj = _require_dict(obj, name)
    allowed = set(cls.__dataclass_fields__)
    _reject_unknown(obj, allowed, name)
    try:
        return cls(**obj)
    except TypeError as exc:
        raise TypeError(f"Invalid {name}: {exc}") from exc


def _is_finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_threshold(value, name: str) -> None:
    if value is None:
        return
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be None or a finite value in [0,1].")


def _validate_model_section(model: dict) -> dict:
    allowed = {
        "task_mode",
        "bpe_path",
        "checkpoint_path",
        "load_from_hf",
        "eval_mode",
        "compile",
        "prompt_chunk_size",
        "freeze",
        "remoteclip",
        "aggregation",
    }
    _reject_unknown(model, allowed, "model")

    if str(model.get("task_mode", "")).lower() != "semantic":
        raise ValueError("Only task_mode='semantic' is implemented.")

    for flag in ("load_from_hf", "eval_mode", "compile"):
        if not isinstance(model.get(flag, None), bool):
            raise TypeError(f"model.{flag} must be bool.")

    prompt_chunk_size = model.get("prompt_chunk_size")
    if prompt_chunk_size is not None:
        if (
            isinstance(prompt_chunk_size, bool)
            or not isinstance(prompt_chunk_size, int)
            or prompt_chunk_size <= 0
        ):
            raise ValueError("model.prompt_chunk_size must be None or a positive int.")

    remoteclip = _coerce_dataclass(
        model["remoteclip"], RemoteCLIPConfig, "model.remoteclip"
    )
    if not isinstance(remoteclip.enabled, bool):
        raise TypeError("model.remoteclip.enabled must be bool.")
    if remoteclip.enabled:
        if not remoteclip.pretrained:
            raise ValueError(
                "model.remoteclip.pretrained is required when enabled."
            )
        if len(remoteclip.prompt_templates) != 64:
            raise ValueError(
                "model.remoteclip.prompt_templates must contain 64 templates."
            )
        for idx, template in enumerate(remoteclip.prompt_templates):
            if "{}" not in str(template):
                raise ValueError(
                    "model.remoteclip.prompt_templates"
                    f"[{idx}] must contain '{{}}'."
                )
        if list(remoteclip.image_intermediate_layers) != [7, 15]:
            raise ValueError(
                "model.remoteclip.image_intermediate_layers must be [7, 15] "
                "for fixed CLIP spatial guidance."
            )
        if int(remoteclip.image_size) != 504:
            raise ValueError("model.remoteclip.image_size must be 504.")
        if int(remoteclip.text_prompt_batch_size) <= 0:
            raise ValueError(
                "model.remoteclip.text_prompt_batch_size must be positive."
            )
        if remoteclip.default_output != "feat_map":
            raise ValueError(
                "model.remoteclip.default_output must be 'feat_map'."
            )
        if not isinstance(remoteclip.normalize_label_for_clip, bool):
            raise TypeError(
                "model.remoteclip.normalize_label_for_clip must be bool."
            )
        if not isinstance(remoteclip.text_prompt_use_checkpoint, bool):
            raise TypeError(
                "model.remoteclip.text_prompt_use_checkpoint must be bool."
            )

    aggregation = _coerce_dataclass(
        model["aggregation"], AggregationConfig, "model.aggregation"
    )
    if not aggregation.enabled:
        raise ValueError(
            "model.aggregation.enabled=False is not supported."
        )
    if aggregation.num_blocks <= 0:
        raise ValueError("model.aggregation.num_blocks must be positive.")
    if aggregation.num_heads <= 0:
        raise ValueError("model.aggregation.num_heads must be positive.")
    if not 0.0 <= float(aggregation.dropout) <= 1.0:
        raise ValueError("model.aggregation.dropout must be in [0,1].")
    if not isinstance(aggregation.use_checkpoint, bool):
        raise TypeError("model.aggregation.use_checkpoint must be bool.")
    if aggregation.feature_dim != 256:
        raise ValueError("model.aggregation.feature_dim must be 256.")
    if aggregation.semantic_dim != 256:
        raise ValueError("model.aggregation.semantic_dim must be 256.")
    if aggregation.aggregation_hw != 36:
        raise ValueError("model.aggregation.aggregation_hw must be 36.")
    if aggregation.encoder_hw != 72:
        raise ValueError("model.aggregation.encoder_hw must be 72.")
    if aggregation.aggregation_hw * 2 != aggregation.encoder_hw:
        raise ValueError(
            "model.aggregation.aggregation_hw * 2 must equal encoder_hw."
        )
    if (
        aggregation.window_size <= 0
        or aggregation.aggregation_hw % aggregation.window_size != 0
    ):
        raise ValueError(
            "model.aggregation.window_size must divide aggregation_hw."
        )
    if not 0 <= aggregation.shift_size < aggregation.window_size:
        raise ValueError(
            "model.aggregation.shift_size must satisfy "
            "0 <= shift_size < window_size."
        )

    freeze = _coerce_dataclass(
        model["freeze"], FreezeConfig, "model.freeze"
    )
    for name in (
        "remoteclip_text_finetune",
        "remoteclip_image_finetune",
    ):
        mode = str(getattr(freeze, name)).lower()
        if mode not in SUPPORTED_FINETUNE_MODES:
            raise ValueError(
                f"model.freeze.{name} must be one of "
                f"{sorted(SUPPORTED_FINETUNE_MODES)}, got {mode!r}."
            )
    if not isinstance(freeze.train_adapters_only, bool):
        raise TypeError("model.freeze.train_adapters_only must be bool.")
    for name in ("trainable_modules", "frozen_modules"):
        value = getattr(freeze, name)
        if not isinstance(value, list) or any(
            not isinstance(x, str) for x in value
        ):
            raise TypeError(f"model.freeze.{name} must be a list of strings.")
    if freeze.train_adapters_only and not freeze.trainable_modules:
        raise ValueError(
            "model.freeze.trainable_modules must not be empty when "
            "train_adapters_only=True."
        )

    model["remoteclip"] = asdict(remoteclip)
    model["aggregation"] = asdict(aggregation)
    model["freeze"] = asdict(freeze)
    return model


def _validate_loss_section(loss: dict) -> dict:
    cfg = _coerce_dataclass(
        loss, SemanticCriterionConfig, "loss"
    )

    if (
        isinstance(cfg.ignore_index, bool)
        or not isinstance(cfg.ignore_index, int)
    ):
        raise TypeError("loss.ignore_index must be int.")

    for name in (
        "seg_bce_weight",
        "seg_dice_weight",
        "reference_distill_weight",
    ):
        value = getattr(cfg, name)
        if not _is_finite_number(value) or float(value) < 0.0:
            raise ValueError(f"loss.{name} must be a finite non-negative value.")

    if not _is_finite_number(cfg.eps) or float(cfg.eps) <= 0.0:
        raise ValueError("loss.eps must be a positive finite value.")
    if (
        isinstance(cfg.reference_boundary_width, bool)
        or not isinstance(cfg.reference_boundary_width, int)
        or cfg.reference_boundary_width < 0
    ):
        raise ValueError(
            "loss.reference_boundary_width must be a non-negative int."
        )

    return asdict(cfg)


def _validate_scheduler(scheduler, max_iters: int) -> None:
    if isinstance(scheduler, dict):
        entries = [scheduler]
    elif isinstance(scheduler, (list, tuple)):
        entries = list(scheduler)
    else:
        raise TypeError("param_scheduler must be a dict or a list of dicts.")

    if not entries:
        raise ValueError("param_scheduler must not be empty.")

    if max_iters <= 0:
        return

    ends = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"param_scheduler[{idx}] must be a dict.")
        if "type" not in entry:
            raise ValueError(f"param_scheduler[{idx}] must define 'type'.")
        end = entry.get("end", None)
        if end is None:
            raise ValueError(
                f"param_scheduler[{idx}] must define 'end' so the "
                "iteration plan can be checked against train_cfg.max_iters."
            )
        if int(end) <= 0:
            raise ValueError(f"param_scheduler[{idx}].end must be positive.")
        ends.append(int(end))

    if int(ends[-1]) != int(max_iters):
        raise ValueError(
            "The last param_scheduler entry must end at "
            f"train_cfg.max_iters: end={ends[-1]}, "
            f"max_iters={max_iters}."
        )
    if ends != sorted(ends) or len(set(ends)) != len(ends):
        raise ValueError(
            "param_scheduler ends must be strictly increasing."
        )

    previous_end = 0
    for idx, entry in enumerate(entries):
        duration = int(ends[idx]) - previous_end
        sched_type = str(entry.get("type"))
        if sched_type == "CosineAnnealingLR":
            if int(entry.get("T_max", -1)) != duration:
                raise ValueError(
                    f"param_scheduler[{idx}] CosineAnnealingLR.T_max must "
                    f"equal its active duration {duration}, got "
                    f"{entry.get('T_max')}."
                )
        elif sched_type == "LinearLR":
            if int(entry.get("total_iters", -1)) != duration:
                raise ValueError(
                    f"param_scheduler[{idx}] LinearLR.total_iters must "
                    f"equal its active duration {duration}, got "
                    f"{entry.get('total_iters')}."
                )
        previous_end = int(ends[idx])


def _validate_optimizer(optimizer: dict) -> None:
    if "type" not in optimizer:
        raise ValueError("optimizer.type is required.")
    lr = optimizer.get("lr", None)
    if lr is None or not _is_finite_number(lr) or float(lr) <= 0.0:
        raise ValueError("optimizer.lr must be a positive finite value.")
    wd = optimizer.get("weight_decay", None)
    if wd is not None and (
        not _is_finite_number(wd) or float(wd) < 0.0
    ):
        raise ValueError("optimizer.weight_decay must be non-negative.")


def _validate_evaluation(evaluation: dict, loss: dict) -> dict:
    cfg = _coerce_dataclass(
        evaluation, EvaluationConfig, "evaluation"
    )

    _validate_threshold(
        cfg.background_prob_threshold,
        "evaluation.background_prob_threshold",
    )
    if not isinstance(cfg.metric_groups, list):
        raise TypeError("evaluation.metric_groups must be a list.")

    postprocess = _require_dict(
        cfg.postprocess, "evaluation.postprocess"
    )
    _reject_unknown(
        postprocess,
        {"prediction_prob_threshold"},
        "evaluation.postprocess",
    )
    _validate_threshold(
        postprocess.get("prediction_prob_threshold"),
        "evaluation.postprocess.prediction_prob_threshold",
    )

    tta = _require_dict(cfg.tta, "evaluation.tta")
    _reject_unknown(
        tta,
        {"enabled", "scales", "flip_modes", "size_divisor"},
        "evaluation.tta",
    )
    scales = [float(x) for x in tta.get("scales", [1.0])]
    if any(abs(x - 1.0) > 1e-8 for x in scales):
        raise ValueError(
            "evaluation.tta.scales only supports 1.0 with the current "
            "fixed 1008×1008 pipeline."
        )
    flip_modes = list(tta.get("flip_modes", ["none"]))
    allowed_flips = {"none", "h", "v", "hv"}
    unknown_flips = sorted(set(flip_modes) - allowed_flips)
    if unknown_flips:
        raise ValueError(
            f"evaluation.tta.flip_modes contains unsupported modes: "
            f"{unknown_flips}."
        )
    if int(tta.get("size_divisor", 14)) <= 0:
        raise ValueError("evaluation.tta.size_divisor must be positive.")
    if not isinstance(tta.get("enabled", False), bool):
        raise TypeError("evaluation.tta.enabled must be bool.")

    geometry = _require_dict(cfg.geometry, "evaluation.geometry")
    _reject_unknown(
        geometry, {"input_hw", "keep_ratio"}, "evaluation.geometry"
    )
    input_hw = geometry.get("input_hw", [1008, 1008])
    if (
        not isinstance(input_hw, (list, tuple))
        or len(input_hw) != 2
        or any(int(x) <= 0 for x in input_hw)
    ):
        raise ValueError(
            "evaluation.geometry.input_hw must be two positive integers."
        )
    if not isinstance(geometry.get("keep_ratio", None), bool):
        raise TypeError("evaluation.geometry.keep_ratio must be bool.")

    if int(cfg.ignore_index) != int(loss["ignore_index"]):
        raise ValueError(
            "evaluation.ignore_index must equal loss.ignore_index."
        )

    return asdict(cfg)


def _validate_logging(logging: dict) -> dict:
    logging = _require_dict(logging, "logging")
    _reject_unknown(logging, {"console", "jsonl", "wandb"}, "logging")

    console = _require_dict(logging["console"], "logging.console")
    _reject_unknown(
        console,
        {
            "enabled",
            "interval",
            "val_interval",
            "print_metric_tables",
            "print_per_class_metrics",
        },
        "logging.console",
    )
    if not isinstance(console.get("enabled", None), bool):
        raise TypeError("logging.console.enabled must be bool.")
    for name in ("interval", "val_interval"):
        if int(console.get(name, 0)) < 0:
            raise ValueError(f"logging.console.{name} must be >= 0.")
    for name in ("print_metric_tables", "print_per_class_metrics"):
        if not isinstance(console.get(name, None), bool):
            raise TypeError(f"logging.console.{name} must be bool.")

    jsonl = _require_dict(logging["jsonl"], "logging.jsonl")
    _reject_unknown(
        jsonl,
        {"enabled", "filename", "train_interval"},
        "logging.jsonl",
    )
    if not isinstance(jsonl.get("enabled", None), bool):
        raise TypeError("logging.jsonl.enabled must be bool.")
    if not str(jsonl.get("filename", "")).strip():
        raise ValueError("logging.jsonl.filename must not be empty.")
    if int(jsonl.get("train_interval", 0)) < 0:
        raise ValueError("logging.jsonl.train_interval must be >= 0.")

    wandb = _require_dict(logging["wandb"], "logging.wandb")
    _reject_unknown(
        wandb,
        {
            "enabled",
            "mode",
            "project",
            "name",
            "group",
            "tags",
            "train_interval",
            "log_per_class",
            "log_extra",
        },
        "logging.wandb",
    )
    mode = str(wandb.get("mode", "offline")).lower()
    if mode not in {"offline", "online"}:
        raise ValueError(
            "logging.wandb.mode must be 'offline' or 'online'."
        )
    if not str(wandb.get("project", "")).strip():
        raise ValueError("logging.wandb.project must not be empty.")
    for name in ("enabled", "log_per_class", "log_extra"):
        if not isinstance(wandb.get(name, None), bool):
            raise TypeError(f"logging.wandb.{name} must be bool.")
    for name in ("name", "group"):
        if wandb.get(name, None) is not None and not isinstance(
            wandb.get(name), str
        ):
            raise TypeError(f"logging.wandb.{name} must be null or string.")
    if int(wandb.get("train_interval", 0)) < 0:
        raise ValueError("logging.wandb.train_interval must be >= 0.")
    if not isinstance(wandb.get("tags", []), list):
        raise TypeError("logging.wandb.tags must be a list.")

    return copy.deepcopy(logging)


def _validate_visualization(visualization: dict) -> dict:
    cfg = _coerce_dataclass(
        visualization, VisualizerConfig, "visualization"
    )
    if not 0.0 <= float(cfg.alpha) <= 1.0:
        raise ValueError("visualization.alpha must be in [0,1].")
    if not 0.0 <= float(cfg.vis_prob) <= 1.0:
        raise ValueError("visualization.vis_prob must be in [0,1].")
    if isinstance(cfg.vis_seed, bool) or not isinstance(cfg.vis_seed, int):
        raise TypeError("visualization.vis_seed must be int.")
    if isinstance(cfg.ignore_index, bool) or not isinstance(
        cfg.ignore_index, int
    ):
        raise TypeError("visualization.ignore_index must be int.")
    if cfg.max_samples_per_epoch is not None and int(
        cfg.max_samples_per_epoch
    ) < 0:
        raise ValueError(
            "visualization.max_samples_per_epoch must be None or >= 0."
        )
    return asdict(cfg)


def _validate_experiment(experiment: dict) -> dict:
    cfg = _coerce_dataclass(
        experiment, ExperimentConfig, "experiment"
    )
    for name in (
        "name",
        "method_id",
        "method_name",
        "variant_id",
        "implementation_id",
        "kind",
    ):
        if not str(getattr(cfg, name)).strip():
            raise ValueError(f"experiment.{name} must not be empty.")
    if not isinstance(cfg.factors, dict):
        raise TypeError("experiment.factors must be a dict.")
    for key, value in cfg.factors.items():
        if not (
            value is None
            or isinstance(value, (bool, int, float, str))
        ):
            raise TypeError(
                "experiment.factors values must be bool/int/float/string/null."
            )
    return asdict(cfg)


def _validate_training(training: dict) -> dict:
    cfg = _coerce_dataclass(
        training, TrainingProtocolConfig, "training"
    )
    if cfg.regime not in {
        "supervised_adaptation",
        "training_free",
        "unknown",
    }:
        raise ValueError(
            "training.regime must be supervised_adaptation, "
            "training_free, or unknown."
        )
    return asdict(cfg)


def _validate_runtime(runtime: dict) -> dict:
    cfg = _coerce_dataclass(runtime, RuntimeConfig, "runtime")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int):
        raise TypeError("runtime.seed must be an int.")
    if not str(cfg.device).strip():
        raise ValueError("runtime.device must not be empty.")
    if not isinstance(cfg.use_amp, bool):
        raise TypeError("runtime.use_amp must be bool.")
    if not str(cfg.work_dir).strip():
        raise ValueError("runtime.work_dir must not be empty.")
    return asdict(cfg)


def _validate_train_cfg(
    train_cfg: dict,
    monitor_default: str,
) -> dict:
    _reject_unknown(train_cfg, TRAIN_CFG_KEYS, "train_cfg")

    max_iters = train_cfg.get("max_iters")
    if (
        isinstance(max_iters, bool)
        or not isinstance(max_iters, int)
        or max_iters < 0
    ):
        raise ValueError("train_cfg.max_iters must be a non-negative int.")

    for name in ("log_window_size", "save_interval", "eval_interval"):
        value = train_cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"train_cfg.{name} must be a non-negative int.")
    if train_cfg["log_window_size"] <= 0:
        raise ValueError("train_cfg.log_window_size must be positive.")

    val_max_iters = train_cfg.get("val_max_iters")
    if val_max_iters is not None and (
        isinstance(val_max_iters, bool)
        or not isinstance(val_max_iters, int)
        or val_max_iters < 0
    ):
        raise ValueError(
            "train_cfg.val_max_iters must be None or a non-negative int."
        )

    monitor = str(train_cfg.get("monitor", monitor_default))
    if not monitor:
        raise ValueError("train_cfg.monitor must not be empty.")
    mode = str(train_cfg.get("monitor_mode", "max")).lower()
    if mode not in {"max", "min"}:
        raise ValueError("train_cfg.monitor_mode must be 'max' or 'min'.")

    max_keep = train_cfg.get("max_keep_ckpts")
    if (
        isinstance(max_keep, bool)
        or not isinstance(max_keep, int)
        or max_keep < 0
    ):
        raise ValueError(
            "train_cfg.max_keep_ckpts must be a non-negative int."
        )

    grad_clip = train_cfg.get("grad_clip_norm")
    if grad_clip is not None and (
        not _is_finite_number(grad_clip) or float(grad_clip) <= 0.0
    ):
        raise ValueError(
            "train_cfg.grad_clip_norm must be None or a positive value."
        )

    train_cfg["monitor"] = monitor
    train_cfg["monitor_mode"] = mode
    return copy.deepcopy(train_cfg)


def _default_sections() -> dict:
    return {
        "experiment": _dataclass_defaults(ExperimentConfig),
        "training": _dataclass_defaults(TrainingProtocolConfig),
        "model": {
            "task_mode": "semantic",
            "bpe_path": None,
            "checkpoint_path": None,
            "load_from_hf": True,
            "eval_mode": True,
            "compile": False,
            "prompt_chunk_size": None,
            "freeze": _dataclass_defaults(FreezeConfig),
            "remoteclip": _dataclass_defaults(RemoteCLIPConfig),
            "aggregation": _dataclass_defaults(AggregationConfig),
        },
        "loss": _dataclass_defaults(SemanticCriterionConfig),
        "train_cfg": {
            "max_iters": 10000,
            "log_window_size": 20,
            "grad_clip_norm": 0.1,
            "save_interval": 1000,
            "eval_interval": 1000,
            "val_max_iters": None,
            "monitor": "semantic.miou",
            "monitor_mode": "max",
            "max_keep_ckpts": 5,
        },
        "evaluation": _dataclass_defaults(EvaluationConfig),
        "runtime": _dataclass_defaults(RuntimeConfig),
        "logging": {
            "console": {
                "enabled": True,
                "interval": 20,
                "val_interval": 50,
                "print_metric_tables": True,
                "print_per_class_metrics": True,
            },
            "jsonl": {
                "enabled": True,
                "filename": "metrics.jsonl",
                "train_interval": 20,
            },
            "wandb": {
                "enabled": True,
                "mode": "offline",
                "project": "reca-sam3",
                "name": None,
                "group": None,
                "tags": [],
                "train_interval": 20,
                "log_per_class": False,
                "log_extra": False,
            },
        },
        "visualization": _dataclass_defaults(VisualizerConfig),
    }


def finalize_config(raw_cfg: dict) -> dict:
    """Fill defaults, validate, and expand the effective config."""
    _reject_unknown(raw_cfg, ROOT_KEYS, "config root")

    cfg = _deep_fill(raw_cfg, _default_sections())

    cfg["experiment"] = _validate_experiment(cfg["experiment"])
    cfg["training"] = _validate_training(cfg["training"])
    cfg["model"] = _validate_model_section(cfg["model"])
    cfg["loss"] = _validate_loss_section(cfg["loss"])
    cfg["evaluation"] = _validate_evaluation(
        cfg["evaluation"], cfg["loss"]
    )
    cfg["runtime"] = _validate_runtime(cfg["runtime"])
    cfg["logging"] = _validate_logging(cfg["logging"])
    cfg["visualization"] = _validate_visualization(
        cfg["visualization"]
    )
    cfg["train_cfg"] = _validate_train_cfg(
        cfg["train_cfg"],
        monitor_default="semantic.miou",
    )

    if "val_dataloader" not in cfg:
        raise ValueError("val_dataloader is required.")

    if "optimizer" in cfg:
        optimizer = _require_dict(cfg["optimizer"], "optimizer")
        _validate_optimizer(optimizer)
    if "param_scheduler" in cfg:
        _validate_scheduler(
            cfg["param_scheduler"],
            int(cfg["train_cfg"]["max_iters"]),
        )

    return cfg


def config_fingerprint(cfg: dict) -> str:
    semantic = copy.deepcopy(cfg)
    for path in SEMANTIC_FINGERPRINT_EXCLUDE:
        parts = path.split(".")
        obj = semantic
        for part in parts[:-1]:
            if not isinstance(obj, dict) or part not in obj:
                obj = None
                break
            obj = obj[part]
        if isinstance(obj, dict):
            obj.pop(parts[-1], None)

    text = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_effective_config(
    path: str | Path,
    cfg_options: Optional[list[str]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> tuple[dict, str]:
    """Load, apply overrides, finalize, and return (cfg, fingerprint).

    Precedence: inherited config → --cfg-options → dedicated CLI options.
    """
    cfg = Config.fromfile(path)
    # Fill defaults before applying overrides so dedicated CLI and
    # --cfg-options can target fields that the entry file omitted.
    cfg = _deep_fill(cfg, _default_sections())
    cfg = apply_cfg_options(cfg, cfg_options)

    if seed is not None:
        cfg.setdefault("runtime", {})["seed"] = int(seed)
    if device is not None:
        cfg.setdefault("runtime", {})["device"] = str(device)
    if work_dir is not None:
        cfg.setdefault("runtime", {})["work_dir"] = str(work_dir)

    cfg = finalize_config(cfg)
    return cfg, config_fingerprint(cfg)


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj
