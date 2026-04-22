from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    from importlib import import_module
    import types

    package_name = "_ovrs_sam3_localpkg"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[package_name] = pkg

    build_dataloader = import_module(f"{package_name}.data.build").build_dataloader
    Config = import_module(f"{package_name}.engine.config").Config
    _opt_mod = import_module(f"{package_name}.engine.optimizer_builder")
    build_optimizer = _opt_mod.build_optimizer
    build_scheduler = _opt_mod.build_scheduler
    _trainer_mod = import_module(f"{package_name}.engine.trainer")
    Trainer = _trainer_mod.Trainer
    TrainerConfig = _trainer_mod.TrainerConfig
    _hooks_mod = import_module(f"{package_name}.engine.hooks")
    LoggerHook = _hooks_mod.LoggerHook
    _vis_mod = import_module(f"{package_name}.engine.visualization")
    VisualizationManager = _vis_mod.VisualizationManager
    _builder_mod = import_module(f"{package_name}.model_builder")
    FreezeConfig = _builder_mod.FreezeConfig
    build_training_components = _builder_mod.build_training_components
else:
    from ..data.build import build_dataloader
    from ..engine.config import Config
    from ..engine.optimizer_builder import build_optimizer, build_scheduler
    from ..engine.trainer import Trainer, TrainerConfig
    from ..engine.visualization import VisualizationManager
    from ..model_builder import FreezeConfig, build_training_components


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_dotdict(obj: Any):
    if isinstance(obj, dict):
        return _DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(x) for x in obj]
    return obj


def build_hooks(cfg) -> List[object]:
    logger_cfg = cfg.default_hooks["logger"]
    return [
        LoggerHook(
            interval=int(logger_cfg["interval"]),
            val_interval=int(logger_cfg["val_interval"]),
            print_metric_tables=bool(logger_cfg.get("print_metric_tables", True)),
            print_per_class_metrics=bool(logger_cfg.get("print_per_class_metrics", True)),
        )
    ]


def build_log_getters(cfg) -> List[object]:
    def project_log_getter(trainer):
        out = {}

        model = trainer.model
        core = getattr(model, "core", None)
        if core is None:
            return out

        if hasattr(core, "presence_query_proj"):
            weight = core.presence_query_proj.weight.detach()
            out["presence_query_proj_weight_abs_mean"] = float(weight.abs().mean().item())

        if hasattr(core, "presence_head") and isinstance(core.presence_head, torch.nn.Sequential):
            last_linear = None
            for module in reversed(core.presence_head):
                if isinstance(module, torch.nn.Linear):
                    last_linear = module
                    break
            if last_linear is not None:
                out["presence_head_last_weight_abs_mean"] = float(
                    last_linear.weight.detach().abs().mean().item()
                )

        return out

    return [project_log_getter]


def _unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]

    if all(isinstance(k, str) for k in obj.keys()):
        return obj

    raise ValueError("Cannot find a valid state_dict in the checkpoint.")


def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_model_weights_only(
    model: torch.nn.Module,
    path: str,
    strict: bool = False,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = _unwrap_state_dict(ckpt)
    state_dict = _strip_prefix_if_present(state_dict, "module.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded model weights from {path}")
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train/Eval SAM3 semantic segmentor with iter-based training."
    )
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--load-model-from", type=str, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true", help="only run validation")
    parser.add_argument(
        "--eval-iter",
        type=int,
        default=0,
        help="iter id used in eval-only outputs/logging",
    )
    args = parser.parse_args()

    if args.resume_from is not None and args.load_model_from is not None:
        raise ValueError("--resume-from and --load-model-from cannot be used together.")

    cfg = Config.fromfile(args.config)
    cfg = _to_dotdict(cfg)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    model_cfg = dict(cfg.model)
    freeze_cfg = FreezeConfig(**model_cfg.pop("freeze_cfg", {}))

    model, criterion = build_training_components(
        **model_cfg,
        freeze_cfg=freeze_cfg,
    )

    if args.load_model_from is not None:
        load_model_weights_only(model=model, path=args.load_model_from, strict=False)

    work_dir = args.work_dir or cfg.get("work_dir", "./work_dirs/default")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    visualizer = VisualizationManager.from_cfg(cfg.get("visualization"), work_dir=work_dir)

    trainer_cfg = TrainerConfig(
        max_iters=int(cfg.train_cfg.max_iters),
        log_window_size=int(cfg.train_cfg.get("log_window_size", 20)),
        use_amp=bool(cfg.train_cfg.get("use_amp", True)),
        grad_clip_norm=cfg.train_cfg.get("grad_clip_norm", 0.1),
        save_dir=str(work_dir),
        save_interval=int(cfg.train_cfg.get("save_interval", 1000)),
        eval_interval=int(cfg.train_cfg.get("eval_interval", 1000)),
        monitor=str(cfg.train_cfg.get("monitor", "semantic.miou")),
        monitor_mode=str(cfg.train_cfg.get("monitor_mode", "max")),
        max_keep_ckpts=int(cfg.train_cfg.get("max_keep_ckpts", 5)),
        device=str(cfg.train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        auto_resume=bool(args.auto_resume or cfg.train_cfg.get("auto_resume", False)),
        tta_cfg=cfg.get("tta_cfg", None),
        eval_cfg=cfg.get("eval_cfg", None),
    )

    if args.eval_only:
        if cfg.get("val_dataloader") is None:
            raise ValueError("val_dataloader is None, cannot run eval-only mode.")

        print("Building val_dataloader (eval-only)...")
        val_loader = build_dataloader(cfg.val_dataloader)

        trainer = Trainer(
            model=model,
            optimizer=None,
            criterion=criterion,
            train_dataloader=None,
            val_dataloader=val_loader,
            lr_scheduler=None,
            cfg=trainer_cfg,
            hooks=build_hooks(cfg),
            visualizer=visualizer,
        )

        if args.resume_from:
            trainer.resume_from(args.resume_from)
        else:
            trainer.global_iter = int(args.eval_iter)

        trainer.val()
        return

    print("Building train_dataloader...")
    train_loader = build_dataloader(cfg.train_dataloader)

    print("Building val_dataloader...")
    val_loader = build_dataloader(cfg.val_dataloader) if cfg.get("val_dataloader") else None

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        cfg=trainer_cfg,
        hooks=build_hooks(cfg),
        visualizer=visualizer,
    )

    for getter in build_log_getters(cfg):
        trainer.register_log_getter(getter)

    if args.resume_from:
        trainer.resume_from(args.resume_from)

    trainer.train()


if __name__ == "__main__":
    main()