from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    from importlib import import_module
    import types

    package_name = "_reca_sam3_localpkg"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[package_name] = pkg

    _config_mod = import_module(f"{package_name}.engine.config")
    build_effective_config = _config_mod.build_effective_config
    get_by_dot_path = _config_mod.get_by_dot_path
    to_builtin = _config_mod.to_builtin
    _result_mod = import_module(f"{package_name}.engine.result_writer")
    ResultWriter = _result_mod.ResultWriter
    build_config_meta = _result_mod.build_config_meta
    collect_git_info = _result_mod.collect_git_info
    load_saved_config = _result_mod.load_saved_config
    save_effective_config = _result_mod.save_effective_config
    write_code_patch = _result_mod.write_code_patch
    _results_utils = import_module(f"{package_name}.utils.experiment_results")
    sha256_text = _results_utils.sha256_text
else:
    from ..engine.config import (
        build_effective_config,
        get_by_dot_path,
        to_builtin,
    )
    from ..engine.result_writer import (
        ResultWriter,
        build_config_meta,
        collect_git_info,
        load_saved_config,
        save_effective_config,
        write_code_patch,
    )
    from ..utils.experiment_results import sha256_text


_RUNTIME_IMPORTS = None


def _runtime_imports() -> dict:
    """Import model/training dependencies only when actually running."""
    global _RUNTIME_IMPORTS
    if _RUNTIME_IMPORTS is not None:
        return _RUNTIME_IMPORTS

    if __package__ in (None, ""):
        package_name = "_reca_sam3_localpkg"
        from importlib import import_module

        mods = {
            "build_dataloader": import_module(
                f"{package_name}.data.build"
            ).build_dataloader,
            "build_optimizer": import_module(
                f"{package_name}.engine.optimizer_builder"
            ).build_optimizer,
            "build_scheduler": import_module(
                f"{package_name}.engine.optimizer_builder"
            ).build_scheduler,
            "Trainer": import_module(
                f"{package_name}.engine.trainer"
            ).Trainer,
            "build_training_components": import_module(
                f"{package_name}.model_builder"
            ).build_training_components,
            "build_train_runtime_components": import_module(
                f"{package_name}.model_builder"
            ).build_train_runtime_components,
            "load_checkpoint_file": import_module(
                f"{package_name}.engine.checkpoint"
            ).load_checkpoint_file,
        }
    else:
        from ..data.build import build_dataloader
        from ..engine.checkpoint import load_checkpoint_file
        from ..engine.optimizer_builder import (
            build_optimizer,
            build_scheduler,
        )
        from ..engine.trainer import Trainer
        from ..model_builder import (
            build_train_runtime_components,
            build_training_components,
        )

        mods = {
            "build_dataloader": build_dataloader,
            "build_optimizer": build_optimizer,
            "build_scheduler": build_scheduler,
            "Trainer": Trainer,
            "build_training_components": build_training_components,
            "build_train_runtime_components": build_train_runtime_components,
            "load_checkpoint_file": load_checkpoint_file,
        }

    _RUNTIME_IMPORTS = mods
    return mods


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize_path_token(text: object, max_len: int = 80) -> str:
    text = str(text).strip()
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    text = text.strip("-_.")
    if not text:
        text = "empty"
    if len(text) > max_len:
        text = text[:max_len].rstrip("-_.")
    return text


def _short_key_name(key: str) -> str:
    return key.split(".")[-1]


def build_work_dir_suffix_from_cfg(cfg, keys: list[str]) -> str:
    if not keys:
        raise ValueError("--work-dir-suffix-keys requires at least one key.")

    parts = []
    for key in keys:
        value = get_by_dot_path(cfg, key)
        short_key = _sanitize_path_token(_short_key_name(key), max_len=48)
        value_token = _sanitize_path_token(value, max_len=48)
        parts.append(f"{short_key}-{value_token}")

    return "__".join(parts)


def maybe_append_work_dir_suffix(
    work_dir: str | None,
    cfg,
    suffix_keys: list[str] | None,
    config_fingerprint: str,
    append_hash: bool = False,
) -> str | None:
    if work_dir is None:
        return None
    if not suffix_keys:
        return work_dir

    suffix = build_work_dir_suffix_from_cfg(cfg, suffix_keys)
    if append_hash:
        suffix = f"{suffix}__{config_fingerprint[:8]}"

    return str(Path(work_dir) / suffix)


def print_config_as_json(cfg) -> None:
    print(json.dumps(to_builtin(cfg), indent=2, ensure_ascii=False))


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
    load_checkpoint_file = _runtime_imports()["load_checkpoint_file"]
    ckpt = load_checkpoint_file(path)
    state_dict = _unwrap_state_dict(ckpt)
    state_dict = _strip_prefix_if_present(state_dict, "module.")

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=strict
    )

    if len(missing_keys) == len(model.state_dict()):
        raise ValueError(
            f"Checkpoint {path} has no matching parameter names for this "
            "model. Old pre-renaming checkpoints are not compatible with "
            "the ReCA-SAM3 parameter paths."
        )

    print(f"Loaded model weights from {path}")
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "global_iter": ckpt.get("global_iter", None),
        "extra": ckpt.get("extra", {}) or {},
        "val_stats": ckpt.get("val_stats", {}) or {},
        "checkpoint_version": ckpt.get("checkpoint_version", None),
    }


def _unwrap_dataset(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    while dataset is not None and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def build_dataset_info(dataloader) -> dict:
    dataset = _unwrap_dataset(dataloader)
    if dataset is None:
        return {
            "class_names": None,
            "include_background": False,
            "expected_samples": None,
            "sample_list_sha256": None,
        }

    class_names = getattr(
        dataset,
        "eval_classes",
        getattr(dataset, "classes", None),
    )
    class_names = (
        [str(x) for x in class_names]
        if class_names is not None else None
    )
    include_background = bool(
        getattr(dataset, "background_enabled", False)
    )

    expected_samples = None
    try:
        expected_samples = int(len(dataset))
    except TypeError:
        pass

    sample_list_sha256 = None
    img_paths = getattr(dataset, "img_paths", None)
    img_dir = getattr(dataset, "img_dir", None)
    if img_paths is not None:
        if img_dir is not None:
            sample_ids = [
                str(Path(p).relative_to(img_dir))
                for p in img_paths
            ]
        else:
            sample_ids = [str(Path(p).name) for p in img_paths]
        sample_list_sha256 = sha256_text(
            "\n".join(sorted(sample_ids))
        )

    return {
        "class_names": class_names,
        "include_background": include_background,
        "expected_samples": expected_samples,
        "sample_list_sha256": sample_list_sha256,
    }


def collect_environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_names": [],
    }
    if torch.cuda.is_available():
        env["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    return env


def _get_optional(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _diff_sections(old: dict, new: dict, paths: list[str]) -> list[dict]:
    diffs = []
    for path in paths:
        old_value = _get_optional(old, path)
        new_value = _get_optional(new, path)
        if old_value != new_value:
            diffs.append({
                "path": path,
                "old": old_value,
                "new": new_value,
            })
    return diffs


def _check_resume_work_dir(resume_from: str, work_dir: str) -> None:
    ckpt_dir = Path(resume_from).resolve().parent
    if ckpt_dir != Path(work_dir).resolve():
        raise ValueError(
            "Resume must use the original work directory. "
            f"Checkpoint directory={ckpt_dir}, work_dir={Path(work_dir).resolve()}. "
            "Use --work-dir to point at the original experiment directory."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate ReCA-SAM3 semantic segmentation with "
            "iteration-based training."
        )
    )
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--load-model-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="only run validation and write one evaluation result",
    )
    parser.add_argument(
        "--eval-iter",
        type=int,
        default=None,
        help=(
            "step recorded for a weight file that has no training metadata. "
            "Required for native evaluation records when the step cannot be "
            "read from the checkpoint."
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help=(
            "Override config options, e.g. "
            "train_cfg.max_iters=1000 loss.seg_dice_weight=0.3. "
            "Dedicated CLI options such as --seed and --device take "
            "precedence over the same field."
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help=(
            "Print the final effective config after defaults, "
            "--cfg-options, dedicated CLI options, and work-dir suffixing."
        ),
    )
    parser.add_argument(
        "--work-dir-suffix-keys",
        nargs="+",
        default=None,
        help=(
            "Append a deterministic subdirectory to the work dir using "
            "values from the effective config."
        ),
    )
    parser.add_argument(
        "--work-dir-suffix-hash",
        action="store_true",
        help=(
            "Append an 8-char semantic-config fingerprint to the generated "
            "work-dir suffix."
        ),
    )
    args = parser.parse_args()

    if args.resume_from is not None and args.load_model_from is not None:
        raise ValueError(
            "--resume-from and --load-model-from cannot be used together."
        )

    cfg, config_fingerprint = build_effective_config(
        args.config,
        cfg_options=args.cfg_options,
        seed=args.seed,
        device=args.device,
        work_dir=args.work_dir,
    )

    work_dir = maybe_append_work_dir_suffix(
        work_dir=cfg["runtime"]["work_dir"],
        cfg=cfg,
        suffix_keys=args.work_dir_suffix_keys,
        config_fingerprint=config_fingerprint,
        append_hash=bool(args.work_dir_suffix_hash),
    )
    cfg["runtime"]["work_dir"] = work_dir

    if args.print_config:
        print_config_as_json(cfg)
        return

    runtime_imports = _runtime_imports()
    build_dataloader = runtime_imports["build_dataloader"]
    build_optimizer = runtime_imports["build_optimizer"]
    build_scheduler = runtime_imports["build_scheduler"]
    Trainer = runtime_imports["Trainer"]
    build_training_components = runtime_imports["build_training_components"]
    build_train_runtime_components = runtime_imports[
        "build_train_runtime_components"
    ]

    if not args.eval_only:
        if cfg.get("train_dataloader") is None:
            raise ValueError("train_dataloader is required for training.")
        if cfg.get("optimizer") is None:
            raise ValueError("optimizer is required for training.")
        if cfg.get("param_scheduler") is None:
            raise ValueError("param_scheduler is required for training.")

    seed = int(cfg["runtime"]["seed"])
    set_seed(seed)

    experiment = cfg["experiment"]
    git_info = collect_git_info(PROJECT_ROOT)

    # ------------------------------------------------------------------
    # Work-dir guard and config.json handling
    # ------------------------------------------------------------------
    if not args.eval_only and args.resume_from is None:
        work_dir_path = Path(work_dir)
        if work_dir_path.is_dir():
            existing = sorted(work_dir_path.glob("iter_*.pth"))
            latest = work_dir_path / "latest.pth"
            best = work_dir_path / "best.pth"
            metrics = work_dir_path / "metrics.jsonl"
            config = work_dir_path / "config.json"
            results = sorted((work_dir_path / "results").glob("*.json"))
            if (
                existing
                or latest.exists()
                or best.exists()
                or metrics.exists()
                or config.exists()
                or results
            ):
                raise RuntimeError(
                    "work_dir already contains training artifacts: "
                    f"{work_dir_path}\n"
                    "Use a new --work-dir or explicitly resume with "
                    "--resume-from."
                )

    resume_diff = None
    if args.resume_from is not None:
        _check_resume_work_dir(args.resume_from, work_dir)
        saved = load_saved_config(work_dir)
        if saved is None:
            raise ValueError(
                "Resume requires the original config.json in the work "
                f"directory: {work_dir}"
            )
        saved = dict(saved)
        saved_meta = dict(saved.pop("meta", {}) or {})
        saved_fingerprint = saved_meta.get("config_fingerprint", None)
        if saved_fingerprint != config_fingerprint:
            raise ValueError(
                "Resume config changes semantic training settings. "
                f"Saved fingerprint={saved_fingerprint}, "
                f"current fingerprint={config_fingerprint}. "
                "Use a new experiment directory for a different protocol."
            )
        run_id = saved_meta.get("run_id") or (
            f"{experiment['name']}-{uuid.uuid4().hex[:8]}"
        )
        resume_diff = _diff_sections(
            saved,
            cfg,
            [
                "runtime",
                "logging",
                "visualization",
                "train_cfg",
                "train_dataloader",
                "val_dataloader",
                "experiment.name",
                "experiment.note",
            ],
        )
        print(
            "Resume config fingerprint verified; non-semantic differences: "
            f"{resume_diff}"
        )
    else:
        run_id = f"{experiment['name']}-{uuid.uuid4().hex[:8]}"

    attempt_id = uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # Model and runtime
    # ------------------------------------------------------------------
    model, criterion = build_training_components(cfg)

    checkpoint_metadata = None
    if args.load_model_from is not None:
        checkpoint_metadata = load_model_weights_only(
            model=model,
            path=args.load_model_from,
            strict=False,
        )

    (
        work_dir,
        trainer_cfg,
        hooks,
        visualizer,
        checkpoint_manager,
    ) = build_train_runtime_components(
        cfg,
        work_dir_override=work_dir,
    )

    print(f"Resolved work_dir: {work_dir}")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    def attach_result_writer(
        dataloader,
        *,
        experiment_cfg=None,
        training_cfg=None,
        seed_value=None,
        fingerprint=None,
        run_id_value=None,
        config_source_path=None,
    ):
        return ResultWriter(
            work_dir=work_dir,
            experiment=experiment_cfg or cfg["experiment"],
            training=training_cfg or cfg["training"],
            evaluation=cfg["evaluation"],
            train_cfg=cfg["train_cfg"],
            dataset_info=build_dataset_info(dataloader),
            seed=seed if seed_value is None else int(seed_value),
            config_fingerprint=fingerprint or config_fingerprint,
            git_info=git_info,
            run_id=run_id_value or run_id,
            attempt_id=attempt_id,
            config_source_path=config_source_path,
        )

    if args.eval_only:
        if cfg.get("val_dataloader") is None:
            raise ValueError(
                "val_dataloader is None, cannot run eval-only mode."
            )

        if args.load_model_from is not None:
            checkpoint_step = checkpoint_metadata.get("global_iter", None)
            if checkpoint_step is None:
                checkpoint_step = args.eval_iter
            checkpoint_path = args.load_model_from
            extra = checkpoint_metadata.get("extra", {}) or {}
            val_stats = checkpoint_metadata.get("val_stats", {}) or {}
        elif args.resume_from is not None:
            checkpoint_metadata = load_model_weights_only(
                model=model,
                path=args.resume_from,
                strict=False,
            )
            checkpoint_step = checkpoint_metadata.get("global_iter", None)
            if checkpoint_step is None:
                checkpoint_step = args.eval_iter
            checkpoint_path = args.resume_from
            extra = checkpoint_metadata.get("extra", {}) or {}
            val_stats = checkpoint_metadata.get("val_stats", {}) or {}
        else:
            checkpoint_step = args.eval_iter
            checkpoint_path = None
            extra = {}
            val_stats = {}

        if checkpoint_step is None:
            raise ValueError(
                "A native evaluation result needs a checkpoint step. "
                "Pass --load-model-from with a training checkpoint, "
                "--resume-from, or an explicit --eval-iter."
            )

        monitor = str(cfg["train_cfg"]["monitor"])
        selection = {
            "dataset": extra.get(
                "selection_dataset", cfg["evaluation"].get("dataset")
            ),
            "split": extra.get(
                "selection_split", cfg["evaluation"].get("split")
            ),
            "metric": extra.get("monitor", monitor),
            "mode": extra.get(
                "monitor_mode", cfg["train_cfg"]["monitor_mode"]
            ),
            "value": extra.get(
                "monitor_value", val_stats.get(monitor, None)
            ),
            "is_best": extra.get("is_best_checkpoint", None),
        }

        eval_experiment = cfg["experiment"]
        eval_training = cfg["training"]
        eval_seed = seed
        eval_fingerprint = config_fingerprint
        eval_run_id = (
            f"{cfg['experiment']['name']}_eval-{uuid.uuid4().hex[:8]}"
        )

        checkpoint_config = None
        if checkpoint_path is not None:
            checkpoint_config = load_saved_config(
                Path(checkpoint_path).resolve().parent
            )
        if checkpoint_config is not None:
            saved_meta = dict(
                checkpoint_config.get("meta", {}) or {}
            )
            saved_cfg = dict(checkpoint_config)
            saved_cfg.pop("meta", None)
            if (
                saved_cfg.get("experiment") is not None
                and saved_cfg.get("training") is not None
                and saved_meta.get("config_fingerprint") is not None
            ):
                eval_experiment = saved_cfg["experiment"]
                eval_training = saved_cfg["training"]
                eval_seed = int(saved_meta.get("seed", seed))
                eval_fingerprint = saved_meta["config_fingerprint"]
                eval_run_id = (
                    f"{eval_experiment.get('name', 'eval')}"
                    f"_eval-{uuid.uuid4().hex[:8]}"
                )
                print(
                    "Loaded training metadata from "
                    f"{Path(checkpoint_path).resolve().parent / 'config.json'}"
                )
            else:
                print(
                    "[warning] Checkpoint config.json lacks the new "
                    "meta/experiment/training fields; evaluation metadata "
                    "falls back to the test config."
                )
        else:
            print(
                "[warning] No config.json found next to the checkpoint; "
                "evaluation metadata falls back to the test config."
            )

        print("Building val_dataloader (eval-only)...")
        val_loader = build_dataloader(cfg["val_dataloader"], seed=seed)

        trainer = Trainer(
            model=model,
            optimizer=None,
            criterion=criterion,
            train_dataloader=None,
            val_dataloader=val_loader,
            lr_scheduler=None,
            cfg=trainer_cfg,
            hooks=hooks,
            checkpoint_manager=checkpoint_manager,
            visualizer=visualizer,
            raw_cfg_for_logging=to_builtin(cfg),
            result_writer=attach_result_writer(
                val_loader,
                experiment_cfg=eval_experiment,
                training_cfg=eval_training,
                seed_value=eval_seed,
                fingerprint=eval_fingerprint,
                run_id_value=eval_run_id,
                config_source_path=(
                    Path(checkpoint_path).resolve().parent / "config.json"
                    if checkpoint_path is not None
                    else None
                ),
            ),
            is_eval_only=True,
            attempt_id=attempt_id,
            run_id=eval_run_id,
        )
        trainer.seed = eval_seed
        trainer.experiment_cfg = eval_experiment
        trainer.config_fingerprint = eval_fingerprint
        trainer.global_iter = int(checkpoint_step)

        trainer.hook_manager.call("before_run", trainer)
        try:
            trainer.val()
        except BaseException as exc:
            trainer.hook_manager.call("on_exception", trainer, exc)
            raise
        else:
            trainer.write_eval_result(
                checkpoint_path=checkpoint_path,
                checkpoint_step=int(checkpoint_step),
                selection=selection,
            )
            trainer.hook_manager.call("after_run", trainer)

        return

    if args.resume_from is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    print("Building train_dataloader...")
    train_loader = build_dataloader(cfg["train_dataloader"], seed=seed)

    print("Building val_dataloader...")
    val_loader = (
        build_dataloader(cfg["val_dataloader"], seed=seed)
        if cfg.get("val_dataloader") else None
    )

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
        hooks=hooks,
        checkpoint_manager=checkpoint_manager,
        visualizer=visualizer,
        raw_cfg_for_logging=to_builtin(cfg),
        result_writer=attach_result_writer(val_loader),
        is_eval_only=False,
        attempt_id=attempt_id,
        run_id=run_id,
    )
    trainer.seed = seed
    trainer.experiment_cfg = cfg["experiment"]
    trainer.config_fingerprint = config_fingerprint
    trainer.selection_dataset = cfg["evaluation"].get("dataset")
    trainer.selection_split = cfg["evaluation"].get("split")
    trainer.resume_diff = resume_diff

    if args.resume_from:
        trainer.resume_from(args.resume_from)
    else:
        meta = build_config_meta(
            git_info=git_info,
            seed=seed,
            attempt_id=attempt_id,
            run_id=run_id,
            config_fingerprint=config_fingerprint,
            environment=collect_environment(),
        )
        save_effective_config(work_dir, cfg, meta)
        write_code_patch(work_dir, git_info)
        print(f"Saved effective config to {Path(work_dir) / 'config.json'}")

    trainer.train()


if __name__ == "__main__":
    main()
