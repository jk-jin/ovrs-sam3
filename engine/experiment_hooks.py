from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from .hooks import Hook
from .runtime_state import capture_rng_state, restore_rng_state


WANDB_TRAIN_KEYS = {
    "total_loss",
    "loss_seg_bce",
    "loss_reference_distill_bce",
    "loss_reference_distill_weighted",
}


def _jsonable(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return None
    except Exception:
        pass

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            vv = _jsonable(v)
            if vv is not None:
                out[str(k)] = vv
        return out

    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            vv = _jsonable(v)
            if vv is not None:
                out.append(vv)
        return out

    return None


@contextmanager
def _preserve_rng():
    """Keep logging/tracking initialization from consuming training RNG."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def _experiment_from_trainer(trainer) -> dict:
    cfg = getattr(trainer, "experiment_cfg", None)
    if cfg is None:
        raw_cfg = getattr(trainer, "raw_cfg_for_logging", None) or {}
        cfg = raw_cfg.get("experiment", {}) if isinstance(
            raw_cfg, dict
        ) else {}
    return dict(cfg or {})


def _seed_from_trainer(trainer) -> Optional[int]:
    seed = getattr(trainer, "seed", None)
    if seed is not None:
        return int(seed)
    raw_cfg = getattr(trainer, "raw_cfg_for_logging", None)
    if isinstance(raw_cfg, dict):
        seed = raw_cfg.get("runtime", {}).get("seed", None)
        if seed is not None:
            return int(seed)
    return None


def build_train_event(
    trainer,
    global_iter: int,
) -> Dict[str, Any]:
    state = getattr(trainer, "log_state", {}) or {}
    log_vars = dict(state.get("log_vars", {}) or {})
    extra_log_vars = dict(state.get("extra_log_vars", {}) or {})

    return {
        "mode": "train",
        "iter": int(global_iter),
        "max_iters": int(getattr(trainer.cfg, "max_iters", 0)),
        "seed": _seed_from_trainer(trainer),
        "attempt_id": getattr(trainer, "attempt_id", None),
        "data_cycle": state.get("data_cycle"),
        "iter_time": state.get("iter_time"),
        "data_time": state.get("data_time"),
        "memory_mb": state.get("memory_mb"),
        "lrs": state.get("lrs"),
        "log_vars": log_vars,
        "extra_log_vars": extra_log_vars,
    }


def build_val_event(
    trainer,
    global_iter: int,
    val_stats: Dict[str, float],
) -> Dict[str, Any]:
    metrics = {
        str(k): v for k, v in val_stats.items()
        if not str(k).startswith("_")
    }
    return {
        "mode": "val",
        "iter": int(global_iter),
        "seed": _seed_from_trainer(trainer),
        "attempt_id": getattr(trainer, "attempt_id", None),
        "metrics": metrics,
    }


class MetricsJsonlHook(Hook):
    def __init__(
        self,
        enabled: bool = True,
        filename: str = "metrics.jsonl",
        train_interval: int = 20,
        priority: int = 80,
    ):
        self.enabled = bool(enabled)
        self.filename = str(filename)
        self.train_interval = int(train_interval)
        self.priority = int(priority)
        self.path: Optional[Path] = None

    def before_run(self, trainer):
        if not self.enabled:
            return

        if getattr(trainer, "is_eval_only", False):
            # Independent evaluation writes results/<eval_id>.json and must
            # never reopen or truncate the training JSONL.
            self.enabled = False
            return

        save_dir = Path(trainer.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.path = save_dir / self.filename

        mode = "a" if getattr(trainer, "is_resumed", False) else "w"
        with self.path.open(mode, encoding="utf-8") as f:
            f.write(json.dumps({
                "mode": "meta",
                "event": "before_run",
                "iter": int(getattr(trainer, "global_iter", 0)),
                "run_id": getattr(trainer, "run_id", None),
                "attempt_id": getattr(trainer, "attempt_id", None),
                "parent_attempt_id": getattr(
                    trainer, "parent_attempt_id", None
                ),
                "resume_from_step": getattr(
                    trainer, "resume_from_step", None
                ),
                "resume_diff": getattr(trainer, "resume_diff", None),
                "seed": _seed_from_trainer(trainer),
                "experiment": _experiment_from_trainer(trainer),
                "config_fingerprint": getattr(
                    trainer, "config_fingerprint", None
                ),
            }, ensure_ascii=False) + "\n")

    def _write(self, record: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return

        clean = {}
        for k, v in record.items():
            vv = _jsonable(v)
            if vv is not None:
                clean[str(k)] = vv

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    def after_train_iter(
        self,
        trainer,
        global_iter: int,
        batch,
        outputs: Dict[str, float],
    ):
        if not self.enabled:
            return

        if self.train_interval <= 0:
            return
        if (
            global_iter % self.train_interval != 0
            and global_iter != trainer.cfg.max_iters
        ):
            return

        event = build_train_event(trainer, global_iter)
        record = {
            "mode": event["mode"],
            "iter": event["iter"],
            "max_iters": event["max_iters"],
            "seed": event["seed"],
            "attempt_id": event["attempt_id"],
            "data_cycle": event["data_cycle"],
            "iter_time": event["iter_time"],
            "data_time": event["data_time"],
            "memory_mb": event["memory_mb"],
            "lrs": event["lrs"],
        }
        record.update(event["log_vars"])
        record.update({
            f"extra/{k}": v
            for k, v in event["extra_log_vars"].items()
        })
        self._write(record)

    def after_val(
        self,
        trainer,
        global_iter: int,
        val_stats: Dict[str, float],
    ):
        if not self.enabled or not val_stats:
            return

        event = build_val_event(trainer, global_iter, val_stats)
        record = {
            "mode": "val",
            "iter": event["iter"],
            "seed": event["seed"],
            "attempt_id": event["attempt_id"],
        }
        record.update(event["metrics"])
        self._write(record)

    def after_run(self, trainer):
        if not self.enabled:
            return

        self._write({
            "mode": "meta",
            "event": "after_run",
            "iter": int(getattr(trainer, "global_iter", 0)),
            "attempt_id": getattr(trainer, "attempt_id", None),
        })


class WandbHook(Hook):
    def __init__(
        self,
        enabled: bool = True,
        project: str = "reca-sam3",
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list[str]] = None,
        mode: str = "offline",
        train_interval: int = 20,
        log_per_class: bool = False,
        log_extra: bool = False,
        priority: int = 90,
    ):
        self.enabled = bool(enabled)
        self.project = str(project)
        self.name = name
        self.group = group
        self.tags = list(tags or [])
        self.mode = str(mode)
        self.train_interval = int(train_interval)
        self.log_per_class = bool(log_per_class)
        self.log_extra = bool(log_extra)
        self.priority = int(priority)

        self.run_name: Optional[str] = None

        self._wandb = None
        self._run = None
        self._metric_defined = False
        self._defined_extra_metrics: set[str] = set()
        self._failure_reported = False

    def _disable_after_failure(self, exc: Exception, stage: str) -> None:
        self._run = None
        self.enabled = False
        if not self._failure_reported:
            print(
                f"[W&B] {stage} failed; disabling W&B for this process. "
                f"Local JSONL logging remains active. Error: {exc}"
            )
            self._failure_reported = True

    def _log_payload(self, payload: Dict[str, Any]) -> None:
        if self._run is None:
            return

        try:
            with _preserve_rng():
                self._run.log(payload, commit=True)
        except Exception as exc:
            self._disable_after_failure(exc, "log")

    def before_run(self, trainer):
        if not self.enabled:
            return

        try:
            with _preserve_rng():
                import wandb
        except Exception as exc:
            self._disable_after_failure(exc, "import")
            return

        self._wandb = wandb

        config = {}
        raw_cfg = getattr(trainer, "raw_cfg_for_logging", None)
        if raw_cfg is not None:
            config = _jsonable(raw_cfg) or {}
        config["_attempt"] = {
            "attempt_id": getattr(trainer, "attempt_id", None),
            "resume_from_step": getattr(
                trainer, "resume_from_step", None
            ),
            "parent_attempt_id": getattr(
                trainer, "parent_attempt_id", None
            ),
            "run_id": getattr(trainer, "run_id", None),
            "seed": _seed_from_trainer(trainer),
        }

        experiment = _experiment_from_trainer(trainer)
        seed = _seed_from_trainer(trainer)

        if self.run_name is None:
            self.run_name = self.name
        if self.run_name is None:
            self.run_name = str(experiment.get("name", "reca_sam3"))
            if seed is not None:
                self.run_name = f"{self.run_name}_seed{seed}"

        group = self.group
        if group is None:
            group = experiment.get("group", None)
        if group is None:
            group = experiment.get("method_id", None)

        try:
            with _preserve_rng():
                self._run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    group=group,
                    tags=self.tags,
                    mode=self.mode,
                    config=config,
                    dir=str(trainer.cfg.save_dir),
                    resume="never",
                )
        except Exception as exc:
            self._disable_after_failure(exc, "init")
            return

        self.run_name = self._run.name or self.run_name

        try:
            with _preserve_rng():
                self._define_metrics()
        except Exception as exc:
            self._disable_after_failure(exc, "define_metric")

    def _define_metrics(self) -> None:
        if self._metric_defined or self._run is None:
            return

        self._run.define_metric("trainer/global_iter")
        self._run.define_metric(
            "train/*",
            step_metric="trainer/global_iter",
            step_sync=True,
        )
        self._run.define_metric(
            "val/*",
            step_metric="trainer/global_iter",
            step_sync=True,
        )
        self._metric_defined = True

    def _define_extra_metric(self, metric_name: str) -> None:
        if self._run is None:
            return
        if metric_name in self._defined_extra_metrics:
            return

        with _preserve_rng():
            self._run.define_metric(
                metric_name,
                step_metric="trainer/global_iter",
                step_sync=True,
                overwrite=True,
            )
        self._defined_extra_metrics.add(metric_name)

    def after_train_iter(
        self,
        trainer,
        global_iter: int,
        batch,
        outputs: Dict[str, float],
    ):
        if not self.enabled or self._run is None:
            return

        if self.train_interval <= 0:
            return
        if (
            global_iter % self.train_interval != 0
            and global_iter != trainer.cfg.max_iters
        ):
            return

        event = build_train_event(trainer, global_iter)
        payload = {
            "trainer/global_iter": int(global_iter),
        }
        for key in ("iter_time", "data_time", "memory_mb"):
            if event[key] is not None:
                payload[f"train/{key}"] = event[key]

        lrs = event["lrs"]
        if isinstance(lrs, list) and len(lrs) > 0:
            payload["train/lr_min"] = min(float(x) for x in lrs)
            payload["train/lr_max"] = max(float(x) for x in lrs)

        for key, value in event["log_vars"].items():
            if key in WANDB_TRAIN_KEYS:
                payload[f"train/{key}"] = value

        if self.log_extra:
            for key, value in event["extra_log_vars"].items():
                if _jsonable(value) is None:
                    continue
                metric_name = f"extra/{key}"
                try:
                    self._define_extra_metric(metric_name)
                except Exception as exc:
                    self._disable_after_failure(exc, "define_metric")
                    return
                payload[metric_name] = value

        self._log_payload(payload)

    def after_val(
        self,
        trainer,
        global_iter: int,
        val_stats: Dict[str, float],
    ):
        if not self.enabled or self._run is None or not val_stats:
            return

        event = build_val_event(trainer, global_iter, val_stats)
        payload = {"trainer/global_iter": int(global_iter)}

        for key, value in event["metrics"].items():
            if _jsonable(value) is None:
                continue
            if key in {
                "semantic.miou",
                "semantic.macc",
                "semantic.pixel_acc",
            }:
                payload[f"val/{key}"] = value
            elif self.log_per_class and (
                key.startswith("semantic.iou_class_")
                or key.startswith("semantic.acc_class_")
            ):
                payload[f"val/{key}"] = value
            elif self.log_extra:
                payload[f"val/{key}"] = value

        self._log_payload(payload)

    def on_exception(self, trainer, exception):
        if not self.enabled or self._run is None:
            return
        run = self._run
        self._run = None
        try:
            with _preserve_rng():
                run.finish(exit_code=1)
        except Exception as exc:
            self._disable_after_failure(exc, "finish")

    def after_run(self, trainer):
        if not self.enabled or self._run is None:
            return
        run = self._run
        self._run = None
        try:
            with _preserve_rng():
                run.finish(exit_code=0)
        except Exception as exc:
            self._disable_after_failure(exc, "finish")
