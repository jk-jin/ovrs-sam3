from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .hooks import Hook


def _jsonable(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return None
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
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

        save_dir = Path(trainer.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.path = save_dir / self.filename

        mode = "a" if getattr(trainer, "is_resumed", False) else "w"
        with self.path.open(mode, encoding="utf-8") as f:
            f.write(json.dumps({
                "mode": "meta",
                "event": "before_run",
                "iter": int(getattr(trainer, "global_iter", 0)),
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

    def after_train_iter(self, trainer, global_iter: int, batch, outputs: Dict[str, float]):
        if not self.enabled:
            return

        if self.train_interval <= 0:
            return
        if global_iter % self.train_interval != 0 and global_iter != trainer.cfg.max_iters:
            return

        state = getattr(trainer, "log_state", {}) or {}
        log_vars = dict(state.get("log_vars", {}) or {})
        extra_log_vars = dict(state.get("extra_log_vars", {}) or {})

        record = {
            "mode": "train",
            "iter": int(global_iter),
            "max_iters": int(getattr(trainer.cfg, "max_iters", 0)),
            "data_cycle": state.get("data_cycle"),
            "iter_time": state.get("iter_time"),
            "data_time": state.get("data_time"),
            "memory_mb": state.get("memory_mb"),
            "lrs": state.get("lrs"),
        }
        record.update(log_vars)
        record.update({f"extra/{k}": v for k, v in extra_log_vars.items()})

        self._write(record)

    def after_val(self, trainer, global_iter: int, val_stats: Dict[str, float]):
        if not self.enabled or not val_stats:
            return

        record = {
            "mode": "val",
            "iter": int(global_iter),
        }

        for k, v in val_stats.items():
            if str(k).startswith("_"):
                continue
            record[str(k)] = v

        self._write(record)

    def after_run(self, trainer):
        if not self.enabled:
            return

        self._write({
            "mode": "meta",
            "event": "after_run",
            "iter": int(getattr(trainer, "global_iter", 0)),
        })


def _get_by_dot_path(obj: dict, key: str):
    parts = key.split(".")
    cur = obj
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def _short_key_name(key: str) -> str:
    return key.split(".")[-1]


def _compact_key_name(key: str) -> str:
    mapping = {
        "final_bce_weight": "bce",
        "final_dice_weight": "dice",
        "bce_valid_pixel_weight": "valid",
        "bce_ignore_pixel_weight": "ignore",
        "bce_absent_class_weight": "abs",
        "score_embed_dim": "score_dim",
        "layer_scale_init": "lsi",
        "openclip_text_finetune": "text",
        "openclip_image_finetune": "image",
    }
    short = _short_key_name(key)
    return mapping.get(short, short)


def _sanitize_run_name_token(x) -> str:
    text = str(x)
    text = text.replace("/", "-").replace("\\", "-")
    text = text.replace(" ", "")
    return text


def _build_auto_run_name(raw_cfg: dict, keys: list[str], prefix: Optional[str] = None) -> Optional[str]:
    if not keys:
        return None

    parts = []
    if prefix:
        parts.append(_sanitize_run_name_token(prefix))

    for key in keys:
        value = _get_by_dot_path(raw_cfg, key)
        if value is None:
            continue
        k = _compact_key_name(key)
        v = _sanitize_run_name_token(value)
        parts.append(f"{k}{v}")

    if not parts:
        return None

    return "_".join(parts)


class WandbHook(Hook):
    def __init__(
        self,
        enabled: bool = False,
        project: str = "ovrs-sam3",
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list[str]] = None,
        mode: str = "online",
        train_interval: int = 20,
        priority: int = 90,
        name_from_config_keys: Optional[list[str]] = None,
        name_prefix: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.project = str(project)
        self.name = name
        self.group = group
        self.tags = list(tags or [])
        self.mode = str(mode)
        self.train_interval = int(train_interval)
        self.priority = int(priority)
        self.name_from_config_keys = list(name_from_config_keys or [])
        self.name_prefix = name_prefix

        self.run_id: Optional[str] = None
        self.run_name: Optional[str] = None
        self.entity: Optional[str] = None
        self.last_history_step: int = -1

        self._wandb = None
        self._run = None
        self._metric_defined = False
        self._defined_extra_metrics: set[str] = set()

        self._resume_project: Optional[str] = None
        self._resume_entity: Optional[str] = None

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------

    def state_dict(self):
        return {
            "run_id": self.run_id,
            "project": self.project,
            "entity": self.entity,
            "run_name": self.run_name,
            "last_history_step": int(self.last_history_step),
        }

    def load_state_dict(self, state):
        self.run_id = state.get("run_id", None)
        self.run_name = state.get("run_name", None)
        self._resume_project = state.get("project", None)
        self._resume_entity = state.get("entity", None)

        if "last_history_step" not in state:
            raise ValueError(
                "Checkpoint W&B state is missing "
                "'last_history_step'. This checkpoint predates "
                "precise W&B rewind support and cannot resume "
                "the same run safely."
            )

        self.last_history_step = int(
            state["last_history_step"]
        )
        if self.last_history_step < -1:
            raise ValueError(
                "W&B last_history_step must be >= -1, got "
                f"{self.last_history_step}."
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_payload(self, payload: Dict[str, Any]) -> None:
        if self._run is None:
            raise RuntimeError(
                "W&B run is not initialized."
            )

        next_history_step = self.last_history_step + 1
        self._run.log(
            payload,
            step=next_history_step,
            commit=True,
        )
        self.last_history_step = next_history_step

    # ------------------------------------------------------------------
    # Remote history query
    # ------------------------------------------------------------------

    def _get_remote_last_history_step(
        self,
        wandb,
        entity: str,
    ) -> int:
        if self.run_id is None:
            raise ValueError(
                "Cannot query W&B history without run_id."
            )

        api = wandb.Api()
        remote_run = api.run(
            f"{entity}/{self.project}/{self.run_id}"
        )
        remote_last = remote_run.lastHistoryStep
        if remote_last is None:
            return -1
        return int(remote_last)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def before_run(self, trainer):
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbHook enabled but wandb is not installed. "
                "Install with: pip install wandb"
            ) from exc

        self._wandb = wandb

        is_resumed = bool(getattr(trainer, "is_resumed", False))
        has_wandb_run_id = (
            is_resumed and self.run_id is not None
        )

        config = {}
        raw_cfg = getattr(trainer, "raw_cfg_for_logging", None)
        if raw_cfg is not None:
            config = _jsonable(raw_cfg) or {}

        if self.run_name is None:
            self.run_name = self.name
        if self.run_name is None:
            self.run_name = _build_auto_run_name(
                raw_cfg=config,
                keys=self.name_from_config_keys,
                prefix=self.name_prefix,
            )

        resume_entity = self._resume_entity

        init_kwargs: Dict[str, Any] = {
            "project": self.project,
            "entity": resume_entity,
            "name": self.run_name,
            "group": self.group,
            "tags": self.tags,
            "mode": self.mode,
            "config": config,
        }

        # --- Case A: fresh training ---
        if not has_wandb_run_id:
            self.run_id = wandb.util.generate_id()
            self.last_history_step = -1

            init_kwargs.update({
                "id": self.run_id,
                "resume": "never",
            })

        # --- Case C: checkpoint has a W&B run ID ---
        else:
            if self._resume_project is not None and self._resume_project != self.project:
                raise ValueError(
                    f"W&B project mismatch: checkpoint saved under "
                    f"{self._resume_project!r}, current config is {self.project!r}."
                )
            if resume_entity is None:
                raise ValueError(
                    "Checkpoint has a W&B entity but none was configured. "
                    "Set entity in the W&B config."
                )

            remote_last = self._get_remote_last_history_step(
                wandb=wandb,
                entity=resume_entity,
            )

            if remote_last > self.last_history_step:
                # Remote run has history past the checkpoint — rewind.
                if self.last_history_step < 0:
                    raise RuntimeError(
                        "The checkpoint was saved before its first W&B "
                        "history row, but the remote run already contains "
                        "history. There is no valid rewind target."
                    )

                rewind_target = (
                    f"{self.run_id}"
                    f"?_step={self.last_history_step}"
                )
                init_kwargs["resume_from"] = rewind_target

                print(
                    "WandbHook: rewinding run "
                    f"{self.run_id} from remote history step "
                    f"{remote_last} to checkpoint history step "
                    f"{self.last_history_step}."
                )
            else:
                # Remote has not progressed past checkpoint — normal resume.
                init_kwargs.update({
                    "id": self.run_id,
                    "resume": "must",
                })

        # --- Initialisation ---
        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:
            if "resume_from" in init_kwargs:
                raise RuntimeError(
                    "W&B run rewind failed. The project requires "
                    "resume_from support to truncate history after "
                    "the checkpoint. Do not fall back to resume='must', "
                    "because that would keep stale future metrics."
                ) from exc
            raise

        # Verify rewind kept the same run.
        expected_run_id = init_kwargs.get("id", self.run_id)
        if expected_run_id is not None and self._run.id != expected_run_id:
            raise RuntimeError(
                "W&B rewind returned a different run id: "
                f"expected {expected_run_id!r}, "
                f"got {self._run.id!r}."
            )

        # Sync from the actual run.
        self.run_id = self._run.id
        self.project = self._run.project or self.project
        self.entity = self._run.entity
        self.run_name = self._run.name or self.run_name

        self._define_metrics()

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

        self._run.define_metric(
            metric_name,
            step_metric="trainer/global_iter",
            step_sync=True,
            overwrite=True,
        )
        self._defined_extra_metrics.add(metric_name)

    def after_train_iter(self, trainer, global_iter: int, batch, outputs: Dict[str, float]):
        if not self.enabled or self._wandb is None:
            return

        if self.train_interval <= 0:
            return
        if global_iter % self.train_interval != 0 and global_iter != trainer.cfg.max_iters:
            return

        state = getattr(trainer, "log_state", {}) or {}
        log_vars = dict(state.get("log_vars", {}) or {})
        extra_log_vars = dict(state.get("extra_log_vars", {}) or {})

        payload = {
            "trainer/global_iter": int(global_iter),
            "train/iter_time": state.get("iter_time"),
            "train/data_time": state.get("data_time"),
            "train/memory_mb": state.get("memory_mb"),
        }

        lrs = state.get("lrs", None)
        if isinstance(lrs, list) and len(lrs) > 0:
            payload["train/lr_min"] = min(float(x) for x in lrs)
            payload["train/lr_max"] = max(float(x) for x in lrs)

        for k, v in log_vars.items():
            payload[f"train/{k}"] = v

        for k, v in extra_log_vars.items():
            if _jsonable(v) is None:
                continue

            metric_name = f"extra/{k}"
            self._define_extra_metric(metric_name)
            payload[metric_name] = v

        self._log_payload(payload)

    def after_val(self, trainer, global_iter: int, val_stats: Dict[str, float]):
        if not self.enabled or self._wandb is None or not val_stats:
            return

        payload = {"trainer/global_iter": int(global_iter)}
        for k, v in val_stats.items():
            if str(k).startswith("_"):
                continue
            if _jsonable(v) is not None:
                payload[f"val/{k}"] = v

        self._log_payload(payload)

    def on_exception(self, trainer, exception):
        if self.enabled and self._run is not None:
            run = self._run
            self._run = None
            run.finish(exit_code=1)

    def after_run(self, trainer):
        if self.enabled and self._run is not None:
            run = self._run
            self._run = None
            run.finish(exit_code=0)
