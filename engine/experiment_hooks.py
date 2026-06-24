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
        val_interval: int = 1,
        priority: int = 80,
    ):
        self.enabled = bool(enabled)
        self.filename = str(filename)
        self.train_interval = int(train_interval)
        self.val_interval = int(val_interval)
        self.priority = int(priority)
        self.path: Optional[Path] = None

    def before_run(self, trainer):
        if not self.enabled:
            return

        save_dir = Path(trainer.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.path = save_dir / self.filename

        with self.path.open("a", encoding="utf-8") as f:
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
        "num_query_tokens": "q",
        "clip_score_embed_dim": "score_dim",
        "clip_score_conv_kernel": "k",
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
        log_val_iter: bool = False,
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
        self.log_val_iter = bool(log_val_iter)
        self.priority = int(priority)
        self.name_from_config_keys = list(name_from_config_keys or [])
        self.name_prefix = name_prefix
        self._wandb = None
        self._run = None

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

        config = {}
        raw_cfg = getattr(trainer, "raw_cfg_for_logging", None)
        if raw_cfg is not None:
            config = _jsonable(raw_cfg) or {}

        run_name = self.name
        if run_name is None:
            run_name = _build_auto_run_name(
                raw_cfg=config,
                keys=self.name_from_config_keys,
                prefix=self.name_prefix,
            )

        self._run = wandb.init(
            project=self.project,
            name=run_name,
            group=self.group,
            tags=self.tags,
            mode=self.mode,
            config=config,
        )

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
            "iter": int(global_iter),
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
            payload[f"extra/{k}"] = v

        payload = {k: v for k, v in payload.items() if _jsonable(v) is not None}
        self._wandb.log(payload, step=int(global_iter))

    def after_val(self, trainer, global_iter: int, val_stats: Dict[str, float]):
        if not self.enabled or self._wandb is None or not val_stats:
            return

        payload = {"iter": int(global_iter)}
        for k, v in val_stats.items():
            if str(k).startswith("_"):
                continue
            if _jsonable(v) is not None:
                payload[f"val/{k}"] = v

        self._wandb.log(payload, step=int(global_iter))

    def after_run(self, trainer):
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
