from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..config_dataclasses import CheckpointManagerConfig
from .optimizer_builder import enforce_optimizer_param_group_invariants

class CheckpointManager:
    def __init__(self, cfg: CheckpointManagerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.save_dir / "checkpoint_meta.json"
        self.best_score = None
        self.best_path = None
        self._load_meta()

    def _load_meta(self) -> None:
        if not self.meta_path.exists():
            return
        try:
            meta = json.loads(self.meta_path.read_text())
            self.best_score = meta.get("best_score")
            self.best_path = meta.get("best_path")
        except Exception:
            self.best_score = None
            self.best_path = None

    def _write_meta(self) -> None:
        meta = {
            "best_score": self.best_score,
            "best_path": self.best_path,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.cfg.mode == "min":
            return score < self.best_score
        if self.cfg.mode == "max":
            return score > self.best_score
        raise ValueError(f"Unsupported mode: {self.cfg.mode}")

    def _checkpoint_path(self, global_iter: int) -> Path:
        return self.save_dir / f"iter_{int(global_iter):07d}.pth"

    def _build_payload(
        self,
        global_iter: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        train_stats: Optional[Dict[str, float]] = None,
        val_stats: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "global_iter": int(global_iter),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None and hasattr(scheduler, "state_dict")
                else None
            ),
            "train_stats": dict(train_stats or {}),
            "val_stats": dict(val_stats or {}),
            "extra": dict(extra or {}),
        }

    def _save_payload(self, payload: Dict[str, Any], path: Path) -> Path:
        torch.save(payload, path)
        return path

    def _update_latest(self, ckpt_path: Path) -> None:
        if not self.cfg.save_latest:
            return
        latest_path = self.save_dir / "latest.pth"
        shutil.copyfile(ckpt_path, latest_path)

    def _update_best_from_checkpoint(self, ckpt_path: Path, payload: Dict[str, Any]) -> None:
        if not self.cfg.save_best:
            return

        val_stats = payload.get("val_stats", {}) or {}
        if self.cfg.monitor not in val_stats:
            return

        score = float(val_stats[self.cfg.monitor])
        if self._is_better(score):
            self.best_score = score
            best_path = self.save_dir / "best.pth"
            shutil.copyfile(ckpt_path, best_path)
            self.best_path = str(best_path)
            self._write_meta()

    def save_before_validation(
        self,
        global_iter: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        train_stats: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ckpt_path = self._checkpoint_path(global_iter)

        merged_extra = dict(extra or {})
        merged_extra["val_status"] = "pending"

        payload = self._build_payload(
            global_iter=global_iter,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            train_stats=train_stats,
            val_stats={},
            extra=merged_extra,
        )

        self._save_payload(payload, ckpt_path)
        self._update_latest(ckpt_path)
        self._prune_old_checkpoints()
        return ckpt_path

    def finalize_after_validation(
        self,
        ckpt_path: str | Path,
        val_stats: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu")

        payload["val_stats"] = dict(val_stats or {})

        merged_extra = dict(payload.get("extra", {}) or {})
        merged_extra.update(dict(extra or {}))
        merged_extra["val_status"] = "done"
        payload["extra"] = merged_extra

        self._save_payload(payload, ckpt_path)
        self._update_best_from_checkpoint(ckpt_path, payload)
        return ckpt_path

    def _prune_old_checkpoints(self) -> None:
        ckpts = sorted(self.save_dir.glob("iter_*.pth"))

        if self.cfg.max_keep <= 0:
            to_remove = ckpts
        else:
            if len(ckpts) <= self.cfg.max_keep:
                return
            to_remove = ckpts[:-self.cfg.max_keep]

        for p in to_remove:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=strict)

        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            enforce_optimizer_param_group_invariants(optimizer)

        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        if (
            scheduler is not None
            and ckpt.get("scheduler") is not None
            and hasattr(scheduler, "load_state_dict")
        ):
            scheduler.load_state_dict(ckpt["scheduler"])

        return ckpt

    def resume_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        latest_path = self.save_dir / "latest.pth"
        if not latest_path.exists():
            return None
        return self.load(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            strict=strict,
        )