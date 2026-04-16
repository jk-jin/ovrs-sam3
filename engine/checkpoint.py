from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class CheckpointManagerConfig:
    save_dir: str
    monitor: str = "total_loss"
    mode: str = "min"
    max_keep: int = 5
    save_latest: bool = True
    save_best: bool = True


class CheckpointManager:
    def __init__(self, cfg: CheckpointManagerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.save_dir / "checkpoint_meta.json"
        self.best_score = None
        self.best_path = None
        self._load_meta()

    def _load_meta(self):
        if not self.meta_path.exists():
            return
        try:
            meta = json.loads(self.meta_path.read_text())
            self.best_score = meta.get("best_score")
            self.best_path = meta.get("best_path")
        except Exception:
            self.best_score = None
            self.best_path = None

    def _write_meta(self):
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

    def save(
        self,
        global_iter: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        train_stats: Optional[Dict[str, float]] = None,
        val_stats: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ckpt_path = self.save_dir / f"iter_{global_iter:07d}.pth"
        payload = {
            "global_iter": int(global_iter),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None and hasattr(scheduler, "state_dict")
                else None
            ),
            "train_stats": train_stats or {},
            "val_stats": val_stats or {},
            "extra": extra or {},
        }
        torch.save(payload, ckpt_path)

        if self.cfg.save_latest:
            latest_path = self.save_dir / "latest.pth"
            shutil.copyfile(ckpt_path, latest_path)

        if self.cfg.save_best:
            score_source = val_stats if val_stats else (train_stats or {})
            if self.cfg.monitor in score_source:
                score = float(score_source[self.cfg.monitor])
                if self._is_better(score):
                    self.best_score = score
                    best_path = self.save_dir / "best.pth"
                    shutil.copyfile(ckpt_path, best_path)
                    self.best_path = str(best_path)
                    self._write_meta()

        self._prune_old_checkpoints()
        return ckpt_path

    def _prune_old_checkpoints(self):
        ckpts = sorted(self.save_dir.glob("iter_*.pth"))
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