from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..config_dataclasses import CheckpointManagerConfig
from .optimizer_builder import enforce_optimizer_param_group_invariants


_CHECKPOINT_VERSION = 4


# ---------------------------------------------------------------------------
# Unified safe checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint_file(path: str | Path) -> Dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a dict, got {type(checkpoint)}."
        )
    return checkpoint


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, cfg: CheckpointManagerConfig):
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = None

    # ------------------------------------------------------------------
    # Atomic file operations
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_save(payload: Dict[str, Any], target: Path) -> Path:
        target = Path(target)
        target_dir = target.parent
        fd, tmp_path = tempfile.mkstemp(
            suffix=".pth",
            prefix=".tmp_ckpt_",
            dir=str(target_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_path)
        try:
            torch.save(payload, tmp_path)
            os.replace(str(tmp_path), str(target))
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return target

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> Path:
        source = Path(source)
        target = Path(target)
        target_dir = target.parent
        fd, tmp_path = tempfile.mkstemp(
            suffix=".pth",
            prefix=".tmp_latest_",
            dir=str(target_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_path)
        try:
            shutil.copy2(str(source), str(tmp_path))
            os.replace(str(tmp_path), str(target))
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return target

    # ------------------------------------------------------------------
    # Checkpoint manager state (persisted inside checkpoint)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_score": (
                None if self.best_score is None else float(self.best_score)
            ),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        best_score = state.get("best_score")
        self.best_score = (
            None if best_score is None else float(best_score)
        )

    # ------------------------------------------------------------------
    # Score comparison
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Unified save
    # ------------------------------------------------------------------

    def save(
        self,
        global_iter: int,
        model: torch.nn.Module,
        checkpoint_reason: str = "periodic",
        val_status: str = "not_due",
        runtime_state: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        train_stats: Optional[Dict[str, float]] = None,
        val_stats: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ckpt_path = self._checkpoint_path(global_iter)

        merged_extra = dict(extra or {})
        merged_extra["checkpoint_reason"] = str(checkpoint_reason)
        merged_extra["val_status"] = str(val_status)

        payload = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "global_iter": int(global_iter),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None and hasattr(scheduler, "state_dict")
                else None
            ),
            "runtime_state": dict(runtime_state or {}),
            "checkpoint_manager": self.state_dict(),
            "train_stats": dict(train_stats or {}),
            "val_stats": dict(val_stats or {}),
            "extra": merged_extra,
        }

        self._atomic_save(payload, ckpt_path)
        self._update_latest(ckpt_path)
        self._prune_old_checkpoints()
        return ckpt_path

    # ------------------------------------------------------------------
    # Finalize after validation
    # ------------------------------------------------------------------

    def finalize_after_validation(
        self,
        ckpt_path: str | Path,
        val_stats: Optional[Dict[str, float]] = None,
        runtime_state: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        payload = load_checkpoint_file(ckpt_path)

        # 1. Write val_stats.
        payload["val_stats"] = dict(val_stats or {})

        # 2. Determine if new best — compare ONCE before updating best_score.
        is_new_best = False
        val_stats_dict = payload["val_stats"] or {}
        if self.cfg.monitor in val_stats_dict:
            score = float(val_stats_dict[self.cfg.monitor])
            if self._is_better(score):
                self.best_score = score
                is_new_best = True

        # 3. Update checkpoint_manager state in payload.
        payload["checkpoint_manager"] = self.state_dict()

        # 4. Update extra.
        merged_extra = dict(payload.get("extra", {}) or {})
        merged_extra.update(dict(extra or {}))
        merged_extra["val_status"] = "done"
        payload["extra"] = merged_extra

        # 5. Update runtime_state if provided.
        if runtime_state is not None:
            payload["runtime_state"] = dict(runtime_state)

        # 6. Atomic save.
        self._atomic_save(payload, ckpt_path)

        # 7. Update latest.
        self._update_latest(ckpt_path)

        # 8. Copy to best if this is a new best.
        if is_new_best:
            best_path = self.save_dir / "best.pth"
            self._atomic_copy(ckpt_path, best_path)

        return ckpt_path

    # ------------------------------------------------------------------
    # latest.pth — atomic file copy
    # ------------------------------------------------------------------

    def _update_latest(self, ckpt_path: Path) -> None:
        latest_path = self.save_dir / "latest.pth"
        self._atomic_copy(ckpt_path, latest_path)

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Load (strict resume only)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_version(ckpt: Dict[str, Any], path: str | Path) -> None:
        version = ckpt.get("checkpoint_version", None)
        if version != _CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint {path} has checkpoint_version={version}, "
                f"but this code requires version {_CHECKPOINT_VERSION}. "
                "Old checkpoints cannot be used for full resume."
            )

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        ckpt = load_checkpoint_file(path)
        self._validate_version(ckpt, path)

        # --- Schema validation for resume ---
        # A checkpoint used for full resume must contain runtime_state
        # (with rng), checkpoint_manager, and WandbHook.last_history_step.
        runtime_state = ckpt.get("runtime_state")
        if runtime_state is None:
            raise ValueError(
                f"Checkpoint {path} does not contain 'runtime_state'. "
                "This is required for full resume."
            )
        rng_state = runtime_state.get("rng")
        if rng_state is None:
            raise ValueError(
                f"Checkpoint {path} runtime_state does not contain 'rng'. "
                "This is required for full resume."
            )
        data_state = runtime_state.get("data")
        if data_state is None or data_state.get("sampler") is None:
            raise ValueError(
                f"Checkpoint {path} runtime_state does not contain "
                "'data.sampler'. This is required for full resume."
            )

        ckpt_mgr_state = ckpt.get("checkpoint_manager")
        if ckpt_mgr_state is None:
            raise ValueError(
                f"Checkpoint {path} does not contain 'checkpoint_manager'. "
                "This is required for full resume."
            )
        self.load_state_dict(ckpt_mgr_state)

        # Check that optimizer/scheduler/scaler state matches presence.
        if optimizer is not None and ckpt.get("optimizer") is None:
            raise ValueError(
                f"Checkpoint {path} has no optimizer state, "
                "but this trainer has an optimizer."
            )
        if optimizer is None and ckpt.get("optimizer") is not None:
            raise ValueError(
                f"Checkpoint {path} has optimizer state, "
                "but this trainer has no optimizer."
            )
        if scheduler is not None and ckpt.get("scheduler") is None:
            raise ValueError(
                f"Checkpoint {path} has no scheduler state, "
                "but this trainer has a scheduler."
            )

        # --- Restore ---
        model.load_state_dict(ckpt["model"], strict=True)

        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            enforce_optimizer_param_group_invariants(optimizer)

        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        if (
            scheduler is not None
            and hasattr(scheduler, "load_state_dict")
        ):
            scheduler.load_state_dict(ckpt["scheduler"])

        return ckpt
