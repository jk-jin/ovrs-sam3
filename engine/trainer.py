from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.amp import GradScaler, autocast

from .checkpoint import CheckpointManager, CheckpointManagerConfig
from .evaluator import (
    MulticlassSemanticEvaluator,
    extract_class_names_from_batch,
    extract_semantic_targets_from_batch,
    inference_with_tta,
)
from .hooks import Hook, HookManager
from .visualization import VisualizationManager


@dataclass
class TrainerConfig:
    max_epochs: int = 12
    log_window_size: int = 20
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 0.1
    save_dir: str = "./work_dirs/default"
    save_interval: int = 1
    eval_interval: int = 1
    monitor: str = "semantic.miou"
    monitor_mode: str = "max"
    max_keep_ckpts: int = 5
    device: str = "cuda"
    auto_resume: bool = False
    tta_cfg: Optional[Dict] = None
    eval_cfg: Optional[Dict] = None


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        criterion: torch.nn.Module,
        train_dataloader: Optional[Iterable],
        val_dataloader: Optional[Iterable] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        cfg: Optional[TrainerConfig] = None,
        hooks: Optional[Sequence[Hook]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        visualizer: Optional[VisualizationManager] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg or TrainerConfig()
        self.device = torch.device(self.cfg.device)
        self.scaler = GradScaler(
            device="cuda",
            enabled=self.cfg.use_amp and self.device.type == "cuda",
        )
        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        self.visualizer = visualizer

        self.hook_manager = HookManager(hooks or [])
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            CheckpointManagerConfig(
                save_dir=str(self.save_dir),
                monitor=self.cfg.monitor,
                mode=self.cfg.monitor_mode,
                max_keep=self.cfg.max_keep_ckpts,
                save_latest=True,
                save_best=True,
            )
        )

        self.start_epoch = 1

        self.iters_per_epoch = None
        if self.train_dataloader is not None and hasattr(self.train_dataloader, "__len__"):
            self.iters_per_epoch = len(self.train_dataloader)

        self.max_iters = None
        if self.iters_per_epoch is not None:
            self.max_iters = self.cfg.max_epochs * self.iters_per_epoch

        self.global_iter = 0
        self.log_state: Dict[str, object] = {}
        self._log_getters = []

        self._iter_time_history = deque(maxlen=self.cfg.log_window_size)
        self._data_time_history = deque(maxlen=self.cfg.log_window_size)
        self._train_stat_history = deque(maxlen=self.cfg.log_window_size)

        self.val_iters_per_epoch = None
        if self.val_dataloader is not None and hasattr(self.val_dataloader, "__len__"):
            self.val_iters_per_epoch = len(self.val_dataloader)

        self._val_iter_time_history = deque(maxlen=self.cfg.log_window_size)
        self._val_data_time_history = deque(maxlen=self.cfg.log_window_size)
        self._val_stat_history = deque(maxlen=self.cfg.log_window_size)

    def maybe_resume_latest(self):
        if not self.cfg.auto_resume:
            return None

        ckpt = self.checkpoint_manager.resume_latest(
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.lr_scheduler,
            strict=False,
        )
        if ckpt is not None:
            self.start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"Auto resumed from latest checkpoint, starting at epoch={self.start_epoch}")
        return ckpt

    def resume_from(self, path: str):
        ckpt = self.checkpoint_manager.load(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.lr_scheduler,
            strict=False,
        )
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {path}, starting at epoch={self.start_epoch}")
        return ckpt

    def _move_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)

        if is_dataclass(obj):
            for field in fields(obj):
                setattr(obj, field.name, self._move_to_device(getattr(obj, field.name)))
            return obj

        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)

        return obj

    @staticmethod
    def _empty_loss_sums() -> Dict[str, float]:
        return {
            "loss_semantic_bce": 0.0,
            "loss_semantic_dice": 0.0,
            "loss_instance_bce": 0.0,
            "loss_instance_dice": 0.0,
            "loss_presence_bce": 0.0,
            "total_loss": 0.0,
        }

    @staticmethod
    def _normalize_loss_sums(
        loss_sums: Dict[str, float],
        total_valid_pixels: int,
    ) -> Dict[str, float]:
        if total_valid_pixels <= 0:
            return {
                "loss_semantic_bce": 0.0,
                "loss_semantic_dice": 0.0,
                "loss_instance_bce": 0.0,
                "loss_instance_dice": 0.0,
                "loss_presence_bce": 0.0,
                "total_loss": 0.0,
            }

        return {
            key: float(value) / float(total_valid_pixels)
            for key, value in loss_sums.items()
        }

    def _compute_chunk_loss_sums(
        self,
        batch,
        do_backward: bool,
    ) -> tuple[Dict[str, float], int, bool]:
        if not hasattr(self.model, "iter_chunk_outputs"):
            raise AttributeError(
                "Model does not provide iter_chunk_outputs(batch). "
                "The chunked training pipeline requires this interface."
            )

        label_map = batch.find_targets[0].semantic_label_map
        use_amp = self.cfg.use_amp and self.device.type == "cuda"

        loss_sums = self._empty_loss_sums()
        total_valid_pixels = 0
        did_backward = False

        chunk_iter = self.model.iter_chunk_outputs(batch)

        while True:
            try:
                with autocast(device_type=self.device.type, enabled=use_amp):
                    chunk = next(chunk_iter)
                    loss_dict = self.criterion(
                        chunk["train_outputs"],
                        {"label_map": label_map},
                        chunk_class_ids=chunk["chunk_class_ids"],
                        reduction="sum",
                    )
                    chunk_total_loss = loss_dict["total_loss"]
            except StopIteration:
                break

            chunk_num_valid = int(loss_dict["num_valid"].detach().item())

            loss_sums["loss_semantic_bce"] += float(loss_dict["loss_semantic_bce"].detach().item())
            loss_sums["loss_semantic_dice"] += float(loss_dict["loss_semantic_dice"].detach().item())
            loss_sums["loss_instance_bce"] += float(loss_dict["loss_instance_bce"].detach().item())
            loss_sums["loss_instance_dice"] += float(loss_dict["loss_instance_dice"].detach().item())
            loss_sums["loss_presence_bce"] += float(loss_dict["loss_presence_bce"].detach().item())
            loss_sums["total_loss"] += float(loss_dict["total_loss"].detach().item())

            total_valid_pixels += chunk_num_valid

            if do_backward and chunk_num_valid > 0:
                self.scaler.scale(chunk_total_loss).backward()
                did_backward = True

            del chunk
            del loss_dict
            del chunk_total_loss

        return loss_sums, total_valid_pixels, did_backward

    def train_step(self, batch) -> Dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is None, cannot run train_step().")

        batch = self._move_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)

        loss_sums, total_valid_pixels, did_backward = self._compute_chunk_loss_sums(
            batch=batch,
            do_backward=True,
        )

        if did_backward and total_valid_pixels > 0:
            self.scaler.unscale_(self.optimizer)

            grad_scale = 1.0 / float(total_valid_pixels)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(grad_scale)

            if self.cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        return self._normalize_loss_sums(
            loss_sums=loss_sums,
            total_valid_pixels=total_valid_pixels,
        )

    def _forward_val_outputs(self, batch) -> Dict[str, torch.Tensor]:
        use_amp = self.cfg.use_amp and self.device.type == "cuda"
        with autocast(device_type=self.device.type, enabled=use_amp):
            outputs = inference_with_tta(self.model, batch, tta_cfg=self.cfg.tta_cfg)
        return outputs

    def _compute_val_losses(self, batch) -> Dict[str, float]:
        loss_sums, total_valid_pixels, _ = self._compute_chunk_loss_sums(
            batch=batch,
            do_backward=False,
        )

        return self._normalize_loss_sums(
            loss_sums=loss_sums,
            total_valid_pixels=total_valid_pixels,
        )

    @staticmethod
    def _average_stats(stats_list: list[Dict[str, float]]) -> Dict[str, float]:
        if not stats_list:
            return {}

        keys = sorted({k for stats in stats_list for k in stats.keys()})
        out: Dict[str, float] = {}
        for k in keys:
            vals = [s[k] for s in stats_list if k in s and not math.isnan(s[k])]
            if vals:
                out[k] = sum(vals) / len(vals)
        return out

    def _get_current_lrs(self) -> list[float]:
        if self.optimizer is None:
            return []
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _get_memory_mb(self) -> Optional[int]:
        if self.device.type != "cuda":
            return None
        return int(torch.cuda.max_memory_allocated(self.device) / 1024 / 1024)

    @staticmethod
    def _mean_of_history(values) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def register_log_getter(self, fn):
        if fn is None:
            return
        self._log_getters.append(fn)

    def _to_loggable_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().item())
            return None
        if isinstance(value, (float, int, bool, str)):
            return value
        return None

    def _collect_extra_log_vars(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for fn in self._log_getters:
            try:
                values = fn(self)
            except Exception as e:
                out[f"log_getter_error_{len(out)}"] = str(e)
                continue

            if not isinstance(values, dict):
                continue

            for k, v in values.items():
                vv = self._to_loggable_scalar(v)
                if vv is not None:
                    out[str(k)] = vv
        return out

    def _update_train_log_state(
        self,
        epoch: int,
        step: int,
        stats: Dict[str, float],
        data_time: float,
        iter_time: float,
    ) -> None:
        self._data_time_history.append(float(data_time))
        self._iter_time_history.append(float(iter_time))
        self._train_stat_history.append(dict(stats))

        avg_data_time = self._mean_of_history(self._data_time_history)
        avg_iter_time = self._mean_of_history(self._iter_time_history)
        avg_stats = self._average_stats(list(self._train_stat_history))

        eta_seconds = None
        if self.max_iters is not None:
            remaining_iters = max(self.max_iters - self.global_iter, 0)
            eta_seconds = avg_iter_time * remaining_iters

        self.log_state = {
            "mode": "train",
            "epoch": int(epoch),
            "max_epochs": int(self.cfg.max_epochs),
            "iter": int(step),
            "iters_per_epoch": self.iters_per_epoch,
            "global_iter": int(self.global_iter),
            "max_iters": self.max_iters,
            "lrs": self._get_current_lrs(),
            "eta_seconds": eta_seconds,
            "iter_time": avg_iter_time,
            "data_time": avg_data_time,
            "memory_mb": self._get_memory_mb(),
            "log_vars": avg_stats,
            "extra_log_vars": self._collect_extra_log_vars(),
        }

    def _update_val_log_state(
        self,
        epoch: int,
        step: int,
        stats: Dict[str, float],
        data_time: float,
        iter_time: float,
    ) -> None:
        self._val_data_time_history.append(float(data_time))
        self._val_iter_time_history.append(float(iter_time))
        self._val_stat_history.append(dict(stats))

        avg_data_time = self._mean_of_history(self._val_data_time_history)
        avg_iter_time = self._mean_of_history(self._val_iter_time_history)
        avg_stats = self._average_stats(list(self._val_stat_history))

        eta_seconds = None
        if self.val_iters_per_epoch is not None:
            remaining_iters = max(self.val_iters_per_epoch - step, 0)
            eta_seconds = avg_iter_time * remaining_iters

        self.log_state = {
            "mode": "val_loss",
            "epoch": int(epoch),
            "max_epochs": int(self.cfg.max_epochs),
            "iter": int(step),
            "iters_per_epoch": self.val_iters_per_epoch,
            "eta_seconds": eta_seconds,
            "iter_time": avg_iter_time,
            "data_time": avg_data_time,
            "log_vars": avg_stats,
            "extra_log_vars": self._collect_extra_log_vars(),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.train_dataloader is None:
            return {}

        self.hook_manager.call("before_train_epoch", self, epoch)
        self.model.train()
        epoch_stats: list[Dict[str, float]] = []

        end = time.perf_counter()

        for it, batch in enumerate(self.train_dataloader, start=1):
            data_time = time.perf_counter() - end

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            self.hook_manager.call("before_train_iter", self, epoch, it, batch)

            stats = self.train_step(batch)
            epoch_stats.append(stats)

            iter_time = time.perf_counter() - end

            self.global_iter += 1
            self._update_train_log_state(
                epoch=epoch,
                step=it,
                stats=stats,
                data_time=data_time,
                iter_time=iter_time,
            )

            self.hook_manager.call("after_train_iter", self, epoch, it, batch, stats)

            end = time.perf_counter()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        train_stats = self._average_stats(epoch_stats)
        self.hook_manager.call("after_train_epoch", self, epoch, train_stats)
        return train_stats

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}

        self.hook_manager.call("before_val_epoch", self, epoch)

        self.model.eval()
        self._val_iter_time_history.clear()
        self._val_data_time_history.clear()
        self._val_stat_history.clear()

        eval_cfg = dict(self.cfg.eval_cfg or {})
        evaluator = MulticlassSemanticEvaluator(**eval_cfg)
        stats_list: list[Dict[str, float]] = []
        class_names = None

        end = time.perf_counter()

        for it, batch in enumerate(self.val_dataloader, start=1):
            data_time = time.perf_counter() - end

            batch = self._move_to_device(batch)

            outputs = self._forward_val_outputs(batch)
            loss_stats = self._compute_val_losses(batch)
            stats_list.append(loss_stats)

            targets = extract_semantic_targets_from_batch(batch)
            evaluator.update(outputs, targets)

            if class_names is None:
                class_names = extract_class_names_from_batch(batch)

            if self.visualizer is not None:
                self.visualizer.run(
                    model=self.model,
                    batch=batch,
                    semantic_outputs=outputs,
                    semantic_targets=targets,
                    epoch=epoch,
                    stage="val",
                )

            iter_time = time.perf_counter() - end

            self._update_val_log_state(
                epoch=epoch,
                step=it,
                stats=loss_stats,
                data_time=data_time,
                iter_time=iter_time,
            )

            self.hook_manager.call("after_val_iter", self, epoch, it, batch, loss_stats)

            end = time.perf_counter()

        loss_stats = self._average_stats(stats_list)
        metric_stats = evaluator.compute()

        stats = {**loss_stats, **metric_stats}
        if class_names is not None:
            stats["_class_names"] = class_names

        self.hook_manager.call("after_val_epoch", self, epoch, stats)
        return stats

    def save_checkpoint(
        self,
        epoch: int,
        train_stats: Dict[str, float],
        val_stats: Optional[Dict[str, float]] = None,
    ) -> Path:
        ckpt_path = self.checkpoint_manager.save(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.lr_scheduler,
            train_stats=train_stats,
            val_stats=val_stats or {},
            extra={
                "monitor": self.cfg.monitor,
                "monitor_mode": self.cfg.monitor_mode,
            },
        )
        return ckpt_path

    def train(self):
        self.hook_manager.call("before_run", self)
        self.maybe_resume_latest()

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            train_stats = self.train_epoch(epoch)

            should_save = (epoch % self.cfg.save_interval == 0)
            should_eval = (
                    self.val_dataloader is not None
                    and epoch % self.cfg.eval_interval == 0
            )

            if should_save:
                path = self.save_checkpoint(epoch, train_stats, {})
                print(f"saved checkpoint before validation: {path}")

            val_stats = {}
            if should_eval:
                val_stats = self.val_epoch(epoch)

                if should_save:
                    path = self.save_checkpoint(epoch, train_stats, val_stats)
                    print(f"updated checkpoint after validation: {path}")

        self.hook_manager.call("after_run", self)