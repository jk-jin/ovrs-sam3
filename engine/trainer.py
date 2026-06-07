from __future__ import annotations

import time
from collections import deque
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.amp import GradScaler, autocast

from ..models.task_modes import OUTPUT_KEYS
from ..config_dataclasses import TrainerConfig
from .checkpoint import CheckpointManager
from .evaluator import (
    MulticlassSemanticEvaluator,
    extract_class_names_from_batch,
    extract_semantic_targets_from_batch,
    inference_with_tta,
)
from .hooks import Hook, HookManager
from .visualization import VisualizationManager

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        criterion: torch.nn.Module,
        train_dataloader: Optional[Iterable],
        checkpoint_manager: CheckpointManager,
        val_dataloader: Optional[Iterable] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        cfg: Optional[TrainerConfig] = None,
        hooks: Optional[Sequence[Hook]] = None,
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
        self.checkpoint_manager = checkpoint_manager

        self.global_iter = 0

        self.iters_per_cycle = None
        if self.train_dataloader is not None and hasattr(self.train_dataloader, "__len__"):
            self.iters_per_cycle = len(self.train_dataloader)

        self.val_iters_per_epoch = None
        if self.val_dataloader is not None and hasattr(self.val_dataloader, "__len__"):
            self.val_iters_per_epoch = len(self.val_dataloader)

        self.log_state: Dict[str, object] = {}
        self._log_getters = []

        self._iter_time_history = deque(maxlen=self.cfg.log_window_size)
        self._data_time_history = deque(maxlen=self.cfg.log_window_size)
        self._train_stat_history = deque(maxlen=self.cfg.log_window_size)

        self._val_iter_time_history = deque(maxlen=self.cfg.log_window_size)
        self._val_data_time_history = deque(maxlen=self.cfg.log_window_size)
        self._val_metric_history = deque(maxlen=self.cfg.log_window_size)

        self._train_iterator = None
        self._data_cycle = 0

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
            self.global_iter = int(ckpt.get("global_iter", 0))
            print(f"Auto resumed from latest checkpoint, starting at iter={self.global_iter}")
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
        self.global_iter = int(ckpt.get("global_iter", 0))
        print(f"Resumed from {path}, starting at iter={self.global_iter}")
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

    def _compute_train_loss(self, batch) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if not hasattr(self.model, "build_encoder_refiner_cache"):
            raise AttributeError(
                "Model must provide build_encoder_refiner_cache(batch)."
            )

        if not hasattr(self.model, "run_encoder_refiner_from_cache"):
            raise AttributeError(
                "Model must provide run_encoder_refiner_from_cache(encoder_refiner_cache, batch)."
            )

        label_map = batch.find_targets[0].semantic_label_map
        use_amp = self.cfg.use_amp and self.device.type == "cuda"

        with autocast(device_type=self.device.type, enabled=use_amp):
            encoder_refiner_cache = self.model.build_encoder_refiner_cache(batch)

            final_raw_outputs = self.model.run_encoder_refiner_from_cache(
                encoder_refiner_cache=encoder_refiner_cache,
                batch=batch,
            )

            final_outputs = self.model.adapter(
                raw_outputs=final_raw_outputs,
                batch=batch,
                expected_num_classes=None,
                output_mode="final",
            )

            loss_dict = self.criterion(
                final_outputs,
                {"label_map": label_map},
                reduction="mean",
            )

        if "total_loss" not in loss_dict:
            raise ValueError("Criterion must return 'total_loss'.")
        if "num_valid" not in loss_dict:
            raise ValueError("Criterion must return 'num_valid'.")

        return loss_dict, loss_dict["total_loss"]

    def train_step(self, batch) -> tuple[Dict[str, float], bool]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is None, cannot run train_step().")

        batch = self._move_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)

        loss_dict, total_loss = self._compute_train_loss(batch)

        num_valid = int(loss_dict["num_valid"].detach().item())
        did_step = False

        if num_valid > 0:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.grad_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            did_step = True

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        stats = {}
        for key, value in loss_dict.items():
            if key == "num_valid":
                continue
            if torch.is_tensor(value):
                stats[key] = float(value.detach().item())
            else:
                stats[key] = float(value)

        return stats, did_step

    def _forward_val_outputs(self, batch) -> Dict[str, torch.Tensor]:
        use_amp = self.cfg.use_amp and self.device.type == "cuda"
        with autocast(device_type=self.device.type, enabled=use_amp):
            outputs = inference_with_tta(self.model, batch, tta_cfg=self.cfg.tta_cfg)
        return outputs

    @staticmethod
    def _average_stats(stats_list: list[Dict[str, float]]) -> Dict[str, float]:
        if not stats_list:
            return {}

        keys = sorted({k for stats in stats_list for k in stats.keys()})
        out: Dict[str, float] = {}
        for k in keys:
            vals = [s[k] for s in stats_list if k in s]
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

    @staticmethod
    def _extract_class_names_from_dataloader(dataloader) -> Optional[list[str]]:
        if dataloader is None:
            return None

        dataset = getattr(dataloader, "dataset", None)
        while dataset is not None and hasattr(dataset, "dataset"):
            dataset = dataset.dataset

        if dataset is None:
            return None

        classes = getattr(dataset, "classes", None)
        if classes is None:
            return None

        classes = [str(x) for x in classes]
        if len(classes) == 0:
            return None

        return classes

    def _prepare_text_cache_for_dataloader(
        self,
        dataloader,
        force: bool = False,
    ) -> None:
        if dataloader is None:
            return

        if not hasattr(self.model, "prepare_text_cache"):
            return

        class_names = self._extract_class_names_from_dataloader(dataloader)
        if class_names is None:
            return

        self.model.prepare_text_cache(
            class_names=class_names,
            device=self.device,
            force=force,
        )

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

    def _estimate_data_cycle(self) -> Optional[int]:
        if self.iters_per_cycle is None or self.iters_per_cycle <= 0:
            return None
        return (self.global_iter // self.iters_per_cycle) + 1

    def _update_train_log_state(
        self,
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

        remaining_iters = max(self.cfg.max_iters - self.global_iter, 0)
        eta_seconds = avg_iter_time * remaining_iters

        self.log_state = {
            "mode": "train",
            "iter": int(self.global_iter),
            "max_iters": int(self.cfg.max_iters),
            "data_cycle": self._estimate_data_cycle(),
            "iters_per_cycle": self.iters_per_cycle,
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
        val_step: int,
        metric_stats_snapshot: Dict[str, float],
        data_time: float,
        iter_time: float,
    ) -> None:
        self._val_data_time_history.append(float(data_time))
        self._val_iter_time_history.append(float(iter_time))
        self._val_metric_history.append(dict(metric_stats_snapshot))

        avg_data_time = self._mean_of_history(self._val_data_time_history)
        avg_iter_time = self._mean_of_history(self._val_iter_time_history)
        avg_metrics = self._average_stats(list(self._val_metric_history))

        eta_seconds = None
        if self.val_iters_per_epoch is not None:
            remaining_iters = max(self.val_iters_per_epoch - val_step, 0)
            eta_seconds = avg_iter_time * remaining_iters

        self.log_state = {
            "mode": "val",
            "iter": int(self.global_iter),
            "max_iters": int(self.cfg.max_iters),
            "val_iter": int(val_step),
            "val_total_iters": self.val_iters_per_epoch,
            "eta_seconds": eta_seconds,
            "iter_time": avg_iter_time,
            "data_time": avg_data_time,
            "log_vars": avg_metrics,
            "extra_log_vars": self._collect_extra_log_vars(),
        }

    def _set_dataloader_cycle(self, cycle_index: int) -> None:
        if self.train_dataloader is None:
            return

        sampler = getattr(self.train_dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(cycle_index)

        batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(cycle_index)

    def _build_train_iterator(self) -> None:
        if self.train_dataloader is None:
            self._train_iterator = None
            return

        if self.iters_per_cycle is None or self.iters_per_cycle <= 0:
            self._data_cycle = 0
            self._set_dataloader_cycle(self._data_cycle)
            self._train_iterator = iter(self.train_dataloader)
            return

        completed_cycles = self.global_iter // self.iters_per_cycle
        offset_in_cycle = self.global_iter % self.iters_per_cycle

        self._data_cycle = int(completed_cycles)
        self._set_dataloader_cycle(self._data_cycle)
        self._train_iterator = iter(self.train_dataloader)

        for _ in range(offset_in_cycle):
            try:
                next(self._train_iterator)
            except StopIteration:
                self._data_cycle += 1
                self._set_dataloader_cycle(self._data_cycle)
                self._train_iterator = iter(self.train_dataloader)

    def _next_train_batch(self):
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is None, cannot fetch training batch.")

        if self._train_iterator is None:
            self._build_train_iterator()

        try:
            return next(self._train_iterator)
        except StopIteration:
            self._data_cycle += 1
            self._set_dataloader_cycle(self._data_cycle)
            self._train_iterator = iter(self.train_dataloader)
            return next(self._train_iterator)

    def _save_checkpoint_before_validation(
        self,
        train_stats: Dict[str, float],
    ) -> Path:
        return self.checkpoint_manager.save_before_validation(
            global_iter=self.global_iter,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.lr_scheduler,
            train_stats=train_stats,
            extra={
                "monitor": self.cfg.monitor,
                "monitor_mode": self.cfg.monitor_mode,
            },
        )

    def _finalize_checkpoint_after_validation(
        self,
        ckpt_path: Path,
        val_stats: Dict[str, float],
    ) -> Path:
        return self.checkpoint_manager.finalize_after_validation(
            ckpt_path=ckpt_path,
            val_stats=val_stats,
            extra={
                "monitor": self.cfg.monitor,
                "monitor_mode": self.cfg.monitor_mode,
            },
        )

    @torch.no_grad()
    def val(self) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}

        self._prepare_text_cache_for_dataloader(self.val_dataloader, force=False)
        self.hook_manager.call("before_val", self, self.global_iter)

        self.model.eval()
        self._val_iter_time_history.clear()
        self._val_data_time_history.clear()
        self._val_metric_history.clear()

        eval_cfg = dict(self.cfg.eval_cfg or {})
        evaluator = MulticlassSemanticEvaluator(**eval_cfg)
        class_names = None

        end = time.perf_counter()

        for it, batch in enumerate(self.val_dataloader, start=1):
            data_time = time.perf_counter() - end

            batch = self._move_to_device(batch)

            outputs = self._forward_val_outputs(batch)

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
                    epoch=self.global_iter,
                    stage="val",
                )

            metric_snapshot = evaluator.compute()
            iter_time = time.perf_counter() - end

            self._update_val_log_state(
                val_step=it,
                metric_stats_snapshot=metric_snapshot,
                data_time=data_time,
                iter_time=iter_time,
            )

            self.hook_manager.call("after_val_iter", self, self.global_iter, it, batch, metric_snapshot)

            end = time.perf_counter()

        stats = evaluator.compute()
        if class_names is not None:
            stats["_class_names"] = class_names

        self.hook_manager.call("after_val", self, self.global_iter, stats)
        return stats

    def train(self):
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is None, cannot run train().")

        self.hook_manager.call("before_run", self)
        self.maybe_resume_latest()
        self._prepare_text_cache_for_dataloader(self.train_dataloader, force=False)
        self._build_train_iterator()

        self.model.train()
        self._iter_time_history.clear()
        self._data_time_history.clear()
        self._train_stat_history.clear()

        train_stats_window: list[Dict[str, float]] = []

        end = time.perf_counter()

        while self.global_iter < self.cfg.max_iters:
            data_time = time.perf_counter() - end

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            batch = self._next_train_batch()
            next_iter = self.global_iter + 1

            self.hook_manager.call("before_train_iter", self, next_iter, batch)

            stats, _ = self.train_step(batch)
            train_stats_window.append(stats)

            self.global_iter = next_iter

            iter_time = time.perf_counter() - end

            self._update_train_log_state(
                stats=stats,
                data_time=data_time,
                iter_time=iter_time,
            )

            self.hook_manager.call("after_train_iter", self, self.global_iter, batch, stats)

            should_eval = (
                self.val_dataloader is not None
                and self.cfg.eval_interval > 0
                and self.global_iter % self.cfg.eval_interval == 0
            )
            should_save = (
                self.cfg.save_interval > 0
                and self.global_iter % self.cfg.save_interval == 0
            )

            averaged_train_stats = self._average_stats(train_stats_window)

            if should_save:
                ckpt_path = self._save_checkpoint_before_validation(averaged_train_stats)
                print(f"saved training-state checkpoint at iter={self.global_iter}: {ckpt_path}")
            else:
                ckpt_path = None

            if should_eval:
                val_stats = self.val()
                self.model.train()
            else:
                val_stats = {}

            if ckpt_path is not None and should_eval:
                self._finalize_checkpoint_after_validation(ckpt_path, val_stats)
                print(f"finalized checkpoint with validation stats at iter={self.global_iter}: {ckpt_path}")

            end = time.perf_counter()

        final_train_stats = self._average_stats(train_stats_window)

        need_final_save = (
            self.cfg.save_interval <= 0
            or self.global_iter % self.cfg.save_interval != 0
        )

        final_ckpt_path = None
        if need_final_save:
            final_ckpt_path = self._save_checkpoint_before_validation(final_train_stats)
            print(f"saved final training-state checkpoint at iter={self.global_iter}: {final_ckpt_path}")

        need_final_eval = (
            self.val_dataloader is not None
            and (
                self.cfg.eval_interval <= 0
                or self.global_iter % self.cfg.eval_interval != 0
            )
        )

        if need_final_eval:
            final_val_stats = self.val()
            self.model.train()
        else:
            final_val_stats = {}

        if final_ckpt_path is not None and need_final_eval:
            self._finalize_checkpoint_after_validation(final_ckpt_path, final_val_stats)
            print(f"finalized final checkpoint with validation stats at iter={self.global_iter}: {final_ckpt_path}")

        self.hook_manager.call("after_run", self)