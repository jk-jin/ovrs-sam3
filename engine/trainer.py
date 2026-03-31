from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.amp import GradScaler, autocast

from ..losses.target_converter import TargetConverter
from .checkpoint import CheckpointManager, CheckpointManagerConfig
from .evaluator import evaluate_model
from .hooks import Hook, HookManager, LoggerHook
from .visualization import VisualizationManager


@dataclass
class TrainerConfig:
    task: str = 'instance'  # instance | semantic | hybrid
    max_epochs: int = 12
    log_interval: int = 20
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 0.1
    save_dir: str = './work_dirs/default'
    save_interval: int = 1
    eval_interval: int = 1
    monitor: str = 'total_loss'
    monitor_mode: str = 'min'
    max_keep_ckpts: int = 5
    device: str = 'cuda'
    auto_resume: bool = False


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: Iterable,
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
        self.scaler = GradScaler(device='cuda', enabled=self.cfg.use_amp and self.device.type == 'cuda')
        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        self.visualizer = visualizer

        self.hook_manager = HookManager(hooks or [LoggerHook(interval=self.cfg.log_interval)])
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
            self.start_epoch = int(ckpt.get('epoch', 0)) + 1
            print(f'Auto resumed from latest checkpoint, starting at epoch={self.start_epoch}')
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
        self.start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f'Resumed from {path}, starting at epoch={self.start_epoch}')
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

    def _compute_losses(self, batch) -> Dict[str, torch.Tensor]:
        task = self.cfg.task
        if task == 'instance':
            outputs = self.model(batch, mode='instance')
            targets = TargetConverter.from_batch(batch, task='instance')
            return self.criterion(outputs['instance_outputs'], targets)
        if task == 'semantic':
            outputs = self.model(batch, mode='semantic')
            targets = TargetConverter.from_batch(batch, task='semantic')
            return self.criterion(outputs['semantic_outputs'], targets)
        if task == 'hybrid':
            outputs = self.model(batch, mode='hybrid')
            instance_targets = TargetConverter.from_batch(batch, task='instance')
            semantic_targets = TargetConverter.from_batch(batch, task='semantic')
            return self.criterion(outputs, instance_targets, semantic_targets)
        raise ValueError(f'Unsupported task: {task}')

    def train_step(self, batch) -> Dict[str, float]:
        self.model.train()
        batch = self._move_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)

        use_amp = self.cfg.use_amp and self.device.type == 'cuda'
        with autocast(device_type=self.device.type, enabled=use_amp):
            loss_dict = self._compute_losses(batch)
            total_loss = loss_dict['total_loss']

        self.scaler.scale(total_loss).backward()
        if self.cfg.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {k: float(v.detach().item()) for k, v in loss_dict.items() if torch.is_tensor(v) and v.ndim == 0}

    @torch.no_grad()
    def val_step(self, batch) -> Dict[str, float]:
        self.model.eval()
        batch = self._move_to_device(batch)
        use_amp = self.cfg.use_amp and self.device.type == 'cuda'
        with autocast(device_type=self.device.type, enabled=use_amp):
            loss_dict = self._compute_losses(batch)
        return {k: float(v.detach().item()) for k, v in loss_dict.items() if torch.is_tensor(v) and v.ndim == 0}

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

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.hook_manager.call('before_train_epoch', self, epoch)
        epoch_stats: list[Dict[str, float]] = []
        tic = time.time()
        for it, batch in enumerate(self.train_dataloader, start=1):
            self.hook_manager.call('before_train_iter', self, epoch, it, batch)
            stats = self.train_step(batch)
            epoch_stats.append(stats)
            self.hook_manager.call('after_train_iter', self, epoch, it, batch, stats)
            if it % self.cfg.log_interval == 0:
                elapsed = time.time() - tic
                msg = f'epoch={epoch} iter={it} time={elapsed:.1f}s'
                if 'total_loss' in stats:
                    msg += f" total_loss={stats['total_loss']:.4f}"
                print(msg)
                tic = time.time()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        train_stats = self._average_stats(epoch_stats)
        self.hook_manager.call('after_train_epoch', self, epoch, train_stats)
        return train_stats

    def val_epoch(self, epoch: int) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}
        stats_list = [self.val_step(batch) for batch in self.val_dataloader]
        loss_stats = self._average_stats(stats_list)
        metric_stats = evaluate_model(
            self.model,
            self.val_dataloader,
            task=self.cfg.task,
            device=self.device,
            visualizer=self.visualizer,
            epoch=epoch,
            stage='val',
        )
        stats = {**loss_stats, **metric_stats}
        if stats:
            printable = ', '.join(f'{k}={v:.4f}' for k, v in stats.items())
            print(f'val epoch={epoch}: {printable}')
        self.hook_manager.call('after_val_epoch', self, epoch, stats)
        return stats

    def save_checkpoint(self, epoch: int, train_stats: Dict[str, float], val_stats: Optional[Dict[str, float]] = None) -> Path:
        ckpt_path = self.checkpoint_manager.save(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.lr_scheduler,
            train_stats=train_stats,
            val_stats=val_stats or {},
            extra={'task': self.cfg.task},
        )
        self.hook_manager.call('after_save_checkpoint', self, epoch, str(ckpt_path))
        return ckpt_path

    def train(self):
        self.hook_manager.call('before_run', self)
        self.maybe_resume_latest()
        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            train_stats = self.train_epoch(epoch)
            val_stats = {}
            if self.val_dataloader is not None and epoch % self.cfg.eval_interval == 0:
                val_stats = self.val_epoch(epoch)
            if epoch % self.cfg.save_interval == 0:
                path = self.save_checkpoint(epoch, train_stats, val_stats)
                print(f'saved checkpoint: {path}')
        self.hook_manager.call('after_run', self)
