from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence
from .evaluator import format_semantic_metric_tables


class Hook:
    priority = 50

    def before_run(self, trainer):
        pass

    def after_run(self, trainer):
        pass

    def before_train_epoch(self, trainer, epoch: int):
        pass

    def after_train_epoch(self, trainer, epoch: int, train_stats: Dict[str, float]):
        pass

    def before_train_iter(self, trainer, epoch: int, step: int, batch):
        pass

    def after_train_iter(self, trainer, epoch: int, step: int, batch, outputs: Dict[str, float]):
        pass

    def before_val_epoch(self, trainer, epoch: int):
        pass

    def after_val_iter(self, trainer, epoch: int, step: int, batch, outputs: Dict[str, float]):
        pass

    def after_val_epoch(self, trainer, epoch: int, val_stats: Dict[str, float]):
        pass


class HookManager:
    def __init__(self, hooks: Optional[Sequence[Hook]] = None):
        self.hooks = sorted(list(hooks or []), key=lambda h: getattr(h, 'priority', 50))

    def call(self, fn_name: str, *args, **kwargs):
        for hook in self.hooks:
            fn = getattr(hook, fn_name, None)
            if fn is not None:
                fn(*args, **kwargs)


@dataclass
class LoggerHook(Hook):
    interval: int = 20
    val_interval: int = 50
    print_metric_tables: bool = True
    print_per_class_metrics: bool = True
    priority: int = 70

    @staticmethod
    def _get_class_names_from_trainer(trainer):
        dataloader = getattr(trainer, 'val_dataloader', None)
        if dataloader is None:
            return None

        dataset = getattr(dataloader, 'dataset', None)
        while dataset is not None and hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if dataset is None:
            return None

        classes = getattr(dataset, 'classes', None)
        if classes is None:
            return None

        return [str(x) for x in classes]

    @staticmethod
    def _format_seconds(seconds) -> str:
        if seconds is None:
            return 'N/A'
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f'{h:02d}:{m:02d}:{s:02d}'

    @staticmethod
    def _format_lr(lrs) -> str:
        if not lrs:
            return 'N/A'
        lrs = [float(x) for x in lrs]
        if len(lrs) == 1:
            return f'{lrs[0]:.3e}'
        lr_min = min(lrs)
        lr_max = max(lrs)
        if abs(lr_min - lr_max) < 1e-12:
            return f'{lr_min:.3e}'
        return f'{lr_min:.3e}~{lr_max:.3e}'

    @staticmethod
    def _format_extra_value(v) -> str:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v)

    def after_train_iter(self, trainer, epoch: int, step: int, batch, outputs: Dict[str, float]):
        state = getattr(trainer, 'log_state', None)
        if not state or state.get('mode') != 'train':
            return

        iters_per_epoch = state.get('iters_per_epoch', None)
        should_log = (step % self.interval == 0)
        if iters_per_epoch is not None and step == iters_per_epoch:
            should_log = True
        if not should_log:
            return

        epoch_id = state.get('epoch', epoch)
        max_epochs = state.get('max_epochs', '?')
        global_iter = state.get('global_iter', '?')
        max_iters = state.get('max_iters', '?')
        iter_time = state.get('iter_time', 0.0)
        data_time = state.get('data_time', 0.0)
        eta = self._format_seconds(state.get('eta_seconds', None))
        lr_str = self._format_lr(state.get('lrs', []))
        memory_mb = state.get('memory_mb', None)
        log_vars = state.get('log_vars', {})
        extra_log_vars = state.get('extra_log_vars', {})

        iter_part = f'[{step}/{iters_per_epoch}]' if iters_per_epoch is not None else f'[{step}/?]'
        global_part = f'[{global_iter}/{max_iters}]' if max_iters is not None else f'[{global_iter}/?]'

        msg = (
            f"Epoch [{epoch_id}/{max_epochs}]"
            f"{iter_part} "
            f"iter {global_part} "
            f"lr: {lr_str} "
            f"eta: {eta} "
            f"time: {iter_time:.3f} "
            f"data: {data_time:.3f}"
        )

        if memory_mb is not None:
            msg += f" mem: {memory_mb}"

        for k, v in sorted(log_vars.items()):
            msg += f" {k}: {v:.4f}"

        for k, v in sorted(extra_log_vars.items()):
            msg += f" {k}: {self._format_extra_value(v)}"

        print(msg)

    def after_val_iter(self, trainer, epoch: int, step: int, batch, outputs: Dict[str, float]):
        state = getattr(trainer, 'log_state', None)
        if not state or state.get('mode') != 'val_loss':
            return

        total_iters = state.get('iters_per_epoch', None)
        should_log = (step % self.val_interval == 0)
        if total_iters is not None and step == total_iters:
            should_log = True
        if not should_log:
            return

        epoch_id = state.get('epoch', epoch)
        max_epochs = state.get('max_epochs', '?')
        eta = self._format_seconds(state.get('eta_seconds', None))
        iter_time = state.get('iter_time', 0.0)
        data_time = state.get('data_time', 0.0)
        log_vars = state.get('log_vars', {})
        extra_log_vars = state.get('extra_log_vars', {})

        iter_part = f'[{step}/{total_iters}]' if total_iters is not None else f'[{step}/?]'

        msg = (
            f'[val] Epoch [{epoch_id}/{max_epochs}]'
            f'{iter_part} '
            f'eta: {eta} '
            f'time: {iter_time:.3f} '
            f'data: {data_time:.3f}'
        )

        for k, v in sorted(log_vars.items()):
            msg += f' {k}: {v:.4f}'

        for k, v in sorted(extra_log_vars.items()):
            msg += f' {k}: {self._format_extra_value(v)}'

        print(msg)

    def after_val_epoch(self, trainer, epoch: int, val_stats: Dict[str, float]):
        if not val_stats:
            return

        loss_msg = f'[val] epoch={epoch}'
        for k, v in sorted(val_stats.items()):
            if not isinstance(v, (int, float)):
                continue
            if 'loss' not in k.lower():
                continue
            loss_msg += f' {k}={v:.4f}'
        print(loss_msg)

        if not self.print_metric_tables:
            return

        class_names = val_stats.get('_class_names', None)
        if class_names is None:
            class_names = self._get_class_names_from_trainer(trainer)
        summary_table, per_class_table = format_semantic_metric_tables(
            metric_stats=val_stats,
            class_names=class_names,
        )

        if summary_table:
            print(summary_table)

        if self.print_per_class_metrics and per_class_table:
            print(per_class_table)
