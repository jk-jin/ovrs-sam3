from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


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

    def after_val_epoch(self, trainer, epoch: int, val_stats: Dict[str, float]):
        pass

    def after_save_checkpoint(self, trainer, epoch: int, ckpt_path: str):
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
    priority: int = 70

    def after_train_iter(self, trainer, epoch: int, step: int, batch, outputs: Dict[str, float]):
        if step % self.interval != 0:
            return
        msg = f'[train] epoch={epoch} iter={step}'
        for k, v in sorted(outputs.items()):
            msg += f' {k}={v:.4f}'
        print(msg)

    def after_val_epoch(self, trainer, epoch: int, val_stats: Dict[str, float]):
        if not val_stats:
            return
        msg = f'[val] epoch={epoch}'
        for k, v in sorted(val_stats.items()):
            msg += f' {k}={v:.4f}'
        print(msg)


@dataclass
class EvalHook(Hook):
    priority: int = 60

    def after_val_epoch(self, trainer, epoch: int, val_stats: Dict[str, float]):
        # Placeholder hook for future custom evaluator integration.
        # Keeping it as a dedicated object now makes the engine easier to extend later.
        return None


@dataclass
class CheckpointHook(Hook):
    interval: int = 1
    save_best: bool = True
    monitor: str = 'total_loss'
    mode: str = 'min'
    priority: int = 80

    def after_train_epoch(self, trainer, epoch: int, train_stats: Dict[str, float]):
        return None

    def after_save_checkpoint(self, trainer, epoch: int, ckpt_path: str):
        print(f'[ckpt] epoch={epoch} path={ckpt_path}')
