from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[1]
    from importlib import import_module
    import types

    package_name = "_ovrs_sam3_localpkg"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[package_name] = pkg

    build_dataloader = import_module(f'{package_name}.data.build').build_dataloader
    Config = import_module(f'{package_name}.engine.config').Config
    _opt_mod = import_module(f'{package_name}.engine.optimizer_builder')
    build_optimizer = _opt_mod.build_optimizer
    build_scheduler = _opt_mod.build_scheduler
    _trainer_mod = import_module(f'{package_name}.engine.trainer')
    Trainer = _trainer_mod.Trainer
    TrainerConfig = _trainer_mod.TrainerConfig
    _hooks_mod = import_module(f'{package_name}.engine.hooks')
    LoggerHook = _hooks_mod.LoggerHook
    _vis_mod = import_module(f'{package_name}.engine.visualization')
    VisualizationManager = _vis_mod.VisualizationManager
    _sem_mod = import_module(f'{package_name}.losses.semantic_criterion')
    SemanticCriterion = _sem_mod.SemanticCriterion
    SemanticLossWeights = _sem_mod.SemanticLossWeights
    _builder_mod = import_module(f'{package_name}.model_builder')
    FreezeConfig = _builder_mod.FreezeConfig
    build_segmentor_model = _builder_mod.build_segmentor_model
else:
    from ..data.build import build_dataloader
    from ..engine.config import Config
    from ..engine.optimizer_builder import build_optimizer, build_scheduler
    from ..engine.trainer import Trainer, TrainerConfig
    from ..engine.visualization import VisualizationManager
    from ..losses.semantic_criterion import SemanticCriterion, SemanticLossWeights
    from ..model_builder import FreezeConfig, build_segmentor_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_dotdict(obj: Any):
    if isinstance(obj, dict):
        return _DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(x) for x in obj]
    return obj


def build_criterion(cfg: Dict[str, Any]):
    criterion_cfg = cfg.get('criterion', {})
    weights = SemanticLossWeights(**criterion_cfg.get('semantic', {}))
    ignore_index = int(criterion_cfg.get('ignore_index', 255))
    return SemanticCriterion(weights=weights, ignore_index=ignore_index)


def build_hooks(cfg) -> List[object]:
    logger_cfg = cfg.default_hooks['logger']
    return [
        LoggerHook(
            interval=int(logger_cfg['interval']),
            val_interval=int(logger_cfg['val_interval']),
            print_metric_tables=bool(logger_cfg.get('print_metric_tables', True)),
            print_per_class_metrics=bool(logger_cfg.get('print_per_class_metrics', True)),
        )
    ]

def build_log_getters(cfg) -> List[object]:
    def project_log_getter(trainer):
        out = {}

        model = trainer.model
        core = getattr(model, 'core', None)
        if core is None:
            return out

        if hasattr(core, 'clip_text_token_gate'):
            gate = core.clip_text_token_gate.detach()
            if gate.numel() == 1:
                out['clip_gate_raw'] = float(gate.item())
                out['clip_gate_tanh'] = float(torch.tanh(gate).item())

        return out

    return [project_log_getter]


def main():
    parser = argparse.ArgumentParser(description='Train/Eval SAM3 semantic-only segmentor with simple mmseg-style config.')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--eval-only', action='store_true', help='only run validation')
    parser.add_argument('--eval-epoch', type=int, default=0, help='epoch id used in eval-only outputs')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg = _to_dotdict(cfg)

    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))
    set_seed(seed)

    model_cfg = dict(cfg.model)
    freeze_cfg = FreezeConfig(**model_cfg.pop('freeze_cfg', {}))
    model = build_segmentor_model(**model_cfg, freeze_cfg=freeze_cfg)

    work_dir = args.work_dir or cfg.get('work_dir', './work_dirs/default')
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    visualizer = VisualizationManager.from_cfg(cfg.get('visualization'), work_dir=work_dir)
    criterion = build_criterion(cfg)

    trainer_cfg = TrainerConfig(
        max_epochs=int(cfg.train_cfg.max_epochs),
        log_window_size=int(cfg.train_cfg.get('log_window_size', 20)),
        use_amp=bool(cfg.train_cfg.get('use_amp', True)),
        grad_clip_norm=cfg.train_cfg.get('grad_clip_norm', 0.1),
        save_dir=str(work_dir),
        save_interval=int(cfg.train_cfg.get('save_interval', 1)),
        eval_interval=int(cfg.train_cfg.get('eval_interval', 1)),
        monitor=str(cfg.train_cfg.get('monitor', 'total_loss')),
        monitor_mode=str(cfg.train_cfg.get('monitor_mode', 'min')),
        max_keep_ckpts=int(cfg.train_cfg.get('max_keep_ckpts', 5)),
        device=str(cfg.train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        auto_resume=bool(args.auto_resume or cfg.train_cfg.get('auto_resume', False)),
        tta_cfg=cfg.get('tta_cfg', None),
        eval_cfg=cfg.get('eval_cfg', None),
    )

    if args.eval_only:
        if cfg.get('val_dataloader') is None:
            raise ValueError('val_dataloader is None, cannot run eval-only mode.')

        print('Building val_dataloader (eval-only)...')
        val_loader = build_dataloader(cfg.val_dataloader)

        trainer = Trainer(
            model=model,
            optimizer=None,
            criterion=criterion,
            train_dataloader=None,
            val_dataloader=val_loader,
            lr_scheduler=None,
            cfg=trainer_cfg,
            hooks=build_hooks(cfg),
            visualizer=visualizer,
        )

        if args.resume_from:
            trainer.resume_from(args.resume_from)

        trainer.val_epoch(epoch=args.eval_epoch)
        return

    print('Building train_dataloader...')
    train_loader = build_dataloader(cfg.train_dataloader)

    print('Building val_dataloader...')
    val_loader = build_dataloader(cfg.val_dataloader) if cfg.get('val_dataloader') else None

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        cfg=trainer_cfg,
        hooks=build_hooks(cfg),
        visualizer=visualizer,
    )

    for getter in build_log_getters(cfg):
        trainer.register_log_getter(getter)

    if args.resume_from:
        trainer.resume_from(args.resume_from)

    trainer.train()


if __name__ == '__main__':
    main()
