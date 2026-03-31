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
    sys.path.insert(0, str(repo_root.parent))
    package_name = repo_root.name
    from importlib import import_module

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
    EvalHook = _hooks_mod.EvalHook
    CheckpointHook = _hooks_mod.CheckpointHook
    _vis_mod = import_module(f'{package_name}.engine.visualization')
    VisualizationManager = _vis_mod.VisualizationManager
    _hybrid_mod = import_module(f'{package_name}.losses.hybrid_criterion')
    HybridCriterion = _hybrid_mod.HybridCriterion
    HybridLossWeights = _hybrid_mod.HybridLossWeights
    _inst_mod = import_module(f'{package_name}.losses.instance_criterion')
    InstanceCriterion = _inst_mod.InstanceCriterion
    InstanceLossWeights = _inst_mod.InstanceLossWeights
    _sem_mod = import_module(f'{package_name}.losses.semantic_criterion')
    SemanticCriterion = _sem_mod.SemanticCriterion
    SemanticLossWeights = _sem_mod.SemanticLossWeights
    _builder_mod = import_module(f'{package_name}.model_builder')
    FreezeConfig = _builder_mod.FreezeConfig
    build_segmentor_model = _builder_mod.build_segmentor_model
else:
    from ..data.build import build_dataloader
    from ..engine.config import Config
    from ..engine.hooks import CheckpointHook, EvalHook, LoggerHook
    from ..engine.optimizer_builder import build_optimizer, build_scheduler
    from ..engine.trainer import Trainer, TrainerConfig
    from ..engine.visualization import VisualizationManager
    from ..losses.hybrid_criterion import HybridCriterion, HybridLossWeights
    from ..losses.instance_criterion import InstanceCriterion, InstanceLossWeights
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
    task = cfg['train_cfg']['task']
    criterion_cfg = cfg.get('criterion', {})
    if task == 'instance':
        weights = InstanceLossWeights(**criterion_cfg.get('instance', {}))
        return InstanceCriterion(weights=weights)
    if task == 'semantic':
        weights = SemanticLossWeights(**criterion_cfg.get('semantic', {}))
        return SemanticCriterion(weights=weights)
    if task == 'hybrid':
        inst = InstanceCriterion(weights=InstanceLossWeights(**criterion_cfg.get('instance', {})))
        sem = SemanticCriterion(weights=SemanticLossWeights(**criterion_cfg.get('semantic', {})))
        hybrid_weights = HybridLossWeights(**criterion_cfg.get('hybrid', {}))
        return HybridCriterion(inst, sem, hybrid_weights)
    raise ValueError(f'Unsupported task: {task}')


def build_hooks(cfg) -> List[object]:
    hooks = []
    default_hooks = cfg.get('default_hooks', {}) or {}
    logger_cfg = default_hooks.get('logger') or {'interval': int(cfg.train_cfg.get('log_interval', 20))}
    hooks.append(LoggerHook(interval=int(logger_cfg.get('interval', cfg.train_cfg.get('log_interval', 20)))))

    if cfg.get('val_dataloader') is not None:
        hooks.append(EvalHook())

    ckpt_cfg = default_hooks.get('checkpoint') or {}
    hooks.append(
        CheckpointHook(
            interval=int(ckpt_cfg.get('interval', cfg.train_cfg.get('save_interval', 1))),
            save_best=bool(ckpt_cfg.get('save_best', True)),
            monitor=str(ckpt_cfg.get('monitor', cfg.train_cfg.get('monitor', 'total_loss'))),
            mode=str(ckpt_cfg.get('mode', cfg.train_cfg.get('monitor_mode', 'min'))),
        )
    )
    return hooks


def main():
    parser = argparse.ArgumentParser(description='Train/Eval SAM3 segmentor with simple mmseg-style config.')
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
        task=cfg.train_cfg.task,
        max_epochs=int(cfg.train_cfg.max_epochs),
        log_interval=int(cfg.train_cfg.get('log_interval', 20)),
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
            hooks=[],
            visualizer=visualizer,
        )

        if args.resume_from:
            trainer.resume_from(args.resume_from)

        stats = trainer.val_epoch(epoch=args.eval_epoch)
        print('eval-only stats:', stats)
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

    if args.resume_from:
        trainer.resume_from(args.resume_from)

    trainer.train()


if __name__ == '__main__':
    main()
