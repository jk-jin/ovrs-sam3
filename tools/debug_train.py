from __future__ import annotations

"""Tiny debugging entrypoint.

Use this when you only want to verify one forward/backward pass with your config:
python tools/debug_train.py configs/sam3_instance.py
"""

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root.parent))
    package_name = repo_root.name
    from importlib import import_module

    train_mod = import_module(f'{package_name}.tools.train')
else:
    from . import train as train_mod

Config = train_mod.Config
_to_dotdict = train_mod._to_dotdict
build_criterion = train_mod.build_criterion
build_hooks = train_mod.build_hooks
build_segmentor_model = train_mod.build_segmentor_model
build_dataloader = train_mod.build_dataloader
build_optimizer = train_mod.build_optimizer
build_scheduler = train_mod.build_scheduler
FreezeConfig = train_mod.FreezeConfig
Trainer = train_mod.Trainer
TrainerConfig = train_mod.TrainerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    cfg = _to_dotdict(Config.fromfile(args.config))
    model_cfg = dict(cfg.model)
    freeze_cfg = FreezeConfig(**model_cfg.pop('freeze_cfg', {}))
    model = build_segmentor_model(**model_cfg, freeze_cfg=freeze_cfg)
    train_loader = build_dataloader(cfg.train_dataloader)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_criterion(cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=None,
        lr_scheduler=scheduler,
        cfg=TrainerConfig(
            task=cfg.train_cfg.task,
            max_epochs=1,
            log_interval=1,
            use_amp=bool(cfg.train_cfg.get('use_amp', True)),
            grad_clip_norm=cfg.train_cfg.get('grad_clip_norm', 0.1),
            save_dir=str(Path('./work_dirs/debug')),
            save_interval=1000,
            eval_interval=1000,
            device=str(cfg.train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        ),
        hooks=build_hooks(cfg),
    )

    batch = next(iter(train_loader))
    stats = trainer.train_step(batch)
    print('debug train_step ok:', stats)


if __name__ == '__main__':
    main()
