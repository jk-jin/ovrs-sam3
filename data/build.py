from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .resumable_sampler import ResumableRandomBatchSampler


ConfigDict = Dict[str, Any]


def get_obj_from_string(path: str):
    module_name, obj_name = path.rsplit('.', 1)

    try:
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    except ModuleNotFoundError as e:
        original_error = e

    root_pkg = __package__.split('.')[0]

    fallback_module_name = f'{root_pkg}.{module_name}'
    try:
        module = importlib.import_module(fallback_module_name)
        return getattr(module, obj_name)
    except ModuleNotFoundError:
        raise original_error


def instantiate(cfg: Any, **extra_kwargs):
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        return cfg
    cfg = dict(cfg)
    obj_type = cfg.pop('type')
    cls = get_obj_from_string(obj_type) if isinstance(obj_type, str) else obj_type
    cfg.update(extra_kwargs)
    return cls(**cfg)


def build_dataset(cfg: ConfigDict):
    return instantiate(cfg)


def build_collate_fn(cfg: Optional[ConfigDict]):
    if cfg is None:
        return None
    return instantiate(cfg)


def build_dataloader(cfg: ConfigDict, seed: int = 42):
    cfg = dict(cfg)
    dataset_cfg = cfg.pop('dataset')
    collate_fn_cfg = cfg.pop('collate_fn', None)
    dataset = build_dataset(dataset_cfg)
    collate_fn = build_collate_fn(collate_fn_cfg)

    batch_size = int(cfg.pop('batch_size', 1))
    num_workers = int(cfg.pop('num_workers', 0))
    drop_last = bool(cfg.pop('drop_last', False))
    pin_memory = bool(cfg.pop('pin_memory', False))
    persistent_workers = bool(cfg.pop('persistent_workers', False))
    prefetch_factor = cfg.pop('prefetch_factor', None)
    shuffle = bool(cfg.pop('shuffle', False))

    dataloader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if prefetch_factor is not None and num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    if shuffle:
        train_generator = torch.Generator()
        train_generator.manual_seed(int(seed))

        dataloader_kwargs["batch_sampler"] = ResumableRandomBatchSampler(
            dataset_size=len(dataset),
            batch_size=batch_size,
            drop_last=drop_last,
            generator=train_generator,
        )
    else:
        val_generator = torch.Generator()
        val_generator.manual_seed(int(seed) + 1)

        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = False
        dataloader_kwargs["drop_last"] = drop_last
        dataloader_kwargs["generator"] = val_generator

    return DataLoader(dataset=dataset, **dataloader_kwargs)
