from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader


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


def build_dataloader(cfg: ConfigDict):
    cfg = dict(cfg)
    dataset_cfg = cfg.pop('dataset')
    collate_fn_cfg = cfg.pop('collate_fn', None)
    dataset = build_dataset(dataset_cfg)
    collate_fn = build_collate_fn(collate_fn_cfg)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **cfg)
