from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional

import torch


ConfigDict = Dict[str, Any]


def _get_obj_from_string(path: str):
    module_name, obj_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


class OptimizerBuilder:
    """Build optimizer and lr scheduler from mmseg-style config dicts.

    Supported optimizer `type` values:
    - AdamW
    - Adam
    - SGD
    - fully-qualified import path, e.g. `torch.optim.AdamW`

    Supported scheduler `type` values:
    - CosineAnnealingLR
    - MultiStepLR
    - StepLR
    - PolynomialLR
    - LinearLR
    - SequentialLR
    - fully-qualified import path
    """

    @staticmethod
    def _resolve_cls(type_name: str, default_module: str):
        if '.' in type_name:
            return _get_obj_from_string(type_name)
        module = importlib.import_module(default_module)
        return getattr(module, type_name)

    @staticmethod
    def _build_param_groups(model: torch.nn.Module, optim_cfg: ConfigDict):
        base_lr = float(optim_cfg.get('lr', 1e-4))
        base_wd = float(optim_cfg.get('weight_decay', 0.05))
        paramwise_cfg = optim_cfg.get('paramwise_cfg') or {}
        custom_keys = paramwise_cfg.get('custom_keys') or {}
        norm_decay_mult = float(paramwise_cfg.get('norm_decay_mult', 1.0))
        bias_lr_mult = float(paramwise_cfg.get('bias_lr_mult', 1.0))
        bias_decay_mult = float(paramwise_cfg.get('bias_decay_mult', 1.0))

        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            group = {
                'params': [param],
                'lr': base_lr,
                'weight_decay': base_wd,
            }

            if name.endswith('.bias'):
                group['lr'] = base_lr * bias_lr_mult
                group['weight_decay'] = base_wd * bias_decay_mult

            lowered = name.lower()
            if any(token in lowered for token in ('norm', 'bn', 'ln', 'gn')):
                group['weight_decay'] = base_wd * norm_decay_mult

            for key, rule in custom_keys.items():
                if key in name:
                    if 'lr_mult' in rule:
                        group['lr'] = base_lr * float(rule['lr_mult'])
                    if 'decay_mult' in rule:
                        group['weight_decay'] = base_wd * float(rule['decay_mult'])

            if getattr(param, "_ovrs_disable_weight_decay", False):
                group["weight_decay"] = 0.0

            params.append(group)
        return params

    @classmethod
    def build_optimizer(cls, model: torch.nn.Module, cfg: ConfigDict) -> torch.optim.Optimizer:
        wrapper_cfg = cfg.get('optim_wrapper', cfg)
        optim_cfg = dict(wrapper_cfg.get('optimizer', wrapper_cfg.get('optimizer_cfg', {})))
        if not optim_cfg:
            raise ValueError('Optimizer config is empty. Expected optim_wrapper.optimizer.')

        optim_type = optim_cfg.pop('type', 'AdamW')
        optimizer_cls = cls._resolve_cls(optim_type, 'torch.optim')
        param_groups = cls._build_param_groups(model, optim_cfg)

        # remove keys used only for param grouping
        optim_cfg.pop('paramwise_cfg', None)
        optimizer = optimizer_cls(param_groups, **optim_cfg)
        return enforce_optimizer_param_group_invariants(optimizer)

    @classmethod
    def build_scheduler(cls, optimizer: torch.optim.Optimizer, cfg: ConfigDict):
        sched_cfg = cfg.get('param_scheduler', None)
        if sched_cfg is None:
            return None

        if isinstance(sched_cfg, (list, tuple)):
            schedulers = [cls._build_single_scheduler(optimizer, sc) for sc in sched_cfg]
            if len(schedulers) == 1:
                return schedulers[0]
            milestones = []
            current = 0
            for sc in sched_cfg[:-1]:
                current += int(sc.get('end', 0) or 0)
                milestones.append(current)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones,
            )
        return cls._build_single_scheduler(optimizer, sched_cfg)

    @classmethod
    def _build_single_scheduler(cls, optimizer: torch.optim.Optimizer, cfg: ConfigDict):
        sched_cfg = dict(cfg)
        sched_type = sched_cfg.pop('type', 'CosineAnnealingLR')
        sched_cfg.pop('by_epoch', None)
        sched_cfg.pop('begin', None)
        sched_cfg.pop('end', None)
        scheduler_cls = cls._resolve_cls(sched_type, 'torch.optim.lr_scheduler')
        return scheduler_cls(optimizer, **sched_cfg)


def build_optimizer(model: torch.nn.Module, cfg: ConfigDict) -> torch.optim.Optimizer:
    return OptimizerBuilder.build_optimizer(model, cfg)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: ConfigDict):
    return OptimizerBuilder.build_scheduler(optimizer, cfg)


def enforce_optimizer_param_group_invariants(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    """Re-apply hard constraints on optimizer param groups after load_state_dict.

    When an optimizer is restored from checkpoint, load_state_dict() may
    overwrite per-group hyperparameters (e.g. weight_decay).  This function
    ensures that fused QKV parameters marked with _ovrs_disable_weight_decay
    always have weight_decay=0.0, regardless of what the checkpoint contains.
    """
    for group in optimizer.param_groups:
        params = list(group.get("params", []))
        if any(
            getattr(param, "_ovrs_disable_weight_decay", False)
            for param in params
        ):
            group["weight_decay"] = 0.0
    return optimizer
