from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .adapters.instance_adapter import QueryMaskInstanceAdapter
from .adapters.semantic_adapter import QueryMaskSemanticAdapter
from .data_misc import BatchedDatapoint
from .sam3_core import Sam3Core


class SAM3Segmentor(nn.Module):
    """Unified wrapper for raw / instance / semantic modes."""

    def __init__(
        self,
        core: Sam3Core,
        instance_adapter: nn.Module | None = None,
        semantic_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.core = core
        self.instance_adapter = instance_adapter or QueryMaskInstanceAdapter()
        self.semantic_adapter = semantic_adapter or QueryMaskSemanticAdapter()

    def forward(
        self,
        batch: BatchedDatapoint,
        mode: str = 'raw',
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        raw_outputs = self.core(batch)

        if mode == 'raw':
            return {'raw_outputs': raw_outputs}
        if mode == 'instance':
            return {
                'raw_outputs': raw_outputs,
                'instance_outputs': self.instance_adapter(raw_outputs),
            }
        if mode == 'semantic':
            return {
                'raw_outputs': raw_outputs,
                'semantic_outputs': self.semantic_adapter(raw_outputs),
            }
        if mode == 'hybrid':
            return {
                'raw_outputs': raw_outputs,
                'instance_outputs': self.instance_adapter(raw_outputs),
                'semantic_outputs': self.semantic_adapter(raw_outputs),
            }
        raise ValueError(f'Unsupported mode: {mode}')
