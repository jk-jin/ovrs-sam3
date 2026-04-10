from __future__ import annotations

import torch
import torch.nn as nn

from .adapters.semantic_adapter import QueryMaskSemanticAdapter
from .data_misc import BatchedDatapoint
from .sam3_image import Sam3Image

class SAM3Segmentor(nn.Module):
    def __init__(
        self,
        core: Sam3Image,
        semantic_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.core = core
        self.semantic_adapter = semantic_adapter or QueryMaskSemanticAdapter()

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        raw_outputs = self.core(batch)
        return self.semantic_adapter(raw_outputs, batch=batch)