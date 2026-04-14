from __future__ import annotations

from typing import Any, Dict, Iterator

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

    def iter_chunk_outputs(
        self,
        batch: BatchedDatapoint,
    ) -> Iterator[Dict[str, Any]]:
        for chunk in self.core.iter_chunk_raw_outputs(batch):
            raw_outputs = chunk["raw_outputs"]
            chunk_class_ids = chunk["chunk_class_ids"]

            adapted = self.semantic_adapter(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=len(chunk_class_ids),
            )

            yield {
                "chunk_start": chunk["chunk_start"],
                "chunk_end": chunk["chunk_end"],
                "chunk_class_ids": chunk_class_ids,
                "chunk_class_names": chunk["chunk_class_names"],
                "raw_outputs": raw_outputs,
                "train_outputs": adapted["train_outputs"],
                "inference_outputs": adapted["inference_outputs"],
            }

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        raw_outputs = self.core(batch)
        adapted = self.semantic_adapter(
            raw_outputs=raw_outputs,
            batch=batch,
            expected_num_classes=None,
        )
        return adapted["inference_outputs"]