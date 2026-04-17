from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .data_misc import BatchedDatapoint
from .sam3_image import Sam3Image
from .task_modes import normalize_task_mode


class SAM3Segmentor(nn.Module):
    def __init__(
        self,
        core: Sam3Image,
        adapter: nn.Module,
        task_mode: str,
    ):
        super().__init__()
        self.core = core
        self.adapter = adapter
        self.task_mode = normalize_task_mode(task_mode)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def clear_text_cache(self) -> None:
        self.core.clear_text_cache()

    def prepare_text_cache(
        self,
        class_names: List[str],
        device: Optional[torch.device] = None,
        force: bool = False,
    ) -> None:
        self.core.prepare_text_cache(
            class_texts=class_names,
            device=device,
            force=force,
        )

    def iter_chunk_outputs(
        self,
        batch: BatchedDatapoint,
    ) -> Iterator[Dict[str, Any]]:
        for chunk in self.core.iter_chunk_raw_outputs(batch):
            raw_outputs = chunk["raw_outputs"]
            chunk_class_ids = chunk["chunk_class_ids"]

            train_outputs = self.adapter(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=len(chunk_class_ids),
                output_mode="train",
            )

            yield {
                "chunk_start": chunk["chunk_start"],
                "chunk_end": chunk["chunk_end"],
                "chunk_class_ids": chunk_class_ids,
                "chunk_class_names": chunk["chunk_class_names"],
                "raw_outputs": raw_outputs,
                "train_outputs": train_outputs,
            }

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        raw_outputs = self.core(batch)
        final_outputs = self.adapter(
            raw_outputs=raw_outputs,
            batch=batch,
            expected_num_classes=None,
            output_mode="infer",
        )
        return final_outputs