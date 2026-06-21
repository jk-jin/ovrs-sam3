from __future__ import annotations

from typing import Dict, List, Optional, Any

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

    def build_template_guided_refiner_cache(
        self,
        batch: BatchedDatapoint,
    ) -> Dict[str, Any]:
        return self.core.build_template_guided_refiner_cache(batch)

    def run_template_guided_refiner_from_cache(
        self,
        template_guided_refiner_cache: Dict[str, Any],
        batch: BatchedDatapoint,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.core.run_template_guided_refiner_from_cache(
            template_guided_refiner_cache=template_guided_refiner_cache,
            batch=batch,
            return_debug=return_debug,
        )

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        refiner_cache = self.build_template_guided_refiner_cache(batch)

        final_raw_outputs = self.run_template_guided_refiner_from_cache(
            template_guided_refiner_cache=refiner_cache,
            batch=batch,
        )

        return self.adapter(
            raw_outputs=final_raw_outputs,
            batch=batch,
            expected_num_classes=None,
            output_mode="infer",
        )
