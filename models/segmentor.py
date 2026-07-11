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

        core = self.core

        # Clear RemoteCLIP text cache on every train/eval mode switch.
        # In validation this ensures the first image re-encodes templates
        # at current weights; subsequent images reuse the cache within the
        # same validation pass.
        core.clear_remoteclip_text_cache()

        # Frozen SAM3 modules must stay in eval mode even during training.
        core.backbone.eval()
        core.transformer.eval()
        core.geometry_encoder.eval()
        core.segmentation_head.eval()

        # OpenCLIP image encoder stays eval even when partially trainable:
        # we train selected weights, not dropout / patch-dropout behavior.
        if getattr(core, "clip_image_encoder", None) is not None:
            core.clip_image_encoder.eval()

        # OpenCLIP text encoder stays eval even when q/v parameters are trainable.
        # We only train selected weights, not dropout behavior.
        if getattr(core, "clip_text_encoder", None) is not None:
            core.clip_text_encoder.eval()

        # Encoder refiner is the trainable module.
        if mode:
            core.encoder_refiner.train()
        else:
            core.encoder_refiner.eval()

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

    def build_encoder_refiner_cache(
        self,
        batch: BatchedDatapoint,
    ) -> Dict[str, Any]:
        return self.core.build_encoder_refiner_cache(batch)

    def run_encoder_refiner_from_cache(
        self,
        encoder_refiner_cache: Dict[str, Any],
        batch: BatchedDatapoint,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.core.run_encoder_refiner_from_cache(
            encoder_refiner_cache=encoder_refiner_cache,
            batch=batch,
            return_debug=return_debug,
        )

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        encoder_refiner_cache = self.build_encoder_refiner_cache(batch)

        final_raw_outputs = self.run_encoder_refiner_from_cache(
            encoder_refiner_cache=encoder_refiner_cache,
            batch=batch,
        )

        output_mode = "final" if self.training else "infer"

        return self.adapter(
            raw_outputs=final_raw_outputs,
            batch=batch,
            expected_num_classes=None,
            output_mode=output_mode,
        )