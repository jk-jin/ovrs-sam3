from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from .data_misc import BatchedDatapoint
from .sam3_image import Sam3Image
from .task_modes import normalize_task_mode


class ReCASAM3(nn.Module):
    """ReCA-SAM3 segmentor: frozen SAM3 core plus RemoteCLIP-guided aggregation."""

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

        # RemoteCLIP-guided aggregation is the trainable module.
        if mode:
            core.feature_aggregation.train()
        else:
            core.feature_aggregation.eval()

        return self

    def clear_text_cache(self) -> None:
        self.core.clear_text_cache()

    def prepare_text_cache(
        self,
        prompt_names: List[str],
        device: Optional[torch.device] = None,
        force: bool = False,
    ) -> None:
        self.core.prepare_text_cache(
            prompt_names=prompt_names,
            device=device,
            force=force,
        )

    def build_aggregation_cache(
        self,
        batch: BatchedDatapoint,
    ) -> Dict[str, Any]:
        return self.core.build_aggregation_cache(batch)

    def run_aggregation_lowres_from_cache(
        self,
        aggregation_cache: Dict[str, Any],
        batch: BatchedDatapoint,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.core.run_aggregation_lowres_from_cache(
            aggregation_cache=aggregation_cache,
            batch=batch,
            return_debug=return_debug,
        )

    def decode_aggregation_chunk_from_cache(
        self,
        aggregation_cache: Dict[str, Any],
        aggregated_feature_36_chunk: torch.Tensor,
        class_start: int,
        class_end: int,
        return_teacher_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.core.decode_aggregation_chunk_from_cache(
            aggregation_cache=aggregation_cache,
            aggregated_feature_36_chunk=aggregated_feature_36_chunk,
            class_start=class_start,
            class_end=class_end,
            return_teacher_logits=return_teacher_logits,
        )

    def run_aggregation_from_cache(
        self,
        aggregation_cache: Dict[str, Any],
        batch: BatchedDatapoint,
        return_debug: bool = False,
        return_reference_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.core.run_aggregation_from_cache(
            aggregation_cache=aggregation_cache,
            batch=batch,
            return_debug=return_debug,
            return_reference_logits=return_reference_logits,
        )

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError(
                "ReCASAM3.forward() must not be called during training. "
                "The Trainer must use the streaming chunk path: "
                "build_aggregation_cache() + "
                "run_aggregation_lowres_from_cache() + per-chunk "
                "decode_aggregation_chunk_from_cache()."
            )

        aggregation_cache = self.build_aggregation_cache(batch)

        # The reference gate alone decides whether the original path is
        # decoded; training distillation is irrelevant here.
        raw_outputs = self.run_aggregation_from_cache(
            aggregation_cache=aggregation_cache,
            batch=batch,
            return_reference_logits=(
                getattr(self.adapter, "reference_prob_threshold", None)
                is not None
            ),
        )

        # Inference: go through adapter for sigmoid, absolute gates, argmax.
        return self.adapter(
            raw_outputs=raw_outputs,
            batch=batch,
            output_mode="infer",
        )
