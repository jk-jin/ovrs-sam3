from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .data_misc import BatchedDatapoint
from .sam3_image import Sam3Image
from .task_modes import OUTPUT_KEYS, normalize_task_mode


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

    def build_shared_clip_feature(
        self,
        batch: BatchedDatapoint,
    ) -> torch.Tensor:
        return self.core.build_shared_clip_feature(batch)

    def build_sam3_pixel_feature(
        self,
        batch: BatchedDatapoint,
    ) -> torch.Tensor:
        return self.core.build_sam3_pixel_feature(batch)

    @staticmethod
    def _build_mixer_cache_item(
        outputs: Dict[str, torch.Tensor],
        chunk_class_ids: List[int],
        detach_score_maps: bool,
    ) -> Dict[str, torch.Tensor | list[int]]:
        required_keys = (
            OUTPUT_KEYS.semantic_logits,
            OUTPUT_KEYS.class_tokens,
        )

        for key in required_keys:
            if key not in outputs:
                raise ValueError(
                    f"Chunk outputs must contain '{key}' for final mixer."
                )

        semantic_logits = outputs[OUTPUT_KEYS.semantic_logits]
        class_tokens = outputs[OUTPUT_KEYS.class_tokens]

        if detach_score_maps:
            semantic_logits = semantic_logits.detach()

        return {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.class_tokens: class_tokens,
            "chunk_class_ids": list(chunk_class_ids),
        }

    def iter_chunk_outputs(
        self,
        batch: BatchedDatapoint,
        shared_clip_feature: Optional[torch.Tensor] = None,
    ) -> Iterator[Dict[str, Any]]:
        for chunk in self.core.iter_chunk_raw_outputs(
            batch,
            shared_clip_feature=shared_clip_feature,
        ):
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

    def run_final_mixer_from_chunks(
        self,
        mixer_cache: List[Dict[str, torch.Tensor | list[int]]],
        batch: Optional[BatchedDatapoint] = None,
        shared_clip_feature: Optional[torch.Tensor] = None,
        sam3_feature_high: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.core.run_final_mixer_from_chunks(
            mixer_cache=mixer_cache,
            batch=batch,
            shared_clip_feature=shared_clip_feature,
            sam3_feature_high=sam3_feature_high,
        )

    def forward(self, batch: BatchedDatapoint) -> dict[str, torch.Tensor]:
        shared_clip_feature = self.build_shared_clip_feature(batch)
        sam3_feature_high = self.build_sam3_pixel_feature(batch)

        mixer_cache = []

        for chunk in self.core.iter_chunk_raw_outputs(
                batch,
                shared_clip_feature=shared_clip_feature,
        ):
            raw_outputs = chunk["raw_outputs"]

            chunk_outputs = self.adapter(
                raw_outputs=raw_outputs,
                batch=batch,
                expected_num_classes=len(chunk["chunk_class_ids"]),
                output_mode="train",
            )

            mixer_cache.append(
                self._build_mixer_cache_item(
                    outputs=chunk_outputs,
                    chunk_class_ids=chunk["chunk_class_ids"],
                    detach_score_maps=True,
                )
            )

        final_raw_outputs = self.run_final_mixer_from_chunks(
            mixer_cache=mixer_cache,
            batch=batch,
            shared_clip_feature=shared_clip_feature,
            sam3_feature_high=sam3_feature_high,
        )

        return self.adapter(
            raw_outputs=final_raw_outputs,
            batch=batch,
            expected_num_classes=None,
            output_mode="infer",
        )