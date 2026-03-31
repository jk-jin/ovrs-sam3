from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .adapters.instance_adapter import QueryMaskInstanceAdapter
from .adapters.semantic_adapter import QueryMaskSemanticAdapter
from .sam3_core import Sam3Core
from .segmentor import SAM3Segmentor


def convert_sam3_image_to_core(model: nn.Module) -> Sam3Core:
    """In-place convert an already-built Sam3Image-like model into Sam3Core.

    This keeps the loaded official weights and reuses all submodules.
    """
    model.__class__ = Sam3Core
    model.matcher = None
    return model


def build_segmentor_from_sam3_image(
    sam3_image_model: nn.Module,
    semantic_topk: Optional[int] = 20,
    semantic_aggregation: str = 'weighted_sum',
    instance_topk: Optional[int] = 100,
    instance_score_threshold: float = 0.0,
    instance_mask_threshold: float = 0.5,
) -> SAM3Segmentor:
    core = convert_sam3_image_to_core(sam3_image_model)
    return SAM3Segmentor(
        core=core,
        instance_adapter=QueryMaskInstanceAdapter(
            topk=instance_topk,
            score_threshold=instance_score_threshold,
            mask_threshold=instance_mask_threshold,
            return_binary_masks=False,
        ),
        semantic_adapter=QueryMaskSemanticAdapter(
            topk=semantic_topk,
            aggregation=semantic_aggregation,
        ),
    )
