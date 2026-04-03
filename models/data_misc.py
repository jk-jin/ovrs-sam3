from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch


MyTensor = Union[torch.Tensor, List[Any]]


@dataclass
class FindStage:
    img_ids: Optional[MyTensor] = None
    text_ids: Optional[MyTensor] = None

    input_boxes: Optional[MyTensor] = None
    input_boxes__type = torch.float

    input_boxes_mask: Optional[MyTensor] = None
    input_boxes_mask__type = torch.bool

    input_boxes_label: Optional[MyTensor] = None
    input_boxes_label__type = torch.long

    input_points: Optional[MyTensor] = None
    input_points__type = torch.float

    input_points_mask: Optional[MyTensor] = None
    input_points_mask__type = torch.bool


@dataclass
class BatchedFindTarget:
    semantic_label_map: MyTensor
    semantic_label_map__type = torch.long


@dataclass
class BatchedInferenceMetadata:
    original_image_id: MyTensor
    original_image_id__type = torch.long

    original_size: MyTensor
    original_size__type = torch.long

    num_classes: int
    class_names: List[str]


@dataclass
class BatchedDatapoint:
    img_batch: torch.Tensor

    find_text_batch: List[str]

    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]

    raw_images: Optional[List[Any]] = None