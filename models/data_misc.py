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

    semantic_eval_label_map: Optional[MyTensor] = None
    semantic_eval_label_map__type = torch.long


@dataclass
class BatchedInferenceMetadata:
    original_image_id: MyTensor
    original_image_id__type = torch.long

    original_size: MyTensor
    original_size__type = torch.long

    # Original forward class space after optional background exclusion.
    num_classes: int
    class_names: List[str]

    # Expanded text-prompt space used by SAM3 / RemoteCLIP / Refiner.
    num_prompts: int
    prompt_names: List[str]

    prompt_to_class_id: MyTensor
    prompt_to_class_id__type = torch.long

    # Dataset evaluation class space.
    eval_num_classes: int
    eval_class_names: List[str]

    background_enabled: bool = False
    background_class_id: int = 0
    background_class_name: Optional[str] = None
    background_exclude_from_forward: bool = False


@dataclass
class BatchedDatapoint:
    img_batch: torch.Tensor

    find_text_batch: List[str]

    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]

    raw_images: Optional[List[Any]] = None
    raw_images_original: Optional[List[Any]] = None