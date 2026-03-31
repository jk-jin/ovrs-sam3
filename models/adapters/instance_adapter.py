from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..box_ops import box_cxcywh_to_xyxy


class QueryMaskInstanceAdapter(nn.Module):
    """Prepare raw SAM3 query outputs for instance-style training and inference.

    Design goals:
    - keep the original tensors (`pred_logits`, `pred_boxes`, `pred_masks`) intact
    - expose convenience tensors for debugging / visualization / quick evaluation
    - avoid changing the loss interface used by ``InstanceCriterion``

    This adapter is intentionally lightweight. It does *not* perform NMS because
    the current model is query-based and training still relies on matching.
    """

    def __init__(
        self,
        topk: Optional[int] = 100,
        score_threshold: float = 0.0,
        mask_threshold: float = 0.5,
        return_binary_masks: bool = False,
        keep_aux_outputs: bool = True,
    ):
        super().__init__()
        self.topk = topk
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.return_binary_masks = bool(return_binary_masks)
        self.keep_aux_outputs = bool(keep_aux_outputs)

    @staticmethod
    def _take_last_if_stacked(x: torch.Tensor) -> torch.Tensor:
        if x.dim() in (4, 5):
            return x[-1]
        return x

    def _select_topk(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_queries = scores.shape
        if self.topk is None:
            topk = num_queries
        else:
            topk = min(int(self.topk), int(num_queries))
        topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
        return topk_scores, topk_indices

    @staticmethod
    def _gather_boxes(boxes: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(boxes, 1, topk_indices[..., None].expand(-1, -1, boxes.size(-1)))

    @staticmethod
    def _gather_masks(masks: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            masks,
            1,
            topk_indices[..., None, None].expand(-1, -1, masks.size(-2), masks.size(-1)),
        )

    def forward(self, raw_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_logits = self._take_last_if_stacked(raw_outputs['pred_logits'])
        pred_boxes = self._take_last_if_stacked(raw_outputs['pred_boxes'])
        pred_masks = self._take_last_if_stacked(raw_outputs['pred_masks'])

        instance_scores = pred_logits.squeeze(-1).sigmoid()  # [B, Q]
        topk_scores, topk_indices = self._select_topk(instance_scores)
        topk_boxes = self._gather_boxes(pred_boxes, topk_indices)
        topk_masks = self._gather_masks(pred_masks, topk_indices)

        keep_mask = topk_scores >= self.score_threshold
        if self.return_binary_masks:
            topk_binary_masks = topk_masks.sigmoid() >= self.mask_threshold
        else:
            topk_binary_masks = None

        outputs: Dict[str, torch.Tensor] = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'pred_boxes_xyxy': box_cxcywh_to_xyxy(pred_boxes),
            'pred_masks': pred_masks,
            'instance_scores': instance_scores,
            'topk_scores': topk_scores,
            'topk_indices': topk_indices,
            'topk_keep_mask': keep_mask,
            'topk_boxes': topk_boxes,
            'topk_boxes_xyxy': box_cxcywh_to_xyxy(topk_boxes),
            'topk_masks': topk_masks,
        }

        if topk_binary_masks is not None:
            outputs['topk_binary_masks'] = topk_binary_masks

        if self.keep_aux_outputs and 'aux_outputs' in raw_outputs:
            outputs['aux_outputs'] = raw_outputs['aux_outputs']
        if 'queries' in raw_outputs:
            outputs['queries'] = raw_outputs['queries']
        if 'prompt' in raw_outputs:
            outputs['prompt'] = raw_outputs['prompt']
        if 'prompt_mask' in raw_outputs:
            outputs['prompt_mask'] = raw_outputs['prompt_mask']
        if 'encoder_hidden_states' in raw_outputs:
            outputs['encoder_hidden_states'] = raw_outputs['encoder_hidden_states']
        if 'prev_encoder_out' in raw_outputs:
            outputs['prev_encoder_out'] = raw_outputs['prev_encoder_out']
        return outputs
