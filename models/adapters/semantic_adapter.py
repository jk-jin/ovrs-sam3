from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class QueryMaskSemanticAdapter(nn.Module):
    """Convert query-wise SAM3 masks into a dense semantic logit map.

    First version goal: run a text-conditioned binary semantic segmentation task.
    One text prompt -> one dense mask.
    """

    def __init__(self, topk: Optional[int] = None, aggregation: str = "weighted_sum"):
        super().__init__()
        self.topk = topk
        self.aggregation = aggregation

    @staticmethod
    def _take_last_if_aux(x: torch.Tensor) -> torch.Tensor:
        # pred_masks may be [L, B, Q, H, W]; pred_logits may be [L, B, Q, 1]
        if x.dim() >= 5:
            return x[-1]
        if x.dim() == 4 and x.shape[-1] == 1:
            return x[-1]
        return x

    def _select_topk(
        self, query_scores: torch.Tensor, query_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.topk is None:
            return query_scores, query_masks

        q = query_scores.shape[1]
        k = min(self.topk, q)
        topk_scores, topk_idx = torch.topk(query_scores, k=k, dim=1)
        gather_idx = topk_idx[..., None, None].expand(-1, -1, query_masks.size(-2), query_masks.size(-1))
        topk_masks = torch.gather(query_masks, dim=1, index=gather_idx)
        return topk_scores, topk_masks

    def forward(self, raw_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_masks = raw_outputs["pred_masks"]
        pred_logits = raw_outputs["pred_logits"]

        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        if pred_logits.dim() == 4:
            pred_logits = pred_logits[-1]

        query_scores = pred_logits.squeeze(-1).sigmoid()  # [B, Q]
        query_scores, pred_masks = self._select_topk(query_scores, pred_masks)

        if self.aggregation == "max":
            weighted_masks = pred_masks * query_scores[..., None, None]
            semantic_logits = weighted_masks.max(dim=1, keepdim=True).values
        elif self.aggregation == "logsumexp":
            weighted_masks = pred_masks + query_scores[..., None, None].log().clamp(min=-20.0)
            semantic_logits = torch.logsumexp(weighted_masks, dim=1, keepdim=True)
        elif self.aggregation == "weighted_sum":
            weights = query_scores / query_scores.sum(dim=1, keepdim=True).clamp(min=1e-6)
            semantic_logits = (pred_masks * weights[..., None, None]).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return {
            "semantic_logits": semantic_logits,
            "query_scores": query_scores,
            "raw_pred_masks": pred_masks,
        }
