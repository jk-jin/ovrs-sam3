from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_misc import BatchedDatapoint


class QueryMaskSemanticAdapter(nn.Module):
    """
    Consume class-structured SAM3 outputs and produce multi-class semantic logits.

    Expected raw output shapes from sam3_image.py:
        pred_masks:        [B, C, Q, H, W]
        pred_logits:       [B, C, Q, 1]
        semantic_seg:      [B, C, 1, H, W]  or [B, C, H, W]
        presence_logit:    [B, C] or [B, C, 1]
        presence_logit_dec:[B, C, Q]

    Output:
        semantic_logits:   [B, C, H, W]
    """

    def __init__(
        self,
        topk: Optional[int] = None,
        aggregation: str = "weighted_sum",
        use_query_branch: bool = True,
        use_semantic_branch: bool = True,
        fusion_mode: str = "max",
        use_presence_score: bool = True,
        presence_reduce: str = "max",
    ):
        super().__init__()
        self.topk = topk
        self.aggregation = aggregation

        self.use_query_branch = bool(use_query_branch)
        self.use_semantic_branch = bool(use_semantic_branch)
        self.fusion_mode = fusion_mode

        self.use_presence_score = bool(use_presence_score)
        self.presence_reduce = presence_reduce

    def _select_topk(
        self,
        query_scores: torch.Tensor,   # [B, C, Q]
        query_masks: torch.Tensor,    # [B, C, Q, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.topk is None:
            return query_scores, query_masks

        num_queries = int(query_scores.shape[2])
        k = min(int(self.topk), num_queries)
        if k <= 0 or k >= num_queries:
            return query_scores, query_masks

        topk_scores, topk_idx = torch.topk(query_scores, k=k, dim=2)
        gather_idx = topk_idx[..., None, None].expand(
            -1, -1, -1, query_masks.size(-2), query_masks.size(-1)
        )
        topk_masks = torch.gather(query_masks, dim=2, index=gather_idx)
        return topk_scores, topk_masks

    def _aggregate_query_logits(
        self,
        pred_logits: torch.Tensor,    # [B, C, Q, 1]
        pred_masks: torch.Tensor,     # [B, C, Q, H, W]
    ) -> torch.Tensor:
        if pred_logits.dim() != 4:
            raise ValueError(
                f"Expected pred_logits as [B, C, Q, 1], got {tuple(pred_logits.shape)}"
            )
        if pred_masks.dim() != 5:
            raise ValueError(
                f"Expected pred_masks as [B, C, Q, H, W], got {tuple(pred_masks.shape)}"
            )

        if pred_logits.shape[-1] != 1:
            raise ValueError(
                f"Expected pred_logits last dim = 1, got {tuple(pred_logits.shape)}"
            )

        query_scores = pred_logits.squeeze(-1).sigmoid()  # [B, C, Q]
        query_scores, pred_masks = self._select_topk(query_scores, pred_masks)

        if self.aggregation == "max":
            weighted_masks = pred_masks * query_scores[..., None, None]
            query_logits = weighted_masks.max(dim=2).values

        elif self.aggregation == "logsumexp":
            weighted_masks = pred_masks + query_scores.clamp(min=1e-6).log()[..., None, None]
            query_logits = torch.logsumexp(weighted_masks, dim=2)

        elif self.aggregation == "weighted_sum":
            weights = query_scores / query_scores.sum(dim=2, keepdim=True).clamp(min=1e-6)
            query_logits = (pred_masks * weights[..., None, None]).sum(dim=2)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return query_logits  # [B, C, H, W]

    @staticmethod
    def _extract_semantic_branch(raw_outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        semantic_seg = raw_outputs.get("semantic_seg", None)
        if semantic_seg is None:
            return None

        if semantic_seg.dim() == 5:
            # [B, C, 1, H, W] -> [B, C, H, W]
            if semantic_seg.shape[2] != 1:
                raise ValueError(
                    f"Expected semantic_seg as [B, C, 1, H, W], got {tuple(semantic_seg.shape)}"
                )
            semantic_seg = semantic_seg[:, :, 0]

        elif semantic_seg.dim() == 4:
            # already [B, C, H, W]
            pass

        else:
            raise ValueError(
                f"Expected semantic_seg as [B, C, 1, H, W] or [B, C, H, W], got {tuple(semantic_seg.shape)}"
            )

        return semantic_seg

    def _extract_presence_prob(
        self,
        raw_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_presence_score:
            return None

        presence_logit = raw_outputs.get("presence_logit", None)
        if presence_logit is not None:
            if presence_logit.dim() == 3:
                if presence_logit.shape[-1] != 1:
                    raise ValueError(
                        f"Expected presence_logit as [B, C, 1], got {tuple(presence_logit.shape)}"
                    )
                presence_logit = presence_logit.squeeze(-1)
            elif presence_logit.dim() != 2:
                raise ValueError(
                    f"Expected presence_logit as [B, C] or [B, C, 1], got {tuple(presence_logit.shape)}"
                )
            return presence_logit.sigmoid()  # [B, C]

        presence_logit_dec = raw_outputs.get("presence_logit_dec", None)
        if presence_logit_dec is not None:
            if presence_logit_dec.dim() == 4:
                if presence_logit_dec.shape[-1] != 1:
                    raise ValueError(
                        f"Expected presence_logit_dec as [B, C, Q, 1], got {tuple(presence_logit_dec.shape)}"
                    )
                presence_logit_dec = presence_logit_dec.squeeze(-1)

            if presence_logit_dec.dim() != 3:
                raise ValueError(
                    f"Expected presence_logit_dec as [B, C, Q], got {tuple(presence_logit_dec.shape)}"
                )

            presence_prob_dec = presence_logit_dec.sigmoid()  # [B, C, Q]

            if self.presence_reduce == "max":
                return presence_prob_dec.max(dim=2).values   # [B, C]
            elif self.presence_reduce == "mean":
                return presence_prob_dec.mean(dim=2)         # [B, C]
            else:
                raise ValueError(f"Unknown presence_reduce: {self.presence_reduce}")

        return None

    @staticmethod
    def _resize_to_match(
        x: Optional[torch.Tensor],
        target_hw: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if tuple(x.shape[-2:]) == tuple(target_hw):
            return x
        return F.interpolate(
            x,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    def _fuse_branches(
        self,
        query_logits: Optional[torch.Tensor],      # [B, C, H, W] or None
        semantic_logits: Optional[torch.Tensor],   # [B, C, H, W] or None
    ) -> torch.Tensor:
        if query_logits is None and semantic_logits is None:
            raise ValueError("At least one of query branch or semantic branch must be enabled.")

        if self.fusion_mode == "query_only":
            if query_logits is None:
                raise ValueError("fusion_mode='query_only' but query branch is missing.")
            return query_logits

        if self.fusion_mode == "semantic_only":
            if semantic_logits is None:
                raise ValueError("fusion_mode='semantic_only' but semantic branch is missing.")
            return semantic_logits

        if self.fusion_mode == "max":
            if query_logits is None:
                return semantic_logits
            if semantic_logits is None:
                return query_logits
            return torch.maximum(query_logits, semantic_logits)

        if self.fusion_mode == "sum":
            if query_logits is None:
                return semantic_logits
            if semantic_logits is None:
                return query_logits
            return query_logits + semantic_logits

        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def forward(self, raw_outputs, batch: BatchedDatapoint):
        query_branch_logits = None
        if self.use_query_branch:
            pred_masks = raw_outputs.get("pred_masks", None)
            pred_logits = raw_outputs.get("pred_logits", None)
            if pred_masks is None or pred_logits is None:
                raise ValueError(
                    "Query branch is enabled, but pred_masks or pred_logits is missing."
                )
            query_branch_logits = self._aggregate_query_logits(pred_logits, pred_masks)

        semantic_branch_logits = None
        if self.use_semantic_branch:
            semantic_branch_logits = self._extract_semantic_branch(raw_outputs)

        target_hw = None
        if query_branch_logits is not None:
            target_hw = tuple(query_branch_logits.shape[-2:])
        elif semantic_branch_logits is not None:
            target_hw = tuple(semantic_branch_logits.shape[-2:])

        if target_hw is None:
            raise ValueError("Cannot determine target spatial size for semantic logits.")

        query_branch_logits = self._resize_to_match(query_branch_logits, target_hw)
        semantic_branch_logits = self._resize_to_match(semantic_branch_logits, target_hw)

        fused_logits = self._fuse_branches(
            query_logits=query_branch_logits,
            semantic_logits=semantic_branch_logits,
        )

        presence_prob = self._extract_presence_prob(raw_outputs)
        if presence_prob is not None:
            fused_logits = fused_logits * presence_prob[..., None, None]

        out = {
            "semantic_logits": fused_logits,   # [B, C, H, W]
        }

        if query_branch_logits is not None:
            out["semantic_logits_query"] = query_branch_logits
        if semantic_branch_logits is not None:
            out["semantic_logits_dense"] = semantic_branch_logits
        if presence_prob is not None:
            out["presence_prob"] = presence_prob

        if len(batch.find_metadatas) > 0:
            meta = batch.find_metadatas[0]
            expected_num_classes = int(meta.num_classes)
            actual_num_classes = int(fused_logits.shape[1])
            if expected_num_classes != actual_num_classes:
                raise ValueError(
                    f"Class count mismatch: metadata says {expected_num_classes}, "
                    f"but semantic_logits has {actual_num_classes} channels."
                )

        return out