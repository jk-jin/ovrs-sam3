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
        aggregation: str = "max",
        use_query_branch: bool = True,
        use_semantic_branch: bool = True,
        fusion_mode: str = "max",
        use_presence_score: bool = True,
        confidence_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.aggregation = aggregation

        self.use_query_branch = bool(use_query_branch)
        self.use_semantic_branch = bool(use_semantic_branch)
        self.fusion_mode = fusion_mode

        self.use_presence_score = bool(use_presence_score)
        self.confidence_threshold = confidence_threshold

    def _extract_query_presence_prob(
            self,
            raw_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        presence_logit_dec = raw_outputs.get("presence_logit_dec", None)
        if presence_logit_dec is None:
            return None

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

        return presence_logit_dec.sigmoid()  # [B, C, Q]

    def _aggregate_query_logits(
            self,
            pred_logits: torch.Tensor,  # [B, C, Q, 1]
            pred_masks: torch.Tensor,  # [B, C, Q, H, W]
            raw_outputs: Dict[str, torch.Tensor],
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

        query_presence = self._extract_query_presence_prob(raw_outputs)
        if self.use_presence_score and query_presence is not None:
            query_scores = query_scores * query_presence

        pred_masks = pred_masks.sigmoid()  # [B, C, Q, H, W]

        if self.confidence_threshold is not None:
            keep = query_scores > float(self.confidence_threshold)  # [B, C, Q]
        else:
            keep = torch.ones_like(query_scores, dtype=torch.bool)

        keep_f = keep.to(dtype=pred_masks.dtype)

        if self.aggregation == "max":
            weighted_masks = pred_masks * query_scores[..., None, None] * keep_f[..., None, None]
            query_logits = weighted_masks.max(dim=2).values  # [B, C, H, W]

        elif self.aggregation == "weighted_sum":
            filtered_scores = query_scores * keep_f
            weights = filtered_scores / filtered_scores.sum(dim=2, keepdim=True).clamp(min=1e-6)
            query_logits = (pred_masks * weights[..., None, None]).sum(dim=2)

        elif self.aggregation == "logsumexp":
            minus_inf = torch.full_like(pred_masks, -1e4)
            masked = torch.where(
                keep[..., None, None],
                pred_masks * query_scores[..., None, None],
                minus_inf,
            )
            query_logits = torch.logsumexp(masked, dim=2)

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

        return semantic_seg.sigmoid()

    def _extract_presence_prob(
            self,
            raw_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_presence_score:
            return None

        presence_logit = raw_outputs.get("presence_logit", None)
        if presence_logit is None:
            return None

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
            query_branch_logits = self._aggregate_query_logits(
                pred_logits=pred_logits,
                pred_masks=pred_masks,
                raw_outputs=raw_outputs,
            )

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

        out = {
            "semantic_logits": fused_logits,
            "semantic_score_map": fused_logits,
        }

        if query_branch_logits is not None:
            out["semantic_logits_query"] = query_branch_logits
        if semantic_branch_logits is not None:
            out["semantic_logits_dense"] = semantic_branch_logits

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