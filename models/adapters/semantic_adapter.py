from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_misc import BatchedDatapoint


class QueryMaskSemanticAdapter(nn.Module):
    def __init__(
        self,
        use_instance_branch: bool = True,
        use_semantic_branch: bool = True,
        fusion_mode: str = "max",
        use_presence_score: bool = True,
        query_confidence_threshold: Optional[float] = None,
        instance_train_aggregation: str = "logsumexp",
        instance_infer_aggregation: str = "max",
    ):
        super().__init__()

        self.use_instance_branch = bool(use_instance_branch)
        self.use_semantic_branch = bool(use_semantic_branch)
        self.fusion_mode = str(fusion_mode)

        self.use_presence_score = bool(use_presence_score)
        self.query_confidence_threshold = query_confidence_threshold

        self.instance_train_aggregation = str(instance_train_aggregation)
        self.instance_infer_aggregation = str(instance_infer_aggregation)

    def _extract_presence_logits(
        self,
        raw_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        presence_logits = raw_outputs.get("presence_logit_dec", None)
        if presence_logits is None:
            presence_logits = raw_outputs.get("presence_logit", None)
        if presence_logits is None:
            return None

        if presence_logits.dim() == 3:
            if presence_logits.shape[-1] != 1:
                raise ValueError(
                    f"Expected presence logits as [B, C, 1], got {tuple(presence_logits.shape)}"
                )
            presence_logits = presence_logits.squeeze(-1)
        elif presence_logits.dim() != 2:
            raise ValueError(
                f"Expected presence logits as [B, C] or [B, C, 1], got {tuple(presence_logits.shape)}"
            )

        return presence_logits

    def _extract_presence_prob(
        self,
        raw_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_presence_score:
            return None

        presence_logits = self._extract_presence_logits(raw_outputs)
        if presence_logits is None:
            return None

        return presence_logits.sigmoid()

    @staticmethod
    def _extract_semantic_branch_logits(
        raw_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        semantic_seg = raw_outputs.get("semantic_seg", None)
        if semantic_seg is None:
            return None

        if semantic_seg.dim() == 5:
            if semantic_seg.shape[2] != 1:
                raise ValueError(
                    f"Expected semantic_seg as [B, C, 1, H, W], got {tuple(semantic_seg.shape)}"
                )
            semantic_seg = semantic_seg[:, :, 0]
        elif semantic_seg.dim() != 4:
            raise ValueError(
                f"Expected semantic_seg as [B, C, 1, H, W] or [B, C, H, W], got {tuple(semantic_seg.shape)}"
            )

        # 这里返回原始 logits，不做 sigmoid
        return semantic_seg

    @staticmethod
    def _validate_instance_tensors(
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
    ) -> None:
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

        if pred_logits.shape[:3] != pred_masks.shape[:3]:
            raise ValueError(
                "pred_logits and pred_masks shape mismatch: "
                f"{tuple(pred_logits.shape)} vs {tuple(pred_masks.shape)}"
            )

    @staticmethod
    def _build_instance_query_logits_for_training(
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
    ) -> torch.Tensor:
        # pred_logits: [B, C, Q, 1]
        # pred_masks:  [B, C, Q, H, W]
        #
        # 训练时在 logit 空间里聚合：
        # 每个 query 的像素级 logit = mask_logit + query_score_logit
        query_score_logits = pred_logits.squeeze(-1)[..., None, None]  # [B, C, Q, 1, 1]
        return pred_masks + query_score_logits  # [B, C, Q, H, W]

    def _aggregate_instance_logits_for_training(
        self,
        instance_query_logits: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.instance_train_aggregation

        if mode == "logsumexp":
            return torch.logsumexp(instance_query_logits, dim=2)

        if mode == "max":
            return instance_query_logits.amax(dim=2)

        if mode == "mean":
            return instance_query_logits.mean(dim=2)

        raise ValueError(
            f"Unknown instance_train_aggregation={mode}. "
            "Supported modes are: logsumexp, max, mean."
        )

    def _build_instance_score_map_for_inference(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        raw_outputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pred_logits: [B, C, Q, 1]
        # pred_masks:  [B, C, Q, H, W]
        #
        # 推理时沿用 score-map 逻辑，query 先做置信度过滤，再沿 query 聚合。
        query_scores = pred_logits.squeeze(-1).sigmoid()  # [B, C, Q]

        presence_prob = self._extract_presence_prob(raw_outputs)  # [B, C] or None
        if presence_prob is not None:
            query_scores = query_scores * presence_prob[..., None]

        if self.query_confidence_threshold is not None:
            query_keep_mask = query_scores > float(self.query_confidence_threshold)
            query_scores = query_scores * query_keep_mask.to(query_scores.dtype)
        else:
            query_keep_mask = torch.ones_like(query_scores, dtype=torch.bool)

        mask_probs = pred_masks.sigmoid()  # [B, C, Q, H, W]
        mode = self.instance_infer_aggregation

        if mode == "max":
            instance_score_map = (mask_probs * query_scores[..., None, None]).amax(dim=2)

        elif mode == "weighted_sum":
            weights = query_scores / query_scores.sum(dim=2, keepdim=True).clamp(min=1e-6)
            instance_score_map = (mask_probs * weights[..., None, None]).sum(dim=2)

        elif mode == "logsumexp":
            weighted = mask_probs * query_scores[..., None, None]
            instance_score_map = torch.logsumexp(weighted.clamp(min=1e-6), dim=2)

        else:
            raise ValueError(
                f"Unknown instance_infer_aggregation={mode}. "
                "Supported modes are: max, weighted_sum, logsumexp."
            )

        return instance_score_map, query_scores, query_keep_mask

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
        instance_score_map: Optional[torch.Tensor],
        semantic_score_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if instance_score_map is None and semantic_score_map is None:
            raise ValueError("At least one branch must be enabled.")

        if self.fusion_mode == "instance_only":
            if instance_score_map is None:
                raise ValueError("fusion_mode='instance_only' but instance branch is missing.")
            return instance_score_map

        if self.fusion_mode == "semantic_only":
            if semantic_score_map is None:
                raise ValueError("fusion_mode='semantic_only' but semantic branch is missing.")
            return semantic_score_map

        if self.fusion_mode == "max":
            if instance_score_map is None:
                return semantic_score_map
            if semantic_score_map is None:
                return instance_score_map
            return torch.maximum(instance_score_map, semantic_score_map)

        if self.fusion_mode == "sum":
            if instance_score_map is None:
                return semantic_score_map
            if semantic_score_map is None:
                return instance_score_map
            return instance_score_map + semantic_score_map

        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    @staticmethod
    def _infer_expected_num_classes(
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Optional[int]:
        if expected_num_classes is not None:
            return int(expected_num_classes)

        if len(batch.find_metadatas) == 0:
            return None

        try:
            return int(batch.find_metadatas[0].num_classes)
        except Exception:
            return None

    @staticmethod
    def _validate_class_count(
        actual_num_classes: int,
        expected_num_classes: Optional[int],
    ) -> None:
        if expected_num_classes is None:
            return

        if actual_num_classes != int(expected_num_classes):
            raise ValueError(
                f"Class count mismatch: expected {expected_num_classes}, "
                f"but got {actual_num_classes} channels."
            )

    def _build_train_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_branch_logits = None
        if self.use_semantic_branch:
            semantic_branch_logits = self._extract_semantic_branch_logits(raw_outputs)
            if semantic_branch_logits is None:
                raise ValueError(
                    "Semantic branch is enabled, but semantic_seg is missing."
                )

        instance_branch_logits = None
        if self.use_instance_branch:
            pred_masks = raw_outputs.get("pred_masks", None)
            pred_logits = raw_outputs.get("pred_logits", None)
            if pred_masks is None or pred_logits is None:
                raise ValueError(
                    "Instance branch is enabled, but pred_masks or pred_logits is missing."
                )

            self._validate_instance_tensors(pred_logits=pred_logits, pred_masks=pred_masks)
            instance_query_logits = self._build_instance_query_logits_for_training(
                pred_logits=pred_logits,
                pred_masks=pred_masks,
            )
            instance_branch_logits = self._aggregate_instance_logits_for_training(
                instance_query_logits
            )

        target_hw = None
        if instance_branch_logits is not None:
            target_hw = tuple(instance_branch_logits.shape[-2:])
        elif semantic_branch_logits is not None:
            target_hw = tuple(semantic_branch_logits.shape[-2:])

        if target_hw is None:
            raise ValueError("Cannot determine target spatial size for training outputs.")

        semantic_branch_logits = self._resize_to_match(semantic_branch_logits, target_hw)
        instance_branch_logits = self._resize_to_match(instance_branch_logits, target_hw)

        actual_num_classes = None
        if instance_branch_logits is not None:
            actual_num_classes = int(instance_branch_logits.shape[1])
        elif semantic_branch_logits is not None:
            actual_num_classes = int(semantic_branch_logits.shape[1])

        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        if actual_num_classes is not None:
            self._validate_class_count(
                actual_num_classes=actual_num_classes,
                expected_num_classes=expected_num_classes,
            )

        out: Dict[str, torch.Tensor] = {}

        if semantic_branch_logits is not None:
            out["semantic_branch_logits"] = semantic_branch_logits

        if instance_branch_logits is not None:
            out["instance_branch_logits"] = instance_branch_logits

        presence_logits = self._extract_presence_logits(raw_outputs)
        if presence_logits is not None:
            out["presence_logits"] = presence_logits

        return out

    def _build_inference_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        semantic_score_map = None
        if self.use_semantic_branch:
            semantic_branch_logits = self._extract_semantic_branch_logits(raw_outputs)
            if semantic_branch_logits is None:
                raise ValueError(
                    "Semantic branch is enabled, but semantic_seg is missing."
                )
            semantic_score_map = semantic_branch_logits.sigmoid()

        instance_score_map = None
        instance_query_scores = None
        instance_query_keep_mask = None
        if self.use_instance_branch:
            pred_masks = raw_outputs.get("pred_masks", None)
            pred_logits = raw_outputs.get("pred_logits", None)
            if pred_masks is None or pred_logits is None:
                raise ValueError(
                    "Instance branch is enabled, but pred_masks or pred_logits is missing."
                )

            self._validate_instance_tensors(pred_logits=pred_logits, pred_masks=pred_masks)
            (
                instance_score_map,
                instance_query_scores,
                instance_query_keep_mask,
            ) = self._build_instance_score_map_for_inference(
                pred_logits=pred_logits,
                pred_masks=pred_masks,
                raw_outputs=raw_outputs,
            )

        target_hw = None
        if instance_score_map is not None:
            target_hw = tuple(instance_score_map.shape[-2:])
        elif semantic_score_map is not None:
            target_hw = tuple(semantic_score_map.shape[-2:])

        if target_hw is None:
            raise ValueError("Cannot determine target spatial size for inference outputs.")

        semantic_score_map = self._resize_to_match(semantic_score_map, target_hw)
        instance_score_map = self._resize_to_match(instance_score_map, target_hw)

        actual_num_classes = None
        if instance_score_map is not None:
            actual_num_classes = int(instance_score_map.shape[1])
        elif semantic_score_map is not None:
            actual_num_classes = int(semantic_score_map.shape[1])

        expected_num_classes = self._infer_expected_num_classes(
            batch=batch,
            expected_num_classes=expected_num_classes,
        )
        if actual_num_classes is not None:
            self._validate_class_count(
                actual_num_classes=actual_num_classes,
                expected_num_classes=expected_num_classes,
            )

        fused_score_map = self._fuse_branches(
            instance_score_map=instance_score_map,
            semantic_score_map=semantic_score_map,
        )

        final_presence = self._extract_presence_prob(raw_outputs)
        if final_presence is not None:
            fused_score_map = fused_score_map * final_presence[..., None, None]

        fused_pred = fused_score_map.argmax(dim=1)

        out: Dict[str, torch.Tensor] = {
            "fused_score_map": fused_score_map,
            "fused_pred": fused_pred,
        }

        if semantic_score_map is not None:
            out["semantic_score_map"] = semantic_score_map

        if instance_score_map is not None:
            out["instance_score_map"] = instance_score_map

        if instance_query_scores is not None:
            out["instance_query_scores"] = instance_query_scores

        if instance_query_keep_mask is not None:
            out["instance_query_keep_mask"] = instance_query_keep_mask

        if final_presence is not None:
            out["presence_prob"] = final_presence

        return out

    def forward(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch: BatchedDatapoint,
        expected_num_classes: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        train_outputs = self._build_train_outputs(
            raw_outputs=raw_outputs,
            batch=batch,
            expected_num_classes=expected_num_classes,
        )

        inference_outputs = self._build_inference_outputs(
            raw_outputs=raw_outputs,
            batch=batch,
            expected_num_classes=expected_num_classes,
        )

        return {
            "train_outputs": train_outputs,
            "inference_outputs": inference_outputs,
        }