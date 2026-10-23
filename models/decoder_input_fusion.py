from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DecoderInputFusion(nn.Module):
    """Three independent semantic/detail/query-guided fusion branches at 72×72.

    The semantic branch retains the cross-attended encoder input. The query
    branch updates raw encoder tokens by attending to all 200 raw queries.
    Each branch pairs its guidance with an independent Refiner projection,
    works in 128 channels and projects back to 256 before summation.
    """

    def __init__(self, use_checkpoint: bool = True, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.semantic_refiner_proj = self._projection()
        self.detail_refiner_proj = self._projection()
        self.query_refiner_proj = self._projection()
        self.encoder_proj = self._projection()
        self.fpn_proj = self._projection()
        self.query_encoder_proj = self._projection()
        self.query_norm_encoder = nn.LayerNorm(256)
        self.query_norm_memory = nn.LayerNorm(256)
        self.query_cross_attention = nn.MultiheadAttention(
            256, num_heads, dropout=dropout, batch_first=True,
        )
        self.query_dropout = nn.Dropout(dropout)
        self.semantic_block = self._branch_block()
        self.detail_block = self._branch_block()
        self.query_block = self._branch_block()
        self.semantic_out_proj = nn.Conv2d(128, 256, 1, bias=False)
        self.detail_out_proj = nn.Conv2d(128, 256, 1, bias=False)
        self.query_out_proj = nn.Conv2d(128, 256, 1, bias=False)
        self.fusion_out_proj = nn.Conv2d(256, 256, 1, bias=False)

    @staticmethod
    def _projection() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
        )

    @staticmethod
    def _branch_block() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
        )

    def _forward_impl(
        self,
        refiner_feature_36: torch.Tensor,
        original_encoder_feature_72: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        raw_encoder_feature_72: torch.Tensor,
        instance_queries: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sam_fpn_72.shape[0]
        num_pairs = refiner_feature_36.shape[0]
        num_prompts = num_pairs // batch_size

        refiner_72 = F.interpolate(
            refiner_feature_36,
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        )
        semantic_input = (
            self.semantic_refiner_proj(refiner_72)
            + self.encoder_proj(original_encoder_feature_72)
        )
        detail_refiner = self.detail_refiner_proj(refiner_72).reshape(
            batch_size, num_prompts, 128, 72, 72
        )
        # FPN is projected per image and broadcast over the prompt dimension.
        detail_input = (
            detail_refiner + self.fpn_proj(sam_fpn_72)[:, None]
        ).reshape(num_pairs, 128, 72, 72)

        query_guidance = self.attend_instance_queries(raw_encoder_feature_72, instance_queries)
        query_input = (
            self.query_refiner_proj(refiner_72)
            + self.query_encoder_proj(query_guidance)
        )

        semantic_feature = semantic_input + self.semantic_block(semantic_input)
        detail_feature = detail_input + self.detail_block(detail_input)
        query_feature = query_input + self.query_block(query_input)
        return self.fusion_out_proj(
            self.semantic_out_proj(semantic_feature)
            + self.detail_out_proj(detail_feature)
            + self.query_out_proj(query_feature)
        )

    def attend_instance_queries(self, raw_encoder_feature_72, instance_queries):
        """Update raw encoder tokens; the frozen queries provide keys/values."""
        tokens = raw_encoder_feature_72.flatten(2).transpose(1, 2)
        memory = self.query_norm_memory(instance_queries)
        update = self.query_cross_attention(
            self.query_norm_encoder(tokens), memory, memory, need_weights=False,
        )[0]
        updated = tokens + self.query_dropout(update)
        return updated.transpose(1, 2).reshape_as(raw_encoder_feature_72)

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        original_encoder_feature_72: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        raw_encoder_feature_72: torch.Tensor,
        instance_queries: torch.Tensor,
    ) -> torch.Tensor:
        if (
            refiner_feature_36.ndim != 4
            or tuple(refiner_feature_36.shape[1:]) != (256, 36, 36)
        ):
            raise ValueError("refiner_feature_36 must be [N, 256, 36, 36].")
        num_pairs = refiner_feature_36.shape[0]
        if tuple(original_encoder_feature_72.shape) != (num_pairs, 256, 72, 72):
            raise ValueError("original_encoder_feature_72 must be [N, 256, 72, 72].")
        if tuple(raw_encoder_feature_72.shape) != (num_pairs, 256, 72, 72):
            raise ValueError("raw_encoder_feature_72 must be [N, 256, 72, 72].")
        if tuple(instance_queries.shape) != (num_pairs, 200, 256):
            raise ValueError("instance_queries must be [N, 200, 256].")
        if sam_fpn_72.ndim != 4 or tuple(sam_fpn_72.shape[1:]) != (256, 72, 72):
            raise ValueError("sam_fpn_72 must be [B, 256, 72, 72].")
        batch_size = sam_fpn_72.shape[0]
        if batch_size <= 0 or num_pairs <= 0 or num_pairs % batch_size:
            raise ValueError("N must be positive and divisible by positive B.")
        if not (
            refiner_feature_36.device
            == original_encoder_feature_72.device
            == sam_fpn_72.device
            == raw_encoder_feature_72.device
            == instance_queries.device
        ):
            raise ValueError("Fusion inputs must be on the same device.")

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                self._forward_impl,
                refiner_feature_36,
                original_encoder_feature_72,
                sam_fpn_72,
                raw_encoder_feature_72,
                instance_queries,
                use_reentrant=False,
            )
        return self._forward_impl(
            refiner_feature_36,
            original_encoder_feature_72,
            sam_fpn_72,
            raw_encoder_feature_72,
            instance_queries,
        )
