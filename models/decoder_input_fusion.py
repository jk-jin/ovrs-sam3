from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DecoderInputFusionLayer(nn.Module):
    """One 72×72 layer: encoder concatenation, then semantic/detail fusion."""

    def __init__(self):
        super().__init__()
        self.input_fusion = nn.Conv2d(512, 256, 1, bias=False)
        self.semantic_refiner_proj = self._projection()
        self.detail_refiner_proj = self._projection()
        self.encoder_proj = self._projection()
        self.fpn_proj = self._projection()
        self.semantic_block = self._branch_block()
        self.detail_block = self._branch_block()
        self.semantic_out_proj = nn.Conv2d(128, 256, 1, bias=False)
        self.detail_out_proj = nn.Conv2d(128, 256, 1, bias=False)
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

    def forward(
        self,
        input_feature_72: torch.Tensor,
        original_encoder_feature_72: torch.Tensor,
        sam_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sam_fpn_72.shape[0]
        num_pairs = input_feature_72.shape[0]
        num_prompts = num_pairs // batch_size

        refiner_72 = self.input_fusion(torch.cat(
            [input_feature_72, original_encoder_feature_72], dim=1
        ))
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

        semantic_feature = semantic_input + self.semantic_block(semantic_input)
        detail_feature = detail_input + self.detail_block(detail_input)
        return self.fusion_out_proj(
            self.semantic_out_proj(semantic_feature)
            + self.detail_out_proj(detail_feature)
        )


class DecoderInputFusion(nn.Module):
    """Upsample Refiner36 once and apply independent 72×72 fusion layers."""

    def __init__(self, use_checkpoint: bool = True, num_layers: int = 2):
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        self.use_checkpoint = bool(use_checkpoint)
        self.layers = nn.ModuleList([DecoderInputFusionLayer() for _ in range(num_layers)])

    def _forward_impl(self, refiner_feature_36, original_encoder_feature_72, sam_fpn_72):
        feature_72 = F.interpolate(refiner_feature_36, size=(72, 72), mode="bilinear", align_corners=False)
        for layer in self.layers:
            feature_72 = layer(feature_72, original_encoder_feature_72, sam_fpn_72)
        return feature_72

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        original_encoder_feature_72: torch.Tensor,
        sam_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        if (
            refiner_feature_36.ndim != 4
            or tuple(refiner_feature_36.shape[1:]) != (256, 36, 36)
        ):
            raise ValueError("refiner_feature_36 must be [N, 256, 36, 36].")
        num_pairs = refiner_feature_36.shape[0]
        if tuple(original_encoder_feature_72.shape) != (num_pairs, 256, 72, 72):
            raise ValueError("original_encoder_feature_72 must be [N, 256, 72, 72].")
        if sam_fpn_72.ndim != 4 or tuple(sam_fpn_72.shape[1:]) != (256, 72, 72):
            raise ValueError("sam_fpn_72 must be [B, 256, 72, 72].")
        batch_size = sam_fpn_72.shape[0]
        if batch_size <= 0 or num_pairs <= 0 or num_pairs % batch_size:
            raise ValueError("N must be positive and divisible by positive B.")
        if not (
            refiner_feature_36.device
            == original_encoder_feature_72.device
            == sam_fpn_72.device
        ):
            raise ValueError("Fusion inputs must be on the same device.")

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                self._forward_impl,
                refiner_feature_36,
                original_encoder_feature_72,
                sam_fpn_72,
                use_reentrant=False,
            )
        return self._forward_impl(
            refiner_feature_36,
            original_encoder_feature_72,
            sam_fpn_72,
        )
