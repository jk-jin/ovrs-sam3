from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DecoderInputFusion(nn.Module):
    """Fuse Refiner36, original encoder72 and image-level FPN72.

    The output replaces the last FPN level consumed by the original SAM3
    Pixel Decoder. Semantic branch pairs the Refiner feature with the
    original encoder72 feature; detail branch pairs the Refiner feature with
    the image-level FPN72 feature. Both branches operate in 128-channel
    compact space with independent standard 3×3 conv blocks and independent
    Refiner projections, are summed with equal weight after projecting back
    to 256 channels, and pass through a final 1×1 Conv.
    """

    def __init__(self, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
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

    def _forward_impl(
        self,
        refiner_feature_36: torch.Tensor,
        original_encoder_feature_72: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        image_ids: torch.Tensor | None = None,
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
        detail_refiner = self.detail_refiner_proj(refiner_72)
        # Project before selecting pairs: never duplicate 256-channel FPN.
        fpn_projected = self.fpn_proj(sam_fpn_72)
        if image_ids is None:
            detail_input = (
                detail_refiner.reshape(batch_size, num_prompts, 128, 72, 72)
                + fpn_projected[:, None]
            ).reshape(num_pairs, 128, 72, 72)
        else:
            detail_input = detail_refiner + fpn_projected[image_ids]

        semantic_feature = semantic_input + self.semantic_block(semantic_input)
        detail_feature = detail_input + self.detail_block(detail_input)
        return self.fusion_out_proj(
            self.semantic_out_proj(semantic_feature)
            + self.detail_out_proj(detail_feature)
        )

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        original_encoder_feature_72: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        image_ids: torch.Tensor | None = None,
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
        if batch_size <= 0 or num_pairs <= 0:
            raise ValueError("Batch and pair counts must be positive.")
        if image_ids is None:
            if num_pairs % batch_size:
                raise ValueError("N must be divisible by B without explicit image_ids.")
        elif image_ids.shape != (num_pairs,) or image_ids.dtype != torch.long:
            raise ValueError("image_ids must be a long tensor of shape [N].")
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
                image_ids,
                use_reentrant=False,
            )
        return self._forward_impl(
            refiner_feature_36,
            original_encoder_feature_72,
            sam_fpn_72,
            image_ids,
        )
