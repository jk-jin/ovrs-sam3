from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DecoderInputFusion(nn.Module):
    """Pair each refined stream with its original 72×72 source, then sum/fuse."""

    def __init__(self, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.semantic_branch = self._branch()
        self.detail_branch = self._branch()
        self.fusion_out_proj = nn.Conv2d(256, 256, 1, bias=False)

    @staticmethod
    def _branch():
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )

    def _forward_impl(
        self,
        refined_encoder_36,
        refined_fpn_36,
        original_encoder_feature_72,
        sam_fpn_72,
    ):
        batch_size = sam_fpn_72.shape[0]
        pairs = refined_encoder_36.shape[0]
        prompts = pairs // batch_size
        encoder_72 = F.interpolate(
            refined_encoder_36, (72, 72), mode="bilinear", align_corners=False
        )
        fpn_72 = F.interpolate(
            refined_fpn_36, (72, 72), mode="bilinear", align_corners=False
        )
        original_fpn = sam_fpn_72[:, None].expand(-1, prompts, -1, -1, -1)
        # Concatenate in [batch, prompt, channel, height, width] order so that
        # batches > 1 keep their own image-level FPN for every prompt.
        detail_input = torch.cat(
            (fpn_72.reshape(batch_size, prompts, 256, 72, 72), original_fpn), dim=2
        ).reshape(pairs, 512, 72, 72)
        semantic = self.semantic_branch(
            torch.cat((encoder_72, original_encoder_feature_72), dim=1)
        )
        detail = self.detail_branch(detail_input)
        return self.fusion_out_proj(semantic + detail)

    def forward(
        self,
        refined_encoder_36,
        refined_fpn_36,
        original_encoder_feature_72,
        sam_fpn_72,
    ):
        if refined_encoder_36.ndim != 4 or tuple(refined_encoder_36.shape[1:]) != (
            256,
            36,
            36,
        ):
            raise ValueError("refined_encoder_36 must be [N, 256, 36, 36]")
        if refined_fpn_36.shape != refined_encoder_36.shape:
            raise ValueError("refined_fpn_36 must match refined_encoder_36")
        pairs = refined_encoder_36.shape[0]
        if tuple(original_encoder_feature_72.shape) != (pairs, 256, 72, 72):
            raise ValueError("original_encoder_feature_72 must be [N, 256, 72, 72]")
        if sam_fpn_72.ndim != 4 or tuple(sam_fpn_72.shape[1:]) != (256, 72, 72):
            raise ValueError("sam_fpn_72 must be [B, 256, 72, 72]")
        batch_size = sam_fpn_72.shape[0]
        if batch_size <= 0 or pairs <= 0 or pairs % batch_size:
            raise ValueError("N must be positive and divisible by positive B")
        inputs = (
            refined_encoder_36,
            refined_fpn_36,
            original_encoder_feature_72,
            sam_fpn_72,
        )
        if any(x.device != refined_encoder_36.device for x in inputs):
            raise ValueError("Fusion inputs must be on the same device")
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, *inputs, use_reentrant=False)
        return self._forward_impl(*inputs)
