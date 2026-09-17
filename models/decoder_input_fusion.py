from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderInputFusion72(nn.Module):
    """Fuse Refiner, original encoder and FPN features at 72x72.

    Class-conditioned inputs use image-major, prompt-minor pair order.
    FPN stays image-level until its projected feature is broadcast.
    """

    def __init__(self):
        super().__init__()
        self.semantic_refiner_proj = self._make_input_proj()
        self.detail_refiner_proj = self._make_input_proj()
        self.encoder_proj = self._make_input_proj()
        self.fpn_proj = self._make_input_proj()

        self.semantic_block = self._make_branch_block()
        self.detail_block = self._make_branch_block()

        self.semantic_out_proj = nn.Conv2d(128, 256, 1, bias=False)
        self.detail_out_proj = nn.Conv2d(128, 256, 1, bias=False)
        self.fusion_out_proj = nn.Conv2d(256, 256, 1, bias=False)

    @staticmethod
    def _make_input_proj() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
        )

    @staticmethod
    def _make_branch_block() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
        )

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        encoder_feature_72: torch.Tensor,
        sam_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        if (
            refiner_feature_36.ndim != 4
            or tuple(refiner_feature_36.shape[1:]) != (256, 36, 36)
        ):
            raise ValueError("refiner_feature_36 must be [N, 256, 36, 36].")

        num_pairs = refiner_feature_36.shape[0]
        if tuple(encoder_feature_72.shape) != (num_pairs, 256, 72, 72):
            raise ValueError("encoder_feature_72 must be [N, 256, 72, 72].")
        if (
            sam_fpn_72.ndim != 4
            or tuple(sam_fpn_72.shape[1:]) != (256, 72, 72)
        ):
            raise ValueError("sam_fpn_72 must be [B, 256, 72, 72].")

        batch_size = sam_fpn_72.shape[0]
        if batch_size <= 0 or num_pairs <= 0 or num_pairs % batch_size != 0:
            raise ValueError("N and B must be positive, and N must be divisible by B.")
        if not (
            refiner_feature_36.device
            == encoder_feature_72.device
            == sam_fpn_72.device
        ):
            raise ValueError("All fusion inputs must be on the same device.")

        prompts_per_image = num_pairs // batch_size
        refiner_feature_72 = F.interpolate(
            refiner_feature_36,
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        )

        semantic_input = (
            self.semantic_refiner_proj(refiner_feature_72)
            + self.encoder_proj(encoder_feature_72)
        )

        detail_refiner = self.detail_refiner_proj(refiner_feature_72)
        fpn_compact = self.fpn_proj(sam_fpn_72)
        detail_input = (
            detail_refiner.reshape(batch_size, prompts_per_image, 128, 72, 72)
            + fpn_compact[:, None]
        ).reshape(num_pairs, 128, 72, 72)

        semantic_feature = semantic_input + self.semantic_block(semantic_input)
        detail_feature = detail_input + self.detail_block(detail_input)

        semantic_out = self.semantic_out_proj(semantic_feature)
        detail_out = self.detail_out_proj(detail_feature)
        return self.fusion_out_proj(semantic_out + detail_out)
