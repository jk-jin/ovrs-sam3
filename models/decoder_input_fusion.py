from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DecoderInputFusion(nn.Module):
    """Fuse Refiner36 and image-level FPN72 in one 256-channel branch.

    Upsampled Refiner features are concatenated with the matching image's
    FPN72, then mixed by 1×1 Conv/GN/GELU and a plain 3×3 output Conv.
    Pairs must be flattened in image-major, prompt-minor order. The output
    feeds the original frozen SAM3 Pixel Decoder with input gradients kept.
    """

    def __init__(self, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
        )

    def _forward_impl(
        self,
        refiner_feature_36: torch.Tensor,
        sam_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sam_fpn_72.shape[0]
        num_pairs = refiner_feature_36.shape[0]
        num_prompts = num_pairs // batch_size
        refiner_72 = F.interpolate(
            refiner_feature_36,
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_prompts, 256, 72, 72)
        # expand is a view; cat materializes only the required fusion input.
        fpn_72 = sam_fpn_72[:, None].expand(-1, num_prompts, -1, -1, -1)
        fusion_input = torch.cat((refiner_72, fpn_72), dim=2).reshape(
            num_pairs, 512, 72, 72
        )
        return self.fusion(fusion_input)

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        sam_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        if (
            refiner_feature_36.ndim != 4
            or tuple(refiner_feature_36.shape[1:]) != (256, 36, 36)
        ):
            raise ValueError("refiner_feature_36 must be [N, 256, 36, 36].")
        if sam_fpn_72.ndim != 4 or tuple(sam_fpn_72.shape[1:]) != (256, 72, 72):
            raise ValueError("sam_fpn_72 must be [B, 256, 72, 72].")
        batch_size = sam_fpn_72.shape[0]
        num_pairs = refiner_feature_36.shape[0]
        if batch_size <= 0 or num_pairs <= 0 or num_pairs % batch_size:
            raise ValueError("N must be positive and divisible by positive B.")
        if refiner_feature_36.device != sam_fpn_72.device:
            raise ValueError("Fusion inputs must be on the same device.")

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                self._forward_impl,
                refiner_feature_36,
                sam_fpn_72,
                use_reentrant=False,
            )
        return self._forward_impl(refiner_feature_36, sam_fpn_72)

