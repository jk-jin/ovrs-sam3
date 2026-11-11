from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def aggregate_orientations(
    feature: torch.Tensor,
    block: nn.Module,
) -> torch.Tensor:
    """Apply one shared conv block at 0°/90°/180°/270° and average deltas."""
    spatial_dims = (-2, -1)
    delta = block(feature)
    for quarter_turns in (1, 2, 3):
        rotated_feature = torch.rot90(feature, quarter_turns, spatial_dims)
        rotated_delta = block(rotated_feature)
        # Accumulate every delta back in the input coordinate system.
        delta = delta + torch.rot90(
            rotated_delta, -quarter_turns, spatial_dims
        )
    return feature + delta * 0.25


class SemanticDetailFusion(nn.Module):
    """Fuse aggregated36, prompt-conditioned encoder72 and FPN72 in four orientations.

    The output replaces the last FPN level consumed by the original SAM3
    Pixel Decoder. The semantic branch pairs the aggregated feature with the
    prompt-conditioned encoder72 feature; the detail branch pairs the
    aggregated feature with the image-level SAM3 FPN72 feature. Each branch
    keeps independent 256-channel projections and one 256-channel conv block
    shared across 0°/90°/180°/270° orientations. Inverse-rotated conv deltas
    are averaged back into the branch input, then both branches are projected,
    summed with equal weight and pass through a final 1×1 Conv.
    """

    def __init__(self, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.semantic_feature_proj = self._projection()
        self.detail_feature_proj = self._projection()
        self.encoder_feature_proj = self._projection()
        self.sam3_feature_proj = self._projection()
        self.semantic_block = self._branch_block()
        self.detail_block = self._branch_block()
        self.semantic_out_proj = nn.Conv2d(256, 256, 1, bias=False)
        self.detail_out_proj = nn.Conv2d(256, 256, 1, bias=False)
        self.fusion_out_proj = nn.Conv2d(256, 256, 1, bias=False)

    @staticmethod
    def _projection() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
        )

    @staticmethod
    def _branch_block() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
        )

    def _forward_impl(
        self,
        aggregated_feature_36: torch.Tensor,
        prompt_conditioned_feature_72: torch.Tensor,
        sam3_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sam3_fpn_72.shape[0]
        num_pairs = aggregated_feature_36.shape[0]
        num_prompts = num_pairs // batch_size

        aggregated_72 = F.interpolate(
            aggregated_feature_36,
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        )
        semantic_input = (
            self.semantic_feature_proj(aggregated_72)
            + self.encoder_feature_proj(prompt_conditioned_feature_72)
        )
        detail_aggregated = self.detail_feature_proj(aggregated_72).reshape(
            batch_size, num_prompts, 256, 72, 72
        )
        # SAM3 FPN is projected per image and broadcast over the prompt axis.
        detail_input = (
            detail_aggregated + self.sam3_feature_proj(sam3_fpn_72)[:, None]
        ).reshape(num_pairs, 256, 72, 72)

        semantic_feature = aggregate_orientations(
            semantic_input, self.semantic_block
        )
        detail_feature = aggregate_orientations(
            detail_input, self.detail_block
        )
        return self.fusion_out_proj(
            self.semantic_out_proj(semantic_feature)
            + self.detail_out_proj(detail_feature)
        )

    def forward(
        self,
        aggregated_feature_36: torch.Tensor,
        prompt_conditioned_feature_72: torch.Tensor,
        sam3_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        if (
            aggregated_feature_36.ndim != 4
            or tuple(aggregated_feature_36.shape[1:]) != (256, 36, 36)
        ):
            raise ValueError(
                "aggregated_feature_36 must be [N, 256, 36, 36]."
            )
        num_pairs = aggregated_feature_36.shape[0]
        if tuple(prompt_conditioned_feature_72.shape) != (
            num_pairs, 256, 72, 72
        ):
            raise ValueError(
                "prompt_conditioned_feature_72 must be [N, 256, 72, 72]."
            )
        if sam3_fpn_72.ndim != 4 or tuple(
            sam3_fpn_72.shape[1:]
        ) != (256, 72, 72):
            raise ValueError("sam3_fpn_72 must be [B, 256, 72, 72].")
        batch_size = sam3_fpn_72.shape[0]
        if batch_size <= 0 or num_pairs <= 0 or num_pairs % batch_size:
            raise ValueError("N must be positive and divisible by positive B.")
        if not (
            aggregated_feature_36.device
            == prompt_conditioned_feature_72.device
            == sam3_fpn_72.device
        ):
            raise ValueError("Fusion inputs must be on the same device.")

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                self._forward_impl,
                aggregated_feature_36,
                prompt_conditioned_feature_72,
                sam3_fpn_72,
                use_reentrant=False,
            )
        return self._forward_impl(
            aggregated_feature_36,
            prompt_conditioned_feature_72,
            sam3_fpn_72,
        )
