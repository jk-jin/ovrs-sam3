from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config_dataclasses import WINDOW_SCALES
from .remoteclip_guidance import RemoteCLIPGuidanceEmbedding
from .aggregation_attention import (
    AggregationAttentionBlock,
    apply_layer_norm_bcdhw,
    resize_prompt_features,
)
from .semantic_detail_fusion import SemanticDetailFusion
from .task_modes import OUTPUT_KEYS


class RemoteCLIPGuidedAggregation(nn.Module):
    """Aggregate encoder features at 36×36 with fixed RemoteCLIP Q/K guidance.

    Each block runs cross-class attention and averages five parallel window updates.
    Only encoder features receive residual updates; the semantic embedding
    stays trainable through Q/K. SAM3 FPN guides window Q/K. CLIP intermediate
    features are still extracted by the backbone but are unused here.
    """

    def __init__(
        self,
        clip_text_encoder,
        feature_dim: int = 256,
        clip_dim: int = 768,
        semantic_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 6,
        num_blocks: int = 2,
        dropout: float = 0.1,
        prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        use_checkpoint: bool = True,
        text_prompt_batch_size: int = 64,
        text_prompt_use_checkpoint: bool = True,
        share_window_weights_across_scales: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.use_checkpoint = bool(use_checkpoint)
        self.num_blocks = int(num_blocks)
        if prompt_templates is None:
            raise ValueError(
                "prompt_templates must be a list of 64 prompt templates."
            )
        if len(prompt_templates) != 64:
            raise ValueError(
                f"Expected 64 prompt templates, got {len(prompt_templates)}."
            )

        self.remoteclip_guidance = RemoteCLIPGuidanceEmbedding(
            clip_text_encoder=clip_text_encoder,
            prompt_templates=list(prompt_templates),
            normalize_label=bool(normalize_label_for_clip),
            clip_output_dim=int(clip_dim),
            feature_dim=int(semantic_dim),
            text_prompt_batch_size=int(text_prompt_batch_size),
            text_prompt_use_checkpoint=bool(text_prompt_use_checkpoint),
        )

        self.blocks = nn.ModuleList([
            AggregationAttentionBlock(
                feature_dim=self.feature_dim,
                semantic_dim=self.semantic_dim,
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
                share_window_weights_across_scales=share_window_weights_across_scales,
            )
            for _ in range(self.num_blocks)
        ])
        self.feature_output_norm = nn.LayerNorm(self.feature_dim)
        self.semantic_detail_fusion = SemanticDetailFusion(use_checkpoint=self.use_checkpoint)

    def fuse_semantic_detail_chunk(
        self,
        aggregated_feature: torch.Tensor,
        prompt_conditioned_feature_72: torch.Tensor,
        sam3_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        """Return [N, 256, 72, 72] for the original SAM3 Pixel Decoder.

        Args:
            aggregated_feature:        [N, 256, 36, 36]
            prompt_conditioned_feature_72:[N, 256, 72, 72]
            sam3_fpn_72:                  [B, 256, 72, 72]
        """
        return self.semantic_detail_fusion(
            aggregated_feature=aggregated_feature,
            prompt_conditioned_feature_72=prompt_conditioned_feature_72,
            sam3_fpn_72=sam3_fpn_72,
        )

    def forward(
        self,
        prompt_conditioned_features_72: torch.Tensor,
        remoteclip_feat_map: torch.Tensor,
        class_names: List[str],
        sam3_fpn_72: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full 36×36 aggregation on all prompts simultaneously."""
        if prompt_conditioned_features_72.ndim != 5:
            raise ValueError(
                "prompt_conditioned_features_72 must be [B, P, D, 72, 72], "
                f"got {tuple(prompt_conditioned_features_72.shape)}."
            )

        batch_size, num_prompts, hidden_dim, height, width = (
            prompt_conditioned_features_72.shape
        )

        if hidden_dim != self.feature_dim:
            raise ValueError(
                "prompt_conditioned_features_72 feature_dim mismatch: "
                f"expected {self.feature_dim}, got {hidden_dim}."
            )

        if (height, width) != (72, 72):
            raise ValueError(
                "RemoteCLIPGuidedAggregation expects 72×72 "
                f"encoder features, got {(height, width)}."
            )

        if remoteclip_feat_map.ndim != 4:
            raise ValueError(
                "remoteclip_feat_map must be [B, D_clip, 36, 36], "
                f"got {tuple(remoteclip_feat_map.shape)}."
            )

        if remoteclip_feat_map.shape[0] != batch_size:
            raise ValueError(
                "remoteclip_feat_map batch mismatch: expected "
                f"{batch_size}, got {remoteclip_feat_map.shape[0]}."
            )

        if tuple(remoteclip_feat_map.shape[-2:]) != (36, 36):
            raise ValueError(
                "remoteclip_feat_map must be 36×36, "
                f"got {tuple(remoteclip_feat_map.shape[-2:])}."
            )

        clip_guidance_36, clip_template_scores_36, remoteclip_template_text = self.remoteclip_guidance(
            class_names=class_names, remoteclip_feat_map=remoteclip_feat_map,
        )
        if sam3_fpn_72 is None or tuple(sam3_fpn_72.shape) != (batch_size, 256, 72, 72):
            raise ValueError("SAM3 spatial guidance requires [B, 256, 72, 72] backbone_fpn[2]")
        fpn_36 = F.interpolate(
            sam3_fpn_72, size=(36, 36), mode="bilinear", align_corners=False,
        )[:, None]
        # Guidance is fixed across branches and blocks. Resize FPN per image before broadcasting.
        semantic_scales = tuple(
            resize_prompt_features(clip_guidance_36, size) for size in WINDOW_SCALES
        )
        spatial_scales = tuple(
            resize_prompt_features(fpn_36, size).expand(-1, num_prompts, -1, -1, -1)
            for size in WINDOW_SCALES
        )

        feature_state_36 = resize_prompt_features(prompt_conditioned_features_72, WINDOW_SCALES[0])

        for block in self.blocks:
            block_args = (feature_state_36, semantic_scales, spatial_scales)
            if self.use_checkpoint and self.training:
                feature_state_36 = checkpoint(block, *block_args, use_reentrant=False)
            else:
                feature_state_36 = block(*block_args)

        # Normalize the accumulated feature state once after all blocks.
        feature_state_36 = apply_layer_norm_bcdhw(
            feature_state_36,
            self.feature_output_norm,
        )

        return {
            OUTPUT_KEYS.aggregated_features: feature_state_36,
            OUTPUT_KEYS.semantic_state_36: clip_guidance_36,
            OUTPUT_KEYS.clip_guidance_36: clip_guidance_36,
            OUTPUT_KEYS.clip_template_scores_36: clip_template_scores_36,
            OUTPUT_KEYS.remoteclip_template_text_features:
                remoteclip_template_text,
        }
