from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .remoteclip_guidance import RemoteCLIPGuidanceEmbedding
from .aggregation_attention import (
    CrossClassAggregationBlock,
    IntraClassAggregationBlock,
    apply_layer_norm_bcdhw,
)
from .semantic_detail_fusion import SemanticDetailFusion
from .task_modes import OUTPUT_KEYS


class RemoteCLIPGuidedAggregation(nn.Module):
    """RemoteCLIP-guided cross-class feature aggregation at 36×36.

    Takes the full 6-layer SAM3 encoder output after prompt cross-attention
    as input. The feature state is the bilinear-downsampled encoder feature
    without any FPN injection. Blocks alternate cross-class and intra-class
    attention, starting with cross-class. Fixed RemoteCLIP middle-layer and
    SAM3 FPN guidance extend intra-class attention Q/K. The semantic state
    comes directly from RemoteCLIPGuidanceEmbedding without SAM3 FPN fusion.

    The aggregation module outputs 36×36 features. High-resolution decoding
    is handled by one SemanticDetailFusion at 72×72. After one upsampling,
    independent semantic/detail branches fuse the aggregated feature with
    the prompt-conditioned encoder72 and SAM3 FPN72 respectively, without an
    extra encoder concatenation before the branches. The resulting decoder
    feature replaces the original Pixel Decoder's last FPN input.

    After all blocks, the accumulated 36×36 feature state is normalized once
    with a channel-wise LayerNorm before being returned. The final semantic
    state is not post-normalized.
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
        num_blocks: int = 4,
        dropout: float = 0.1,
        prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        use_checkpoint: bool = True,
        text_prompt_batch_size: int = 64,
        text_prompt_use_checkpoint: bool = True,
        remoteclip_native_dim: int | None = None,
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

        # Alternate cross → intra → cross → intra without sharing parameters.
        blocks = []
        for block_index in range(self.num_blocks):
            if block_index % 2 == 0:
                blocks.append(
                    CrossClassAggregationBlock(
                        feature_dim=self.feature_dim,
                        semantic_dim=self.semantic_dim,
                        num_heads=int(num_heads),
                        dropout=float(dropout),
                    )
                )
            else:
                blocks.append(
                    IntraClassAggregationBlock(
                        feature_dim=self.feature_dim,
                        semantic_dim=self.semantic_dim,
                        num_heads=int(num_heads),
                        window_size=int(window_size),
                        shift_size=int(shift_size),
                        dropout=float(dropout),
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        # Final normalization for the accumulated feature residual stream.
        self.feature_output_norm = nn.LayerNorm(self.feature_dim)

        self.semantic_detail_fusion = SemanticDetailFusion(
            use_checkpoint=self.use_checkpoint,
        )
        if remoteclip_native_dim is None or remoteclip_native_dim <= 0:
            raise ValueError(
                "remoteclip_native_dim must be positive for RemoteCLIP "
                "middle-layer guidance"
            )
        self.remoteclip_native_dim = int(remoteclip_native_dim)
        self.clip_spatial_guidance_proj = nn.Conv2d(
            self.remoteclip_native_dim * 2, 256, kernel_size=1
        )

    def fuse_semantic_detail_chunk(
        self,
        aggregated_feature_36: torch.Tensor,
        prompt_conditioned_feature_72: torch.Tensor,
        sam3_fpn_72: torch.Tensor,
    ) -> torch.Tensor:
        """Return [N, 256, 72, 72] for the original SAM3 Pixel Decoder.

        Args:
            aggregated_feature_36:        [N, 256, 36, 36]
            prompt_conditioned_feature_72:[N, 256, 72, 72]
            sam3_fpn_72:                  [B, 256, 72, 72]
        """
        return self.semantic_detail_fusion(
            aggregated_feature_36=aggregated_feature_36,
            prompt_conditioned_feature_72=prompt_conditioned_feature_72,
            sam3_fpn_72=sam3_fpn_72,
        )

    def forward(
        self,
        prompt_conditioned_features_72: torch.Tensor,
        remoteclip_feat_map: torch.Tensor,
        class_names: List[str],
        clip_spatial_guidance_features: list[torch.Tensor] | None = None,
        clip_spatial_guidance_layer_indices: tuple[int, ...] | None = None,
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

        (
            clip_guidance_36,
            clip_template_scores_36,
            remoteclip_template_text,
        ) = self.remoteclip_guidance(
            class_names=class_names,
            remoteclip_feat_map=remoteclip_feat_map,
        )

        # Project image-level guidance once before broadcasting over prompts.
        if (
            clip_spatial_guidance_features is None
            or clip_spatial_guidance_layer_indices is None
        ):
            raise ValueError(
                "RemoteCLIP middle guidance requires features and layer "
                "indices [7, 15]"
            )
        if len(clip_spatial_guidance_features) != len(
            clip_spatial_guidance_layer_indices
        ) or len(set(clip_spatial_guidance_layer_indices)) != len(
            clip_spatial_guidance_layer_indices
        ):
            raise ValueError(
                "RemoteCLIP middle feature indices must be unique and match "
                "the feature count"
            )
        by_layer = dict(
            zip(
                clip_spatial_guidance_layer_indices,
                clip_spatial_guidance_features,
            )
        )
        if 7 not in by_layer or 15 not in by_layer:
            raise ValueError(
                "RemoteCLIP middle guidance requires indices 7 and 15 "
                "(layers 8 and 16)"
            )
        mids = [by_layer[7], by_layer[15]]
        if any(
            tuple(x.shape) != (batch_size, self.remoteclip_native_dim, 36, 36)
            for x in mids
        ):
            raise ValueError(
                "RemoteCLIP middle features must match native channels and "
                "the 36×36 grid"
            )
        projected = self.clip_spatial_guidance_proj(torch.cat(mids, dim=1))
        clip_spatial_guidance_36 = projected[:, None].expand(
            -1, num_prompts, -1, -1, -1
        )
        if sam3_fpn_72 is None or tuple(sam3_fpn_72.shape) != (
            batch_size, 256, 72, 72
        ):
            raise ValueError(
                "SAM3 spatial guidance requires [B, 256, 72, 72] "
                "backbone_fpn[2]"
            )
        downsampled = F.interpolate(
            sam3_fpn_72,
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )
        sam3_spatial_guidance_36 = downsampled[:, None].expand(
            -1, num_prompts, -1, -1, -1
        )

        initial_feature_36 = F.interpolate(
            prompt_conditioned_features_72.reshape(
                batch_size * num_prompts,
                hidden_dim,
                72,
                72,
            ),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch_size,
            num_prompts,
            hidden_dim,
            36,
            36,
        )

        feature_state_36 = initial_feature_36

        # The semantic state produced by remoteclip_guidance.py is used
        # directly. No SAM3 FPN injection is performed before attention.
        semantic_state_36 = clip_guidance_36

        for block in self.blocks:
            if isinstance(block, IntraClassAggregationBlock):
                block_args = (
                    block,
                    feature_state_36,
                    semantic_state_36,
                    clip_spatial_guidance_36,
                    sam3_spatial_guidance_36,
                )
            else:
                block_args = (
                    block,
                    feature_state_36,
                    semantic_state_36,
                )

            if self.use_checkpoint and self.training:
                feature_state_36, semantic_state_36 = checkpoint(
                    *block_args,
                    use_reentrant=False,
                )
            else:
                feature_state_36, semantic_state_36 = block_args[0](
                    *block_args[1:]
                )

        # Normalize the accumulated feature state once after all blocks.
        feature_state_36 = apply_layer_norm_bcdhw(
            feature_state_36,
            self.feature_output_norm,
        )

        return {
            OUTPUT_KEYS.aggregated_features_36: feature_state_36,
            OUTPUT_KEYS.semantic_state_36: semantic_state_36,
            OUTPUT_KEYS.clip_guidance_36: clip_guidance_36,
            OUTPUT_KEYS.clip_template_scores_36: clip_template_scores_36,
            OUTPUT_KEYS.remoteclip_template_text_features:
                remoteclip_template_text,
        }
