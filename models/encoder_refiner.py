from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .clip_prompt_encoder import SingleTokenClipPromptEncoder
from .clip_score_prior import ClipScorePriorBuilder
from .encoder_query_extractor import EncoderQueryExtractor
from .encoder_refiner_blocks import TextScoreEncoderRefinerLayer


class TextScoreGuidedEncoderRefiner(nn.Module):
    """
    Multi-layer encoder feature refiner with text-conditioned class attention
    and low-res score-guided spatial refiner.

    All sub-modules are created eagerly in __init__ so that parameters are
    visible to apply_freeze_cfg and the optimizer before the first forward.

    Inputs (forward):
        encoder_features:   [B, C, D, H, W]
        clip_image_feat_map: [B, D_clip, Hc, Wc]
        class_names:        list of C strings
        sam_image_last:     [B, D, H, W]

    Output (forward):
        refined_encoder_features_72: [B, C, D, 72, 72]
        class_query_tokens:          [B, C, Q, D]
        dynamic_clip_text:           [B, C, Q, D_clip]
        score_embed_18:              [B, C, D_score, 18, 18]
        score_maps_18:               [B, C, Q, 18, 18]
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        num_query_tokens: int = 32,
        prompt_template: str = "a remote sensing image of {}.",
        normalize_label_for_clip: bool = True,
        score_conv_kernel: int = 7,
        score_base_hw: int = 18,
        spatial_fusion_conv_kernel: int = 7,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.use_checkpoint = bool(use_checkpoint)

        self.query_extractor = EncoderQueryExtractor(
            hidden_dim=self.hidden_dim,
            num_query_tokens=self.num_query_tokens,
            num_heads=int(num_heads),
            dropout=float(dropout),
        )

        self.clip_prompt_encoder = SingleTokenClipPromptEncoder(
            clip_text_encoder=clip_text_encoder,
            prompt_template=str(prompt_template),
            sam_dim=self.hidden_dim,
            normalize_label=bool(normalize_label_for_clip),
            use_checkpoint=bool(use_checkpoint),
            num_attention_heads=int(num_heads),
        )

        self.score_prior_builder = ClipScorePriorBuilder(
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            num_query_tokens=self.num_query_tokens,
            conv_kernel=int(score_conv_kernel),
            base_hw=int(score_base_hw),
        )

        self.layers = nn.ModuleList([
            TextScoreEncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                score_embed_dim=int(score_embed_dim),
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
                spatial_fusion_kernel=int(spatial_fusion_conv_kernel),
            )
            for _ in range(int(fusion_layers))
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_features: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        sam_text_mean: torch.Tensor,
        class_names: List[str],
        sam_image_last: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            refined_encoder_features_72: [B, C, D, 72, 72]
            class_query_tokens:          [B, C, Q, D]
            dynamic_clip_text:           [B, C, Q, D_clip]
            score_embed_18:              [B, C, D_score, 18, 18]
            score_maps_18:               [B, C, Q, 18, 18]
        """
        batch_size, num_classes, hidden_dim, height, width = encoder_features.shape

        if (height, width) != (72, 72):
            raise ValueError(
                f"TextScoreGuidedEncoderRefiner expects 72x72 encoder features, "
                f"got {(height, width)}."
            )

        if tuple(sam_text_mean.shape) != (batch_size, num_classes, hidden_dim):
            raise ValueError(
                f"sam_text_mean must be [{batch_size}, {num_classes}, {hidden_dim}], "
                f"got {tuple(sam_text_mean.shape)}."
            )

        class_query_tokens = self.query_extractor(encoder_features)

        dynamic_clip_text = self.clip_prompt_encoder(
            class_query_tokens=class_query_tokens,
            class_names=class_names,
            clip_image_feat_map=clip_image_feat_map,
        )

        score_embed_18, score_maps_18 = self.score_prior_builder(
            dynamic_clip_text=dynamic_clip_text,
            clip_image_feat_map=clip_image_feat_map,
        )

        # Validate score_embed_18 shape
        expected_score_shape = (
            batch_size,
            num_classes,
            self.score_prior_builder.score_embed_dim,
            18,
            18,
        )
        if tuple(score_embed_18.shape) != expected_score_shape:
            raise ValueError(
                f"score_embed_18 shape mismatch: "
                f"expected {expected_score_shape}, got {tuple(score_embed_18.shape)}."
            )

        refined_encoder_features_72 = encoder_features

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                refined_encoder_features_72 = checkpoint(
                    layer,
                    refined_encoder_features_72,
                    sam_text_mean,
                    sam_image_last,
                    score_embed_18,
                    use_reentrant=False,
                )
            else:
                refined_encoder_features_72 = layer(
                    encoder_features_72=refined_encoder_features_72,
                    sam_text_mean=sam_text_mean,
                    sam_image_last_72=sam_image_last,
                    score_embed_18=score_embed_18,
                )

        return (
            refined_encoder_features_72,
            class_query_tokens,
            dynamic_clip_text,
            score_embed_18,
            score_maps_18,
        )
