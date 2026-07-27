from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .score_embeddings import ClipScoreEmbedding
from .encoder_refiner_attention import EncoderRefinerLayer
from .refined_mask_decoder import RefinerMaskDecoder


class ClassConditionedEncoderRefiner(nn.Module):
    """Encoder feature refiner operating at 36×36.

    Takes the full 6-layer SAM3 encoder output after prompt cross-attention
    as input. The score stream is initialised solely from RemoteCLIP score
    embeddings. High-resolution mask decoding is delegated to
    decode_mask_chunk(), which must be called separately for each class chunk.

    Forward inputs:
        encoder_features_72:  [B, C, 256, 72, 72]  (full encoder + cross-attention)
        clip_image_feat_map:  [B, D_clip, 36, 36]
        sam_text_mean:        [B, C, 256]
        class_names:          list of C class names

    Forward outputs:
        refiner_features_36:  [B, C, 256, 36, 36]
        score_embed_36:       [B, C, 256, 36, 36]
        clip_score_embed_36:  [B, C, 256, 36, 36]
        clip_score_maps_36:   [B, C,  32, 36, 36]
        template_clip_text:   [C, 32, D_clip]
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 6,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        residual_scale_init: float = 0.1,
        use_checkpoint: bool = True,
        text_prompt_batch_size: int = 64,
        text_prompt_use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.use_checkpoint = bool(use_checkpoint)
        self.num_fusion_layers = int(fusion_layers)

        if prompt_templates is None:
            raise ValueError(
                "prompt_templates must be a list of 32 prompt templates."
            )
        if len(prompt_templates) != 32:
            raise ValueError(
                f"Expected 32 prompt templates, got {len(prompt_templates)}."
            )

        self.clip_score_embed = ClipScoreEmbedding(
            clip_text_encoder=clip_text_encoder,
            prompt_templates=list(prompt_templates),
            normalize_label=bool(normalize_label_for_clip),
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            text_prompt_batch_size=int(text_prompt_batch_size),
            text_prompt_use_checkpoint=bool(text_prompt_use_checkpoint),
        )

        self.layers = nn.ModuleList([
            EncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                score_embed_dim=self.score_embed_dim,
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
                residual_scale_init=float(residual_scale_init),
            )
            for _ in range(self.num_fusion_layers)
        ])

        self.mask_decoder = RefinerMaskDecoder(
            hidden_dim=self.hidden_dim,
            use_checkpoint=self.use_checkpoint,
        )

    def decode_mask_chunk(
        self,
        refiner_feature_36: torch.Tensor,
        original_feature_72: torch.Tensor,
        original_feature_144: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> torch.Tensor:
        return self.mask_decoder(
            refiner_feature_36=refiner_feature_36,
            original_feature_72=original_feature_72,
            original_feature_144=original_feature_144,
            original_feature_288=original_feature_288,
        )

    def initialize_mask_head_from_semantic_head(
        self,
        semantic_seg_head: nn.Conv2d,
    ) -> None:
        self.mask_decoder.initialize_mask_head_from_semantic_head(
            semantic_seg_head
        )

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        sam_text_mean: torch.Tensor,
        class_names: List[str],
    ) -> dict[str, torch.Tensor]:
        """Run the full Refiner at 36×36 on all classes at once.

        Args:
            encoder_features_72:  [B, C, 256, 72, 72]
                Full 6-layer encoder output after prompt cross-attention.
            clip_image_feat_map:  [B, D_clip, 36, 36]
            sam_text_mean:        [B, C, 256]
            class_names:          list of C class names

        Returns dict with keys:
            refiner_features_36
            score_embed_36
            clip_score_embed_36
            clip_score_maps_36
            template_clip_text

        The Refiner operates on ALL classes simultaneously so that
        cross-class attention works correctly. High-resolution mask
        decoding is done later per class chunk via decode_mask_chunk().
        """
        batch_size, num_classes, hidden_dim, H, W = encoder_features_72.shape

        if (H, W) != (72, 72):
            raise ValueError(
                f"ClassConditionedEncoderRefiner expects 72×72 encoder features, "
                f"got {(H, W)}."
            )
        if tuple(clip_image_feat_map.shape[-2:]) != (36, 36):
            raise ValueError(
                f"clip_image_feat_map must be 36×36, "
                f"got {tuple(clip_image_feat_map.shape[-2:])}."
            )
        if tuple(sam_text_mean.shape) != (batch_size, num_classes, hidden_dim):
            raise ValueError(
                f"sam_text_mean must be [{batch_size}, {num_classes}, {hidden_dim}], "
                f"got {tuple(sam_text_mean.shape)}."
            )

        # 1. CLIP score embedding at 36×36.
        (
            clip_score_embed_36,
            clip_score_maps_36,
            template_clip_text,
        ) = self.clip_score_embed(
            class_names=class_names,
            remoteclip_feat_map=clip_image_feat_map,
        )

        # 2. Downsample encoder features from 72×72 to 36×36 as the feature stream.
        base_feature_36 = F.interpolate(
            encoder_features_72.reshape(
                batch_size * num_classes,
                hidden_dim,
                72,
                72,
            ),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch_size,
            num_classes,
            hidden_dim,
            36,
            36,
        )

        feature_36 = base_feature_36

        # 3. Score stream is initialised solely from RemoteCLIP.
        score_embed_36 = clip_score_embed_36

        # 4. Run refiner layers.
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                feature_36, score_embed_36 = checkpoint(
                    layer,
                    feature_36,
                    score_embed_36,
                    sam_text_mean,
                    use_reentrant=False,
                )
            else:
                feature_36, score_embed_36 = layer(
                    feature_36=feature_36,
                    score_embed_36=score_embed_36,
                    sam_text_mean=sam_text_mean,
                )

        return {
            "refiner_features_36": feature_36,
            "score_embed_36": score_embed_36,
            "clip_score_embed_36": clip_score_embed_36,
            "clip_score_maps_36": clip_score_maps_36,
            "template_clip_text": template_clip_text,
        }
