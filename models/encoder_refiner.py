from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .clip_score_embedding import ClipScoreEmbeddingBuilder
from .encoder_refiner_attention import EncoderRefinerLayer


class ClassConditionedEncoderRefiner(nn.Module):
    """
    Multi-layer encoder feature refiner with multi-scale spatial window attention.

    All sub-modules are created eagerly in __init__ so that parameters are
    visible to apply_freeze_cfg and the optimizer before the first forward.

    Inputs (forward):
        encoder_features:       [B, C, D, H, W]
        clip_image_feat_map:    [B, D_clip, Hc, Wc]
        clip_mid_features:      list of [B, D_native, Hc, Wc]
        clip_mid_layer_indices: tuple of layer indices
        sam_text_mean:          [B, C, D]
        class_names:            list of C strings
        sam_image_last:         [B, D, H, W]

    Output (forward):
        refined_encoder_features: [B, C, D, 72, 72]
        clip_score_embeds:        {"scale_18": ..., "scale_36": ..., "scale_72": ...}
        clip_score_maps_18:       [B, C, 16, 18, 18]
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        clip_mid_dim: int = 1024,
        score_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        prompt_templates: list[str] | tuple[str, ...] = (),
        normalize_label_for_clip: bool = True,
        score_conv_kernel: int = 7,
        score_base_hw: int = 18,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.score_builder = ClipScoreEmbeddingBuilder(
            clip_text_encoder=clip_text_encoder,
            clip_output_dim=int(clip_dim),
            clip_mid_dim=int(clip_mid_dim),
            score_embed_dim=int(score_embed_dim),
            prompt_templates=list(prompt_templates),
            normalize_label_for_clip=bool(normalize_label_for_clip),
            conv_kernel=int(score_conv_kernel),
            base_hw=int(score_base_hw),
            use_checkpoint=bool(use_checkpoint),
        )

        self.layers = nn.ModuleList([
            EncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                score_embed_dim=int(score_embed_dim),
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
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
        clip_mid_features: list[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
        sam_text_mean: torch.Tensor,
        class_names: List[str],
        sam_image_last: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            refined_encoder_features_72: [B, C, D, 72, 72]
            clip_score_embeds:           dict of 3-scale score embeddings
            clip_score_maps_18:          [B, C, 16, 18, 18]
        """
        batch_size, num_classes, hidden_dim, height, width = encoder_features.shape

        if (height, width) != (72, 72):
            raise ValueError(
                f"ClassConditionedEncoderRefiner expects 72x72 encoder features, "
                f"got {(height, width)}."
            )

        if tuple(sam_text_mean.shape) != (batch_size, num_classes, hidden_dim):
            raise ValueError(
                f"sam_text_mean must be [{batch_size}, {num_classes}, {hidden_dim}], "
                f"got {tuple(sam_text_mean.shape)}."
            )

        clip_score_embeds, clip_score_maps_18 = self.score_builder(
            class_names=class_names,
            clip_image_feat_map=clip_image_feat_map,
            clip_mid_features=clip_mid_features,
            clip_mid_layer_indices=clip_mid_layer_indices,
        )

        # Validate score embedding shapes.
        for key, expected_hw in {
            "scale_18": (18, 18),
            "scale_36": (36, 36),
            "scale_72": (72, 72),
        }.items():
            score_embed = clip_score_embeds[key]
            expected_shape = (
                batch_size,
                num_classes,
                self.score_builder.score_embed_dim,
                expected_hw[0],
                expected_hw[1],
            )
            if tuple(score_embed.shape) != expected_shape:
                raise ValueError(
                    f"clip_score_embeds[{key!r}] shape mismatch: "
                    f"expected {expected_shape}, got {tuple(score_embed.shape)}."
                )

        refined_encoder_features_72 = encoder_features

        clip_score_embed_18 = clip_score_embeds["scale_18"]
        clip_score_embed_36 = clip_score_embeds["scale_36"]
        clip_score_embed_72 = clip_score_embeds["scale_72"]

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                refined_encoder_features_72 = checkpoint(
                    layer,
                    refined_encoder_features_72,
                    sam_text_mean,
                    sam_image_last,
                    clip_score_embed_18,
                    clip_score_embed_36,
                    clip_score_embed_72,
                    use_reentrant=False,
                )
            else:
                refined_encoder_features_72 = layer(
                    encoder_features_72=refined_encoder_features_72,
                    sam_text_mean=sam_text_mean,
                    sam_image_last_72=sam_image_last,
                    clip_score_embed_18=clip_score_embed_18,
                    clip_score_embed_36=clip_score_embed_36,
                    clip_score_embed_72=clip_score_embed_72,
                )

        return (
            refined_encoder_features_72,
            clip_score_embeds,
            clip_score_maps_18,
        )
