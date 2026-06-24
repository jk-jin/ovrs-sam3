from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .clip_prompt_encoder import SingleTokenClipPromptEncoder
from .clip_score_embedding import ClipScoreEmbeddingBuilder
from .encoder_query_extractor import EncoderQueryExtractor
from .encoder_refiner_attention import EncoderRefinerLayer


class ClassConditionedEncoderRefiner(nn.Module):
    """
    Multi-layer encoder feature refiner with multi-scale spatial window attention.

    All sub-modules are created eagerly in __init__ so that parameters are
    visible to apply_freeze_cfg and the optimizer before the first forward.

    Inputs (forward):
        encoder_features:   [B, C, D, H, W]
        clip_image_feat_map: [B, D_clip, Hc, Wc]
        sam_text_mean:      [B, C, D]
        class_names:        list of C strings
        sam_image_last:     [B, D, H, W]
        clip_mid_features:  (optional) list of [B, D_native, Hc, Wc]
        clip_mid_layer_indices: (optional) list of int

    Output (forward):
        refined_encoder_features: [B, C, D, 72, 72]
        class_query_tokens:       [B, C, Q, D]  or None (fixed_templates mode)
        clip_text_features:       [B, C, Q, D_clip]
        clip_score_embeds:        {"scale_18": ..., "scale_36": ..., "scale_72": ...}
        clip_score_maps_18:       [B, C, Q, 18, 18]
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
        use_checkpoint: bool = True,
        # Ablation config
        score_embed_source: str = "learned_query",
        fixed_score_templates: list[str] | None = None,
        score_upsample_fuse_clip_mid: bool = False,
        score_mid_proj_dim: int = 64,
        clip_mid_native_dim: int = 1024,
        clip_mid_layer_for_36: int = 15,
        clip_mid_layer_for_72: int = 7,
        window_attention_scales: list[int] | None = None,
        class_attention_context: str = "sam_text_score",
        spatial_upsample_fuse_sam_fpn: bool = False,
        sam_fpn_fuse_proj_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.use_checkpoint = bool(use_checkpoint)

        self.score_embed_source = str(score_embed_source)
        if self.score_embed_source not in {"learned_query", "fixed_templates"}:
            raise ValueError(
                f"score_embed_source must be 'learned_query' or 'fixed_templates', "
                f"got {self.score_embed_source!r}."
            )

        if window_attention_scales is None:
            window_attention_scales = [36, 18]
        self.window_attention_scales = list(window_attention_scales)

        self.class_attention_context = str(class_attention_context)

        # Always create query_extractor for potential debug/analysis,
        # but it is only used in learned_query mode.
        self.query_extractor = EncoderQueryExtractor(
            hidden_dim=self.hidden_dim,
            num_query_tokens=self.num_query_tokens,
            num_heads=int(num_heads),
            dropout=float(dropout),
        )

        # Always create clip_prompt_encoder for learned_query mode.
        self.clip_prompt_encoder = SingleTokenClipPromptEncoder(
            clip_text_encoder=clip_text_encoder,
            prompt_template=str(prompt_template),
            sam_dim=self.hidden_dim,
            normalize_label=bool(normalize_label_for_clip),
            use_checkpoint=bool(use_checkpoint),
            num_attention_heads=int(num_heads),
        )

        score_builder_num_queries = (
            len(fixed_score_templates)
            if self.score_embed_source == "fixed_templates" and fixed_score_templates
            else self.num_query_tokens
        )

        self.score_builder = ClipScoreEmbeddingBuilder(
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            num_query_tokens=score_builder_num_queries,
            conv_kernel=int(score_conv_kernel),
            base_hw=int(score_base_hw),
            score_upsample_fuse_clip_mid=bool(score_upsample_fuse_clip_mid),
            score_mid_proj_dim=int(score_mid_proj_dim),
            clip_mid_native_dim=int(clip_mid_native_dim),
            clip_mid_layer_for_36=int(clip_mid_layer_for_36),
            clip_mid_layer_for_72=int(clip_mid_layer_for_72),
        )

        self.fixed_score_templates = (
            list(fixed_score_templates)
            if fixed_score_templates is not None
            else []
        )

        self.layers = nn.ModuleList([
            EncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                score_embed_dim=int(score_embed_dim),
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
                class_attention_context=self.class_attention_context,
                window_attention_scales=self.window_attention_scales,
                spatial_upsample_fuse_sam_fpn=bool(spatial_upsample_fuse_sam_fpn),
                sam_fpn_fuse_proj_dim=int(sam_fpn_fuse_proj_dim),
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
        clip_mid_features: list[torch.Tensor] | None = None,
        clip_mid_layer_indices: list[int] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            refined_encoder_features_72: [B, C, D, 72, 72]
            class_query_tokens:          [B, C, Q, D] or None (fixed_templates mode)
            clip_text_features:          [B, C, Q, D_clip]
            clip_score_embeds:           dict of 3-scale score embeddings
            clip_score_maps_18:          [B, C, Q, 18, 18]
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

        if self.score_embed_source == "learned_query":
            class_query_tokens = self.query_extractor(encoder_features)

            clip_text_features = self.clip_prompt_encoder(
                class_query_tokens=class_query_tokens,
                class_names=class_names,
                clip_image_feat_map=clip_image_feat_map,
            )
        elif self.score_embed_source == "fixed_templates":
            class_query_tokens = None

            if not self.fixed_score_templates:
                raise ValueError(
                    "score_embed_source='fixed_templates' but fixed_score_templates is empty."
                )

            trainable_text = any(
                p.requires_grad for p in self.clip_prompt_encoder.clip_text_encoder.parameters()
            )
            use_text_cache = (not self.training) or (not trainable_text)

            fixed_text = self.clip_prompt_encoder.clip_text_encoder.encode_prompt_templates_trainable(
                class_names=class_names,
                templates=self.fixed_score_templates,
                device=encoder_features.device,
                normalize_label=bool(self.clip_prompt_encoder.normalize_label),
                normalize=True,
                use_cache=use_text_cache,
                detach_output=use_text_cache,
            )
            # fixed_text: [C, K, D_clip]
            K = fixed_text.shape[1]
            clip_text_features = fixed_text.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            raise ValueError(
                f"Unknown score_embed_source: {self.score_embed_source!r}"
            )

        clip_score_embeds, clip_score_maps_18 = self.score_builder(
            clip_text_features=clip_text_features,
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
            class_query_tokens,
            clip_text_features,
            clip_score_embeds,
            clip_score_maps_18,
        )
