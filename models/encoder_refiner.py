from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .clip_prompt_encoder import SingleTokenClipPromptEncoder
from .clip_score_embedding import ClipScoreEmbeddingBuilder18
from .encoder_query_extractor import EncoderQueryExtractor
from .encoder_refiner_blocks import (
    FeatureDownsampler72To18,
    LowResRefinerLayer18,
    safe_group_norm,
)
from .guided_refiner_upsampler import GuidedRefinerUpsampler


class LowResGuidedEncoderRefiner(nn.Module):
    """
    Low-resolution (18×18) encoder feature refiner with guided upsampling.

    Flow:
        encoder_features_72 [B, C, D, 72, 72]
        → FeatureDownsampler72To18 → encoder_features_18 [B, C, D, 18, 18]

        → EncoderQueryExtractor → class_query_tokens [B, C, Q, D]
        → SingleTokenClipPromptEncoder → dynamic_clip_text [B, C, Q, D_clip]
        → ClipScoreEmbeddingBuilder18 → clip_score_embed_18, clip_score_maps_18

        → N × LowResRefinerLayer18 (class attention + clip-guided window attention)
        → GuidedRefinerUpsampler (18→36→72, guided by sam_image + CLIP mid layers)
        → refined_encoder_features_72 [B, C, D, 72, 72]
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        clip_native_dim: int = 1024,
        score_embed_dim: int = 128,
        guidance_embed_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        num_query_tokens: int = 32,
        prompt_template: str = "a remote sensing image of {}.",
        normalize_label_for_clip: bool = True,
        score_conv_kernel: int = 7,
        encoder_hw: int = 72,
        refiner_hw: int = 18,
        upsample_mid_hw: int = 36,
        upsample_clip_layer_36: int = 15,
        upsample_clip_layer_72: int = 7,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.use_checkpoint = bool(use_checkpoint)
        self.refiner_hw = int(refiner_hw)
        self.guidance_embed_dim = int(guidance_embed_dim)

        # Downsample: 72×72 → 18×18
        self.downsample_72_to_18 = FeatureDownsampler72To18(hidden_dim=self.hidden_dim)

        # Query extractor (operates on 18×18 features)
        self.query_extractor = EncoderQueryExtractor(
            hidden_dim=self.hidden_dim,
            num_query_tokens=self.num_query_tokens,
            num_heads=int(num_heads),
            dropout=float(dropout),
        )

        # CLIP text prompt encoder
        self.clip_prompt_encoder = SingleTokenClipPromptEncoder(
            clip_text_encoder=clip_text_encoder,
            prompt_template=str(prompt_template),
            sam_dim=self.hidden_dim,
            normalize_label=bool(normalize_label_for_clip),
            use_checkpoint=bool(use_checkpoint),
            num_attention_heads=int(num_heads),
        )

        # CLIP score embedding builder (18×18 only)
        self.score_builder = ClipScoreEmbeddingBuilder18(
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            num_query_tokens=self.num_query_tokens,
            conv_kernel=int(score_conv_kernel),
            base_hw=self.refiner_hw,
        )

        # CLIP final projection for 18×18 guidance
        self.clip_final_proj_18 = nn.Sequential(
            nn.Conv2d(self.clip_dim, self.guidance_embed_dim, kernel_size=1, bias=False),
            safe_group_norm(self.guidance_embed_dim),
            nn.GELU(),
        )

        # Refiner layers (all at 18×18)
        self.layers = nn.ModuleList([
            LowResRefinerLayer18(
                hidden_dim=self.hidden_dim,
                score_embed_dim=int(score_embed_dim),
                guidance_embed_dim=self.guidance_embed_dim,
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
            )
            for _ in range(int(fusion_layers))
        ])

        # Guided upsampler: 18→36→72
        self.upsampler = GuidedRefinerUpsampler(
            hidden_dim=self.hidden_dim,
            clip_native_dim=int(clip_native_dim),
            guidance_embed_dim=self.guidance_embed_dim,
            num_heads=int(num_heads),
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=float(dropout),
            upsample_clip_layer_36=int(upsample_clip_layer_36),
            upsample_clip_layer_72=int(upsample_clip_layer_72),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        clip_mid_features: list[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
        sam_text_mean: torch.Tensor,
        class_names: List[str],
        sam_image_last_72: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            refined_encoder_features_72: [B, C, D, 72, 72]
            class_query_tokens:          [B, C, Q, D]
            dynamic_clip_text:           [B, C, Q, D_clip]
            clip_score_embed_18:         [B, C, D_score, 18, 18]
            clip_score_maps_18:          [B, C, Q, 18, 18]
        """
        B, C, D, H, W = encoder_features_72.shape

        if (H, W) != (72, 72):
            raise ValueError(
                f"LowResGuidedEncoderRefiner expects 72×72 encoder features, "
                f"got {(H, W)}."
            )
        if tuple(sam_text_mean.shape) != (B, C, D):
            raise ValueError(
                f"sam_text_mean must be [{B}, {C}, {D}], "
                f"got {tuple(sam_text_mean.shape)}."
            )
        if len(class_names) != C:
            raise ValueError(
                f"class_names length mismatch: expected {C}, got {len(class_names)}."
            )
        if clip_image_feat_map.ndim != 4:
            raise ValueError(
                f"clip_image_feat_map must be 4D, got {clip_image_feat_map.ndim}D."
            )
        if len(clip_mid_features) != len(clip_mid_layer_indices):
            raise ValueError(
                f"clip_mid_features and clip_mid_layer_indices must have same length, "
                f"got {len(clip_mid_features)} vs {len(clip_mid_layer_indices)}."
            )
        if tuple(sam_image_last_72.shape) != (B, D, 72, 72):
            raise ValueError(
                f"sam_image_last_72 must be [{B}, {D}, 72, 72], "
                f"got {tuple(sam_image_last_72.shape)}."
            )

        # Downsample 72 → 18
        encoder_features_18 = self.downsample_72_to_18(encoder_features_72)

        # Extract class query tokens from 18×18 features
        class_query_tokens = self.query_extractor(encoder_features_18)

        # Build dynamic CLIP text
        dynamic_clip_text = self.clip_prompt_encoder(
            class_query_tokens=class_query_tokens,
            class_names=class_names,
            clip_image_feat_map=clip_image_feat_map,
        )

        # Build 18×18 CLIP score embedding
        clip_score_embed_18, clip_score_maps_18 = self.score_builder(
            dynamic_clip_text=dynamic_clip_text,
            clip_image_feat_map=clip_image_feat_map,
        )

        # Pre-compute CLIP final guidance for 18×18 refiner layers
        clip_final_18 = F.interpolate(
            clip_image_feat_map,
            size=(self.refiner_hw, self.refiner_hw),
            mode="bilinear",
            align_corners=False,
        )
        clip_final_guidance_18 = self.clip_final_proj_18(clip_final_18)

        # Run low-res refiner layers
        x = encoder_features_18
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    layer,
                    x,
                    sam_text_mean,
                    clip_score_embed_18,
                    clip_final_guidance_18,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    encoder_features_18=x,
                    sam_text_mean=sam_text_mean,
                    clip_score_embed_18=clip_score_embed_18,
                    clip_final_guidance_18=clip_final_guidance_18,
                )

        # Guided upsample: 18 → 36 → 72
        refined_encoder_features_72 = self.upsampler(
            refined_features_18=x,
            sam_image_last_72=sam_image_last_72,
            clip_mid_features=clip_mid_features,
            clip_mid_layer_indices=clip_mid_layer_indices,
        )

        return (
            refined_encoder_features_72,
            class_query_tokens,
            dynamic_clip_text,
            clip_score_embed_18,
            clip_score_maps_18,
        )
