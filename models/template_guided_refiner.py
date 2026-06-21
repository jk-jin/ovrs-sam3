from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .guidance_upsampler import ClipGuidanceUpsampler
from .highres_encoder_refiner import HighResEncoderRefiner
from .lowres_score_refiner import LowResScoreRefiner
from .template_score_builder import TemplateScoreBuilder


class TemplateScoreGuidedRefiner(nn.Module):
    """
    Top-level refiner that orchestrates template-score-guided encoder refinement.

    Pipeline:
        class_names + clip_final_map
            → TemplateScoreBuilder
            → template_score_maps_18, lowres_score_embed

        lowres_score_embed + clip_final_feat_18 + sam_fpn_feat_18 + sam_text_mean
            → LowResScoreRefiner (4 layers)
            → refined_score_embed_18

        refined_score_embed_18 + sam_fpn_feat_36/72 + clip_mid_features
            → ClipGuidanceUpsampler
            → clip_guidance_36, clip_guidance_72

        encoder_features_72 + clip_guidance_72
            → HighResEncoderRefiner (2 layers)
            → refined_encoder_features_72
    """

    def __init__(
        self,
        clip_text_encoder: nn.Module,
        clip_image_native_dim: int,
        prompt_templates: List[str],
        normalize_label_for_clip: bool = True,
        hidden_dim: int = 256,
        num_prompt_templates: int = 32,
        lowres_hw: int = 18,
        lowres_layers: int = 4,
        highres_hw: int = 72,
        highres_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_size: int = 9,
        shift_size: int = 4,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lowres_hw = int(lowres_hw)
        self.highres_hw = int(highres_hw)

        if self.lowres_hw != 18:
            raise ValueError(f"lowres_hw must be 18, got {lowres_hw}")
        if self.highres_hw != 72:
            raise ValueError(f"highres_hw must be 72, got {highres_hw}")

        self.template_score_builder = TemplateScoreBuilder(
            clip_text_encoder=clip_text_encoder,
            prompt_templates=list(prompt_templates),
            normalize_label_for_clip=bool(normalize_label_for_clip),
            hidden_dim=self.hidden_dim,
            num_prompt_templates=int(num_prompt_templates),
        )

        # Project CLIP final map from D_clip → hidden_dim for use in lowres refiner.
        self.clip_final_proj = nn.Sequential(
            nn.Conv2d(self._infer_clip_dim(clip_text_encoder), self.hidden_dim, kernel_size=1),
            nn.GroupNorm(min(8, self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )

        # Project SAM3 FPN features to hidden_dim for lowres refiner.
        self.sam18_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.GroupNorm(min(8, self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )

        self.lowres_score_refiner = LowResScoreRefiner(
            hidden_dim=self.hidden_dim,
            num_heads=int(num_heads),
            window_size=int(window_size),
            shift_size=int(shift_size),
            lowres_layers=int(lowres_layers),
            dropout=float(dropout),
            use_checkpoint=bool(use_checkpoint),
        )

        self.guidance_upsampler = ClipGuidanceUpsampler(
            hidden_dim=self.hidden_dim,
            clip_native_dim=int(clip_image_native_dim),
            sam_fpn_dim=self.hidden_dim,
        )

        self.highres_encoder_refiner = HighResEncoderRefiner(
            hidden_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            highres_layers=int(highres_layers),
            use_checkpoint=bool(use_checkpoint),
        )

    @staticmethod
    def _infer_clip_dim(clip_text_encoder: nn.Module) -> int:
        output_dim = getattr(clip_text_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim
        raise AttributeError("clip_text_encoder must expose output_dim.")

    @staticmethod
    def select_feature_by_hw(
        features: List[torch.Tensor],
        target_hw: tuple[int, int],
        fallback_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Select a feature map by spatial size, with optional fallback."""
        for feat in features:
            if tuple(feat.shape[-2:]) == target_hw:
                return feat

        if fallback_hw is not None:
            for feat in features:
                if tuple(feat.shape[-2:]) == fallback_hw:
                    return F.interpolate(
                        feat, size=target_hw,
                        mode="bilinear", align_corners=False,
                    )

        best = min(
            features,
            key=lambda f: abs(f.shape[-2] * f.shape[-1] - target_hw[0] * target_hw[1]),
        )
        return F.interpolate(
            best, size=target_hw,
            mode="bilinear", align_corners=False,
        )

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        sam_text_mean: torch.Tensor,
        class_names: List[str],
        clip_final_map: torch.Tensor,
        clip_mid_features: List[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
        sam_fpn_features: List[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features_72:   [B, C, D, 72, 72]
            sam_text_mean:         [B, C, D]
            class_names:           list of C strings
            clip_final_map:        [B, D_clip, Hc, Wc]
            clip_mid_features:     list of [B, D_native, Hc, Wc]
            clip_mid_layer_indices: tuple of layer indices
            sam_fpn_features:      list of SAM3 FPN feature maps

        Returns:
            dict with:
                refined_encoder_features_72: [B, C, D, 72, 72]
                template_score_maps_18:      [B, C, K, 18, 18]
                lowres_score_embed:          [B, C, D, 18, 18]
                refined_score_embed_18:      [B, C, D, 18, 18]
                clip_guidance_36:            [B, C, D, 36, 36]
                clip_guidance_72:            [B, C, D, 72, 72]
        """
        B, C, D, H, W = encoder_features_72.shape

        if (H, W) != (self.highres_hw, self.highres_hw):
            raise ValueError(
                f"encoder_features_72 must be {self.highres_hw}x{self.highres_hw}, "
                f"got {H}x{W}."
            )

        # ---- 1. TemplateScoreBuilder ----
        template_score_maps_18, lowres_score_embed, _template_text_features = (
            self.template_score_builder(
                class_names=class_names,
                clip_final_map=clip_final_map,
            )
        )

        # ---- 2. Prepare features for LowResScoreRefiner ----
        # CLIP final map → 18x18, projected to hidden_dim.
        clip_final_feat_18 = F.interpolate(
            clip_final_map,
            size=(self.lowres_hw, self.lowres_hw),
            mode="bilinear",
            align_corners=False,
        )
        clip_final_feat_18 = self.clip_final_proj(clip_final_feat_18)  # [B, D, 18, 18]

        # SAM3 FPN → 18x18 feature.
        sam_fpn_feat_18 = self.select_feature_by_hw(
            sam_fpn_features, (18, 18), fallback_hw=(72, 72),
        )
        sam_fpn_feat_18 = self.sam18_proj(sam_fpn_feat_18.to(dtype=lowres_score_embed.dtype))

        # ---- 3. LowResScoreRefiner ----
        refined_score_embed_18 = self.lowres_score_refiner(
            lowres_score_embed=lowres_score_embed,
            sam_text_mean=sam_text_mean,
            clip_final_feat_18=clip_final_feat_18,
            sam_fpn_feat_18=sam_fpn_feat_18,
        )

        # ---- 4. ClipGuidanceUpsampler ----
        sam_fpn_feat_36 = self.select_feature_by_hw(
            sam_fpn_features, (36, 36), fallback_hw=(72, 72),
        )
        sam_fpn_feat_72 = self.select_feature_by_hw(
            sam_fpn_features, (72, 72),
        )

        clip_guidance_36, clip_guidance_72 = self.guidance_upsampler(
            refined_score_embed_18=refined_score_embed_18,
            sam_fpn_feat_36=sam_fpn_feat_36,
            sam_fpn_feat_72=sam_fpn_feat_72,
            clip_mid_features=clip_mid_features,
            clip_mid_layer_indices=clip_mid_layer_indices,
        )

        # ---- 5. HighResEncoderRefiner ----
        refined_encoder_features_72 = self.highres_encoder_refiner(
            encoder_features_72=encoder_features_72,
            clip_guidance_72=clip_guidance_72,
        )

        return {
            "refined_encoder_features_72": refined_encoder_features_72,
            "template_score_maps_18": template_score_maps_18,
            "lowres_score_embed": lowres_score_embed,
            "refined_score_embed_18": refined_score_embed_18,
            "clip_guidance_36": clip_guidance_36,
            "clip_guidance_72": clip_guidance_72,
        }
