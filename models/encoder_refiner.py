from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .score_embeddings import (
    ClipScoreEmbedding,
    CombinedScoreEmbeddingBuilder,
    SamMaskScoreEmbedding,
)
from .encoder_refiner_attention import EncoderRefinerLayer


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


# ---------------------------------------------------------------------------
# EncoderFeatureUpsampler
# ---------------------------------------------------------------------------


class EncoderFeatureUpsampler(nn.Module):
    """
    Upsample refined 36×36 feature back to 72×72 with simple bilinear
    interpolation and 1×1 channel fusion.

    Input:
        refined_feature_36:  [B, C, 256, 36, 36]
        sam_fpn_72:          [B, 256, 72, 72]  # kept for API compatibility, unused here
        original_encoder_72: [B, C, 256, 72, 72]

    Process:
        1. Bilinear interpolate refined_feature_36: 36 → 72
        2. Concatenate with original_encoder_72 along channel dim
        3. 1×1 conv: 512 → 256

    Output:
        refined_feature_72: [B, C, 256, 72, 72]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.fuse = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
            bias=False,
        )

        self._init_original_passthrough()

    def _init_original_passthrough(self) -> None:
        """
        Initialize the 1×1 fusion conv so that the module initially behaves
        like original_encoder_72 passthrough.

        Cat order is [upsampled_refined, original_encoder].
        Therefore the second half of input channels is initialized as identity.
        """
        with torch.no_grad():
            self.fuse.weight.zero_()
            for i in range(self.hidden_dim):
                self.fuse.weight[i, self.hidden_dim + i, 0, 0] = 1.0

    def forward(
        self,
        refined_feature_36: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        original_encoder_72: torch.Tensor,
    ) -> torch.Tensor:
        del sam_fpn_72

        B, C, D, H36, W36 = refined_feature_36.shape

        if D != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got D={D}."
            )
        if (H36, W36) != (36, 36):
            raise ValueError(
                f"Expected 36×36 refined feature, got {(H36, W36)}."
            )
        if tuple(original_encoder_72.shape) != (B, C, D, 72, 72):
            raise ValueError(
                f"original_encoder_72 must be [{B}, {C}, {D}, 72, 72], "
                f"got {tuple(original_encoder_72.shape)}."
            )

        up = F.interpolate(
            refined_feature_36.reshape(B * C, D, 36, 36),
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        )

        orig = original_encoder_72.reshape(B * C, D, 72, 72)

        fused = torch.cat([up, orig], dim=1)
        out = self.fuse(fused)

        return out.reshape(B, C, D, 72, 72).contiguous()


# ---------------------------------------------------------------------------
# ClassConditionedEncoderRefiner
# ---------------------------------------------------------------------------


class ClassConditionedEncoderRefiner(nn.Module):
    """
    Encoder feature refiner operating at 36×36 with combined CLIP + SAM score embedding.

    Forward inputs:
        encoder_features_72:  [B, C, 256, 72, 72]
        clip_image_feat_map:  [B, D_clip, 36, 36]
        sam_text_mean:        [B, C, 256]
        class_names:          list of C class names
        sam_prior_logits:     [B, C, H_mask, W_mask]
        sam_fpn_72:           [B, 256, 72, 72]

    Forward outputs:
        refined_encoder_features_72: [B, C, 256, 72, 72]
        score_embed_36:              [B, C, 256, 36, 36]
        clip_score_embed_36:         [B, C, 192, 36, 36]
        sam_score_embed_36:          [B, C,  64, 36, 36]
        clip_score_maps_36:          [B, C,  32, 36, 36]
        template_clip_text:          [C, 32, D_clip]
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        score_embed_dim: int = 256,
        clip_score_embed_dim: int = 192,
        sam_score_embed_dim: int = 64,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 6,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        clip_score_conv_kernel: int = 7,
        use_checkpoint: bool = True,
        feature_residual_scale: float = 1e-3,
        score_residual_scale: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.use_checkpoint = bool(use_checkpoint)

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
            score_embed_dim=int(clip_score_embed_dim),
            conv_kernel=int(clip_score_conv_kernel),
        )

        self.sam_score_embed = SamMaskScoreEmbedding(
            out_dim=int(sam_score_embed_dim),
        )

        self.score_fusion = CombinedScoreEmbeddingBuilder(
            clip_dim=int(clip_score_embed_dim),
            sam_dim=int(sam_score_embed_dim),
            fused_dim=int(score_embed_dim),
        )

        self.layers = nn.ModuleList([
            EncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                score_embed_dim=self.score_embed_dim,
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
                feature_residual_scale=float(feature_residual_scale),
                score_residual_scale=float(score_residual_scale),
            )
            for _ in range(int(fusion_layers))
        ])

        self.upsampler = EncoderFeatureUpsampler(
            hidden_dim=self.hidden_dim,
        )

        self.clip_score_embed_dim = int(clip_score_embed_dim)
        self.sam_score_embed_dim = int(sam_score_embed_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        sam_text_mean: torch.Tensor,
        class_names: List[str],
        sam_score_embed_36: torch.Tensor,
        sam_fpn_72: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features_72:  [B, C, 256, 72, 72]
            clip_image_feat_map:  [B, D_clip, 36, 36]
            sam_text_mean:        [B, C, 256]
            class_names:          list of C class names
            sam_score_embed_36:   [B, C, 64, 36, 36]  (pre-downsampled)
            sam_fpn_72:           [B, 256, 72, 72]

        Returns dict with keys:
            refined_encoder_features_72
            refiner_features_36
            score_embed_36
            clip_score_embed_36
            sam_score_embed_36
            clip_score_maps_36
            template_clip_text
        """
        batch_size, num_classes, hidden_dim, H, W = encoder_features_72.shape

        if (H, W) != (72, 72):
            raise ValueError(
                f"ClassConditionedEncoderRefiner expects 72×72 encoder features, "
                f"got {(H, W)}."
            )
        if tuple(sam_text_mean.shape) != (batch_size, num_classes, hidden_dim):
            raise ValueError(
                f"sam_text_mean must be [{batch_size}, {num_classes}, {hidden_dim}], "
                f"got {tuple(sam_text_mean.shape)}."
            )
        if tuple(clip_image_feat_map.shape[-2:]) != (36, 36):
            raise ValueError(
                f"clip_image_feat_map must be 36×36, "
                f"got {tuple(clip_image_feat_map.shape[-2:])}."
            )
        if tuple(sam_score_embed_36.shape) != (
            batch_size,
            num_classes,
            self.sam_score_embed_dim,
            36,
            36,
        ):
            raise ValueError(
                f"sam_score_embed_36 must be "
                f"[{batch_size}, {num_classes}, {self.sam_score_embed_dim}, 36, 36], "
                f"got {tuple(sam_score_embed_36.shape)}."
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

        # 2. Fuse CLIP and SAM score embeddings into combined score_embed_36.
        score_embed_36 = self.score_fusion(
            clip_score_embed_36=clip_score_embed_36,
            sam_score_embed_36=sam_score_embed_36,
        )

        # 4. Downsample encoder features from 72×72 to 36×36.
        feature_36 = F.interpolate(
            encoder_features_72.reshape(
                batch_size * num_classes, hidden_dim, 72, 72
            ),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_classes, hidden_dim, 36, 36)

        # Downsample SAM3 FPN feature from 72×72 to 36×36 for window attention Q/K.
        if sam_fpn_72.ndim != 4:
            raise ValueError(
                f"sam_fpn_72 must be [B, D, 72, 72], got {tuple(sam_fpn_72.shape)}."
            )
        if tuple(sam_fpn_72.shape) != (batch_size, hidden_dim, 72, 72):
            raise ValueError(
                f"sam_fpn_72 must be [{batch_size}, {hidden_dim}, 72, 72], "
                f"got {tuple(sam_fpn_72.shape)}."
            )

        sam_fpn_36 = F.interpolate(
            sam_fpn_72.to(device=feature_36.device, dtype=feature_36.dtype),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        ).contiguous()

        # 5. Run refiner layers.
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                feature_36, score_embed_36 = checkpoint(
                    layer,
                    feature_36,
                    score_embed_36,
                    sam_text_mean,
                    sam_fpn_36,
                    use_reentrant=False,
                )
            else:
                feature_36, score_embed_36 = layer(
                    feature_36=feature_36,
                    score_embed_36=score_embed_36,
                    sam_text_mean=sam_text_mean,
                    sam_fpn_36=sam_fpn_36,
                )

        # 6. Upsample refined feature back to 72×72.
        refined_encoder_features_72 = self.upsampler(
            refined_feature_36=feature_36,
            sam_fpn_72=sam_fpn_72,
            original_encoder_72=encoder_features_72,
        )

        return {
            "refined_encoder_features_72": refined_encoder_features_72,
            "refiner_features_36": feature_36,
            "score_embed_36": score_embed_36,
            "clip_score_embed_36": clip_score_embed_36,
            "sam_score_embed_36": sam_score_embed_36,
            "clip_score_maps_36": clip_score_maps_36,
            "template_clip_text": template_clip_text,
        }
