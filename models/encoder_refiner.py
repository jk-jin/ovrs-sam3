from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .score_embeddings import ClipScoreEmbedding
from .encoder_refiner_attention import (
    EncoderRefinerLayer,
    apply_layer_norm_bcdhw,
    make_residual_scale,
)


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


# ---------------------------------------------------------------------------
# ClassConditionedEncoderRefiner
# ---------------------------------------------------------------------------


class ClassConditionedEncoderRefiner(nn.Module):
    """
    Encoder feature refiner operating at 36×36.

    Takes the full 6-layer SAM3 encoder output after prompt cross-attention
    as input. SAM3 FPN is concatenated with downsampled encoder feature at
    36×36, then convolved to produce a residual delta with learnable scale.
    SAM text prompt tokens are reduced to a masked mean before entering
    the refiner. The refiner receives sam_text_mean directly.

    Forward inputs:
        encoder_features_72:  [B, C, 256, 72, 72]  (full encoder + cross-attention)
        clip_image_feat_map:  [B, D_clip, 36, 36]
        sam_text_mean:        [B, C, 256]
        class_names:          list of C class names
        sam_fpn_72:           [B, 256, 72, 72]

    Forward outputs:
        refined_encoder_features_72: [B, C, 256, 72, 72]
        refiner_features_36:         [B, C, 256, 36, 36]
        score_embed_36:              [B, C, 256, 36, 36]
        clip_score_embed_36:         [B, C, 256, 36, 36]
        clip_score_maps_36:          [B, C,  32, 36, 36]
        template_clip_text:          [C, 32, D_clip]
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

        # FPN-to-feature convolution fusion: concatenate
        # [encoder_feature_36, sam_fpn_36] and produce a residual delta.
        self.feature_fpn_fusion = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim * 2,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.hidden_dim),
            nn.GELU(),
        )
        self.feature_fpn_res_scale = make_residual_scale(
            residual_scale_init
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
            for _ in range(int(fusion_layers))
        ])

        self.encoder_feature_fusion_72 = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim * 2,
                self.hidden_dim,
                kernel_size=1,
                bias=True,
            ),
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        self.final_score_norm = nn.LayerNorm(self.score_embed_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fuse_refiner_and_encoder_features_72(
        self,
        feature_36: torch.Tensor,
        encoder_features_72: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, hidden_dim, height, width = (
            encoder_features_72.shape
        )

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected encoder hidden_dim={self.hidden_dim}, "
                f"got {hidden_dim}."
            )
        if (height, width) != (72, 72):
            raise ValueError(
                "Expected encoder_features_72 spatial size 72×72, "
                f"got {(height, width)}."
            )

        expected_refiner_shape = (
            batch_size,
            num_classes,
            hidden_dim,
            36,
            36,
        )
        if tuple(feature_36.shape) != expected_refiner_shape:
            raise ValueError(
                f"feature_36 must be {expected_refiner_shape}, "
                f"got {tuple(feature_36.shape)}."
            )

        refiner_features_72 = F.interpolate(
            feature_36.reshape(
                batch_size * num_classes,
                hidden_dim,
                36,
                36,
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        encoder_features_72_flat = encoder_features_72.reshape(
            batch_size * num_classes,
            hidden_dim,
            height,
            width,
        )

        fusion_input_72 = torch.cat(
            [refiner_features_72, encoder_features_72_flat],
            dim=1,
        )
        fused_encoder_features_72 = self.encoder_feature_fusion_72(
            fusion_input_72
        )

        return fused_encoder_features_72.reshape(
            batch_size,
            num_classes,
            hidden_dim,
            height,
            width,
        ).contiguous()

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        sam_text_mean: torch.Tensor,
        class_names: List[str],
        sam_fpn_72: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features_72:  [B, C, 256, 72, 72]
                Full 6-layer encoder output after prompt cross-attention.
            clip_image_feat_map:  [B, D_clip, 36, 36]
            sam_text_mean:        [B, C, 256]
            class_names:          list of C class names
            sam_fpn_72:           [B, 256, 72, 72]

        Returns dict with keys:
            refined_encoder_features_72
            refiner_features_36
            score_embed_36
            clip_score_embed_36
            clip_score_maps_36
            template_clip_text

        Process:
            1. Build CLIP score embedding at 36×36.
            2. Downsample cross-attended full-encoder features from 72×72 to base_feature_36.
            3. Inject SAM3 FPN into encoder feature via conv residual with learnable scale.
            4. Run refiner layers.
            5. Bilinearly upsample the full refiner feature to 72×72,
               concatenate it with the original cross-attended encoder feature, and fuse
               them with one 1×1 convolution followed by one 3×3 convolution.
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

        # score_embed_36 starts clean — FPN no longer injects into it.
        score_embed_36 = clip_score_embed_36

        # 2. Downsample encoder features from 72×72 to 36×36 (baseline).
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

        # 3. Inject SAM3 FPN into encoder feature via conv residual.
        if tuple(sam_fpn_72.shape) != (batch_size, hidden_dim, 72, 72):
            raise ValueError(
                f"sam_fpn_72 must be [{batch_size}, {hidden_dim}, 72, 72], "
                f"got {tuple(sam_fpn_72.shape)}."
            )

        base_feature_flat = base_feature_36.reshape(
            batch_size * num_classes,
            hidden_dim,
            36,
            36,
        )

        sam_fpn_36 = F.interpolate(
            sam_fpn_72.to(
                device=base_feature_flat.device,
                dtype=base_feature_flat.dtype,
            ),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )

        sam_fpn_36 = (
            sam_fpn_36
            .unsqueeze(1)
            .expand(
                batch_size,
                num_classes,
                hidden_dim,
                36,
                36,
            )
            .reshape(
                batch_size * num_classes,
                hidden_dim,
                36,
                36,
            )
        )

        feature_fpn_input = torch.cat(
            [base_feature_flat, sam_fpn_36],
            dim=1,
        )

        feature_fpn_delta_36 = self.feature_fpn_fusion(
            feature_fpn_input
        ).reshape(
            batch_size,
            num_classes,
            hidden_dim,
            36,
            36,
        )

        feature_36 = (
            base_feature_36
            + self.feature_fpn_res_scale * feature_fpn_delta_36
        )

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

        # Final LayerNorm for score_embed only.
        score_embed_36 = apply_layer_norm_bcdhw(
            score_embed_36,
            self.final_score_norm,
        )

        # 5. Upsample the refiner feature and directly fuse it with the
        # original encoder feature at 72×72.
        if self.use_checkpoint and self.training:
            refined_encoder_features_72 = checkpoint(
                self._fuse_refiner_and_encoder_features_72,
                feature_36,
                encoder_features_72,
                use_reentrant=False,
            )
        else:
            refined_encoder_features_72 = (
                self._fuse_refiner_and_encoder_features_72(
                    feature_36=feature_36,
                    encoder_features_72=encoder_features_72,
                )
            )

        return {
            "refined_encoder_features_72": refined_encoder_features_72,
            "refiner_features_36": feature_36,
            "score_embed_36": score_embed_36,
            "clip_score_embed_36": clip_score_embed_36,
            "clip_score_maps_36": clip_score_maps_36,
            "template_clip_text": template_clip_text,
        }
