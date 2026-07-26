from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model_misc import LayerNorm2d
from .score_embeddings import ClipScoreEmbedding
from .encoder_refiner_attention import (
    EncoderRefinerLayer,
    apply_layer_norm_bcdhw,
)


# ---------------------------------------------------------------------------
# MaskPromptEncoder36
# ---------------------------------------------------------------------------


class MaskPromptEncoder36(nn.Module):
    """Encodes 288×288 raw logits into a 36×36 dense mask prompt embedding.

    Input:  [B, C, 288, 288]
    Output: [B, C, 256, 36, 36]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=2),
            LayerNorm2d(4),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(64, self.hidden_dim, kernel_size=1)

        # Start from a zero mask embedding. The non-zero mask block in the
        # score fusion layer still allows gradients to reach this projection.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, logits_288: torch.Tensor) -> torch.Tensor:
        if logits_288.dim() != 4:
            raise ValueError(
                f"MaskPromptEncoder36 expects [B, C, H, W], got {tuple(logits_288.shape)}."
            )
        B, C, H, W = logits_288.shape
        if H != 288 or W != 288:
            raise ValueError(
                f"MaskPromptEncoder36 expects 288×288 input, got {H}×{W}."
            )

        x = logits_288.reshape(B * C, 1, 288, 288)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.proj(x)

        if tuple(x.shape[-2:]) != (36, 36):
            raise ValueError(
                f"MaskPromptEncoder36 output spatial size mismatch: "
                f"expected 36×36, got {x.shape[-2]}×{x.shape[-1]}."
            )

        return x.reshape(B, C, self.hidden_dim, 36, 36).contiguous()


# ---------------------------------------------------------------------------
# ClassConditionedEncoderRefiner
# ---------------------------------------------------------------------------


class ClassConditionedEncoderRefiner(nn.Module):
    """
    Encoder feature refiner operating at 36×36.

    Takes the full 6-layer SAM3 encoder output after prompt cross-attention
    as input. Original logits from a first-pass Pixel Decoder + Semantic Head
    are encoded via MaskPromptEncoder36 into a mask prompt embedding. The mask
    prompt is fused into the score stream via channel concat followed by a
    1×1 convolution, instead of being added to the feature stream.

    A shared 1×1 Conv produces per-layer score-stream auxiliary logits at
    36×36 for distillation supervision.

    Forward inputs:
        encoder_features_72:  [B, C, 256, 72, 72]  (full encoder + cross-attention)
        clip_image_feat_map:  [B, D_clip, 36, 36]
        sam_text_mean:        [B, C, 256]
        class_names:          list of C class names
        original_logits_288:  [B, C, 288, 288]

    Forward outputs:
        refined_encoder_features_72: [B, C, 256, 72, 72]
        refiner_features_36:         [B, C, 256, 36, 36]
        score_embed_36:              [B, C, 256, 36, 36]
        clip_score_embed_36:         [B, C, 256, 36, 36]
        clip_score_maps_36:          [B, C,  32, 36, 36]
        template_clip_text:          [C, 32, D_clip]
        refiner_aux_logits_36:       [L, B, C, 36, 36]
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

        self.mask_prompt_encoder = MaskPromptEncoder36(
            hidden_dim=self.hidden_dim,
        )

        if self.hidden_dim != self.score_embed_dim:
            raise ValueError(
                "Mask-score fusion requires hidden_dim and score_embed_dim "
                f"to be equal, got {self.hidden_dim} and {self.score_embed_dim}."
            )

        self.mask_score_fusion_36 = nn.Conv2d(
            self.score_embed_dim + self.hidden_dim,
            self.score_embed_dim,
            kernel_size=1,
            bias=True,
        )

        self._init_mask_score_fusion_36()

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

        self.score_aux_mask_head = nn.Conv2d(
            self.score_embed_dim,
            1,
            kernel_size=1,
        )

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
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_mask_score_fusion_36(self) -> None:
        fusion = self.mask_score_fusion_36

        with torch.no_grad():
            fusion.weight.zero_()
            fusion.bias.zero_()

            identity = torch.eye(
                self.score_embed_dim,
                device=fusion.weight.device,
                dtype=fusion.weight.dtype,
            )

            fusion.weight[
                :, :self.score_embed_dim, 0, 0
            ].copy_(identity)

            fusion.weight[
                :, self.score_embed_dim:, 0, 0
            ].copy_(0.1 * identity)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fuse_mask_prompt_into_score_36(
        self,
        clip_score_embed_36: torch.Tensor,
        mask_prompt_embed_36: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, score_dim, height, width = (
            clip_score_embed_36.shape
        )

        expected_score_shape = (
            batch_size,
            num_classes,
            self.score_embed_dim,
            36,
            36,
        )
        if tuple(clip_score_embed_36.shape) != expected_score_shape:
            raise ValueError(
                f"clip_score_embed_36 must be {expected_score_shape}, "
                f"got {tuple(clip_score_embed_36.shape)}."
            )

        expected_mask_shape = (
            batch_size,
            num_classes,
            self.hidden_dim,
            36,
            36,
        )
        if tuple(mask_prompt_embed_36.shape) != expected_mask_shape:
            raise ValueError(
                f"mask_prompt_embed_36 must be {expected_mask_shape}, "
                f"got {tuple(mask_prompt_embed_36.shape)}."
            )

        fusion_input = torch.cat(
            [clip_score_embed_36, mask_prompt_embed_36],
            dim=2,
        ).reshape(
            batch_size * num_classes,
            self.score_embed_dim + self.hidden_dim,
            36,
            36,
        )

        fused_score_embed_36 = self.mask_score_fusion_36(fusion_input)

        return fused_score_embed_36.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            36,
            36,
        ).contiguous()

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
        original_logits_288: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features_72:  [B, C, 256, 72, 72]
                Full 6-layer encoder output after prompt cross-attention.
            clip_image_feat_map:  [B, D_clip, 36, 36]
            sam_text_mean:        [B, C, 256]
            class_names:          list of C class names
            original_logits_288:  [B, C, 288, 288]

        Returns dict with keys:
            refined_encoder_features_72
            refiner_features_36
            score_embed_36
            clip_score_embed_36
            clip_score_maps_36
            template_clip_text
            refiner_aux_logits_36

        Process:
            1. Build CLIP score embedding at 36×36.
            2. Downsample encoder features from 72×72 to 36×36 as the feature stream.
            3. Encode original logits into mask prompt embedding.
            4. Fuse score embedding and mask prompt embedding via channel concat
               followed by 1×1 convolution to produce the initial score stream.
            5. Run refiner layers and produce per-layer score-stream auxiliary logits
               via the shared score_aux_mask_head.
            6. Bilinearly upsample the refiner feature to 72×72, concatenate it
               with the original cross-attended encoder feature, and fuse with
               1×1 + 3×3 convolutions.
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

        if tuple(original_logits_288.shape) != (batch_size, num_classes, 288, 288):
            raise ValueError(
                f"original_logits_288 must be [{batch_size}, {num_classes}, 288, 288], "
                f"got {tuple(original_logits_288.shape)}."
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

        # 3. Encode original logits into mask prompt embedding.
        mask_prompt_logits = original_logits_288.detach().clamp(-32.0, 32.0)
        mask_prompt_embed_36 = self.mask_prompt_encoder(mask_prompt_logits)

        # 4. Fuse score embedding and mask prompt embedding to produce the
        # initial score stream.
        score_embed_36 = self._fuse_mask_prompt_into_score_36(
            clip_score_embed_36=clip_score_embed_36,
            mask_prompt_embed_36=mask_prompt_embed_36,
        )

        # 5. Run refiner layers and collect per-layer aux logits (training only).
        aux_logits_per_layer: list[torch.Tensor] = []

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

            if self.training:
                aux_logit = self.score_aux_mask_head(
                    score_embed_36.reshape(
                        batch_size * num_classes,
                        self.score_embed_dim,
                        36,
                        36,
                    )
                ).reshape(
                    batch_size,
                    num_classes,
                    36,
                    36,
                )
                aux_logits_per_layer.append(aux_logit)

        refiner_aux_logits_36 = torch.stack(
            aux_logits_per_layer,
            dim=0,
        ) if aux_logits_per_layer else torch.empty(
            0, batch_size, num_classes, 36, 36,
            device=feature_36.device,
            dtype=feature_36.dtype,
        )

        # Final LayerNorm for score_embed only.
        score_embed_36 = apply_layer_norm_bcdhw(
            score_embed_36,
            self.final_score_norm,
        )

        # 6. Upsample the refiner feature and directly fuse it with the
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
            "refiner_aux_logits_36": refiner_aux_logits_36,
        }
