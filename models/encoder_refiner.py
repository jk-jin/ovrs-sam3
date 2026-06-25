from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .clip_score_embedding import ClipScoreEmbeddingBuilder
from .encoder_refiner_attention import EncoderRefinerLayer


class ClassConditionedEncoderRefiner(nn.Module):
    """
    多层 encoder 特征精炼器，使用多尺度空间窗口注意力。

    输入 (forward)：
        encoder_features:       [B, C, D, H, W]
        clip_image_feat_map:    [B, D_clip, Hc, Wc]
        clip_mid_features:      CLIP ViT 中间层特征列表
        clip_mid_layer_indices: 中间层编号
        sam_text_mean:          [B, C, D]
        class_names:            C 个类别名的列表
        sam_image_last:         [B, D, H, W]

    输出 (forward)：
        refined_encoder_features_72: [B, C, D, 72, 72]
        template_clip_text:          [C, 32, D_clip]
        clip_score_embeds:           {"scale_18": ..., "scale_36": ..., "scale_72": ...}
        clip_score_maps_18:          [B, C, 32, 18, 18]

    设计：
        32 个固定 prompt 模板 → frozen OpenCLIP text encoder → template_clip_text
        → × clip_image_feat_map → score_maps_18
        → CLIP mid-layer15 融合 → score_embed_36
        → CLIP mid-layer7  融合 → score_embed_72
        → class attention + window attention (FPN 融合上采样)
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        clip_native_dim: int = 1024,
        score_embed_dim: int = 64,
        num_heads: int = 8,
        window_size: int = 9,
        shift_size: int = 4,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        score_conv_kernel: int = 7,
        score_base_hw: int = 18,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
        self.use_checkpoint = bool(use_checkpoint)

        if prompt_templates is None:
            raise ValueError("prompt_templates must be a list of 32 prompt templates.")
        if len(prompt_templates) != 32:
            raise ValueError(f"Expected 32 prompt templates, got {len(prompt_templates)}.")

        self.score_builder = ClipScoreEmbeddingBuilder(
            clip_text_encoder=clip_text_encoder,
            prompt_templates=list(prompt_templates),
            normalize_label=bool(normalize_label_for_clip),
            clip_output_dim=int(clip_dim),
            clip_native_dim=int(clip_native_dim),
            score_embed_dim=int(score_embed_dim),
            conv_kernel=int(score_conv_kernel),
            base_hw=int(score_base_hw),
            clip_mid_proj_dim=32,
            clip_mid_layer_for_36=15,
            clip_mid_layer_for_72=7,
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
        clip_mid_features: List[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
        sam_text_mean: torch.Tensor,
        class_names: List[str],
        sam_image_last: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            refined_encoder_features_72: [B, C, D, 72, 72]
            template_clip_text:          [C, 32, D_clip]
            clip_score_embeds:           dict of 3-scale score embeddings
            clip_score_maps_18:          [B, C, 32, 18, 18]
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

        clip_score_embeds, clip_score_maps_18, template_clip_text = self.score_builder(
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
            template_clip_text,
            clip_score_embeds,
            clip_score_maps_18,
        )
