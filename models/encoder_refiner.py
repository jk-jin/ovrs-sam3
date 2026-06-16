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
        class_names:        list of C strings
        sam_image_last:     [B, D, H, W]

    Output (forward):
        refined_encoder_features: [B, C, D, 72, 72]
        class_query_tokens:       [B, C, Q, D]
        dynamic_clip_text:        [B, C, Q, D_clip]
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
        clip_mid_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.use_checkpoint = bool(use_checkpoint)

        # Centralized input normalization before all refiner attention layers.
        self.sam_image_input_norm = nn.LayerNorm(self.hidden_dim)
        self.sam_image_input_scale = nn.Parameter(torch.tensor(1.0))

        self.clip_score_input_norm_18 = nn.LayerNorm(self.score_embed_dim)
        self.clip_score_input_norm_36 = nn.LayerNorm(self.score_embed_dim)
        self.clip_score_input_norm_72 = nn.LayerNorm(self.score_embed_dim)

        self.clip_score_input_scale_18 = nn.Parameter(torch.tensor(2.0))
        self.clip_score_input_scale_36 = nn.Parameter(torch.tensor(2.0))
        self.clip_score_input_scale_72 = nn.Parameter(torch.tensor(2.0))

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

        self.score_builder = ClipScoreEmbeddingBuilder(
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            num_query_tokens=self.num_query_tokens,
            conv_kernel=int(score_conv_kernel),
            base_hw=int(score_base_hw),
            clip_mid_dim=clip_mid_dim,
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
    # Centralized input normalization
    # ------------------------------------------------------------------

    def _normalize_sam_image_last(
        self,
        sam_image_last: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize SAM image feature before all refiner attention layers.

        LayerNorm over the channel dim: [B, D, H, W] → [B, H, W, D] → norm → back.
        """
        if sam_image_last.ndim != 4:
            raise ValueError(
                f"sam_image_last must be [B, D, H, W], got {tuple(sam_image_last.shape)}."
            )
        if int(sam_image_last.shape[1]) != int(self.hidden_dim):
            raise ValueError(
                f"sam_image_last channel mismatch: "
                f"expected {self.hidden_dim}, got {sam_image_last.shape[1]}."
            )

        x = sam_image_last.permute(0, 2, 3, 1).contiguous()
        x = self.sam_image_input_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return self.sam_image_input_scale * x

    def _normalize_clip_score_embed(
        self,
        score_embed: torch.Tensor,
        norm: nn.LayerNorm,
        scale: torch.Tensor,
        expected_hw: tuple[int, int],
        name: str,
    ) -> torch.Tensor:
        """Normalize CLIP score embedding before all refiner attention layers.

        LayerNorm over the score_embed_dim: [B, C, D_score, H, W] → permute → norm → back.
        """
        if score_embed.ndim != 5:
            raise ValueError(
                f"{name} must be [B, C, D_score, H, W], got {tuple(score_embed.shape)}."
            )
        if int(score_embed.shape[2]) != int(norm.normalized_shape[0]):
            raise ValueError(
                f"{name} channel mismatch: "
                f"expected {norm.normalized_shape[0]}, got {score_embed.shape[2]}."
            )
        if tuple(score_embed.shape[-2:]) != tuple(expected_hw):
            raise ValueError(
                f"{name} spatial size mismatch: "
                f"expected {expected_hw}, got {tuple(score_embed.shape[-2:])}."
            )

        x = score_embed.permute(0, 1, 3, 4, 2).contiguous()
        x = norm(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        return scale * x

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
        clip_mid_features: List[torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            refined_encoder_features_72: [B, C, D, 72, 72]
            class_query_tokens:          [B, C, Q, D]
            dynamic_clip_text:           [B, C, Q, D_clip]
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

        class_query_tokens = self.query_extractor(encoder_features)

        dynamic_clip_text = self.clip_prompt_encoder(
            class_query_tokens=class_query_tokens,
            class_names=class_names,
            clip_image_feat_map=clip_image_feat_map,
        )

        clip_score_embeds, clip_score_maps_18 = self.score_builder(
            dynamic_clip_text=dynamic_clip_text,
            clip_image_feat_map=clip_image_feat_map,
            clip_mid_features=clip_mid_features,
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
        # ------------------------------------------------------------------
        # Centralized input normalization before all refiner attention layers.
        # ------------------------------------------------------------------

        sam_image_last = self._normalize_sam_image_last(sam_image_last)

        clip_score_embed_18 = self._normalize_clip_score_embed(
            score_embed=clip_score_embeds["scale_18"],
            norm=self.clip_score_input_norm_18,
            scale=self.clip_score_input_scale_18,
            expected_hw=(18, 18),
            name="clip_score_embeds['scale_18']",
        )
        clip_score_embed_36 = self._normalize_clip_score_embed(
            score_embed=clip_score_embeds["scale_36"],
            norm=self.clip_score_input_norm_36,
            scale=self.clip_score_input_scale_36,
            expected_hw=(36, 36),
            name="clip_score_embeds['scale_36']",
        )
        clip_score_embed_72 = self._normalize_clip_score_embed(
            score_embed=clip_score_embeds["scale_72"],
            norm=self.clip_score_input_norm_72,
            scale=self.clip_score_input_scale_72,
            expected_hw=(72, 72),
            name="clip_score_embeds['scale_72']",
        )

        clip_score_embeds = {
            "scale_18": clip_score_embed_18,
            "scale_36": clip_score_embed_36,
            "scale_72": clip_score_embed_72,
        }

        refined_encoder_features_72 = encoder_features

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
            dynamic_clip_text,
            clip_score_embeds,
            clip_score_maps_18,
        )
