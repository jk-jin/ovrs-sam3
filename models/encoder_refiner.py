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
    Multi-layer encoder feature refiner.

    All sub-modules are created eagerly in __init__ so that parameters are
    visible to apply_freeze_cfg and the optimizer before the first forward.

    Inputs (forward):
        e:                 [B, C, D, H, W]
        clip_image_feat_map: [B, D_clip, Hc, Wc]
        class_names:       list of C strings
        sam_image_last:    [B, D, H, W]

    Output (forward):
        refined_e:          [B, C, D, H, W]
        class_query_tokens: [B, C, Q, D]
        dynamic_clip_text:  [B, C, Q, D_clip]
        clip_score_embed:   [B, C, D_score, H, W]
        clip_score_maps:    [B, C, Q, Hc, Wc]
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        score_embed_dim: int = 32,
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
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.use_checkpoint = bool(use_checkpoint)

        # Query extractor — created eagerly.
        self.query_extractor = EncoderQueryExtractor(
            hidden_dim=self.hidden_dim,
            num_query_tokens=self.num_query_tokens,
            num_heads=int(num_heads),
            dropout=float(dropout),
        )

        # CLIP prompt encoder — created eagerly so parameters enter optimizer.
        self.clip_prompt_encoder = SingleTokenClipPromptEncoder(
            clip_text_encoder=clip_text_encoder,
            prompt_template=str(prompt_template),
            sam_dim=self.hidden_dim,
            normalize_label=bool(normalize_label_for_clip),
            use_checkpoint=bool(use_checkpoint),
            num_attention_heads=int(num_heads),
        )

        # Score embedding builder.
        self.score_builder = ClipScoreEmbeddingBuilder(
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            num_query_tokens=self.num_query_tokens,
            conv_kernel=int(score_conv_kernel),
            encoder_hw=int(encoder_hw),
        )

        # Refiner layers.
        self.layers = nn.ModuleList([
            EncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                clip_dim=self.clip_dim,
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
        e: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        class_names: List[str],
        sam_image_last: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            refined_e:          [B, C, D, H, W]
            class_query_tokens: [B, C, Q, D]
            dynamic_clip_text:  [B, C, Q, D_clip]
            clip_score_embed:   [B, C, D_score, H, W]
            clip_score_maps:    [B, C, Q, Hc, Wc]
        """
        B, C, D, H, W = e.shape

        class_query_tokens = self.query_extractor(e)

        dynamic_clip_text = self.clip_prompt_encoder(
            class_query_tokens=class_query_tokens,
            class_names=class_names,
            clip_image_feat_map=clip_image_feat_map,
        )

        clip_score_embed, clip_score_maps = self.score_builder(
            dynamic_clip_text, clip_image_feat_map,
            target_hw=(H, W),
        )

        if clip_score_embed.shape[-2:] != (H, W):
            raise ValueError(
                "clip_score_embed spatial size must match e. "
                f"Expected {(H, W)}, got {tuple(clip_score_embed.shape[-2:])}."
            )

        refined_e = e
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                refined_e = checkpoint(
                    layer,
                    refined_e,
                    class_query_tokens,
                    sam_image_last,
                    clip_image_feat_map,
                    clip_score_embed,
                    use_reentrant=False,
                )
            else:
                refined_e = layer(
                    e=refined_e,
                    class_query_tokens=class_query_tokens,
                    sam_image_last=sam_image_last,
                    clip_image_feat_map=clip_image_feat_map,
                    clip_score_embed=clip_score_embed,
                )

        refined_e = refined_e + e

        return (
            refined_e,
            class_query_tokens,
            dynamic_clip_text,
            clip_score_embed,
            clip_score_maps,
        )
