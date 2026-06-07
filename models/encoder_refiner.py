from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .clip_prompt_encoder import SingleTokenClipPromptEncoder
from .clip_score_embedding import ClipScoreEmbeddingBuilder
from .encoder_query_extractor import EncoderQueryExtractor
from .encoder_refiner_attention import EncoderRefinerLayer


class ClassConditionedEncoderRefiner(nn.Module):
    """
    Multi-layer encoder feature refiner.

    Inputs:
        e:                 [B, C, D, H, W]
        sam_text_features: [M, C, D]
        sam_text_mask:     [C, M]
        clip_text_encoder  — OpenCLIP text encoder
        clip_image_feat_map: [B, D_clip, Hc, Wc]
        class_names:       list of C strings
        sam_image_last:    [B, D, H, W]

    Output:
        refined_e:          [B, C, D, H, W]
        class_query_tokens: [B, C, Q, D]
        dynamic_clip_text:  [B, C, Q, D_clip]
        clip_score_embed:   [B, C, D_score, H, W]
    """

    def __init__(
        self,
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
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clip_dim = int(clip_dim)

        self.num_query_tokens = int(num_query_tokens)

        self.query_extractor = EncoderQueryExtractor(
            hidden_dim=self.hidden_dim,
            num_query_tokens=self.num_query_tokens,
            num_heads=int(num_heads),
            dropout=float(dropout),
        )

        self.clip_prompt_encoder = None
        self._prompt_template = str(prompt_template)
        self._normalize_label_for_clip = bool(normalize_label_for_clip)

        self.score_builder = ClipScoreEmbeddingBuilder(
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            conv_kernel=7,
            mid_hw=32,
            encoder_hw=36,
        )

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
    # Lazy CLIP prompt encoder
    # ------------------------------------------------------------------

    def set_clip_prompt_encoder(self, encoder: SingleTokenClipPromptEncoder) -> None:
        self.clip_prompt_encoder = encoder

    def _build_clip_prompt_encoder(self, clip_text_encoder) -> SingleTokenClipPromptEncoder:
        if self.clip_prompt_encoder is not None:
            return self.clip_prompt_encoder

        encoder = SingleTokenClipPromptEncoder(
            clip_text_encoder=clip_text_encoder,
            prompt_template=self._prompt_template,
            sam_dim=self.hidden_dim,
            normalize_label=self._normalize_label_for_clip,
        )
        self.clip_prompt_encoder = encoder
        return encoder

    # ------------------------------------------------------------------
    # Text mean helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_text_mean(
        sam3_text_features: torch.Tensor,
        sam3_text_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute mean of valid SAM3 text tokens per class.

        Args:
            sam3_text_features: [M, C, D]
            sam3_text_mask:     [C, M]  (True = ignore/padded)

        Returns:
            sam_text_mean: [B, C, D]
        """
        M, C, D = sam3_text_features.shape
        feats = sam3_text_features.permute(1, 0, 2)  # [C, M, D]

        valid_mask = (~sam3_text_mask.bool()).float()  # [C, M], 1=valid
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)

        mean = (feats * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_count
        mean = mean.unsqueeze(0).expand(batch_size, C, D)
        return mean.contiguous()

    @staticmethod
    def _compute_clip_text_mean(
        dynamic_clip_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean of Q dynamic CLIP text features per class.

        Args:
            dynamic_clip_text: [B, C, Q, D_clip]

        Returns:
            clip_text_mean: [B, C, D_clip]
        """
        return dynamic_clip_text.mean(dim=2).contiguous()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        e: torch.Tensor,
        sam_text_features: torch.Tensor,
        sam_text_mask: torch.Tensor,
        clip_text_encoder,
        clip_image_feat_map: torch.Tensor,
        class_names: List[str],
        sam_image_last: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            refined_e:          [B, C, D, H, W]
            class_query_tokens: [B, C, Q, D]
            dynamic_clip_text:  [B, C, Q, D_clip]
            clip_score_embed:   [B, C, D_score, H, W]
        """
        B, C, D, H, W = e.shape

        # 1. Extract class query tokens
        class_query_tokens = self.query_extractor(e)

        # 2. Build CLIP prompt encoder on first call
        self._build_clip_prompt_encoder(clip_text_encoder)

        # 3. Build dynamic CLIP text features
        dynamic_clip_text = self.clip_prompt_encoder(
            class_query_tokens, class_names
        )

        # 4. Build CLIP score embedding
        clip_score_embed, clip_score_maps = self.score_builder(
            dynamic_clip_text, clip_image_feat_map
        )

        # 5. Build text means for attention guidance
        sam_text_mean = self._compute_text_mean(
            sam_text_features, sam_text_mask, B
        )
        clip_text_mean = self._compute_clip_text_mean(dynamic_clip_text)

        # 6. Run refiner layers
        refined_e = e
        for layer in self.layers:
            refined_e = layer(
                e=refined_e,
                sam_text_mean=sam_text_mean,
                clip_text_mean=clip_text_mean,
                sam_image_last=sam_image_last,
                clip_score_embed=clip_score_embed,
            )

        return refined_e, class_query_tokens, dynamic_clip_text, clip_score_embed
