from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .score_embeddings import ClipScoreEmbedding
from .encoder_refiner_attention import EncoderRefinerLayer, apply_layer_norm_bcdhw
from .decoder_input_fusion import DecoderInputFusion


class ClassConditionedEncoderRefiner(nn.Module):
    """Refine encoder/FPN streams at 36×36 under unchanged CLIP score guidance.

    Odd layers: FPN intra -> encoder inter. Even layers: encoder intra -> FPN
    inter. Each layer ends with independent FFNs. Both normalized output streams
    enter the per-chunk 72×72 DecoderInputFusion.
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim=256,
        clip_dim=768,
        score_embed_dim=256,
        num_heads=8,
        window_size=12,
        shift_size=6,
        fusion_layers=4,
        dropout=0.1,
        prompt_templates=None,
        normalize_label_for_clip=True,
        use_checkpoint=True,
        text_prompt_batch_size=64,
        text_prompt_use_checkpoint=True,
    ):
        super().__init__()
        if hidden_dim != 256:
            raise ValueError("SAM3 refiner and decoder fusion require hidden_dim=256")
        if fusion_layers <= 0:
            raise ValueError("fusion_layers must be positive")
        self.hidden_dim = hidden_dim
        self.use_checkpoint = bool(use_checkpoint)
        self.clip_score_embed = ClipScoreEmbedding(
            clip_text_encoder=clip_text_encoder,
            prompt_templates=list(prompt_templates or []),
            normalize_label=normalize_label_for_clip,
            clip_output_dim=clip_dim,
            score_embed_dim=score_embed_dim,
            text_prompt_batch_size=text_prompt_batch_size,
            text_prompt_use_checkpoint=text_prompt_use_checkpoint,
        )
        self.layers = nn.ModuleList(
            [
                EncoderRefinerLayer(
                    hidden_dim=hidden_dim,
                    score_embed_dim=score_embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    dropout=dropout,
                    intra_target="fpn" if index % 2 == 0 else "encoder",
                )
                for index in range(fusion_layers)
            ]
        )
        self.encoder_output_norm = nn.LayerNorm(hidden_dim)
        self.fpn_output_norm = nn.LayerNorm(hidden_dim)
        self.decoder_input_fusion = DecoderInputFusion(
            use_checkpoint=self.use_checkpoint
        )

    def fuse_decoder_input_chunk(
        self,
        refined_encoder_36,
        refined_fpn_36,
        original_encoder_feature_72,
        sam_fpn_72,
    ):
        return self.decoder_input_fusion(
            refined_encoder_36=refined_encoder_36,
            refined_fpn_36=refined_fpn_36,
            original_encoder_feature_72=original_encoder_feature_72,
            sam_fpn_72=sam_fpn_72,
        )

    def forward(
        self, encoder_features_72, clip_image_feat_map, class_names, sam_fpn_72
    ):
        if encoder_features_72.ndim != 5 or tuple(encoder_features_72.shape[2:]) != (
            256,
            72,
            72,
        ):
            raise ValueError("encoder_features_72 must be [B, P, 256, 72, 72]")
        batch, prompts = encoder_features_72.shape[:2]
        if batch <= 0 or prompts <= 0 or len(class_names) != prompts:
            raise ValueError(
                "Positive batch/prompt counts and one class name per prompt are required"
            )
        if tuple(sam_fpn_72.shape) != (batch, 256, 72, 72):
            raise ValueError("sam_fpn_72 must be [B, 256, 72, 72]")
        if (
            clip_image_feat_map.ndim != 4
            or clip_image_feat_map.shape[0] != batch
            or tuple(clip_image_feat_map.shape[-2:]) != (36, 36)
        ):
            raise ValueError("clip_image_feat_map must be [B, D_clip, 36, 36]")
        score, score_maps, template_text = self.clip_score_embed(
            class_names=class_names,
            remoteclip_feat_map=clip_image_feat_map,
        )
        encoder = F.interpolate(
            encoder_features_72.reshape(batch * prompts, 256, 72, 72),
            (36, 36),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, prompts, 256, 36, 36)
        fpn = F.interpolate(sam_fpn_72, (36, 36), mode="bilinear", align_corners=False)
        fpn = fpn[:, None].expand(-1, prompts, -1, -1, -1)
        # Score is reused unchanged, with gradients preserved to its generator.
        for layer in self.layers:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                encoder, fpn = checkpoint(
                    layer, encoder, fpn, score, use_reentrant=False
                )
            else:
                encoder, fpn = layer(encoder, fpn, score)
        return {
            "refined_encoder_36": apply_layer_norm_bcdhw(
                encoder, self.encoder_output_norm
            ),
            "refined_fpn_36": apply_layer_norm_bcdhw(fpn, self.fpn_output_norm),
            "score_embed_36": score,
            "clip_score_embed_36": score,
            "clip_score_maps_36": score_maps,
            "template_clip_text": template_text,
        }
