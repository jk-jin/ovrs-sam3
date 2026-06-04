from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lowres_aggregator import LowResClassGuidedAggregator
from .sam_fpn_score_upsampler import SamFpnScoreConcatUpsampler, FinalMaskConvHead
from .task_modes import OUTPUT_KEYS


class LowResClassTokenBuilder(nn.Module):
    """
    Build per-class class tokens in the final fusion stage.

    Flow:
        learnable query → attend SAM3 text tokens → class_token_seed
                       → attend class_feature_low spatial tokens → class_tokens
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_class_tokens: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_class_tokens = int(num_class_tokens)
        self.num_heads = int(num_heads)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}")

        self.query_embed = nn.Parameter(torch.zeros(1, self.num_class_tokens, self.hidden_dim))
        nn.init.normal_(self.query_embed, std=0.02)

        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads,
            dropout=float(dropout), batch_first=True,
        )
        self.text_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        self.feature_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads,
            dropout=float(dropout), batch_first=True,
        )
        self.feature_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _sanitize_mask(mask: Optional[torch.Tensor], expected_shape: tuple) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"Mask shape mismatch: expected {expected_shape}, got {tuple(mask.shape)}."
            )
        mask = mask.detach().bool()
        fully_masked = mask.all(dim=1)
        if fully_masked.any():
            mask = mask.clone()
            mask[fully_masked, 0] = False
        return mask.contiguous()

    def forward(
        self,
        sam3_text_features: torch.Tensor,
        sam3_text_mask: Optional[torch.Tensor],
        class_feature_low: torch.Tensor,
    ) -> torch.Tensor:
        # --- Step 1: attend SAM3 text tokens ---
        text_feats = sam3_text_features.permute(1, 0, 2)  # [C, M, D]
        num_classes, seq_len, _ = text_feats.shape

        text_mask = self._sanitize_mask(sam3_text_mask, expected_shape=(num_classes, seq_len))

        query = self.query_embed.expand(num_classes, self.num_class_tokens, self.hidden_dim)
        query = query.to(device=text_feats.device, dtype=text_feats.dtype)
        text_feats = text_feats.detach()

        attn_out, _ = self.text_cross_attn(
            query=query, key=text_feats, value=text_feats,
            key_padding_mask=text_mask, need_weights=False,
        )
        class_token_seed = self.text_cross_attn_norm(query + self.dropout(attn_out))  # [C, Q, D]

        # --- Step 2: attend class_feature_low spatial tokens ---
        B, C, D, Hc, Wc = class_feature_low.shape
        if C != num_classes:
            raise ValueError(f"Class count mismatch: text={num_classes}, feature_low={C}")

        cf_flat = class_feature_low.flatten(3).permute(0, 1, 3, 2)  # [B, C, N, D]
        N = Hc * Wc

        seed = class_token_seed.unsqueeze(0).expand(B, C, self.num_class_tokens, D)
        seed_bc = seed.reshape(B * C, self.num_class_tokens, D)
        cf_bc = cf_flat.reshape(B * C, N, D)

        attn_out2, _ = self.feature_cross_attn(
            query=seed_bc, key=cf_bc, value=cf_bc, need_weights=False,
        )
        class_tokens = self.feature_cross_attn_norm(seed_bc + self.dropout(attn_out2))
        return class_tokens.reshape(B, C, self.num_class_tokens, D).contiguous()


class ClipScoreEmbedder(nn.Module):
    """
    Build CLIP correlation embedding from dynamic text features and CLIP image features.

    GSNet-style:
        score_maps [B, C, P, H, W]
        -> reshape [B*C, P, H, W]
        -> 7x7 conv
        -> [B, C, score_embed_dim, H, W]
    """

    def __init__(self, clip_output_dim: int = 768, score_embed_dim: int = 32, num_templates: int = 4):
        super().__init__()
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_templates = int(num_templates)

        self.score_conv = nn.Sequential(
            nn.Conv2d(
                self.num_templates,
                self.score_embed_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(min(8, self.score_embed_dim), self.score_embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        dynamic_text_features: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, P, D_clip = dynamic_text_features.shape
        _, D_img, Hc, Wc = clip_image_feat_map.shape

        if P != self.num_templates:
            raise ValueError(
                f"Template count mismatch: dynamic_text_features has P={P}, "
                f"but ClipScoreEmbedder was built with num_templates={self.num_templates}"
            )
        if D_clip != D_img:
            raise ValueError(
                f"CLIP dimension mismatch: text output_dim={D_clip}, "
                f"image feat dim={D_img}"
            )

        text_norm = F.normalize(dynamic_text_features, dim=-1)
        img_norm = F.normalize(clip_image_feat_map, dim=1)

        text_flat = text_norm.reshape(B * C, P, D_clip)
        img_flat = img_norm[:, None].expand(
            B, C, D_img, Hc, Wc
        ).reshape(B * C, D_img, Hc * Wc)

        score_maps = torch.bmm(
            text_flat,
            img_flat,
        ).reshape(B * C, P, Hc, Wc) * 20.0

        clip_score_embed = self.score_conv(score_maps)
        clip_score_embed = clip_score_embed.reshape(
            B,
            C,
            self.score_embed_dim,
            Hc,
            Wc,
        )

        score_maps = score_maps.reshape(B, C, P, Hc, Wc)

        return clip_score_embed.contiguous(), score_maps.contiguous()


class SamScoreEmbedder(nn.Module):
    """Build SAM3 score embedding from semantic_logits using GSNet-style 7x7 conv."""

    def __init__(self, score_embed_dim: int = 32):
        super().__init__()
        self.score_embed_dim = int(score_embed_dim)

        self.score_conv = nn.Sequential(
            nn.Conv2d(
                1,
                self.score_embed_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(min(8, self.score_embed_dim), self.score_embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        semantic_logits: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C = semantic_logits.shape[:2]
        Hc, Wc = int(target_hw[0]), int(target_hw[1])

        score = torch.sigmoid(semantic_logits.detach())
        score_low = F.interpolate(
            score.reshape(B * C, 1, *semantic_logits.shape[-2:]),
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, C, Hc, Wc)

        sam_score_embed = self.score_conv(
            score_low.reshape(B * C, 1, Hc, Wc)
        )

        return (
            sam_score_embed.reshape(
                B,
                C,
                self.score_embed_dim,
                Hc,
                Wc,
            ).contiguous(),
            score_low.contiguous(),
        )


class LowResScoreFusionStem(nn.Module):
    """Fuse CLIP score embed and SAM score embed using GSNet-style 7x7 conv."""

    def __init__(self, score_embed_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        in_ch = int(score_embed_dim) * 2

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_ch,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU(),
        )

    def forward(self, clip_score_embed: torch.Tensor, sam_score_embed: torch.Tensor) -> torch.Tensor:
        B, C, _, Hc, Wc = clip_score_embed.shape
        x = torch.cat([clip_score_embed, sam_score_embed], dim=2)
        x = x.reshape(B * C, -1, Hc, Wc)
        x = self.fusion(x)
        return x.reshape(B, C, -1, Hc, Wc).contiguous()


class ClassFeatureLowProjector(nn.Module):
    """Project frozen SAM3 encoder feature into trainable low-res class feature."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        num_groups = min(8, self.hidden_dim)
        if self.hidden_dim % num_groups != 0:
            num_groups = 1

        self.proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, self.hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        encoder_feature: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if encoder_feature.dim() != 4:
            raise ValueError(
                "encoder_feature must be [B*C, D, H, W], "
                f"got {tuple(encoder_feature.shape)}."
            )

        if int(encoder_feature.shape[1]) != self.hidden_dim:
            raise ValueError(
                "encoder_feature channel mismatch: expected "
                f"{self.hidden_dim}, got {encoder_feature.shape[1]}."
            )

        target_hw = (int(target_hw[0]), int(target_hw[1]))
        x = F.adaptive_avg_pool2d(encoder_feature, target_hw)
        x = self.proj(x)
        return x.contiguous()


class LowResScoreGuidedSamMixer(nn.Module):
    """
    New final mixer for open-vocabulary semantic segmentation.

    Flow:
        1. Build class_tokens from SAM3 text + class_feature_low
        2. class_code = mean(class_tokens, dim=2)
        3. Dynamic CLIP prompt → dynamic_text_features [B, C, G, output_dim]
        4. CLIP score embed → clip_score_embed [B, C, D_score, Hc, Wc]
        5. SAM score embed  → sam_score_embed [B, C, D_score, Hc, Wc]
        6. Fusion stem → [B, C, hidden_dim, Hc, Wc]
        7. Low-res aggregator (L layers) → refined
        8. SAM FPN upsampler → [B, C, out_ch, H, W]
        9. 3×3 conv mask head → final_logits [B, C, H, W]

    All sub-modules are created in __init__ so parameters are registered
    before the optimizer is built.
    """

    def __init__(
        self,
        sam_dim: int = 256,
        fusion_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        clip_text_encoder=None,
        clip_prompt_templates: Optional[List[str]] = None,
        normalize_label_for_clip: bool = True,
        score_embed_dim: int = 32,
        lowres_hidden_dim: int = 256,
        window_size: int = 8,
        shift_size: int = 4,
        tokens_per_template: int = 4,
        upsampler_class_chunk_size: int = 4,
        upsampler_decoder_channels: Optional[List[int]] = None,
        upsampler_sam_guidance_channels: Optional[List[int]] = None,
        upsampler_score_channels: Optional[List[int]] = None,
        upsampler_score_input: str = "score_and_tanh_logit",
        upsampler_upsample_mode: str = "bilinear",
        upsampler_norm: str = "group_norm",
        upsampler_act: str = "gelu",
    ):
        super().__init__()

        self.sam_dim = int(sam_dim)

        prompt_templates = list(clip_prompt_templates or [])
        num_templates = len(prompt_templates)
        if num_templates <= 0:
            raise ValueError("clip_prompt_templates must not be empty")

        # num_class_tokens is derived from template count, not configurable.
        self.num_class_tokens = num_templates * int(tokens_per_template)

        # Read CLIP output dim from the text encoder (avoid hard-coding).
        self.clip_output_dim = int(clip_text_encoder.output_dim)

        # 1. Class token builder
        self.class_token_builder = LowResClassTokenBuilder(
            hidden_dim=self.sam_dim,
            num_class_tokens=self.num_class_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.class_feature_low_projector = ClassFeatureLowProjector(
            hidden_dim=self.sam_dim,
        )

        # 2. Dynamic CLIP prompt encoder
        from .dynamic_clip_prompt import DynamicClipPromptEncoder
        self.dynamic_prompt_encoder = DynamicClipPromptEncoder(
            clip_text_encoder=clip_text_encoder,
            prompt_templates=prompt_templates,
            sam_dim=self.sam_dim,
            tokens_per_template=int(tokens_per_template),
            normalize_label=bool(normalize_label_for_clip),
        )

        # 3. Score embedders
        self.clip_score_embedder = ClipScoreEmbedder(
            clip_output_dim=self.clip_output_dim,
            score_embed_dim=int(score_embed_dim),
            num_templates=num_templates,
        )
        self.sam_score_embedder = SamScoreEmbedder(score_embed_dim=int(score_embed_dim))

        # 4. Fusion stem
        self.fusion_stem = LowResScoreFusionStem(
            score_embed_dim=int(score_embed_dim),
            hidden_dim=int(lowres_hidden_dim),
        )

        # 5. Aggregator
        self.aggregator = LowResClassGuidedAggregator(
            num_layers=int(fusion_layers),
            hidden_dim=int(lowres_hidden_dim),
            num_heads=num_heads,
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=dropout,
        )

        # 6. Upsampler (all params registered at init, target sizes resolved lazily)
        self.upsampler = SamFpnScoreConcatUpsampler(
            in_ch=int(lowres_hidden_dim),
            decoder_channels=(
                list(upsampler_decoder_channels) if upsampler_decoder_channels is not None else None
            ),
            sam_guidance_channels=(
                list(upsampler_sam_guidance_channels) if upsampler_sam_guidance_channels is not None else None
            ),
            score_channels=(
                list(upsampler_score_channels) if upsampler_score_channels is not None else None
            ),
            score_input=str(upsampler_score_input),
            upsample_mode=str(upsampler_upsample_mode),
            norm=str(upsampler_norm),
            act=str(upsampler_act),
            class_chunk_size=int(upsampler_class_chunk_size),
        )

        # 7. Mask head
        self.mask_head = FinalMaskConvHead(
            in_ch=self.upsampler.out_ch,
            class_chunk_size=int(upsampler_class_chunk_size),
        )

    def _build_class_code(self, class_tokens: torch.Tensor) -> torch.Tensor:
        return class_tokens.mean(dim=2)  # [B, C, D]

    def project_class_feature_low(
        self,
        encoder_feature: torch.Tensor,
        target_hw: Tuple[int, int],
        batch_size: int,
        num_classes: int,
    ) -> torch.Tensor:
        num_pairs = int(batch_size) * int(num_classes)

        if int(encoder_feature.shape[0]) != num_pairs:
            raise ValueError(
                "encoder_feature first dimension must equal B*C. "
                f"Expected {num_pairs}, got {encoder_feature.shape[0]}."
            )

        class_feature_low = self.class_feature_low_projector(
            encoder_feature=encoder_feature,
            target_hw=target_hw,
        )

        h_low, w_low = class_feature_low.shape[-2:]
        return class_feature_low.reshape(
            int(batch_size),
            int(num_classes),
            self.sam_dim,
            h_low,
            w_low,
        ).contiguous()

    def forward(
        self,
        semantic_logits: torch.Tensor,
        class_feature_low: torch.Tensor,
        sam3_fpn_features: List[torch.Tensor],
        sam3_text_features: torch.Tensor,
        sam3_text_mask: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        class_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        B, C, final_h, final_w = semantic_logits.shape
        _, _, _, Hc, Wc = class_feature_low.shape

        # 1. Build class tokens
        class_tokens = self.class_token_builder(sam3_text_features, sam3_text_mask, class_feature_low)

        # 2. class_code
        class_code = self._build_class_code(class_tokens)

        # 3. Dynamic CLIP prompt
        dynamic_text_features = self.dynamic_prompt_encoder(class_tokens, class_names)
        # [B, C, G, clip_output_dim]

        # 4. CLIP score embedding
        clip_score_embed, clip_score_maps = self.clip_score_embedder(
            dynamic_text_features, clip_image_feat_map,
        )

        # 5. SAM score embedding
        sam_score_embed, sam3_score_low = self.sam_score_embedder(
            semantic_logits, target_hw=(Hc, Wc),
        )

        # 6. Fuse
        x = self.fusion_stem(clip_score_embed, sam_score_embed)

        # 7. Low-res aggregation
        x = self.aggregator(x, class_feature_low, class_code)

        # 8. Upsample
        x_up = self.upsampler(x, semantic_logits, sam3_fpn_features)

        # 9. Final mask
        final_logits = self.mask_head(x_up)

        return {
            OUTPUT_KEYS.final_logits: final_logits.contiguous(),
            OUTPUT_KEYS.class_tokens: class_tokens.contiguous(),
            OUTPUT_KEYS.class_feature_low: class_feature_low.contiguous(),
            OUTPUT_KEYS.clip_score_maps: clip_score_maps.contiguous(),
            OUTPUT_KEYS.sam3_score_low: sam3_score_low.contiguous(),
        }