from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_guided_upsampler import SamFpnClipGuidedUpsampler
from .final_mask_head import FinalMaskConvHead
from .lowres_aggregator import TextGuidedLowResAggregator
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


class ClipDenseFeatureProjector(nn.Module):
    """
    Project frozen CLIP dense feature map into low-res aggregator hidden space.

    Input:
        clip_image_feat_map: [B, D_clip, Hc, Wc]

    Output:
        clip_dense_low: [B, D_low, Hc, Wc]
    """

    def __init__(
        self,
        clip_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.hidden_dim = int(hidden_dim)

        num_groups = min(8, self.hidden_dim)
        if self.hidden_dim % num_groups != 0:
            num_groups = 1

        self.proj = nn.Sequential(
            nn.Conv2d(
                self.clip_dim,
                self.hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, self.hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        clip_image_feat_map: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if clip_image_feat_map.dim() != 4:
            raise ValueError(
                "clip_image_feat_map must be [B, D_clip, Hc, Wc], "
                f"got {tuple(clip_image_feat_map.shape)}."
            )

        if int(clip_image_feat_map.shape[1]) != self.clip_dim:
            raise ValueError(
                f"clip_image_feat_map channel mismatch: "
                f"expected {self.clip_dim}, got {clip_image_feat_map.shape[1]}."
            )

        x = clip_image_feat_map.detach()

        target_hw = (int(target_hw[0]), int(target_hw[1]))
        if tuple(x.shape[-2:]) != target_hw:
            x = F.interpolate(
                x,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        return self.proj(x).contiguous()


class PresenceHead(nn.Module):
    """
    Predict per-class image-level presence from stage-wise guidance history.

    Input:
        stage_text_guidance_history: [B, C, S, D]

    Output:
        presence_logits: [B, C]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_stages: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_stages = int(num_stages)

        self.stage_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, stage_text_guidance_history: torch.Tensor) -> torch.Tensor:
        if stage_text_guidance_history.dim() != 4:
            raise ValueError(
                "stage_text_guidance_history must be [B, C, S, D], "
                f"got {tuple(stage_text_guidance_history.shape)}."
            )

        B, C, S, D = stage_text_guidance_history.shape

        if S != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} stages, got {S}."
            )

        if D != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {D}."
            )

        t = self.stage_proj(stage_text_guidance_history)
        t = t.mean(dim=2)
        # [B, C, D]

        presence_logits = self.mlp(t).squeeze(-1)
        return presence_logits.contiguous()


class ClassTextGuidanceBuilder(nn.Module):
    """
    Build fused per-class text guidance vector.

    Inputs:
        sam3_text_features:
            [M, C, D_sam]

        sam3_text_mask:
            [C, M]
            True means padding / invalid token.

        dynamic_text_features:
            [B, C, G, D_clip]

    Output:
        class_text_guidance:
            [B, C, D_low]
    """

    def __init__(
        self,
        sam_dim: int = 256,
        clip_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.clip_dim = int(clip_dim)
        self.hidden_dim = int(hidden_dim)

        self.sam_text_proj = nn.Sequential(
            nn.LayerNorm(self.sam_dim),
            nn.Linear(self.sam_dim, self.hidden_dim),
        )

        self.clip_text_proj = nn.Sequential(
            nn.LayerNorm(self.clip_dim),
            nn.Linear(self.clip_dim, self.hidden_dim),
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(
        self,
        sam3_text_features: torch.Tensor,
        sam3_text_mask: torch.Tensor,
        dynamic_text_features: torch.Tensor,
    ) -> torch.Tensor:
        if sam3_text_features.dim() != 3:
            raise ValueError(
                "sam3_text_features must be [M, C, D_sam], "
                f"got {tuple(sam3_text_features.shape)}."
            )

        if sam3_text_mask.dim() != 2:
            raise ValueError(
                "sam3_text_mask must be [C, M], "
                f"got {tuple(sam3_text_mask.shape)}."
            )

        if dynamic_text_features.dim() != 4:
            raise ValueError(
                "dynamic_text_features must be [B, C, G, D_clip], "
                f"got {tuple(dynamic_text_features.shape)}."
            )

        M, C_sam, D_sam = sam3_text_features.shape
        B, C_clip, G, D_clip = dynamic_text_features.shape

        if C_sam != C_clip:
            raise ValueError(
                f"Class count mismatch: SAM3 text has C={C_sam}, "
                f"CLIP text has C={C_clip}."
            )

        if D_sam != self.sam_dim:
            raise ValueError(
                f"SAM3 text dim mismatch: expected {self.sam_dim}, got {D_sam}."
            )

        if D_clip != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, got {D_clip}."
            )

        if tuple(sam3_text_mask.shape) != (C_sam, M):
            raise ValueError(
                f"sam3_text_mask must be [C, M] = {(C_sam, M)}, "
                f"got {tuple(sam3_text_mask.shape)}."
            )

        # SAM3 branch.
        # [M, C, D_sam] → [C, M, D_sam]
        sam_tokens = sam3_text_features.permute(1, 0, 2).detach()
        sam_tokens = self.sam_text_proj(sam_tokens)  # [C, M, D_low]

        valid = (~sam3_text_mask.bool()).to(
            device=sam_tokens.device,
            dtype=sam_tokens.dtype,
        )  # [C, M]

        valid_sum = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        sam_vec = (sam_tokens * valid[:, :, None]).sum(dim=1) / valid_sum
        # [C, D_low]

        # CLIP branch.
        clip_tokens = self.clip_text_proj(dynamic_text_features)
        # [B, C, G, D_low]

        clip_vec = clip_tokens.mean(dim=2)
        # [B, C, D_low]

        sam_vec = sam_vec[None].expand(B, C_sam, self.hidden_dim)
        # [B, C, D_low]

        class_text_guidance = self.fuse(torch.cat([sam_vec, clip_vec], dim=-1))
        return class_text_guidance.contiguous()


class ClipScoreEmbedder(nn.Module):
    """
    Build CLIP correlation embedding from dynamic text features and CLIP image features.

    Design:
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


class ConvGnGelu(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        num_groups = min(8, out_ch)
        if out_ch % num_groups != 0:
            num_groups = 1

        self.block = nn.Sequential(
            nn.Conv2d(
                int(in_ch),
                int(out_ch),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups, int(out_ch)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SamMaskPriorEncoder(nn.Module):
    """
    Encode frozen SAM3 semantic score map into low-res mask-prior embedding.

    Input:
        semantic_logits: [B, C, H, W]

    Output:
        sam_mask_prior_embed: [B, C, D_score, Hc, Wc]
        sam3_score_low:       [B, C, Hc, Wc]

    Only sigmoid(semantic_logits) is used as input.
    No tanh(logit).
    No direct bilinear-downsample-then-7x7-conv shortcut.
    """

    def __init__(
        self,
        score_embed_dim: int = 32,
        hidden_channels: tuple[int, ...] = (16, 32, 32),
    ):
        super().__init__()
        self.score_embed_dim = int(score_embed_dim)

        channels = [1, *[int(x) for x in hidden_channels], self.score_embed_dim]

        blocks = []
        for idx in range(len(channels) - 1):
            in_ch = channels[idx]
            out_ch = channels[idx + 1]

            stride = 2 if idx < len(channels) - 2 else 1

            blocks.append(
                ConvGnGelu(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=3,
                    stride=stride,
                )
            )

        self.encoder = nn.Sequential(*blocks)

        self.refine = nn.Sequential(
            ConvGnGelu(
                in_ch=self.score_embed_dim,
                out_ch=self.score_embed_dim,
                kernel_size=3,
                stride=1,
            ),
            nn.Conv2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                min(8, self.score_embed_dim),
                self.score_embed_dim,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        semantic_logits: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )

        B, C, H, W = semantic_logits.shape
        Hc, Wc = int(target_hw[0]), int(target_hw[1])

        score = torch.sigmoid(semantic_logits.detach())  # [B, C, H, W]

        x = score.reshape(B * C, 1, H, W)
        x = self.encoder(x)

        if tuple(x.shape[-2:]) != (Hc, Wc):
            x = F.interpolate(
                x,
                size=(Hc, Wc),
                mode="bilinear",
                align_corners=False,
            )

        x = self.refine(x)

        sam_mask_prior_embed = x.reshape(
            B,
            C,
            self.score_embed_dim,
            Hc,
            Wc,
        ).contiguous()

        sam3_score_low = F.interpolate(
            score.reshape(B * C, 1, H, W),
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, C, Hc, Wc).contiguous()

        return sam_mask_prior_embed, sam3_score_low


class LowResScoreFusionStem(nn.Module):
    """
    Fuse CLIP correlation embedding and SAM3 mask-prior embedding.

    Input:
        clip_score_embed:      [B, C, D_score, Hc, Wc]
        sam_mask_prior_embed:  [B, C, D_score, Hc, Wc]

    Output:
        x_low: [B, C, D_low, Hc, Wc]
    """

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

    def forward(
        self,
        clip_score_embed: torch.Tensor,
        sam_mask_prior_embed: torch.Tensor,
    ) -> torch.Tensor:
        B, C, _, Hc, Wc = clip_score_embed.shape
        x = torch.cat([clip_score_embed, sam_mask_prior_embed], dim=2)
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


class ClassTextGuidedMaskPriorClipFusionFinalMixer(nn.Module):
    """
    Final mixer for open-vocabulary remote-sensing semantic segmentation.

    Main design:
        1. Build class tokens from SAM3 text tokens and SAM3 class feature maps.
        2. Build dynamic CLIP text features from class tokens.
        3. Build per-class text guidance from valid SAM3 text tokens and dynamic CLIP text features.
        4. Build CLIP text-image correlation embedding.
        5. Build SAM3 mask-prior embedding from sigmoid semantic logits.
        6. Fuse CLIP correlation embedding and SAM3 mask-prior embedding.
        7. Run class-text-guided low-res aggregation with:
               SAM3 class feature guidance
             + CLIP dense feature guidance.
        8. Upsample with:
               SAM FPN guidance
             + CLIP middle-layer guidance.
        9. Predict final mask logits.
       10. Refine class text guidance across upsample stages.
       11. Predict image-level class presence logits.

    Frozen:
        SAM3 backbone and original SAM3 heads.
        OpenCLIP image encoder.
        OpenCLIP text encoder.

    Trainable:
        This final mixer and its submodules.
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
        clip_image_native_dim: Optional[int] = None,
        score_embed_dim: int = 32,
        lowres_hidden_dim: int = 256,
        window_size: int = 8,
        shift_size: int = 4,
        tokens_per_template: int = 4,
        upsampler_class_chunk_size: int = 4,
        upsampler_decoder_channels: Optional[List[int]] = None,
        upsampler_sam_guidance_channels: Optional[List[int]] = None,
        upsampler_clip_guidance_channels: Optional[List[int]] = None,
        upsampler_clip_guidance_stage_indices: Optional[List[int]] = None,
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

        self.clip_dense_projector = ClipDenseFeatureProjector(
            clip_dim=self.clip_output_dim,
            hidden_dim=int(lowres_hidden_dim),
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

        self.text_guidance_builder = ClassTextGuidanceBuilder(
            sam_dim=self.sam_dim,
            clip_dim=self.clip_output_dim,
            hidden_dim=int(lowres_hidden_dim),
            dropout=dropout,
        )

        # 3. Score / mask-prior embedders
        self.clip_score_embedder = ClipScoreEmbedder(
            clip_output_dim=self.clip_output_dim,
            score_embed_dim=int(score_embed_dim),
            num_templates=num_templates,
        )
        self.sam_mask_prior_encoder = SamMaskPriorEncoder(
            score_embed_dim=int(score_embed_dim),
        )

        # 4. Fusion stem
        self.fusion_stem = LowResScoreFusionStem(
            score_embed_dim=int(score_embed_dim),
            hidden_dim=int(lowres_hidden_dim),
        )

        # 5. Aggregator
        self.aggregator = TextGuidedLowResAggregator(
            num_layers=int(fusion_layers),
            hidden_dim=int(lowres_hidden_dim),
            num_heads=num_heads,
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=dropout,
        )

        # 6. Upsampler (CLIP middle features replace semantic score guidance)
        clip_mid_in_ch = (
            int(clip_image_native_dim)
            if clip_image_native_dim is not None
            else int(self.clip_output_dim)
        )

        self.upsampler = SamFpnClipGuidedUpsampler(
            in_ch=int(lowres_hidden_dim),
            decoder_channels=(
                list(upsampler_decoder_channels) if upsampler_decoder_channels is not None else None
            ),
            sam_guidance_channels=(
                list(upsampler_sam_guidance_channels) if upsampler_sam_guidance_channels is not None else None
            ),
            clip_mid_in_ch=clip_mid_in_ch,
            clip_guidance_channels=(
                list(upsampler_clip_guidance_channels) if upsampler_clip_guidance_channels is not None else None
            ),
            clip_guidance_stage_indices=(
                list(upsampler_clip_guidance_stage_indices) if upsampler_clip_guidance_stage_indices is not None else None
            ),
            upsample_mode=str(upsampler_upsample_mode),
            norm=str(upsampler_norm),
            act=str(upsampler_act),
            class_chunk_size=int(upsampler_class_chunk_size),
            presence_text_dim=int(lowres_hidden_dim),
            presence_num_heads=int(num_heads),
            presence_dropout=float(dropout),
        )

        # 7. Mask head
        self.mask_head = FinalMaskConvHead(
            in_ch=self.upsampler.out_ch,
            class_chunk_size=int(upsampler_class_chunk_size),
        )

        # 8. Presence head
        self.presence_head = PresenceHead(
            hidden_dim=int(lowres_hidden_dim),
            num_stages=len(self.upsampler.blocks),
            dropout=dropout,
        )

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
        clip_mid_features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        B, C, final_h, final_w = semantic_logits.shape
        _, _, _, Hc, Wc = class_feature_low.shape

        # 1. Build class tokens for dynamic CLIP prompt
        class_tokens = self.class_token_builder(
            sam3_text_features,
            sam3_text_mask,
            class_feature_low,
        )

        # 2. Dynamic CLIP prompt
        dynamic_text_features = self.dynamic_prompt_encoder(
            class_tokens,
            class_names,
        )
        # [B, C, G, D_clip]

        # 3. Build fused text guidance vector
        class_text_guidance = self.text_guidance_builder(
            sam3_text_features=sam3_text_features,
            sam3_text_mask=sam3_text_mask,
            dynamic_text_features=dynamic_text_features,
        )
        # [B, C, D_low]

        # 4. Project CLIP dense feature for window attention
        clip_dense_low = self.clip_dense_projector(
            clip_image_feat_map=clip_image_feat_map,
            target_hw=(Hc, Wc),
        )
        # [B, D_low, Hc, Wc]

        # 5. CLIP score embedding
        clip_score_embed, clip_score_maps = self.clip_score_embedder(
            dynamic_text_features,
            clip_image_feat_map,
        )

        # 6. SAM3 mask-prior embedding
        sam_mask_prior_embed, sam3_score_low = self.sam_mask_prior_encoder(
            semantic_logits,
            target_hw=(Hc, Wc),
        )

        # 7. Fuse CLIP score embedding and SAM3 mask-prior embedding
        x = self.fusion_stem(
            clip_score_embed,
            sam_mask_prior_embed,
        )

        # 8. Low-res aggregation
        x = self.aggregator(
            x=x,
            class_feature_low=class_feature_low,
            clip_dense_low=clip_dense_low,
            class_text_guidance=class_text_guidance,
        )

        # 9. Upsample with SAM FPN + CLIP middle features + presence refinement
        x_up, stage_text_guidance_history = self.upsampler(
            x_low_refined=x,
            class_text_guidance=class_text_guidance,
            final_hw=tuple(semantic_logits.shape[-2:]),
            sam_fpn_features=sam3_fpn_features,
            clip_mid_features=clip_mid_features,
        )

        # 10. Final mask
        final_logits = self.mask_head(x_up)

        # 11. Presence
        presence_logits = self.presence_head(stage_text_guidance_history)
        presence_score = torch.sigmoid(presence_logits)

        out = {
            OUTPUT_KEYS.final_logits: final_logits.contiguous(),
            OUTPUT_KEYS.class_tokens: class_tokens.contiguous(),
            OUTPUT_KEYS.class_feature_low: class_feature_low.contiguous(),
            OUTPUT_KEYS.clip_score_maps: clip_score_maps.contiguous(),
            OUTPUT_KEYS.sam3_score_low: sam3_score_low.contiguous(),
            OUTPUT_KEYS.class_text_guidance: class_text_guidance.contiguous(),
            OUTPUT_KEYS.stage_text_guidance_history: (
                stage_text_guidance_history.contiguous()
            ),
            OUTPUT_KEYS.presence_logits: presence_logits.contiguous(),
            OUTPUT_KEYS.presence_score: presence_score.contiguous(),
            OUTPUT_KEYS.clip_mid_features: [
                feat.contiguous() for feat in clip_mid_features
            ],
            OUTPUT_KEYS.clip_dense_low: clip_dense_low.contiguous(),
        }

        return out