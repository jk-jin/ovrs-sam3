from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .score_embeddings import ClipScoreEmbedding
from .encoder_refiner_attention import EncoderRefinerLayer


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


# ---------------------------------------------------------------------------
# SAMTextAttentionPooling
# ---------------------------------------------------------------------------


class SAMTextAttentionPooling(nn.Module):
    """Lightweight attention pooling over SAM3 text tokens.

    Given token sequences and a boolean invalid mask, computes a single
    context vector per (batch, class) pair via learned softmax weights.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.token_score = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.token_score[-1].weight)

    def forward(
        self,
        sam_text_tokens: torch.Tensor,
        sam_text_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool SAM text tokens into a context vector.

        Args:
            sam_text_tokens:     [B, C, T, D]
            sam_text_token_mask: [B, C, T], True means invalid / padding.

        Returns:
            sam_text_context: [B, C, D]
        """
        B, C, T, D = sam_text_tokens.shape
        if D != self.hidden_dim:
            raise ValueError(
                f"SAMTextAttentionPooling expects hidden_dim={self.hidden_dim}, "
                f"got D={D}."
            )
        if tuple(sam_text_token_mask.shape) != (B, C, T):
            raise ValueError(
                f"sam_text_token_mask must be [{B}, {C}, {T}], "
                f"got {tuple(sam_text_token_mask.shape)}."
            )

        tokens = sam_text_tokens
        mask = sam_text_token_mask.bool().to(device=tokens.device)

        logits = self.token_score(tokens).squeeze(-1)  # [B, C, T]
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        all_invalid = mask.all(dim=-1, keepdim=True)  # [B, C, 1]
        logits = torch.where(all_invalid, torch.zeros_like(logits), logits)

        weights = torch.softmax(logits, dim=-1)  # [B, C, T]
        weights = weights.masked_fill(mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        context = (tokens * weights.unsqueeze(-1)).sum(dim=2)  # [B, C, D]
        return context.contiguous()


# ---------------------------------------------------------------------------
# EncoderFeatureUpsampler
# ---------------------------------------------------------------------------


class EncoderFeatureUpsampler(nn.Module):
    """
    Upsample refined 36×36 feature back to 72×72 and fuse it with
    original SAM3 encoder feature.

    Inputs:
        refined_feature_36:  [B, C, 256, 36, 36]
        original_encoder_72: [B, C, 256, 72, 72]

    Process:
        1. Bilinear upsample refined_feature_36 to 72×72.
        2. Concatenate [refiner_up_72, original_encoder_72].
        3. Conv fusion 512 → 256.
        4. Concatenate fused_72 with original_encoder_72.
        5. 1×1 conv 512 → 256.

    Output:
        refined_encoder_features_72: [B, C, 256, 72, 72]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        # Stage A: [refiner_up_72, original_encoder_72] = 256 * 2 → 256
        self.local_fusion = nn.Sequential(
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

        # Stage B: [local_fused_72, original_encoder_72] = 256 * 2 → 256
        self.final_fusion = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        refined_feature_36: torch.Tensor,
        original_encoder_72: torch.Tensor,
    ) -> torch.Tensor:
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

        # refined_feature_36: [B, C, D, 36, 36]
        # -> refiner_up_72: [B*C, D, 72, 72]
        refiner_up_72 = F.interpolate(
            refined_feature_36.reshape(B * C, D, 36, 36),
            size=(72, 72),
            mode="bilinear",
            align_corners=False,
        )

        # original_encoder_72: [B, C, D, 72, 72]
        # -> [B*C, D, 72, 72]
        orig_72 = original_encoder_72.reshape(B * C, D, 72, 72)

        # Stage A: fuse upsampled refiner feature with original encoder feature.
        local_in = torch.cat(
            [
                refiner_up_72,
                orig_72.to(device=refiner_up_72.device, dtype=refiner_up_72.dtype),
            ],
            dim=1,
        )
        local_fused_72 = self.local_fusion(local_in)

        # Stage B: fuse the local fused feature with the original encoder feature.
        final_in = torch.cat(
            [
                local_fused_72,
                orig_72.to(device=local_fused_72.device, dtype=local_fused_72.dtype),
            ],
            dim=1,
        )
        out = self.final_fusion(final_in)

        return out.reshape(B, C, D, 72, 72).contiguous()


# ---------------------------------------------------------------------------
# ClassConditionedEncoderRefiner
# ---------------------------------------------------------------------------


class ClassConditionedEncoderRefiner(nn.Module):
    """
    Encoder feature refiner operating at 36×36.

    SAM3 FPN is injected into score_embed before refiner layers
    with a 1+learnable residual scale. SAM text tokens are pooled
    via attention pooling once, then shared across all refiner layers.

    Forward inputs:
        encoder_features_72:  [B, C, 256, 72, 72]
        clip_image_feat_map:  [B, D_clip, 36, 36]
        sam_text_tokens:      [B, C, T, 256]
        sam_text_token_mask:  [B, C, T]
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
        clip_score_conv_kernel: int = 7,
        use_checkpoint: bool = True,
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
            conv_kernel=int(clip_score_conv_kernel),
        )

        self.sam_text_pooler = SAMTextAttentionPooling(hidden_dim=self.hidden_dim)

        self.score_fpn_fusion = nn.Sequential(
            nn.Conv2d(
                self.score_embed_dim + self.hidden_dim,
                self.score_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
            nn.Conv2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )
        self.score_fpn_res_scale = nn.Parameter(torch.zeros(1))

        self.layers = nn.ModuleList([
            EncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                score_embed_dim=self.score_embed_dim,
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
            )
            for _ in range(int(fusion_layers))
        ])

        self.upsampler = EncoderFeatureUpsampler(
            hidden_dim=self.hidden_dim,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        sam_text_tokens: torch.Tensor,
        sam_text_token_mask: torch.Tensor,
        class_names: List[str],
        sam_fpn_72: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_features_72:  [B, C, 256, 72, 72]
            clip_image_feat_map:  [B, D_clip, 36, 36]
            sam_text_tokens:      [B, C, T, 256]
            sam_text_token_mask:  [B, C, T]
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
            2. Inject SAM3 FPN into score embedding with 1+learnable residual scale.
            3. Downsample SAM3 encoder features from 72×72 to 36×36.
            4. Pool SAM text tokens to context via attention pooling.
            5. Run refiner layers.
            6. Upsample refined feature to 72×72 and fuse with original encoder feature.
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

        # Validate sam_text_tokens and sam_text_token_mask.
        if sam_text_tokens.ndim != 4:
            raise ValueError(
                f"sam_text_tokens must be [B, C, T, D], got {tuple(sam_text_tokens.shape)}."
            )
        if sam_text_token_mask.ndim != 3:
            raise ValueError(
                f"sam_text_token_mask must be [B, C, T], got {tuple(sam_text_token_mask.shape)}."
            )

        token_B, token_C, token_T, token_D = sam_text_tokens.shape
        if (token_B, token_C, token_D) != (batch_size, num_classes, hidden_dim):
            raise ValueError(
                f"sam_text_tokens batch/class/dim mismatch: "
                f"expected ({batch_size}, {num_classes}, {hidden_dim}), "
                f"got ({token_B}, {token_C}, {token_D})."
            )
        if tuple(sam_text_token_mask.shape) != (batch_size, num_classes, token_T):
            raise ValueError(
                f"sam_text_token_mask must be [{batch_size}, {num_classes}, {token_T}], "
                f"got {tuple(sam_text_token_mask.shape)}."
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

        # 2. Inject SAM3 FPN into score embedding.
        if tuple(sam_fpn_72.shape) != (batch_size, hidden_dim, 72, 72):
            raise ValueError(
                f"sam_fpn_72 must be [{batch_size}, {hidden_dim}, 72, 72], "
                f"got {tuple(sam_fpn_72.shape)}."
            )

        clip_score_flat = clip_score_embed_36.reshape(
            batch_size * num_classes,
            self.score_embed_dim,
            36,
            36,
        )

        sam_fpn_36 = F.interpolate(
            sam_fpn_72.to(device=clip_score_flat.device, dtype=clip_score_flat.dtype),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )

        sam_fpn_36 = (
            sam_fpn_36
            .unsqueeze(1)
            .expand(batch_size, num_classes, hidden_dim, 36, 36)
            .reshape(batch_size * num_classes, hidden_dim, 36, 36)
        )

        score_fpn_in = torch.cat([clip_score_flat, sam_fpn_36], dim=1)
        score_fpn_delta = self.score_fpn_fusion(score_fpn_in).reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            36,
            36,
        )

        score_embed_36 = (
            clip_score_embed_36
            + (1.0 + self.score_fpn_res_scale) * score_fpn_delta
        )

        # 3. Downsample encoder features from 72×72 to 36×36.
        feature_36 = F.interpolate(
            encoder_features_72.reshape(
                batch_size * num_classes, hidden_dim, 72, 72
            ),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, num_classes, hidden_dim, 36, 36)

        # 4. Pool SAM text tokens to context (once, shared across all layers).
        sam_text_context = self.sam_text_pooler(
            sam_text_tokens=sam_text_tokens,
            sam_text_token_mask=sam_text_token_mask,
        )

        # 5. Run refiner layers.
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                feature_36, score_embed_36 = checkpoint(
                    layer,
                    feature_36,
                    score_embed_36,
                    sam_text_context,
                    use_reentrant=False,
                )
            else:
                feature_36, score_embed_36 = layer(
                    feature_36=feature_36,
                    score_embed_36=score_embed_36,
                    sam_text_context=sam_text_context,
                )

        # 6. Upsample refined feature to 72×72 and fuse with original encoder feature.
        refined_encoder_features_72 = self.upsampler(
            refined_feature_36=feature_36,
            original_encoder_72=encoder_features_72,
        )

        return {
            "refined_encoder_features_72": refined_encoder_features_72,
            "refiner_features_36": feature_36,
            "score_embed_36": score_embed_36,
            "clip_score_embed_36": clip_score_embed_36,
            "clip_score_maps_36": clip_score_maps_36,
            "template_clip_text": template_clip_text,
        }
