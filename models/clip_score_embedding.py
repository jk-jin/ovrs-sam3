from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


# ---------------------------------------------------------------------------
# GuidedScoreFuse
# ---------------------------------------------------------------------------


class GuidedScoreFuse(nn.Module):
    """Residual gated fusion for score embedding and CLIP mid-layer guidance.

    Uses init_scale=0.0 so the initial behaviour is close to the old
    non-guided path — the model learns whether to use the guidance.
    """

    def __init__(
        self,
        score_dim: int,
        guide_dim: int,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.score_dim = int(score_dim)
        self.guide_dim = int(guide_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.score_dim + self.guide_dim,
                self.score_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_dim),
            nn.GELU(),
            nn.Conv2d(
                self.score_dim,
                self.score_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_dim),
            nn.GELU(),
        )

        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if guide.shape[-2:] != x.shape[-2:]:
            guide = F.interpolate(
                guide,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = self.fuse(torch.cat([x, guide], dim=1))
        return x + self.res_scale * fused


# ---------------------------------------------------------------------------
# GuidedScoreUpsample
# ---------------------------------------------------------------------------


class GuidedScoreUpsample(nn.Module):
    """Upsample score embedding and fuse projected CLIP mid-layer guidance.

    Bilinear upsample + 3x3 conv (no ConvTranspose2d) for stability.
    init_scale=0.0 so behaviour starts close to plain upsampling.
    """

    def __init__(
        self,
        score_dim: int,
        guide_dim: int,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.score_dim = int(score_dim)
        self.guide_dim = int(guide_dim)

        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                self.score_dim,
                self.score_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_dim),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.score_dim + self.guide_dim,
                self.score_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_dim),
            nn.GELU(),
            nn.Conv2d(
                self.score_dim,
                self.score_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_dim),
            nn.GELU(),
        )

        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)

        if guide.shape[-2:] != x_up.shape[-2:]:
            guide = F.interpolate(
                guide,
                size=x_up.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = self.fuse(torch.cat([x_up, guide], dim=1))
        return x_up + self.res_scale * fused


# ---------------------------------------------------------------------------
# ClipScoreEmbeddingBuilder
# ---------------------------------------------------------------------------


class ClipScoreEmbeddingBuilder(nn.Module):
    """Build 3-scale CLIP score embeddings from dynamic CLIP text and image features.

    Optional CLIP mid-layer visual guidance refines the upsampling from
    18→36 and 36→72 with shallow / deep ViT features.

    Input:
        dynamic_clip_text: [B, C, Q, D_clip]
        clip_image_feat_map: [B, D_clip, Hc, Wc]  (Hc=Wc=16 for ViT-L/14)
        clip_mid_features (optional): List[[B, D_native, Hc, Wc]]

    Flow (guided, default):
        clip_image_feat_map → bilinear to 18×18
        → text-image dot product → score_maps_18: [B, C, Q, 18, 18]
        → score_conv_18 → raw_18
        → deep mid (layer 15) → proj → GuideScoreFuse → score_embed_18
        → deep mid (layer 15) → proj → GuidedScoreUpsample → score_embed_36
        → shallow mid (layer 7) → proj → GuidedScoreUpsample → score_embed_72

    Flow (fallback, no mid features):
        → ConvTranspose2d ×1 (learnable 2×) → score_embed_36
        → ConvTranspose2d ×1 (learnable 2×) → score_embed_72
    """

    def __init__(
        self,
        clip_output_dim: int = 768,
        score_embed_dim: int = 128,
        num_query_tokens: int = 32,
        conv_kernel: int = 7,
        base_hw: int = 18,
        clip_mid_dim: int | None = None,
    ):
        super().__init__()
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.base_hw = int(base_hw)
        self.clip_mid_dim = None if clip_mid_dim is None else int(clip_mid_dim)

        padding = int(conv_kernel) // 2

        self.score_conv_18 = nn.Sequential(
            nn.Conv2d(
                self.num_query_tokens,
                self.score_embed_dim,
                kernel_size=int(conv_kernel),
                stride=1,
                padding=padding,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        # --- Fallback upsamplers (used when no mid guidance) ---
        self.fallback_up_18_to_36 = nn.Sequential(
            nn.ConvTranspose2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.fallback_up_36_to_72 = nn.Sequential(
            nn.ConvTranspose2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        # --- Guided path (active when clip_mid_dim is set) ---
        if self.clip_mid_dim is not None:
            self.mid_proj_18 = nn.Sequential(
                nn.Conv2d(
                    self.clip_mid_dim,
                    self.score_embed_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                _safe_group_norm(self.score_embed_dim),
                nn.GELU(),
            )
            self.mid_proj_36 = nn.Sequential(
                nn.Conv2d(
                    self.clip_mid_dim,
                    self.score_embed_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                _safe_group_norm(self.score_embed_dim),
                nn.GELU(),
            )
            self.mid_proj_72 = nn.Sequential(
                nn.Conv2d(
                    self.clip_mid_dim,
                    self.score_embed_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                _safe_group_norm(self.score_embed_dim),
                nn.GELU(),
            )

            self.score_fuse_18 = GuidedScoreFuse(
                score_dim=self.score_embed_dim,
                guide_dim=self.score_embed_dim,
                init_scale=0.0,
            )
            self.guided_up_18_to_36 = GuidedScoreUpsample(
                score_dim=self.score_embed_dim,
                guide_dim=self.score_embed_dim,
                init_scale=0.05,
            )
            self.guided_up_36_to_72 = GuidedScoreUpsample(
                score_dim=self.score_embed_dim,
                guide_dim=self.score_embed_dim,
                init_scale=0.05,
            )
        else:
            self.mid_proj_18 = None
            self.mid_proj_36 = None
            self.mid_proj_72 = None
            self.score_fuse_18 = None
            self.guided_up_18_to_36 = None
            self.guided_up_36_to_72 = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_class_expanded_guide(
        self,
        mid_feature: torch.Tensor,
        projection: nn.Module,
        target_hw: tuple[int, int],
        batch_size: int,
        num_classes: int,
    ) -> torch.Tensor:
        """Project a CLIP mid feature, interpolate, and expand per-class.

        Args:
            mid_feature: [B, D_native, Hc, Wc]
            projection: Conv D_native → D_score
            target_hw: output spatial size
            batch_size: B
            num_classes: C

        Returns:
            guide: [B*C, D_score, target_H, target_W]
        """
        if not isinstance(mid_feature, torch.Tensor) or mid_feature.ndim != 4:
            raise ValueError(
                "CLIP mid feature must be a 4D tensor [B, D_mid, Hc, Wc]."
            )
        if int(mid_feature.shape[0]) != int(batch_size):
            raise ValueError(
                "CLIP mid feature batch mismatch: "
                f"expected {batch_size}, got {mid_feature.shape[0]}."
            )

        guide = projection(mid_feature)

        if tuple(guide.shape[-2:]) != tuple(target_hw):
            guide = F.interpolate(
                guide,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        guide = (
            guide[:, None]
            .expand(
                batch_size,
                num_classes,
                self.score_embed_dim,
                target_hw[0],
                target_hw[1],
            )
            .reshape(
                batch_size * num_classes,
                self.score_embed_dim,
                target_hw[0],
                target_hw[1],
            )
        )

        return guide.contiguous()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        dynamic_clip_text: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        clip_mid_features: list[torch.Tensor] | None = None,
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            dynamic_clip_text:  [B, C, Q, D_clip]
            clip_image_feat_map: [B, D_clip, Hc, Wc]
            clip_mid_features:  optional list of [B, D_native, Hc, Wc]

        Returns:
            clip_score_embeds:
                {
                    "scale_18": [B, C, D_score, 18, 18],
                    "scale_36": [B, C, D_score, 36, 36],
                    "scale_72": [B, C, D_score, 72, 72],
                }
            clip_score_maps_18: [B, C, Q, 18, 18]
        """
        batch_size, num_classes, num_queries, clip_dim = dynamic_clip_text.shape
        image_batch_size, image_clip_dim, _, _ = clip_image_feat_map.shape

        if image_batch_size != batch_size:
            raise ValueError(
                f"clip_image_feat_map batch mismatch: "
                f"expected {batch_size}, got {image_batch_size}."
            )
        if clip_dim != image_clip_dim:
            raise ValueError(
                f"CLIP dimension mismatch: text={clip_dim}, image={image_clip_dim}."
            )
        if num_queries != self.num_query_tokens:
            raise ValueError(
                f"Query count mismatch: expected {self.num_query_tokens}, got {num_queries}."
            )

        # --- Score maps from text-image cosine similarity at 18×18 ---

        clip_image_feat_18 = F.interpolate(
            clip_image_feat_map,
            size=(self.base_hw, self.base_hw),
            mode="bilinear",
            align_corners=False,
        )

        text_norm = F.normalize(dynamic_clip_text, dim=-1)
        image_norm = F.normalize(clip_image_feat_18, dim=1)

        text_flat = text_norm.reshape(
            batch_size * num_classes * num_queries,
            clip_dim,
        )

        image_expanded = (
            image_norm[:, None, None]
            .expand(
                batch_size,
                num_classes,
                num_queries,
                image_clip_dim,
                self.base_hw,
                self.base_hw,
            )
            .reshape(
                batch_size * num_classes * num_queries,
                image_clip_dim,
                self.base_hw * self.base_hw,
            )
        )

        score_maps_18 = torch.bmm(
            text_flat.unsqueeze(1),
            image_expanded,
        ).reshape(
            batch_size,
            num_classes,
            num_queries,
            self.base_hw,
            self.base_hw,
        ) * 20.0

        # --- 18×18: base conv → raw embedding ---

        score_embed_18_flat = self.score_conv_18(
            score_maps_18.reshape(
                batch_size * num_classes,
                num_queries,
                self.base_hw,
                self.base_hw,
            )
        )

        # --- Upsample: guided or fallback ---

        use_mid_guidance = (
            clip_mid_features is not None
            and self.clip_mid_dim is not None
            and self.score_fuse_18 is not None
            and len(clip_mid_features) >= 2
        )

        if use_mid_guidance:
            # clip_mid_features[0] = shallow (e.g. layer 7)
            # clip_mid_features[1] = deep   (e.g. layer 15)
            mid_shallow = clip_mid_features[0]
            mid_deep = clip_mid_features[1]

            # 18×18: fuse deep mid guidance
            guide_18 = self._build_class_expanded_guide(
                mid_feature=mid_deep,
                projection=self.mid_proj_18,
                target_hw=(self.base_hw, self.base_hw),
                batch_size=batch_size,
                num_classes=num_classes,
            )
            score_embed_18_flat = self.score_fuse_18(
                score_embed_18_flat,
                guide_18,
            )

            # 18→36: upsample with deep mid guidance
            guide_36 = self._build_class_expanded_guide(
                mid_feature=mid_deep,
                projection=self.mid_proj_36,
                target_hw=(self.base_hw * 2, self.base_hw * 2),
                batch_size=batch_size,
                num_classes=num_classes,
            )
            score_embed_36_flat = self.guided_up_18_to_36(
                score_embed_18_flat,
                guide_36,
            )

            # 36→72: upsample with shallow mid guidance
            guide_72 = self._build_class_expanded_guide(
                mid_feature=mid_shallow,
                projection=self.mid_proj_72,
                target_hw=(self.base_hw * 4, self.base_hw * 4),
                batch_size=batch_size,
                num_classes=num_classes,
            )
            score_embed_72_flat = self.guided_up_36_to_72(
                score_embed_36_flat,
                guide_72,
            )
        else:
            score_embed_36_flat = self.fallback_up_18_to_36(score_embed_18_flat)
            score_embed_72_flat = self.fallback_up_36_to_72(score_embed_36_flat)

        # --- Reshape to [B, C, D_score, H, W] ---

        clip_score_embed_18 = score_embed_18_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            self.base_hw,
            self.base_hw,
        ).contiguous()

        clip_score_embed_36 = score_embed_36_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            self.base_hw * 2,
            self.base_hw * 2,
        ).contiguous()

        clip_score_embed_72 = score_embed_72_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            self.base_hw * 4,
            self.base_hw * 4,
        ).contiguous()

        return {
            "scale_18": clip_score_embed_18,
            "scale_36": clip_score_embed_36,
            "scale_72": clip_score_embed_72,
        }, score_maps_18.contiguous()
