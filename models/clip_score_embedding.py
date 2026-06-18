from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


class ClipScoreEmbeddingBuilder(nn.Module):
    """
    Build 3-scale CLIP score embeddings from multi-template CLIP text and image features.

    Flow:
        class_names + 16 prompt_templates
        → OpenCLIP text encoder
        → template_text_features [C, 16, D_clip]

        clip_image_feat_map → interpolate to 18×18
        → text-image dot product with template_text_features
        → score_maps_18 [B, C, 16, 18, 18]

        score_maps_18 → 7×7 conv blocks → score_embed_18 [B, C, D_score, 18, 18]
        + CLIP final image feat (proj) → inject → score_embed_18

        score_embed_18 + CLIP layer 15 feat (proj) → fuse → ConvTranspose2d
        → score_embed_36 [B, C, D_score, 36, 36]

        score_embed_36 + CLIP layer 7 feat (proj) → fuse → ConvTranspose2d
        → score_embed_72 [B, C, D_score, 72, 72]
    """

    def __init__(
        self,
        clip_text_encoder,
        clip_output_dim: int = 768,
        clip_mid_dim: int = 1024,
        score_embed_dim: int = 128,
        prompt_templates: list[str] | tuple[str, ...] = (),
        normalize_label_for_clip: bool = True,
        conv_kernel: int = 7,
        base_hw: int = 18,
        use_checkpoint: bool = True,
    ):
        super().__init__()

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.clip_output_dim = int(clip_output_dim)
        self.clip_mid_dim = int(clip_mid_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.prompt_templates = tuple(prompt_templates)
        self.normalize_label_for_clip = bool(normalize_label_for_clip)
        self.base_hw = int(base_hw)
        self.use_checkpoint = bool(use_checkpoint)

        num_templates = len(self.prompt_templates)
        if num_templates == 0:
            raise ValueError("prompt_templates must not be empty.")

        padding = int(conv_kernel) // 2

        half_dim = self.score_embed_dim // 2

        self.score_from_templates = nn.Sequential(
            nn.Conv2d(num_templates, half_dim, kernel_size=7, padding=3, bias=False),
            _safe_group_norm(half_dim),
            nn.GELU(),
            nn.Conv2d(half_dim, self.score_embed_dim, kernel_size=7, padding=3, bias=False),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.clip_final_proj = nn.Sequential(
            nn.Conv2d(self.clip_output_dim, self.score_embed_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.clip_mid15_proj = nn.Sequential(
            nn.Conv2d(self.clip_mid_dim, self.score_embed_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.clip_mid7_proj = nn.Sequential(
            nn.Conv2d(self.clip_mid_dim, self.score_embed_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.inject_clip_final_18 = nn.Sequential(
            nn.Conv2d(self.score_embed_dim * 2, self.score_embed_dim, kernel_size=7, padding=3, bias=False),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.fuse_mid15_before_up = nn.Sequential(
            nn.Conv2d(self.score_embed_dim * 2, self.score_embed_dim, kernel_size=7, padding=3, bias=False),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.fuse_mid7_before_up = nn.Sequential(
            nn.Conv2d(self.score_embed_dim * 2, self.score_embed_dim, kernel_size=7, padding=3, bias=False),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.score_up_18_to_36 = nn.Sequential(
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

        self.score_up_36_to_72 = nn.Sequential(
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

    def _expand_per_class(
        self,
        feat: torch.Tensor,
        batch_size: int,
        num_classes: int,
    ) -> torch.Tensor:
        """[B, D, H, W] → [B*C, D, H, W]"""
        return (
            feat[:, None]
            .expand(batch_size, num_classes, feat.shape[1], feat.shape[2], feat.shape[3])
            .reshape(batch_size * num_classes, feat.shape[1], feat.shape[2], feat.shape[3])
        )

    def forward(
        self,
        class_names: list[str],
        clip_image_feat_map: torch.Tensor,
        clip_mid_features: list[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            class_names:             length C
            clip_image_feat_map:     [B, D_clip, Hc, Wc]
            clip_mid_features:       list of [B, D_native, Hc, Wc], must contain layers 7 and 15
            clip_mid_layer_indices:  e.g. (7, 15)

        Returns:
            clip_score_embeds:
                {
                    "scale_18": [B, C, D_score, 18, 18],
                    "scale_36": [B, C, D_score, 36, 36],
                    "scale_72": [B, C, D_score, 72, 72],
                }
            score_maps_18: [B, C, 16, 18, 18]
        """
        batch_size = int(clip_image_feat_map.shape[0])
        num_classes = len(class_names)
        num_templates = len(self.prompt_templates)

        if num_classes == 0:
            raise ValueError("class_names is empty.")

        mid_by_layer = {
            int(layer_idx): feat
            for layer_idx, feat in zip(clip_mid_layer_indices, clip_mid_features)
        }

        if 15 not in mid_by_layer:
            raise ValueError(
                f"clip_mid_features must contain layer 15 feature, "
                f"got layers {sorted(mid_by_layer.keys())}."
            )
        if 7 not in mid_by_layer:
            raise ValueError(
                f"clip_mid_features must contain layer 7 feature, "
                f"got layers {sorted(mid_by_layer.keys())}."
            )

        # --- Step 1: encode 16-template text ---
        trainable_text = any(
            p.requires_grad for p in self.clip_text_encoder.parameters()
        )
        use_text_cache = (not self.training) or (not trainable_text)

        template_text = self.clip_text_encoder.encode_prompt_templates(
            class_names=class_names,
            templates=list(self.prompt_templates),
            device=clip_image_feat_map.device,
            normalize_label=self.normalize_label_for_clip,
            normalize=True,
            use_cache=use_text_cache,
            detach_output=use_text_cache,
            use_checkpoint=self.training and self.use_checkpoint and trainable_text,
        )

        if tuple(template_text.shape) != (num_classes, num_templates, self.clip_output_dim):
            raise ValueError(
                f"template_text shape mismatch: "
                f"expected ({num_classes}, {num_templates}, {self.clip_output_dim}), "
                f"got {tuple(template_text.shape)}."
            )

        # --- Step 2: CLIP image feature → interpolate to 18×18 ---
        clip_image_feat_18 = F.interpolate(
            clip_image_feat_map,
            size=(self.base_hw, self.base_hw),
            mode="bilinear",
            align_corners=False,
        )

        # --- Step 3: text-image dot product → 16-channel score maps ---
        text_norm = F.normalize(template_text, dim=-1)
        image_norm = F.normalize(clip_image_feat_18, dim=1)

        score_maps_18 = torch.einsum(
            "ctd,bdhw->bcthw",
            text_norm.to(dtype=image_norm.dtype),
            image_norm,
        ) * 20.0

        if tuple(score_maps_18.shape) != (batch_size, num_classes, num_templates, 18, 18):
            raise ValueError(
                f"score_maps_18 shape mismatch: "
                f"expected ({batch_size}, {num_classes}, {num_templates}, 18, 18), "
                f"got {tuple(score_maps_18.shape)}."
            )

        D_score = self.score_embed_dim

        # --- Step 4: score maps → conv blocks → score_embed_18 ---
        score_flat = score_maps_18.reshape(batch_size * num_classes, num_templates, 18, 18)
        score_embed_18_flat = self.score_from_templates(score_flat)

        # --- Step 5: inject CLIP final image feature ---
        clip_final_proj = self.clip_final_proj(clip_image_feat_18)
        clip_final_expanded = self._expand_per_class(clip_final_proj, batch_size, num_classes)

        score_embed_18_flat = self.inject_clip_final_18(
            torch.cat([score_embed_18_flat, clip_final_expanded], dim=1)
        )

        clip_score_embed_18 = score_embed_18_flat.reshape(
            batch_size, num_classes, D_score, 18, 18,
        ).contiguous()

        # --- Step 6: 18→36, fuse layer 15 before upsampling ---
        mid15 = F.interpolate(
            mid_by_layer[15],
            size=(18, 18),
            mode="bilinear",
            align_corners=False,
        )
        mid15 = self.clip_mid15_proj(mid15)
        mid15_expanded = self._expand_per_class(mid15, batch_size, num_classes)

        score_embed_18_for_up = self.fuse_mid15_before_up(
            torch.cat([score_embed_18_flat, mid15_expanded], dim=1)
        )
        score_embed_36_flat = self.score_up_18_to_36(score_embed_18_for_up)

        clip_score_embed_36 = score_embed_36_flat.reshape(
            batch_size, num_classes, D_score, 36, 36,
        ).contiguous()

        # --- Step 7: 36→72, fuse layer 7 before upsampling ---
        mid7 = F.interpolate(
            mid_by_layer[7],
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        )
        mid7 = self.clip_mid7_proj(mid7)
        mid7_expanded = self._expand_per_class(mid7, batch_size, num_classes)

        score_embed_36_for_up = self.fuse_mid7_before_up(
            torch.cat([score_embed_36_flat, mid7_expanded], dim=1)
        )
        score_embed_72_flat = self.score_up_36_to_72(score_embed_36_for_up)

        clip_score_embed_72 = score_embed_72_flat.reshape(
            batch_size, num_classes, D_score, 72, 72,
        ).contiguous()

        return {
            "scale_18": clip_score_embed_18,
            "scale_36": clip_score_embed_36,
            "scale_72": clip_score_embed_72,
        }, score_maps_18.contiguous()
