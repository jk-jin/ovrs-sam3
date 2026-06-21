from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


class TemplateScoreBuilder(nn.Module):
    """
    Builds template-based CLIP score maps and low-res score embeddings.

    Input:
        class_names:      list[str]
        clip_final_map:   [B, D_clip, Hc, Wc]
        prompt_templates: list[str]
        clip_text_encoder: OpenCLIPTextEncoder

    Output:
        template_score_maps_18: [B, C, K, 18, 18]
        lowres_score_embed:     [B, C, 256, 18, 18]
        template_text_features: [C, K, D_clip]
    """

    def __init__(
        self,
        clip_text_encoder: nn.Module,
        prompt_templates: List[str],
        normalize_label_for_clip: bool = True,
        hidden_dim: int = 256,
        num_prompt_templates: int = 32,
    ):
        super().__init__()
        # Use object.__setattr__ to avoid registering clip_text_encoder
        # as a submodule.  Otherwise template_guided_refiner.train()
        # would incorrectly switch CLIP text encoder to train mode.
        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)
        self.prompt_templates = list(prompt_templates)
        self.normalize_label_for_clip = bool(normalize_label_for_clip)
        self.hidden_dim = int(hidden_dim)
        self.num_prompt_templates = int(num_prompt_templates)

        if len(self.prompt_templates) != self.num_prompt_templates:
            raise ValueError(
                f"Expected {self.num_prompt_templates} prompt templates, "
                f"got {len(self.prompt_templates)}."
            )

        K = self.num_prompt_templates
        D_out = self.hidden_dim

        self.score_conv = nn.Sequential(
            nn.Conv2d(K, D_out, kernel_size=7, padding=3),
            _safe_group_norm(D_out),
            nn.GELU(),
            nn.Conv2d(D_out, D_out, kernel_size=3, padding=1),
            _safe_group_norm(D_out),
            nn.GELU(),
        )

    @staticmethod
    def _get_trainable_text_params(text_encoder: nn.Module) -> bool:
        return any(p.requires_grad for p in text_encoder.parameters())

    def forward(
        self,
        class_names: List[str],
        clip_final_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            class_names:    list of C class name strings
            clip_final_map: [B, D_clip, Hc, Wc]

        Returns:
            template_score_maps_18: [B, C, K, 18, 18]
            lowres_score_embed:     [B, C, hidden_dim, 18, 18]
            template_text_features: [C, K, D_clip]
        """
        B, D_clip, Hc, Wc = clip_final_map.shape
        C = len(class_names)

        if C == 0:
            raise ValueError("class_names is empty.")

        has_trainable_text = self._get_trainable_text_params(self.clip_text_encoder)

        template_text_features = self.clip_text_encoder.encode_template_prompts(
            class_names=class_names,
            prompt_templates=self.prompt_templates,
            device=clip_final_map.device,
            normalize_label=self.normalize_label_for_clip,
            normalize=True,
            use_cache=not has_trainable_text,
            detach_output=not has_trainable_text,
        )
        # [C, K, D_clip]

        if tuple(template_text_features.shape) != (C, self.num_prompt_templates, D_clip):
            raise ValueError(
                f"template_text_features must be [{C}, {self.num_prompt_templates}, {D_clip}], "
                f"got {tuple(template_text_features.shape)}."
            )

        # Interpolate CLIP final map to 18x18.
        clip_final_map_18 = F.interpolate(
            clip_final_map,
            size=(18, 18),
            mode="bilinear",
            align_corners=False,
        )  # [B, D_clip, 18, 18]

        # Normalize and compute dot-product score maps.
        text_norm = F.normalize(template_text_features, dim=-1)  # [C, K, D_clip]
        image_norm = F.normalize(clip_final_map_18, dim=1)       # [B, D_clip, 18, 18]

        template_score_maps_18 = torch.einsum(
            "ckd,bdhw->bckhw",
            text_norm,
            image_norm,
        ) * 20.0  # [B, C, K, 18, 18]

        # Build lowres score embedding via convolution.
        x = template_score_maps_18.reshape(B * C, self.num_prompt_templates, 18, 18)
        x = self.score_conv(x)  # [B*C, hidden_dim, 18, 18]
        lowres_score_embed = x.reshape(B, C, self.hidden_dim, 18, 18)

        return template_score_maps_18, lowres_score_embed, template_text_features
