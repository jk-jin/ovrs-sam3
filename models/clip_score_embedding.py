from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


class ClipMidFuseUpsample2d(nn.Module):
    """
    将 CLIP 中间层特征投影后与 score embed 拼接，再做融合卷积上采样。

    设计意图：
        score_embed 通道数较小（64），CLIP 中间层通道数很大（1024）。
        先通过 1×1 卷积把 CLIP 中间层降到 32 通道，再与 64 通道 score embed
        在通道维拼接，最后用 3×3 卷积融合。避免高维 CLIP 特征主导融合结果。
    """

    def __init__(
        self,
        score_embed_dim: int,
        clip_native_dim: int,
        clip_mid_proj_dim: int = 32,
    ):
        super().__init__()
        self.score_embed_dim = int(score_embed_dim)
        self.clip_native_dim = int(clip_native_dim)
        self.clip_mid_proj_dim = int(clip_mid_proj_dim)

        self.clip_proj = nn.Sequential(
            nn.Conv2d(self.clip_native_dim, self.clip_mid_proj_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.clip_mid_proj_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                self.score_embed_dim + self.clip_mid_proj_dim,
                self.score_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        score_flat: torch.Tensor,
        clip_mid_feature: torch.Tensor,
        batch_size: int,
        num_classes: int,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        score_up = F.interpolate(
            score_flat,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

        mid = F.interpolate(
            clip_mid_feature.to(device=score_flat.device, dtype=score_flat.dtype),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        mid = self.clip_proj(mid)

        mid = (
            mid[:, None]
            .expand(batch_size, num_classes, self.clip_mid_proj_dim, target_hw[0], target_hw[1])
            .reshape(batch_size * num_classes, self.clip_mid_proj_dim, target_hw[0], target_hw[1])
        )

        return self.fuse(torch.cat([score_up, mid], dim=1))


class ClipScoreEmbeddingBuilder(nn.Module):
    """
    用 32 个固定 prompt 模板构建多尺度 CLIP score embedding。

    输入：
        class_names:         类别名列表，长度 C
        clip_image_feat_map: [B, D_clip, Hc, Wc]   CLIP 图像对齐特征
        clip_mid_features:   CLIP ViT 中间层特征列表
        clip_mid_layer_indices: 中间层编号

    流程：
        class_names × 32 templates → frozen OpenCLIP text encoder
        → template_clip_text [C, 32, D_clip]
        → × clip_image_feat_map (bilinear to 18×18)
        → score_maps_18 [B, C, 32, 18, 18]
        → Conv2d(32 → 64, 7×7) → score_embed_18
        → bilinear + CLIP layer15 fusion → score_embed_36
        → bilinear + CLIP layer7  fusion → score_embed_72
    """

    def __init__(
        self,
        clip_text_encoder,
        prompt_templates: list[str],
        normalize_label: bool = True,
        clip_output_dim: int = 768,
        clip_native_dim: int = 1024,
        score_embed_dim: int = 64,
        conv_kernel: int = 7,
        base_hw: int = 18,
        clip_mid_proj_dim: int = 32,
        clip_mid_layer_for_36: int = 15,
        clip_mid_layer_for_72: int = 7,
    ):
        super().__init__()

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.prompt_templates = list(prompt_templates)
        self.normalize_label = bool(normalize_label)
        self.clip_output_dim = int(clip_output_dim)
        self.clip_native_dim = int(clip_native_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_prompt_templates = len(self.prompt_templates)
        self.base_hw = int(base_hw)
        self.clip_mid_layer_for_36 = int(clip_mid_layer_for_36)
        self.clip_mid_layer_for_72 = int(clip_mid_layer_for_72)

        if self.num_prompt_templates != 32:
            raise ValueError(f"Expected 32 prompt templates, got {self.num_prompt_templates}.")

        padding = int(conv_kernel) // 2

        self.score_conv_18 = nn.Sequential(
            nn.Conv2d(
                self.num_prompt_templates,
                self.score_embed_dim,
                kernel_size=int(conv_kernel),
                stride=1,
                padding=padding,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self.score_up_18_to_36 = ClipMidFuseUpsample2d(
            score_embed_dim=self.score_embed_dim,
            clip_native_dim=self.clip_native_dim,
            clip_mid_proj_dim=clip_mid_proj_dim,
        )
        self.score_up_36_to_72 = ClipMidFuseUpsample2d(
            score_embed_dim=self.score_embed_dim,
            clip_native_dim=self.clip_native_dim,
            clip_mid_proj_dim=clip_mid_proj_dim,
        )

        self._text_feature_cache: dict[tuple, torch.Tensor] = {}

    def _has_trainable_clip_text_params(self) -> bool:
        return any(p.requires_grad for p in self.clip_text_encoder.parameters())

    def _make_text_cache_key(
        self,
        class_names: list[str],
        device: torch.device,
    ) -> tuple:
        return (tuple(class_names), str(device))

    def clear_text_cache(self) -> None:
        self._text_feature_cache.clear()

    def _encode_template_text(
        self,
        class_names: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        trainable = self._has_trainable_clip_text_params()

        if not trainable:
            cache_key = self._make_text_cache_key(class_names, device)
            if cache_key in self._text_feature_cache:
                return self._text_feature_cache[cache_key].to(device=device)

        result = self.clip_text_encoder.encode_prompt_templates(
            class_names=class_names,
            templates=self.prompt_templates,
            device=device,
            normalize_label=self.normalize_label,
            normalize=False,
        )

        if not trainable:
            self._text_feature_cache[cache_key] = result.detach().contiguous()

        return result

    @staticmethod
    def _get_clip_mid_feature(
        clip_mid_features: list[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
        target_layer: int,
    ) -> torch.Tensor:
        indices = tuple(int(x) for x in clip_mid_layer_indices)
        if target_layer not in indices:
            raise ValueError(
                f"Required CLIP intermediate layer {target_layer} not found. "
                f"Available layers: {indices}."
            )
        idx = indices.index(target_layer)
        return clip_mid_features[idx]

    def forward(
        self,
        class_names: list[str],
        clip_image_feat_map: torch.Tensor,
        clip_mid_features: list[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
            clip_score_embeds:
                {
                    "scale_18": [B, C, D_score, 18, 18],
                    "scale_36": [B, C, D_score, 36, 36],
                    "scale_72": [B, C, D_score, 72, 72],
                }
            clip_score_maps_18: [B, C, 32, 18, 18]
            template_clip_text: [C, 32, D_clip]
        """
        batch_size, image_clip_dim, _, _ = clip_image_feat_map.shape

        if image_clip_dim != self.clip_output_dim:
            raise ValueError(
                f"CLIP dimension mismatch: expected {self.clip_output_dim}, "
                f"got {image_clip_dim}."
            )

        num_classes = len(class_names)
        if num_classes == 0:
            raise ValueError("class_names is empty.")

        if not isinstance(clip_mid_features, list):
            raise TypeError("clip_mid_features must be a list of tensors.")

        template_clip_text = self._encode_template_text(
            class_names=class_names,
            device=clip_image_feat_map.device,
        )

        if tuple(template_clip_text.shape) != (
            num_classes,
            self.num_prompt_templates,
            self.clip_output_dim,
        ):
            raise ValueError(
                f"template_clip_text shape mismatch: expected "
                f"({num_classes}, {self.num_prompt_templates}, {self.clip_output_dim}), "
                f"got {tuple(template_clip_text.shape)}."
            )

        clip_image_feat_18 = F.interpolate(
            clip_image_feat_map,
            size=(self.base_hw, self.base_hw),
            mode="bilinear",
            align_corners=False,
        )

        text_norm = F.normalize(
            template_clip_text.to(device=clip_image_feat_map.device, dtype=clip_image_feat_map.dtype),
            dim=-1,
        )
        image_norm = F.normalize(clip_image_feat_18, dim=1)

        score_maps_18 = torch.einsum(
            "ckd,bdhw->bckhw",
            text_norm,
            image_norm,
        ) * 20.0

        score_embed_18_flat = self.score_conv_18(
            score_maps_18.reshape(
                batch_size * num_classes,
                self.num_prompt_templates,
                self.base_hw,
                self.base_hw,
            )
        )

        clip_mid_15 = self._get_clip_mid_feature(
            clip_mid_features,
            clip_mid_layer_indices,
            self.clip_mid_layer_for_36,
        )
        clip_mid_7 = self._get_clip_mid_feature(
            clip_mid_features,
            clip_mid_layer_indices,
            self.clip_mid_layer_for_72,
        )

        score_embed_36_flat = self.score_up_18_to_36(
            score_flat=score_embed_18_flat,
            clip_mid_feature=clip_mid_15,
            batch_size=batch_size,
            num_classes=num_classes,
            target_hw=(self.base_hw * 2, self.base_hw * 2),
        )

        score_embed_72_flat = self.score_up_36_to_72(
            score_flat=score_embed_36_flat,
            clip_mid_feature=clip_mid_7,
            batch_size=batch_size,
            num_classes=num_classes,
            target_hw=(self.base_hw * 4, self.base_hw * 4),
        )

        clip_score_embed_18 = score_embed_18_flat.reshape(
            batch_size, num_classes, self.score_embed_dim, self.base_hw, self.base_hw
        ).contiguous()

        clip_score_embed_36 = score_embed_36_flat.reshape(
            batch_size, num_classes, self.score_embed_dim, self.base_hw * 2, self.base_hw * 2
        ).contiguous()

        clip_score_embed_72 = score_embed_72_flat.reshape(
            batch_size, num_classes, self.score_embed_dim, self.base_hw * 4, self.base_hw * 4
        ).contiguous()

        return {
            "scale_18": clip_score_embed_18,
            "scale_36": clip_score_embed_36,
            "scale_72": clip_score_embed_72,
        }, score_maps_18.contiguous(), template_clip_text.contiguous()
