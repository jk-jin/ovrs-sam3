from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_L2_NORM_EPS = 1e-6
_SCORE_SCALE = 20.0
_NUM_PROMPT_TEMPLATES = 64


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_channels = int(num_channels)
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


class _ScoreConditionBranch(nn.Module):
    """Fuse scores with a pre-normalized CLIP condition using one spatial conv."""

    def __init__(self, clip_dim: int, embed_dim: int):
        super().__init__()
        # 64-channel template score maps → 256-channel intermediate feature 1.
        self.score_stem = nn.Sequential(
            nn.Conv2d(
                _NUM_PROMPT_TEMPLATES,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(embed_dim),
            nn.GELU(),
        )

        # Normalized score feature + normalized image or text condition.
        self.condition_fusion = nn.Sequential(
            nn.Conv2d(
                embed_dim + clip_dim,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(embed_dim),
            nn.GELU(),
        )

        # Normalized intermediate feature 1 + normalized intermediate feature 2.
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(
                embed_dim * 2,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(embed_dim),
            nn.GELU(),
        )

    def forward(self, scores: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        score_feature = F.normalize(
            self.score_stem(scores), dim=1, eps=_L2_NORM_EPS,
        )
        fused = self.condition_fusion(torch.cat([score_feature, condition], dim=1))
        fused = F.normalize(fused, dim=1, eps=_L2_NORM_EPS)
        return self.spatial_fusion(torch.cat([score_feature, fused], dim=1))


class ClipScoreEmbedding(nn.Module):
    """Build a 36×36 score embedding from RemoteCLIP text scores and features.

    Inputs:
        remoteclip_feat_map: [B, D_clip, 36, 36]
        template text:       [C, 64, D_clip]

    Process:
        1. Compute 64 normalized template score maps.
        2. Independently fuse scores with dense image features and with the
           mean pooled template text, broadcast over the batch and 36×36 grid.
        3. Each branch has its own score stem, condition fusion, and a single
           3×3 spatial convolution, with channel L2 normalization before fusion.
        4. Concatenate both branch outputs and apply a bare 1×1 convolution.

    Outputs:
        clip_score_embed_36: [B, C, 256, 36, 36]
        score_maps_36:       [B, C, 64, 36, 36]
        template_clip_text:  [C, 64, D_clip]
    """

    def __init__(
        self,
        clip_text_encoder,
        prompt_templates: list[str],
        normalize_label: bool = True,
        clip_output_dim: int = 768,
        score_embed_dim: int = 256,
        text_prompt_batch_size: int = 64,
        text_prompt_use_checkpoint: bool = True,
    ):
        super().__init__()

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.prompt_templates = list(prompt_templates)
        self.normalize_label = bool(normalize_label)
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_prompt_templates = len(self.prompt_templates)
        self.text_prompt_batch_size = int(text_prompt_batch_size)
        self.text_prompt_use_checkpoint = bool(
            text_prompt_use_checkpoint
        )

        if self.num_prompt_templates != _NUM_PROMPT_TEMPLATES:
            raise ValueError(
                f"Expected {_NUM_PROMPT_TEMPLATES} prompt templates, "
                f"got {self.num_prompt_templates}."
            )

        if self.clip_output_dim <= 0:
            raise ValueError(
                "clip_output_dim must be positive, "
                f"got {self.clip_output_dim}."
            )

        if self.score_embed_dim <= 0:
            raise ValueError(
                "score_embed_dim must be positive, "
                f"got {self.score_embed_dim}."
            )

        self.image_branch = _ScoreConditionBranch(
            self.clip_output_dim, self.score_embed_dim,
        )
        self.text_branch = _ScoreConditionBranch(
            self.clip_output_dim, self.score_embed_dim,
        )
        self.output_fusion = nn.Conv2d(
            self.score_embed_dim * 2, self.score_embed_dim, kernel_size=1,
        )

        self._text_feature_cache: dict[tuple, torch.Tensor] = {}

    def _has_trainable_clip_text_params(self) -> bool:
        return any(
            parameter.requires_grad
            for parameter in self.clip_text_encoder.parameters()
        )

    def _make_text_cache_key(
        self,
        class_names: list[str],
        device: torch.device,
    ) -> tuple:
        return tuple(class_names), str(device)

    def clear_text_cache(self) -> None:
        self._text_feature_cache.clear()

    def _encode_template_text(
        self,
        class_names: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        trainable = self._has_trainable_clip_text_params()
        grad_enabled = torch.is_grad_enabled()
        cache_allowed = (not trainable) or (not grad_enabled)

        if cache_allowed:
            cache_key = self._make_text_cache_key(
                class_names=class_names,
                device=device,
            )
            cached = self._text_feature_cache.get(cache_key)
            if cached is not None:
                return cached.to(device=device)

        result = self.clip_text_encoder.encode_prompt_templates(
            class_names=class_names,
            templates=self.prompt_templates,
            device=device,
            normalize_label=self.normalize_label,
            normalize=False,
            prompt_batch_size=self.text_prompt_batch_size,
            use_checkpoint=self.text_prompt_use_checkpoint,
        )

        if cache_allowed:
            cached = result.detach().contiguous()
            self._text_feature_cache[cache_key] = cached
            return cached.to(device=device)

        return result

    def forward(
        self,
        class_names: list[str],
        remoteclip_feat_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if remoteclip_feat_map.ndim != 4:
            raise ValueError(
                "remoteclip_feat_map must be [B, D_clip, H, W], "
                f"got {tuple(remoteclip_feat_map.shape)}."
            )

        batch_size, image_clip_dim, height, width = (
            remoteclip_feat_map.shape
        )

        if image_clip_dim != self.clip_output_dim:
            raise ValueError(
                "CLIP dimension mismatch: expected "
                f"{self.clip_output_dim}, got {image_clip_dim}."
            )

        if (height, width) != (36, 36):
            raise ValueError(
                "Expected a 36×36 RemoteCLIP feature map, "
                f"got {(height, width)}."
            )

        num_classes = len(class_names)
        if num_classes == 0:
            raise ValueError("class_names is empty.")

        template_clip_text = self._encode_template_text(
            class_names=class_names,
            device=remoteclip_feat_map.device,
        )

        expected_text_shape = (
            num_classes,
            self.num_prompt_templates,
            self.clip_output_dim,
        )
        if tuple(template_clip_text.shape) != expected_text_shape:
            raise ValueError(
                "template_clip_text shape mismatch: expected "
                f"{expected_text_shape}, "
                f"got {tuple(template_clip_text.shape)}."
            )

        template_clip_text = template_clip_text.to(
            device=remoteclip_feat_map.device,
            dtype=remoteclip_feat_map.dtype,
        )

        # Normalize every template text vector along D_clip.
        text_norm = F.normalize(
            template_clip_text,
            p=2,
            dim=-1,
            eps=_L2_NORM_EPS,
        )

        # Normalize every spatial CLIP vector along D_clip.
        image_norm = F.normalize(
            remoteclip_feat_map,
            p=2,
            dim=1,
            eps=_L2_NORM_EPS,
        )

        score_maps_36 = (
            torch.einsum(
                "ckd,bdhw->bckhw",
                text_norm,
                image_norm,
            )
            * _SCORE_SCALE
        )

        score_flat = score_maps_36.reshape(
            batch_size * num_classes,
            self.num_prompt_templates,
            height,
            width,
        )

        # The class dimension is shared with score_flat's [batch, class] order.
        # Normalize image/text conditions before broadcasting to avoid repeats.
        image_condition = (
            image_norm[:, None]
            .expand(batch_size, num_classes, self.clip_output_dim, height, width)
            .reshape(batch_size * num_classes, self.clip_output_dim, height, width)
        )

        # Average raw pooled sentence embeddings across templates only.
        # Normalize before broadcasting, and preserve the text encoder graph.
        text_mean = F.normalize(
            template_clip_text.mean(dim=1), dim=-1, eps=_L2_NORM_EPS,
        )
        text_condition = (
            text_mean[None, :, :, None, None]
            .expand(batch_size, num_classes, self.clip_output_dim, height, width)
            .reshape(batch_size * num_classes, self.clip_output_dim, height, width)
        )

        image_features = self.image_branch(score_flat, image_condition)
        text_features = self.text_branch(score_flat, text_condition)
        clip_score_flat = self.output_fusion(
            torch.cat([image_features, text_features], dim=1)
        )

        clip_score_embed_36 = clip_score_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            height,
            width,
        ).contiguous()

        return (
            clip_score_embed_36,
            score_maps_36.contiguous(),
            template_clip_text.contiguous(),
        )
