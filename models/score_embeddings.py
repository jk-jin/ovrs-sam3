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


# ---------------------------------------------------------------------------
# Multi-scale score encoder
# ---------------------------------------------------------------------------


class _ScoreConvBranch(nn.Module):
    """Depthwise 3×3 conv branch with dilation for multi-scale receptive fields.

    Depthwise Conv 3×3 (dilation) → GroupNorm → GELU
    → Pointwise Conv 1×1 → GroupNorm → GELU
    """

    def __init__(
        self,
        channels: int,
        branch_channels: int,
        dilation: int,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm1 = _safe_group_norm(channels)
        self.act1 = nn.GELU()

        self.pointwise = nn.Conv2d(
            channels,
            branch_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm2 = _safe_group_norm(branch_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pointwise(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class MultiScaleScoreEncoder(nn.Module):
    """Encode 32-template score maps into a 256-channel embedding via multi-scale conv.

    32 templates → 1×1 stem (256 ch)
    → three parallel depthwise branches (dilations 1, 2, 3; each 256→128)
    → concat (384 ch) → 1×1 fuse (384→256)
    → residual add with stem → GroupNorm + GELU
    """

    def __init__(
        self,
        num_templates: int = 32,
        score_embed_dim: int = 256,
    ):
        super().__init__()

        if score_embed_dim <= 0 or score_embed_dim % 2 != 0:
            raise ValueError(
                f"score_embed_dim must be a positive even integer, got {score_embed_dim}."
            )

        branch_channels = score_embed_dim // 2

        self.stem = nn.Sequential(
            nn.Conv2d(
                num_templates,
                score_embed_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(score_embed_dim),
            nn.GELU(),
        )

        self.branches = nn.ModuleList([
            _ScoreConvBranch(
                channels=score_embed_dim,
                branch_channels=branch_channels,
                dilation=1,
            ),
            _ScoreConvBranch(
                channels=score_embed_dim,
                branch_channels=branch_channels,
                dilation=2,
            ),
            _ScoreConvBranch(
                channels=score_embed_dim,
                branch_channels=branch_channels,
                dilation=3,
            ),
        ])

        fused_channels = branch_channels * 3

        self.fuse = nn.Conv2d(
            fused_channels,
            score_embed_dim,
            kernel_size=1,
            bias=False,
        )

        self.output_norm = _safe_group_norm(score_embed_dim)
        self.output_act = nn.GELU()

    def forward(self, score_maps: torch.Tensor) -> torch.Tensor:
        if score_maps.ndim != 4:
            raise ValueError(
                f"score_maps must be 4D [N, C_in, H, W], got shape {tuple(score_maps.shape)}."
            )
        if score_maps.shape[1] != 32:
            raise ValueError(
                f"score_maps must have 32 input channels, got {score_maps.shape[1]}."
            )

        base = self.stem(score_maps)

        branch_outputs = [
            branch(base)
            for branch in self.branches
        ]

        multiscale = torch.cat(branch_outputs, dim=1)
        delta = self.fuse(multiscale)

        output = self.output_norm(base + delta)
        output = self.output_act(output)

        return output


# ---------------------------------------------------------------------------
# ClipScoreEmbedding
# ---------------------------------------------------------------------------


class ClipScoreEmbedding(nn.Module):
    """
    Build CLIP-based score embedding at 36×36 from RemoteCLIP dense features.

    Input:
        remoteclip_feat_map: [B, D_clip, 36, 36]
        template_text:       [C, 32, D_clip]

    Process:
        template_text × remoteclip_feat_map → score_maps [B, C, 32, 36, 36]
        → multi-scale score encoder → clip_score_embed [B, C, 256, 36, 36]
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
        self.text_prompt_use_checkpoint = bool(text_prompt_use_checkpoint)

        if self.num_prompt_templates != 32:
            raise ValueError(
                f"Expected 32 prompt templates, got {self.num_prompt_templates}."
            )

        self.score_encoder = MultiScaleScoreEncoder(
            num_templates=self.num_prompt_templates,
            score_embed_dim=self.score_embed_dim,
        )

        self._text_feature_cache: dict[tuple, torch.Tensor] = {}

    def _has_trainable_clip_text_params(self) -> bool:
        return any(
            p.requires_grad for p in self.clip_text_encoder.parameters()
        )

    def _make_text_cache_key(
        self, class_names: list[str], device: torch.device
    ) -> tuple:
        return (tuple(class_names), str(device))

    def clear_text_cache(self) -> None:
        self._text_feature_cache.clear()

    def _encode_template_text(
        self, class_names: list[str], device: torch.device
    ) -> torch.Tensor:
        trainable = self._has_trainable_clip_text_params()
        grad_enabled = torch.is_grad_enabled()
        cache_allowed = (not trainable) or (not grad_enabled)

        if cache_allowed:
            cache_key = self._make_text_cache_key(class_names, device)
            if cache_key in self._text_feature_cache:
                return self._text_feature_cache[cache_key].to(device=device)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            class_names:          list of C class names.
            remoteclip_feat_map:  [B, D_clip, 36, 36]

        Returns:
            clip_score_embed_36: [B, C, score_embed_dim, 36, 36]
            score_maps_36:       [B, C, 32, 36, 36]
            template_clip_text:  [C, 32, D_clip]
        """
        batch_size, image_clip_dim, H, W = remoteclip_feat_map.shape

        if image_clip_dim != self.clip_output_dim:
            raise ValueError(
                f"CLIP dimension mismatch: expected {self.clip_output_dim}, "
                f"got {image_clip_dim}."
            )
        if (H, W) != (36, 36):
            raise ValueError(
                f"Expected 36×36 feature map, got {(H, W)}."
            )

        num_classes = len(class_names)
        if num_classes == 0:
            raise ValueError("class_names is empty.")

        template_clip_text = self._encode_template_text(
            class_names=class_names,
            device=remoteclip_feat_map.device,
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

        text_norm = F.normalize(
            template_clip_text.to(
                device=remoteclip_feat_map.device,
                dtype=remoteclip_feat_map.dtype,
            ),
            dim=-1,
        )
        image_norm = F.normalize(remoteclip_feat_map, dim=1)

        score_maps_36 = torch.einsum(
            "ckd,bdhw->bckhw",
            text_norm,
            image_norm,
        ) * 20.0

        score_flat = score_maps_36.reshape(
            batch_size * num_classes,
            self.num_prompt_templates,
            H,
            W,
        )

        clip_score_flat = self.score_encoder(score_flat)

        clip_score_embed_36 = clip_score_flat.reshape(
            batch_size, num_classes, self.score_embed_dim, H, W
        ).contiguous()

        return (
            clip_score_embed_36,
            score_maps_36.contiguous(),
            template_clip_text.contiguous(),
        )