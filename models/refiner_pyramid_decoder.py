from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


class PixelDetailInjectionStage(nn.Module):
    """Single-scale detail injection from frozen Pixel Decoder features.

    The 256-channel upsampled refiner feature is the main path and passes
    straight to the output. The frozen Pixel Decoder feature only injects
    detail through a 64-channel compact branch using depthwise 3×3 conv,
    avoiding expensive 3×3 convs on the full 256-channel path.
    """

    def __init__(self, hidden_dim: int = 256, detail_dim: int = 64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.detail_dim = int(detail_dim)

        self.refiner_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.detail_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.detail_dim),
            nn.GELU(),
        )

        self.pixel_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.detail_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.detail_dim),
            nn.GELU(),
        )

        self.detail_block = nn.Sequential(
            nn.Conv2d(
                self.detail_dim,
                self.detail_dim,
                kernel_size=3,
                padding=1,
                groups=self.detail_dim,
                bias=False,
            ),
            _safe_group_norm(self.detail_dim),
            nn.GELU(),
            nn.Conv2d(
                self.detail_dim,
                self.detail_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(self.detail_dim),
            nn.GELU(),
        )

        self.detail_out = nn.Conv2d(
            self.detail_dim,
            self.hidden_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        refiner_feature: torch.Tensor,
        original_pixel_feature: torch.Tensor,
    ) -> torch.Tensor:
        upsampled_refiner = F.interpolate(
            refiner_feature,
            size=original_pixel_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        refiner_compact = self.refiner_proj(upsampled_refiner)
        pixel_compact = self.pixel_proj(original_pixel_feature)

        detail_input = refiner_compact + pixel_compact
        detail_compact = detail_input + self.detail_block(detail_input)
        detail_update = self.detail_out(detail_compact)

        return upsampled_refiner + detail_update


class FinalPixelFeatureFusion288(nn.Module):
    """Gated fusion of refined and original 288×288 features.

    Learns a per-pixel gate to blend the two features, plus a delta head
    for explicit correction. The gate bias is zero-initialised so the
    initial blend is ~0.5. ``delta_head`` is NOT zero-initialised so it
    can actively correct the SAM3 feature from the start.
    """

    def __init__(self, hidden_dim: int = 256, fusion_dim: int = 96):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fusion_dim = int(fusion_dim)

        self.refined_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.fusion_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.fusion_dim),
            nn.GELU(),
        )

        self.original_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.fusion_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.fusion_dim),
            nn.GELU(),
        )

        self.fusion_block = nn.Sequential(
            nn.Conv2d(
                self.fusion_dim * 2,
                self.fusion_dim * 2,
                kernel_size=3,
                padding=1,
                groups=self.fusion_dim * 2,
                bias=False,
            ),
            _safe_group_norm(self.fusion_dim * 2),
            nn.GELU(),
            nn.Conv2d(
                self.fusion_dim * 2,
                self.fusion_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(self.fusion_dim),
            nn.GELU(),
        )

        self.gate_head = nn.Conv2d(
            self.fusion_dim,
            1,
            kernel_size=1,
            bias=True,
        )

        self.delta_head = nn.Conv2d(
            self.fusion_dim,
            self.hidden_dim,
            kernel_size=1,
            bias=False,
        )

        self._init_gate_bias()

    def _init_gate_bias(self) -> None:
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        refined_feature_288: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> torch.Tensor:
        refined_compact = self.refined_proj(refined_feature_288)
        original_compact = self.original_proj(original_feature_288)

        fusion_input = torch.cat(
            [refined_compact, original_compact],
            dim=1,
        )

        fusion_update = self.fusion_block(fusion_input)

        fusion_compact = (
            refined_compact
            + original_compact
            + fusion_update
        )

        gate = torch.sigmoid(self.gate_head(fusion_compact))

        base_feature = (
            gate * refined_feature_288
            + (1.0 - gate) * original_feature_288
        )

        delta = self.delta_head(fusion_compact)

        return base_feature + delta


class RefinerPyramidDecoder(nn.Module):
    """Three-stage lightweight detail injection pyramid + final 288 fusion.

    Stages:
        stage_72:  36→72  + detail from original_pixel_feature_72
        stage_144: 72→144 + detail from original_pixel_feature_144
        stage_288: 144→288 + detail from original_pixel_feature_288
        final_fusion_288: gated base + delta correction at 288

    The entire pyramid is wrapped in a single non-reentrant checkpoint
    during training when ``use_checkpoint=True``.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        detail_dim: int = 64,
        fusion_dim: int = 96,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.stage_72 = PixelDetailInjectionStage(
            hidden_dim=self.hidden_dim,
            detail_dim=int(detail_dim),
        )
        self.stage_144 = PixelDetailInjectionStage(
            hidden_dim=self.hidden_dim,
            detail_dim=int(detail_dim),
        )
        self.stage_288 = PixelDetailInjectionStage(
            hidden_dim=self.hidden_dim,
            detail_dim=int(detail_dim),
        )
        self.final_fusion_288 = FinalPixelFeatureFusion288(
            hidden_dim=self.hidden_dim,
            fusion_dim=int(fusion_dim),
        )

    def _forward_impl(
        self,
        refiner_feature_36: torch.Tensor,
        original_feature_72: torch.Tensor,
        original_feature_144: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> torch.Tensor:
        refined_72 = self.stage_72(
            refiner_feature_36,
            original_feature_72,
        )

        refined_144 = self.stage_144(
            refined_72,
            original_feature_144,
        )

        refined_288 = self.stage_288(
            refined_144,
            original_feature_288,
        )

        return self.final_fusion_288(
            refined_288,
            original_feature_288,
        )

    def _validate_inputs(
        self,
        refiner_feature_36: torch.Tensor,
        original_feature_72: torch.Tensor,
        original_feature_144: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> None:
        """Check shapes before entering checkpoint so errors are clear."""
        expected = [
            (refiner_feature_36, 36, "refiner_feature_36"),
            (original_feature_72, 72, "original_feature_72"),
            (original_feature_144, 144, "original_feature_144"),
            (original_feature_288, 288, "original_feature_288"),
        ]

        N_ref = None
        device_ref = None

        for tensor, hw, name in expected:
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must be [N, 256, {hw}, {hw}], "
                    f"got {tuple(tensor.shape)}."
                )
            N, C, H, W = tensor.shape
            if C != self.hidden_dim:
                raise ValueError(
                    f"{name} channel must be {self.hidden_dim}, got {C}."
                )
            if (H, W) != (hw, hw):
                raise ValueError(
                    f"{name} spatial size must be {hw}×{hw}, got {(H, W)}."
                )
            if N_ref is None:
                N_ref = N
                device_ref = tensor.device
            else:
                if N != N_ref:
                    raise ValueError(
                        f"All pyramid inputs must share N, but "
                        f"{name} has N={N} (expected {N_ref})."
                    )
                if tensor.device != device_ref:
                    raise ValueError(
                        f"All pyramid inputs must be on the same device, "
                        f"but {name} is on {tensor.device} "
                        f"(expected {device_ref})."
                    )

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        original_feature_72: torch.Tensor,
        original_feature_144: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            refiner_feature_36=refiner_feature_36,
            original_feature_72=original_feature_72,
            original_feature_144=original_feature_144,
            original_feature_288=original_feature_288,
        )

        if self.use_checkpoint and self.training:
            return checkpoint(
                self._forward_impl,
                refiner_feature_36,
                original_feature_72,
                original_feature_144,
                original_feature_288,
                use_reentrant=False,
            )
        return self._forward_impl(
            refiner_feature_36=refiner_feature_36,
            original_feature_72=original_feature_72,
            original_feature_144=original_feature_144,
            original_feature_288=original_feature_288,
        )
