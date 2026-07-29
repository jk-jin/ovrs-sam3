from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


class PixelDecoderInputFusion72(nn.Module):
    """Fuse refiner 72 and original encoder 72 before the shared Pixel Decoder.

    Stage A (local spatial fusion):
        concat(refiner_72, original_72) [N, 512, 72, 72]
          → 3×3 Conv(512→256, bias=False)
          → GroupNorm → GELU
          → 3×3 Conv(256→256, bias=False)
          → GroupNorm → GELU
          → local_fused_72 [N, 256, 72, 72]

    Stage B (semantic anchor):
        concat(local_fused_72, original_72) [N, 512, 72, 72]
          → 1×1 Conv(512→256, bias=False)
          → fused_pixel_decoder_input_72 [N, 256, 72, 72]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

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

        self.final_fusion = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        refiner_feature_72: torch.Tensor,
        original_feature_72: torch.Tensor,
    ) -> torch.Tensor:
        if refiner_feature_72.ndim != 4:
            raise ValueError(
                f"refiner_feature_72 must be [N, C, H, W], "
                f"got {tuple(refiner_feature_72.shape)}."
            )
        if original_feature_72.ndim != 4:
            raise ValueError(
                f"original_feature_72 must be [N, C, H, W], "
                f"got {tuple(original_feature_72.shape)}."
            )

        if refiner_feature_72.shape != original_feature_72.shape:
            raise ValueError(
                f"Shape mismatch: refiner {tuple(refiner_feature_72.shape)} "
                f"vs original {tuple(original_feature_72.shape)}."
            )

        N, C, H, W = refiner_feature_72.shape
        if C != self.hidden_dim:
            raise ValueError(
                f"Channel count must be {self.hidden_dim}, got {C}."
            )
        if (H, W) != (72, 72):
            raise ValueError(
                f"Spatial size must be 72×72, got {(H, W)}."
            )

        local_input = torch.cat(
            [refiner_feature_72, original_feature_72],
            dim=1,
        )
        local_fused_72 = self.local_fusion(local_input)

        final_input = torch.cat(
            [local_fused_72, original_feature_72],
            dim=1,
        )
        return self.final_fusion(final_input)


class FinalFeatureFusion288(nn.Module):
    """Fuse two 288×288 features via Stage A local fusion + Stage B semantic anchor.

    Stage A (local spatial fusion):
        concat(refined_288, original_288) [N, 512, 288, 288]
          → 3×3 Conv(512→256, bias=False)
          → GroupNorm → GELU
          → 3×3 Conv(256→256, bias=False)
          → GroupNorm → GELU
          → local_fused_288 [N, 256, 288, 288]

    Stage B (semantic anchor):
        concat(local_fused_288, original_288) [N, 512, 288, 288]
          → 1×1 Conv(512→256, bias=False)
          → fused_feature_288 [N, 256, 288, 288]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

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

        self.final_fusion = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        refined_feature_288: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> torch.Tensor:
        if refined_feature_288.ndim != 4:
            raise ValueError(
                f"refined_feature_288 must be [N, C, H, W], "
                f"got {tuple(refined_feature_288.shape)}."
            )
        if original_feature_288.ndim != 4:
            raise ValueError(
                f"original_feature_288 must be [N, C, H, W], "
                f"got {tuple(original_feature_288.shape)}."
            )

        if refined_feature_288.shape != original_feature_288.shape:
            raise ValueError(
                f"Shape mismatch: refined {tuple(refined_feature_288.shape)} "
                f"vs original {tuple(original_feature_288.shape)}."
            )

        N, C, H, W = refined_feature_288.shape
        if C != self.hidden_dim:
            raise ValueError(
                f"Channel count must be {self.hidden_dim}, got {C}."
            )
        if (H, W) != (288, 288):
            raise ValueError(
                f"Spatial size must be 288×288, got {(H, W)}."
            )

        local_input = torch.cat(
            [refined_feature_288, original_feature_288],
            dim=1,
        )
        local_fused_288 = self.local_fusion(local_input)

        final_input = torch.cat(
            [local_fused_288, original_feature_288],
            dim=1,
        )
        return self.final_fusion(final_input)


class RefinerMaskDecoder(nn.Module):
    """72×72 input fusion and 288×288 final feature fusion.

    Both fusion modules use the same Stage A + Stage B structure with
    independent parameters:

    Stage A: 3×3 Conv → GN → GELU → 3×3 Conv → GN → GELU (local spatial)
    Stage B: concat with original feature → 1×1 Conv (semantic anchor)

    Pixel Decoder input path:
        refiner_feature_72 + original_feature_72
          → PixelDecoderInputFusion72
          → fused_pixel_decoder_input_72
          → (external shared frozen Pixel Decoder)
          → refined_pixel_feature_288

    Final 288×288 path (via forward()):
        refined_pixel_feature_288 [N, 256, 288, 288]
        original_pixel_feature_288 [N, 256, 288, 288]
          → FinalFeatureFusion288
          → fused_feature_288 [N, 256, 288, 288]

    The final mask logits are produced externally by the frozen SAM3
    semantic_seg_head applied to the fused 288 feature.
    """

    def __init__(self, hidden_dim: int = 256, use_checkpoint: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.pixel_decoder_input_fusion_72 = (
            PixelDecoderInputFusion72(
                hidden_dim=self.hidden_dim,
            )
        )
        self.final_fusion_288 = FinalFeatureFusion288(
            hidden_dim=self.hidden_dim,
        )

    def fuse_pixel_decoder_input_72(
        self,
        refiner_feature_72: torch.Tensor,
        original_feature_72: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse refiner 72 and original encoder 72 before the Pixel Decoder."""
        if self.use_checkpoint and self.training:
            return checkpoint(
                self.pixel_decoder_input_fusion_72,
                refiner_feature_72,
                original_feature_72,
                use_reentrant=False,
            )
        return self.pixel_decoder_input_fusion_72(
            refiner_feature_72=refiner_feature_72,
            original_feature_72=original_feature_72,
        )

    def forward(
        self,
        refined_feature_288: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            fused_feature_288 = checkpoint(
                self.final_fusion_288,
                refined_feature_288,
                original_feature_288,
                use_reentrant=False,
            )
        else:
            fused_feature_288 = self.final_fusion_288(
                refined_feature_288=refined_feature_288,
                original_feature_288=original_feature_288,
            )

        return fused_feature_288
