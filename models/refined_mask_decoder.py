from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PixelDecoderInputFusion72(nn.Module):
    """Fuse refiner 72 and original encoder 72 before the shared Pixel Decoder.

    Input:
        refiner_feature_72   [N, 256, 72, 72]
        original_feature_72  [N, 256, 72, 72]

    Output:
        [N, 256, 72, 72]

    Uses 1×1 Conv + 3×3 Conv without norm or activation so that the
    fused output preserves the value range of the original encoder
    features for the downstream frozen Pixel Decoder.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.conv_1x1 = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
            bias=True,
        )
        self.conv_3x3 = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        nn.init.xavier_uniform_(self.conv_1x1.weight)
        nn.init.zeros_(self.conv_1x1.bias)
        nn.init.xavier_uniform_(self.conv_3x3.weight)
        nn.init.zeros_(self.conv_3x3.bias)

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

        fused = torch.cat(
            [refiner_feature_72, original_feature_72],
            dim=1,
        )
        fused = self.conv_1x1(fused)
        fused = self.conv_3x3(fused)
        return fused


class FinalFeatureFusion288(nn.Module):
    """Fuse two 288×288 features via concat + 1×1 Conv + 3×3 Conv + GroupNorm + ReLU.

    Input:
        refined_feature_288  [N, 256, 288, 288]
        original_feature_288 [N, 256, 288, 288]

    Output:
        [N, 256, 288, 288]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.conv_1x1 = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=1,
            bias=True,
        )
        self.conv_3x3 = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.norm = nn.GroupNorm(8, self.hidden_dim)

        nn.init.kaiming_normal_(
            self.conv_1x1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.conv_1x1.bias)
        nn.init.kaiming_normal_(
            self.conv_3x3.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.conv_3x3.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

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

        fused = torch.cat(
            [refined_feature_288, original_feature_288],
            dim=1,
        )
        fused = self.conv_1x1(fused)
        fused = self.conv_3x3(fused)
        fused = self.norm(fused)
        fused = F.relu(fused, inplace=False)
        return fused


class RefinerMaskDecoder(nn.Module):
    """72×72 input fusion and 288×288 final feature fusion.

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
