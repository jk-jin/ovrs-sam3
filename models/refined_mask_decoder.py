from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
    """Fuse two 288×288 features and predict final mask logits.

    Input:
        refined_feature_288  [N, 256, 288, 288]  — Refiner branch
        original_feature_288 [N, 256, 288, 288]  — frozen original branch

    Output:
        [N, 1, 288, 288]
    """

    def __init__(self, hidden_dim: int = 256, use_checkpoint: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.final_fusion_288 = FinalFeatureFusion288(
            hidden_dim=self.hidden_dim,
        )
        self.mask_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True)

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

        return self.mask_head(fused_feature_288)

    @torch.no_grad()
    def initialize_mask_head_from_semantic_head(
        self,
        semantic_seg_head: nn.Conv2d,
    ) -> None:
        if not isinstance(semantic_seg_head, nn.Conv2d):
            raise TypeError(
                f"semantic_seg_head must be nn.Conv2d, got {type(semantic_seg_head)}."
            )

        src_weight = semantic_seg_head.weight
        src_bias = semantic_seg_head.bias
        dst_weight = self.mask_head.weight
        dst_bias = self.mask_head.bias

        if src_weight.shape != dst_weight.shape:
            raise ValueError(
                f"weight shape mismatch: source {tuple(src_weight.shape)} "
                f"vs destination {tuple(dst_weight.shape)}."
            )
        if src_bias is None or dst_bias is None:
            raise ValueError(
                "Both source and destination Conv2d must have bias."
            )
        if src_bias.shape != dst_bias.shape:
            raise ValueError(
                f"bias shape mismatch: source {tuple(src_bias.shape)} "
                f"vs destination {tuple(dst_bias.shape)}."
            )

        dst_weight.copy_(src_weight)
        dst_bias.copy_(src_bias)
