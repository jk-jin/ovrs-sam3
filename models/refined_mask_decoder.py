from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConcatUpsampleBlock(nn.Module):
    """Upsample low_feature and fuse with skip_feature via concat + two convs.

    Input:
        low_feature  [N, 256, H, W]
        skip_feature [N, 256, 2H, 2W]

    Output:
        [N, 256, 2H, 2W]
    """

    def __init__(self):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(512, 256, kernel_size=1, bias=True)
        self.conv_3x3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.norm = nn.GroupNorm(8, 256)

        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            # Kaiming init for ReLU activations
            nn.init.kaiming_normal_(self.conv_1x1.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.conv_1x1.bias)
            nn.init.kaiming_normal_(self.conv_3x3.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.conv_3x3.bias)
            # GroupNorm weight=1, bias=0 by default, confirm it
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(
        self,
        low_feature: torch.Tensor,
        skip_feature: torch.Tensor,
    ) -> torch.Tensor:
        if low_feature.ndim != 4:
            raise ValueError(
                f"low_feature must be [N, C, H, W], got {tuple(low_feature.shape)}."
            )
        if skip_feature.ndim != 4:
            raise ValueError(
                f"skip_feature must be [N, C, H, W], got {tuple(skip_feature.shape)}."
            )

        N_low, C_low, H_low, W_low = low_feature.shape
        N_skip, C_skip, H_skip, W_skip = skip_feature.shape

        if C_low != 256 or C_skip != 256:
            raise ValueError(
                f"Both low_feature and skip_feature must have 256 channels, "
                f"got {C_low} and {C_skip}."
            )
        if N_low != N_skip:
            raise ValueError(
                f"Batch/pair count mismatch: low_feature has {N_low}, "
                f"skip_feature has {N_skip}."
            )
        if H_skip != 2 * H_low or W_skip != 2 * W_low:
            raise ValueError(
                f"skip_feature spatial size must be 2× low_feature, "
                f"got low=({H_low},{W_low}), skip=({H_skip},{W_skip})."
            )

        upsampled = F.interpolate(
            low_feature,
            size=(H_skip, W_skip),
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat([upsampled, skip_feature], dim=1)
        fused = self.conv_1x1(fused)
        fused = self.conv_3x3(fused)
        fused = self.norm(fused)
        fused = F.relu(fused, inplace=False)

        return fused


class RefinerMaskDecoder(nn.Module):
    """Multi-scale mask decoder that upsamples refiner features in three stages.

    36×36 → fusion_72 → 72×72
    72×72 → fusion_144 → 144×144
    144×144 → fusion_288 → 288×288
    288×288 → mask_head → [N, 1, 288, 288]
    """

    def __init__(self, hidden_dim: int = 256, use_checkpoint: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)

        self.fusion_72 = ConcatUpsampleBlock()
        self.fusion_144 = ConcatUpsampleBlock()
        self.fusion_288 = ConcatUpsampleBlock()
        self.mask_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True)

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        original_feature_72: torch.Tensor,
        original_feature_144: torch.Tensor,
        original_feature_288: torch.Tensor,
    ) -> torch.Tensor:
        N, C_36, H_36, W_36 = refiner_feature_36.shape
        if (C_36, H_36, W_36) != (256, 36, 36):
            raise ValueError(
                f"refiner_feature_36 must be [N, 256, 36, 36], "
                f"got {tuple(refiner_feature_36.shape)}."
            )

        for name, feat in [
            ("original_feature_72", original_feature_72),
            ("original_feature_144", original_feature_144),
            ("original_feature_288", original_feature_288),
        ]:
            if feat.ndim != 4 or feat.shape[1] != 256:
                raise ValueError(
                    f"{name} must be [N, 256, H, W], "
                    f"got {tuple(feat.shape)}."
                )
            if feat.shape[0] != N:
                raise ValueError(
                    f"{name} batch/pair count {feat.shape[0]} != refiner N={N}."
                )

        if tuple(original_feature_72.shape[-2:]) != (72, 72):
            raise ValueError(
                f"original_feature_72 must be 72×72, "
                f"got {tuple(original_feature_72.shape[-2:])}."
            )
        if tuple(original_feature_144.shape[-2:]) != (144, 144):
            raise ValueError(
                f"original_feature_144 must be 144×144, "
                f"got {tuple(original_feature_144.shape[-2:])}."
            )
        if tuple(original_feature_288.shape[-2:]) != (288, 288):
            raise ValueError(
                f"original_feature_288 must be 288×288, "
                f"got {tuple(original_feature_288.shape[-2:])}."
            )

        if self.use_checkpoint and self.training:
            feature_72 = checkpoint(
                self.fusion_72,
                refiner_feature_36,
                original_feature_72,
                use_reentrant=False,
            )
            feature_144 = checkpoint(
                self.fusion_144,
                feature_72,
                original_feature_144,
                use_reentrant=False,
            )
            feature_288 = checkpoint(
                self.fusion_288,
                feature_144,
                original_feature_288,
                use_reentrant=False,
            )
        else:
            feature_72 = self.fusion_72(refiner_feature_36, original_feature_72)
            feature_144 = self.fusion_144(feature_72, original_feature_144)
            feature_288 = self.fusion_288(feature_144, original_feature_288)

        return self.mask_head(feature_288)

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
