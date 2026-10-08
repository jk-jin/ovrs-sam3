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


class DecoderInputFusion(nn.Module):
    """Single-scale dual-branch fusion: semantic (Refiner + encoder)
    and detail (Refiner + original SAM3 FPN).

    Both branches operate in 128-channel compact space with independent
    standard 3×3 conv blocks. Each branch uses its own dedicated Refiner
    projection. After projecting back to 256 channels, the two branches
    are summed with equal weight and passed through a final 1×1 Conv.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        branch_dim: int = 128,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if branch_dim <= 0:
            raise ValueError(f"branch_dim must be > 0, got {branch_dim}")

        self.use_checkpoint = bool(use_checkpoint)
        self.hidden_dim = int(hidden_dim)
        self.branch_dim = int(branch_dim)

        # Four independent 256→128 projections (no activation).
        self.semantic_refiner_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.detail_refiner_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.encoder_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.fpn_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.semantic_block = self._make_branch_block()
        self.detail_block = self._make_branch_block()

        self.semantic_out_proj = nn.Conv2d(
            self.branch_dim, self.hidden_dim, kernel_size=1, bias=False,
        )
        self.detail_out_proj = nn.Conv2d(
            self.branch_dim, self.hidden_dim, kernel_size=1, bias=False,
        )

        self.fusion_out_proj = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False,
        )

    def _make_branch_block(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                self.branch_dim,
                self.branch_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.branch_dim),
            nn.GELU(),
            nn.Conv2d(
                self.branch_dim,
                self.branch_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(self.branch_dim),
        )

    def _forward_impl(
        self,
        refiner_feature: torch.Tensor,
        original_encoder_feature: torch.Tensor,
        sam_fpn: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one dual-branch fusion stage.

        Args:
            refiner_feature:        [N, 256, H_prev, W_prev]
            original_encoder_feature: [N, 256, H, W]
            sam_fpn:                [B, 256, H, W]  (image-level, no class dim)

        Returns:
            output: [N, 256, H, W]
        """
        N, _, H_prev, W_prev = refiner_feature.shape
        target_hw = original_encoder_feature.shape[-2:]

        # 1. Bilinear upsample refiner to current resolution.
        upsampled_refiner = F.interpolate(
            refiner_feature,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )  # [N, 256, H, W]

        # 2. Project all inputs to 128 channels with dedicated projections.
        semantic_refiner_compact = self.semantic_refiner_proj(
            upsampled_refiner
        )  # [N, 128, H, W]
        detail_refiner_compact = self.detail_refiner_proj(
            upsampled_refiner
        )  # [N, 128, H, W]
        encoder_compact = self.encoder_proj(original_encoder_feature)  # [N, 128, H, W]
        fpn_compact = self.fpn_proj(sam_fpn)                     # [B, 128, H, W]

        # 3. Recover batch/image layout for broadcast-based fusion.
        B = sam_fpn.shape[0]
        if N % B != 0:
            raise ValueError(
                f"refiner N={N} must be divisible by batch B={B}. "
                f"refiner: {tuple(refiner_feature.shape)}, "
                f"sam_fpn: {tuple(sam_fpn.shape)}"
            )
        C_chunk = N // B
        _, branch_dim, H, W = semantic_refiner_compact.shape

        semantic_refiner_compact_5d = semantic_refiner_compact.reshape(
            B, C_chunk, branch_dim, H, W
        )
        detail_refiner_compact_5d = detail_refiner_compact.reshape(
            B, C_chunk, branch_dim, H, W
        )
        encoder_compact_5d = encoder_compact.reshape(B, C_chunk, branch_dim, H, W)
        # fpn_compact stays [B, branch_dim, H, W] and broadcasts over classes.

        # 4. Build semantic and detail inputs via broadcasting.
        semantic_input = (
            semantic_refiner_compact_5d + encoder_compact_5d
        ).reshape(N, branch_dim, H, W)

        detail_input = (
            detail_refiner_compact_5d + fpn_compact[:, None]
        ).reshape(N, branch_dim, H, W)

        # 5. Branch blocks with internal residual.
        semantic_feature = semantic_input + self.semantic_block(semantic_input)
        detail_feature = detail_input + self.detail_block(detail_input)

        # 6. Project back to 256, sum with equal weight, final projection.
        semantic_out = self.semantic_out_proj(semantic_feature)
        detail_out = self.detail_out_proj(detail_feature)

        fused_out = semantic_out + detail_out
        return self.fusion_out_proj(fused_out)


    def forward(self, refiner_feature: torch.Tensor, original_encoder_feature: torch.Tensor,
                sam_fpn: torch.Tensor) -> torch.Tensor:
        if refiner_feature.ndim != 4 or original_encoder_feature.ndim != 4 or sam_fpn.ndim != 4:
            raise ValueError("Fusion inputs must be four-dimensional feature maps")
        n = refiner_feature.shape[0]
        if tuple(refiner_feature.shape[1:]) != (self.hidden_dim, 36, 36):
            raise ValueError("Refiner fusion input must be [N, hidden_dim, 36, 36]")
        if tuple(original_encoder_feature.shape) != (n, self.hidden_dim, 72, 72):
            raise ValueError("Encoder fusion input must be [N, hidden_dim, 72, 72]")
        if tuple(sam_fpn.shape[1:]) != (self.hidden_dim, 72, 72) or sam_fpn.shape[0] <= 0 or n % sam_fpn.shape[0]:
            raise ValueError("FPN fusion input must be [B, hidden_dim, 72, 72] with N divisible by B")
        if not (refiner_feature.device == original_encoder_feature.device == sam_fpn.device):
            raise ValueError("Fusion inputs must be on the same device")
        inputs = (refiner_feature, original_encoder_feature, sam_fpn)
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, *inputs, use_reentrant=False)
        return self._forward_impl(*inputs)
