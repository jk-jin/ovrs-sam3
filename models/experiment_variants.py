from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SimpleAdapter(nn.Module):
    """Shared residual adapter on the original prompt-conditioned 72x72 grid."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
        )

    def fuse_semantic_detail_chunk(self, aggregated_feature, **kwargs):
        return aggregated_feature + self.block(aggregated_feature)


class ConcatFusion(nn.Module):
    """Three 256-channel inputs with one shared convolutional fusion."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )

    def forward(self, aggregated_feature, prompt_conditioned_feature_72, sam3_fpn_72):
        count = aggregated_feature.shape[0] // sam3_fpn_72.shape[0]
        fpn = sam3_fpn_72.repeat_interleave(count, dim=0)
        upsampled = F.interpolate(aggregated_feature, (72, 72), mode="bilinear", align_corners=False)
        return self.block(torch.cat([upsampled, prompt_conditioned_feature_72, fpn], dim=1))


def semantic_presence_logits(semantic_logits, presence_logits):
    """Return logits whose sigmoid is semantic probability times presence once.

    Compute in float32 to keep tiny probabilities accurate under AMP. The
    caller still merges aliases only after this per-prompt modulation.
    """
    log_probability = F.logsigmoid(semantic_logits.float()) + F.logsigmoid(presence_logits.float())
    return log_probability - torch.log(-torch.expm1(log_probability))
