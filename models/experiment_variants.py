from __future__ import annotations

import torch
from torch.nn import functional as F


def semantic_presence_logits(semantic_logits, presence_logits):
    """Return logits whose sigmoid is semantic probability times presence once.

    Compute in float32 to keep tiny probabilities accurate under AMP. The
    caller still merges aliases only after this per-prompt modulation.
    """
    log_probability = F.logsigmoid(semantic_logits.float()) + F.logsigmoid(presence_logits.float())
    return log_probability - torch.log(-torch.expm1(log_probability))
