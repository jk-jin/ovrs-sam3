"""Probability aggregation shared by training and inference."""
import math

import torch
import torch.nn.functional as F


def aggregate_instances(semantic_logits, mask_logits, query_logits, presence_logits, threshold):
    """Inputs use [B, P, (Q), H, W]; no pixel binarization or NMS.

    The training union uses every query, weighted by its own matching
    probability. The final union filters on score * presence and uses the
    retained masks without multiplying their pixel probabilities by scores.
    Log space keeps both BCE and threshold comparisons stable under AMP.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("instance_score_threshold must be in [0, 1].")
    with torch.autocast(device_type=mask_logits.device.type, enabled=False):
        mask_log_prob = F.logsigmoid(mask_logits.float())
        score_log_prob = F.logsigmoid(query_logits.float())
        union_log_prob = (mask_log_prob + score_log_prob[..., None, None]).amax(dim=2)
        joint_log_prob = score_log_prob + F.logsigmoid(presence_logits.float())[..., None]
        log_threshold = math.log(threshold) if threshold > 0 else -math.inf
        keep = joint_log_prob > log_threshold
        kept_log_prob = mask_log_prob.masked_fill(~keep[..., None, None], -torch.inf).amax(dim=2)
        fused_log_prob = torch.maximum(F.logsigmoid(semantic_logits.float()), kept_log_prob)
    return union_log_prob, fused_log_prob


def binary_cross_entropy_log_prob(log_prob, target):
    """BCE on aggregated log probabilities, stable even for near-certain masks."""
    # log1p(-exp(x)) loses precision near zero; -expm1(x) does not.
    tiny = torch.finfo(log_prob.dtype).tiny
    log_not_prob = (-torch.expm1(log_prob)).clamp_min(tiny).log()
    return -target * log_prob - (1 - target) * log_not_prob


def log_prob_to_logits(log_prob):
    tiny = torch.finfo(log_prob.dtype).tiny
    return log_prob - (-torch.expm1(log_prob)).clamp_min(tiny).log()
