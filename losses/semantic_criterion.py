from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Optional

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config_dataclasses import SemanticCriterionConfig
from ..models.instance_utils import binary_cross_entropy_log_prob
from ..models.task_modes import OUTPUT_KEYS


@dataclass
class SemanticStreamingContext:
    label_map: torch.Tensor
    valid_mask: torch.Tensor
    prompt_to_class_id: torch.Tensor
    presence_target: torch.Tensor
    presence_known: torch.Tensor
    distill_outer_boundary_by_class: torch.Tensor
    num_prompt_channels: int
    num_present_pairs: int
    num_presence_pairs: int
    num_absent_pairs: int
    total_valid_pixels: int
    distill_pixel_denom: int


@torch.no_grad()
def match_instances(student_masks, student_scores, teacher_masks, teacher_scores,
                    pixel_mask, num_points):
    """Hungarian assignment using only mask BCE and score BCE (no Dice).

    All queries participate. Deterministic samples are shared by every pair;
    the Q x Q cost never materializes Q x Q x H x W masks.
    """
    scores = student_scores.detach().float()
    targets = teacher_scores.detach().float().sigmoid()
    cost = F.softplus(scores)[:, None] - scores[:, None] * targets[None]
    positions = pixel_mask.flatten().nonzero(as_tuple=True)[0]
    if positions.numel():
        sample = torch.linspace(0, positions.numel() - 1,
                                min(num_points, positions.numel()),
                                device=positions.device).long()
        positions = positions[sample]
        sm = student_masks.detach().flatten(1)[:, positions].float()
        tm = teacher_masks.detach().flatten(1)[:, positions].float().sigmoid()
        cost = cost + F.softplus(sm).mean(1)[:, None] - sm @ tm.T / positions.numel()
    rows, cols = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(rows, device=scores.device),
            torch.as_tensor(cols, device=scores.device))


def weighted_mask_distill(student, teacher, rows, cols, pixels, gt, gt_valid, confidence, eps):
    """Confidence/purity weighted mask BCE; recomputed during backward to save memory."""
    sp = student[rows].float()
    tp = teacher[cols].detach().float().sigmoid()
    with torch.no_grad():
        fg = (tp > 0.5) & gt_valid[None]
        count = fg.flatten(1).sum(1)
        purity = (fg * gt[None]).flatten(1).sum(1) / count.clamp_min(1)
        reliability = confidence * torch.where(count > 0, purity, torch.ones_like(purity))
    ce = F.binary_cross_entropy_with_logits(sp, tp, reduction='none')
    means = (ce * pixels[None]).flatten(1).sum(1) / pixels.sum()
    return (means * reliability).sum() / reliability.sum().clamp_min(eps)


class SemanticCriterion(nn.Module):
    """BCE-only dual-head supervision; every distillation term is present-only.

    Global denominators make summed chunk contributions independent of prompt
    chunk size. Dense losses are normalized over valid pixels; instance KD is
    normalized per present image-prompt pair, after confidence weighting queries.
    """

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()
        for field in fields(self.cfg):
            if field.name.endswith('_weight'):
                value = float(getattr(self.cfg, field.name))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'{field.name} must be finite and non-negative.')
        width = self.cfg.sam3_mask_distill_boundary_width
        if isinstance(width, bool) or not isinstance(width, int) or width < 0:
            raise ValueError('sam3_mask_distill_boundary_width must be a non-negative integer.')
        points = self.cfg.instance_match_num_points
        if isinstance(points, bool) or not isinstance(points, int) or points <= 0:
            raise ValueError('instance_match_num_points must be a positive integer.')
        if not 0 < self.cfg.eps < 0.5:
            raise ValueError('eps must be in (0, 0.5).')

    def needs_teacher(self, context):
        return context.num_present_pairs > 0 and any(w > 0 for w in (
            self.cfg.sam3_mask_distill_weight, self.cfg.instance_mask_distill_weight,
            self.cfg.instance_score_distill_weight))

    def forward(self, outputs, targets, prompt_to_class_id, num_label_classes, reduction='mean'):
        if reduction != 'mean':
            raise ValueError("Only reduction='mean' is supported.")
        logits = outputs[OUTPUT_KEYS.final_logits]
        context = self.prepare_streaming_context(
            targets, logits.shape[1], prompt_to_class_id, num_label_classes, logits.shape[-2:])
        return self.forward_chunk(outputs, context, 0, logits.shape[1])

    def prepare_streaming_context(self, targets, num_prompt_channels, prompt_to_class_id,
                                  num_label_classes, target_hw=(288, 288)):
        source = targets['label_map'].long()
        if source.ndim == 4 and source.shape[1] == 1:
            source = source[:, 0]
        if source.ndim != 3:
            raise ValueError('label_map must be [B, H, W] or [B, 1, H, W].')
        if num_prompt_channels <= 0 or num_label_classes <= 0:
            raise ValueError('Class and prompt counts must be positive.')
        mapping = torch.as_tensor(prompt_to_class_id, dtype=torch.long, device=source.device)
        if mapping.ndim != 1 or mapping.numel() != num_prompt_channels:
            raise ValueError('prompt_to_class_id must contain one id per prompt.')
        if mapping.min() < 0 or mapping.max() >= num_label_classes:
            raise ValueError('Prompt class id is out of range.')
        if (torch.bincount(mapping, minlength=num_label_classes) == 0).any():
            raise ValueError('Every original class must own at least one prompt.')
        source_valid = source != self.cfg.ignore_index
        labels = source[source_valid]
        if labels.numel() and (labels.min() < 0 or labels.max() >= num_label_classes):
            raise ValueError('Labels must be valid class ids or ignore_index.')

        # Presence is computed BEFORE resizing, preserving tiny cropped objects.
        class_ids = torch.arange(num_label_classes, device=source.device)
        original_presence = (source[:, None] == class_ids[None, :, None, None]).flatten(2).any(2)
        presence = original_presence[:, mapping]
        known = torch.ones_like(presence) if self.cfg.presence_annotations_complete else presence.clone()
        label_map = F.interpolate(source[:, None].float(), size=target_hw, mode='nearest')[:, 0].long()
        valid = label_map != self.cfg.ignore_index
        class_masks = label_map[:, None] == class_ids[None, :, None, None]
        outer = self._build_outer_distill_boundary_by_class(
            class_masks, ~valid, self.cfg.sam3_mask_distill_boundary_width)
        support = valid[:, None] | outer[:, mapping]
        return SemanticStreamingContext(
            label_map=label_map, valid_mask=valid, prompt_to_class_id=mapping,
            presence_target=presence, presence_known=known,
            distill_outer_boundary_by_class=outer, num_prompt_channels=num_prompt_channels,
            num_present_pairs=int(presence.sum()), num_presence_pairs=int(known.sum()),
            num_absent_pairs=int((known & ~presence).sum()),
            total_valid_pixels=int(valid.sum()) * num_prompt_channels,
            distill_pixel_denom=int((support & presence[..., None, None]).sum()),
        )

    @staticmethod
    def _build_outer_distill_boundary_by_class(class_mask_by_class, ignore_mask, boundary_width):
        if boundary_width == 0:
            return torch.zeros_like(class_mask_by_class)
        b, c, h, w = class_mask_by_class.shape
        dilated = F.max_pool2d(class_mask_by_class.float().reshape(b * c, 1, h, w),
                              2 * boundary_width + 1, stride=1, padding=boundary_width)
        return dilated.reshape(b, c, h, w).bool() & ~class_mask_by_class & ignore_mask[:, None]

    def forward_chunk(self, outputs, context, class_start, class_end):
        logits = outputs[OUTPUT_KEYS.final_logits]
        if logits.ndim != 4 or not 0 <= class_start < class_end <= context.num_prompt_channels:
            raise ValueError('Invalid logits shape or prompt chunk range.')
        b, c, h, w = logits.shape
        if c != class_end - class_start or (h, w) != tuple(context.label_map.shape[-2:]):
            raise ValueError('Chunk shape does not match the streaming context.')
        ids = context.prompt_to_class_id[class_start:class_end]
        target = (context.label_map[:, None] == ids[None, :, None, None]).float()
        valid = context.valid_mask[:, None]
        present = context.presence_target[:, class_start:class_end]
        known = context.presence_known[:, class_start:class_end]
        support = valid | context.distill_outer_boundary_by_class[:, ids]
        zero = logits.float().sum() * 0
        losses = {}
        weights = {}

        def add(name, value, weight):
            losses['loss_' + name] = value
            weights[name] = weight

        # Explicit float32 BCE is safe under the trainer's outer autocast.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            semantic_bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
            add('final_bce', (semantic_bce * valid).sum() / max(context.total_valid_pixels, 1),
                self.cfg.final_balanced_bce_weight)
            for key, name, weight in (
                ('instance_union_log_prob', 'instance_bce', self.cfg.instance_bce_weight),
                ('fused_log_prob', 'fusion_bce', self.cfg.fusion_bce_weight),
            ):
                value = zero
                if weight > 0:
                    value = (binary_cross_entropy_log_prob(outputs[key].float(), target) * valid).sum()
                    value = value / max(context.total_valid_pixels, 1)
                add(name, value, weight)

            value = zero
            if self.cfg.presence_bce_weight > 0:
                ce = F.binary_cross_entropy_with_logits(outputs['presence_logits'].float(),
                                                         present.float(), reduction='none')
                value = (ce * known).sum() / max(context.num_presence_pairs, 1)
            add('presence_bce', value, self.cfg.presence_bce_weight)

            scores = outputs['query_logits'].float()
            value = zero
            if self.cfg.absent_query_bce_weight > 0:
                ce = F.binary_cross_entropy_with_logits(scores, torch.zeros_like(scores), reduction='none').mean(-1)
                value = (ce * (known & ~present)).sum() / max(context.num_absent_pairs, 1)
            add('absent_query_bce', value, self.cfg.absent_query_bce_weight)

            semantic_kd, mask_kd, score_kd = zero, zero, zero
            if self.needs_teacher(context) and present.any():
                indices = outputs.get('teacher_pair_indices')
                expected = present.flatten().nonzero(as_tuple=True)[0]
                if indices is None or not torch.equal(indices, expected):
                    raise ValueError('Teacher outputs must cover exactly the present pairs of this chunk.')
                sm = outputs['instance_logits'].flatten(0, 1)
                ss = scores.flatten(0, 1)
                sl = logits.flatten(0, 1).float()
                for teacher_id, pair in enumerate(indices.tolist()):
                    bi, ci = divmod(pair, c)
                    pixel_mask = support[bi, ci]
                    if self.cfg.sam3_mask_distill_weight > 0:
                        teacher_prob = outputs['teacher_semantic_logits'][teacher_id].detach().float().sigmoid()
                        ce = F.binary_cross_entropy_with_logits(sl[pair], teacher_prob, reduction='none')
                        semantic_kd = semantic_kd + (ce * pixel_mask).sum() / max(context.distill_pixel_denom, 1)
                    if self.cfg.instance_mask_distill_weight > 0 or self.cfg.instance_score_distill_weight > 0:
                        tm = outputs['teacher_instance_logits'][teacher_id].detach()
                        ts = outputs['teacher_query_logits'][teacher_id].detach()
                        rows, cols = match_instances(sm[pair], ss[pair], tm, ts, pixel_mask,
                                                      self.cfg.instance_match_num_points)
                        teacher_prob = ts[cols].float().sigmoid()
                        if self.cfg.instance_score_distill_weight > 0:
                            ce = F.binary_cross_entropy_with_logits(ss[pair, rows], teacher_prob, reduction='none')
                            # Retain low-confidence candidates, at reduced weight.
                            confidence = 0.1 + 0.9 * teacher_prob
                            score_kd = score_kd + (ce * confidence).sum() / confidence.sum() / context.num_present_pairs
                        if self.cfg.instance_mask_distill_weight > 0 and pixel_mask.any():
                            args = (sm[pair], tm, rows, cols, pixel_mask, target[bi, ci],
                                    context.valid_mask[bi], teacher_prob, self.cfg.eps)
                            if torch.is_grad_enabled() and sm.requires_grad:
                                value = checkpoint(weighted_mask_distill, *args, use_reentrant=False)
                            else:
                                value = weighted_mask_distill(*args)
                            mask_kd = mask_kd + value / context.num_present_pairs

            add('sam3_mask_distill_bce', semantic_kd, self.cfg.sam3_mask_distill_weight)
            add('instance_mask_distill_bce', mask_kd, self.cfg.instance_mask_distill_weight)
            add('instance_score_distill_bce', score_kd, self.cfg.instance_score_distill_weight)
            total = zero
            for name, weight in weights.items():
                contribution = losses['loss_' + name] * weight
                losses['loss_' + name + '_weighted'] = contribution
                total = total + contribution
            losses['total_loss'] = total
        return losses


class HybridCriterion(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError('HybridCriterion is not implemented.')
