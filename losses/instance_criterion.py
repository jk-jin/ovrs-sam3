from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..models.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


TensorDict = Dict[str, torch.Tensor]
MatchIndices = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class InstanceLossWeights:
    loss_cls: float = 2.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    loss_mask: float = 5.0
    loss_dice: float = 5.0
    no_object_weight: float = 0.1


class SimpleHungarianMatcher(nn.Module):
    """A small DETR-style matcher for the simplified SAM3 trainer.

    This version is intentionally lightweight:
    - one binary text-conditioned class channel
    - box cost + giou cost + simple classification cost
    - optional mask cost can be added later if needed
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: TensorDict, targets: TensorDict) -> MatchIndices:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        if pred_logits.dim() == 4:
            pred_logits = pred_logits[-1]
        if pred_boxes.dim() == 4:
            pred_boxes = pred_boxes[-1]

        batch_size = pred_logits.shape[0]
        target_boxes = targets["boxes"]
        num_boxes = targets["num_boxes"]

        if isinstance(num_boxes, torch.Tensor):
            num_boxes_per_image = [int(x) for x in num_boxes.view(-1).tolist()]
        else:
            num_boxes_per_image = [int(num_boxes)]

        matches: MatchIndices = []
        start = 0
        for b in range(batch_size):
            n_gt = num_boxes_per_image[b] if b < len(num_boxes_per_image) else 0
            if n_gt <= 0:
                matches.append(
                    (
                        torch.zeros(0, dtype=torch.long, device=pred_boxes.device),
                        torch.zeros(0, dtype=torch.long, device=pred_boxes.device),
                    )
                )
                continue

            tgt_boxes_b = target_boxes[start : start + n_gt]
            start += n_gt

            pred_prob_b = pred_logits[b].squeeze(-1).sigmoid()  # [Q]
            pred_boxes_b = pred_boxes[b]  # [Q, 4]

            cost_class = -pred_prob_b[:, None].expand(-1, n_gt)
            cost_bbox = torch.cdist(pred_boxes_b, tgt_boxes_b, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes_b),
                box_cxcywh_to_xyxy(tgt_boxes_b),
            )
            total_cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            src_idx, tgt_idx = linear_sum_assignment(total_cost.detach().cpu().numpy())
            matches.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.long, device=pred_boxes.device),
                    torch.as_tensor(tgt_idx, dtype=torch.long, device=pred_boxes.device),
                )
            )

        return matches


class InstanceCriterion(nn.Module):
    """Minimal instance criterion for the simplified SAM3 trainer.

    Expected outputs:
        pred_logits: [B, Q, 1] or [L, B, Q, 1]
        pred_boxes:  [B, Q, 4] or [L, B, Q, 4]
        pred_masks:  [B, Q, H, W] or [L, B, Q, H, W]

    Expected targets:
        boxes: packed ground-truth boxes across the batch, [sum(N_i), 4]
        num_boxes: per-image counts, [B]
        masks: padded masks, [B, N_max, H, W] or packed masks [sum(N_i), H, W]
    """

    def __init__(
        self,
        matcher: nn.Module | None = None,
        weights: InstanceLossWeights | None = None,
        compute_aux_loss: bool = True,
    ):
        super().__init__()
        self.matcher = matcher or SimpleHungarianMatcher()
        self.weights = weights or InstanceLossWeights()
        self.compute_aux_loss = compute_aux_loss

    @staticmethod
    def _take_last_if_stacked(x: torch.Tensor) -> torch.Tensor:
        if x.dim() in (4, 5):
            return x[-1]
        return x

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid()
        probs = probs.flatten(1)
        targets = targets.float().flatten(1)
        numerator = 2 * (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
        return loss.mean()

    def _split_targets_by_image(self, targets: TensorDict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        boxes = targets["boxes"]
        masks = targets["masks"]
        num_boxes = targets["num_boxes"]
        if isinstance(num_boxes, torch.Tensor):
            num_boxes_list = [int(x) for x in num_boxes.view(-1).tolist()]
        else:
            num_boxes_list = [int(num_boxes)]

        box_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []

        if masks is None:
            masks = torch.zeros((boxes.shape[0], 1, 1), device=boxes.device, dtype=torch.bool)

        if masks.dim() == 4 and masks.shape[0] == len(num_boxes_list):
            # padded representation: [B, N_max, H, W]
            start = 0
            for b, n_gt in enumerate(num_boxes_list):
                box_list.append(boxes[start : start + n_gt])
                mask_list.append(masks[b, :n_gt])
                start += n_gt
        elif masks.dim() == 3:
            # packed representation: [sum(N_i), H, W]
            start = 0
            for n_gt in num_boxes_list:
                box_list.append(boxes[start : start + n_gt])
                mask_list.append(masks[start : start + n_gt])
                start += n_gt
        else:
            raise ValueError(f"Unsupported target mask shape: {tuple(masks.shape)}")

        return box_list, mask_list

    def _classification_loss(self, pred_logits: torch.Tensor, indices: MatchIndices) -> torch.Tensor:
        batch_size, num_queries, _ = pred_logits.shape
        target_classes = torch.zeros(
            (batch_size, num_queries),
            device=pred_logits.device,
            dtype=pred_logits.dtype,
        )
        for b, (src_idx, _) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = 1.0

        loss_per_query = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1),
            target_classes,
            reduction="none",
        )
        weights = torch.full_like(loss_per_query, self.weights.no_object_weight)
        weights[target_classes > 0] = 1.0
        return (loss_per_query * weights).mean()

    def _box_losses(
        self,
        pred_boxes: torch.Tensor,
        target_boxes_by_image: Sequence[torch.Tensor],
        indices: MatchIndices,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_boxes: List[torch.Tensor] = []
        tgt_boxes: List[torch.Tensor] = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_boxes.append(pred_boxes[b, src_idx])
            tgt_boxes.append(target_boxes_by_image[b][tgt_idx])

        if not src_boxes:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        src_boxes_cat = torch.cat(src_boxes, dim=0)
        tgt_boxes_cat = torch.cat(tgt_boxes, dim=0)
        loss_bbox = F.l1_loss(src_boxes_cat, tgt_boxes_cat, reduction="mean")
        loss_giou = 1.0 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes_cat),
                box_cxcywh_to_xyxy(tgt_boxes_cat),
            )
        ).mean()
        return loss_bbox, loss_giou

    def _mask_losses(
        self,
        pred_masks: torch.Tensor,
        target_masks_by_image: Sequence[torch.Tensor],
        indices: MatchIndices,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_masks: List[torch.Tensor] = []
        tgt_masks: List[torch.Tensor] = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_masks.append(pred_masks[b, src_idx])
            tgt_masks.append(target_masks_by_image[b][tgt_idx].float())

        if not src_masks:
            zero = pred_masks.sum() * 0.0
            return zero, zero

        src_masks_cat = torch.cat(src_masks, dim=0)
        tgt_masks_cat = torch.cat(tgt_masks, dim=0)

        if src_masks_cat.shape[-2:] != tgt_masks_cat.shape[-2:]:
            src_masks_cat = F.interpolate(
                src_masks_cat[:, None],
                size=tgt_masks_cat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        loss_mask = F.binary_cross_entropy_with_logits(src_masks_cat, tgt_masks_cat, reduction="mean")
        loss_dice = self._dice_loss(src_masks_cat, tgt_masks_cat)
        return loss_mask, loss_dice

    def _compute_single(self, outputs: TensorDict, targets: TensorDict, prefix: str = "") -> TensorDict:
        pred_logits = self._take_last_if_stacked(outputs["pred_logits"])
        pred_boxes = self._take_last_if_stacked(outputs["pred_boxes"])
        pred_masks = self._take_last_if_stacked(outputs["pred_masks"])

        target_boxes_by_image, target_masks_by_image = self._split_targets_by_image(targets)
        indices = self.matcher(
            {
                "pred_logits": pred_logits,
                "pred_boxes": pred_boxes,
            },
            targets,
        )

        loss_cls = self._classification_loss(pred_logits, indices)
        loss_bbox, loss_giou = self._box_losses(pred_boxes, target_boxes_by_image, indices)
        loss_mask, loss_dice = self._mask_losses(pred_masks, target_masks_by_image, indices)

        weighted_total = (
            self.weights.loss_cls * loss_cls
            + self.weights.loss_bbox * loss_bbox
            + self.weights.loss_giou * loss_giou
            + self.weights.loss_mask * loss_mask
            + self.weights.loss_dice * loss_dice
        )

        return {
            f"{prefix}loss_cls": loss_cls,
            f"{prefix}loss_bbox": loss_bbox,
            f"{prefix}loss_giou": loss_giou,
            f"{prefix}loss_mask": loss_mask,
            f"{prefix}loss_dice": loss_dice,
            f"{prefix}loss_total": weighted_total,
            f"{prefix}indices": indices,
        }

    def forward(self, outputs: TensorDict, targets: TensorDict) -> TensorDict:
        losses = self._compute_single(outputs, targets, prefix="")
        total = losses["loss_total"]

        if self.compute_aux_loss and "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_single(aux_out, targets, prefix=f"aux_{i}.")
                total = total + aux_losses[f"aux_{i}.loss_total"]
                losses.update(aux_losses)

        losses["total_loss"] = total
        return losses
