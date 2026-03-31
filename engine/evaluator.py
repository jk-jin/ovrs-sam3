from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..losses.target_converter import TargetConverter
from ..models.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .visualization import VisualizationManager

TensorDict = Dict[str, torch.Tensor]


def move_batch_to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if is_dataclass(obj):
        for field in fields(obj):
            setattr(obj, field.name, move_batch_to_device(getattr(obj, field.name), device))
        return obj
    if isinstance(obj, dict):
        return {k: move_batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_batch_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_batch_to_device(v, device) for v in obj)
    return obj


@dataclass
class BinarySemanticMetrics:
    pixel_acc: float
    iou: float
    dice: float


class BinarySemanticEvaluator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0
        self.correct = 0.0
        self.total = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0

    @staticmethod
    def _prepare_target(semantic_masks: torch.Tensor, out_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
        if semantic_masks.dim() == 2:
            semantic_masks = semantic_masks[None, None]
        elif semantic_masks.dim() == 3:
            semantic_masks = semantic_masks[:, None]
        elif semantic_masks.dim() == 4 and semantic_masks.shape[1] != 1:
            semantic_masks = semantic_masks.any(dim=1, keepdim=True)
        semantic_masks = semantic_masks.float().to(device)
        if semantic_masks.shape[-2:] != out_hw:
            semantic_masks = F.interpolate(semantic_masks, size=out_hw, mode='nearest')
        return semantic_masks

    def update(self, outputs: TensorDict, targets: TensorDict):
        logits = outputs['semantic_logits']
        preds = logits.sigmoid() >= self.threshold
        target = self._prepare_target(targets['semantic_masks'], logits.shape[-2:], logits.device) >= 0.5
        self.intersection += float((preds & target).sum().item())
        self.union += float((preds | target).sum().item())
        self.correct += float((preds == target).sum().item())
        self.total += float(target.numel())
        self.pred_sum += float(preds.sum().item())
        self.target_sum += float(target.sum().item())

    def compute(self) -> Dict[str, float]:
        pixel_acc = self.correct / max(self.total, 1.0)
        iou = self.intersection / max(self.union, 1.0)
        dice = (2.0 * self.intersection) / max(self.pred_sum + self.target_sum, 1.0)
        return {
            'semantic.pixel_acc': pixel_acc,
            'semantic.iou': iou,
            'semantic.dice': dice,
        }


class QueryMaskInstanceEvaluator:
    def __init__(self, score_threshold: float = 0.3, mask_threshold: float = 0.5, match_iou_threshold: float = 0.5):
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.match_iou_threshold = float(match_iou_threshold)
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.sum_mask_iou = 0.0
        self.sum_box_iou = 0.0
        self.num_matches = 0

    @staticmethod
    def _take_last_if_stacked(x: torch.Tensor) -> torch.Tensor:
        if x.dim() in (4, 5):
            return x[-1]
        return x

    @staticmethod
    def _to_mask_list(masks: torch.Tensor, num_boxes: Sequence[int]) -> List[torch.Tensor]:
        if masks is None:
            return [torch.zeros((0, 1, 1), dtype=torch.bool)] * len(num_boxes)
        if masks.dim() == 4 and masks.shape[0] == len(num_boxes):
            return [masks[b, :n] for b, n in enumerate(num_boxes)]
        if masks.dim() == 3:
            out = []
            start = 0
            for n in num_boxes:
                out.append(masks[start:start + n])
                start += n
            return out
        raise ValueError(f'Unsupported gt mask shape: {tuple(masks.shape)}')

    @staticmethod
    def _to_box_list(boxes: torch.Tensor, num_boxes: Sequence[int]) -> List[torch.Tensor]:
        out = []
        start = 0
        for n in num_boxes:
            out.append(boxes[start:start + n])
            start += n
        return out

    @staticmethod
    def _pairwise_mask_iou(pred_masks: torch.Tensor, tgt_masks: torch.Tensor) -> torch.Tensor:
        pred = pred_masks.flatten(1).bool()
        tgt = tgt_masks.flatten(1).bool()
        inter = (pred[:, None] & tgt[None]).sum(dim=-1).float()
        union = (pred[:, None] | tgt[None]).sum(dim=-1).float().clamp(min=1.0)
        return inter / union

    def update(self, outputs: TensorDict, targets: TensorDict):
        pred_logits = self._take_last_if_stacked(outputs['pred_logits'])
        pred_boxes = self._take_last_if_stacked(outputs['pred_boxes'])
        pred_masks = self._take_last_if_stacked(outputs['pred_masks'])

        scores = pred_logits.squeeze(-1).sigmoid()
        num_boxes = targets['num_boxes']
        if isinstance(num_boxes, torch.Tensor):
            num_boxes = [int(x) for x in num_boxes.view(-1).tolist()]
        else:
            num_boxes = [int(num_boxes)]

        gt_boxes_list = self._to_box_list(targets['boxes'], num_boxes)
        gt_masks_list = self._to_mask_list(targets['masks'], num_boxes)

        for b in range(pred_logits.shape[0]):
            keep = scores[b] >= self.score_threshold
            pred_scores_b = scores[b][keep]
            pred_boxes_b = pred_boxes[b][keep]
            pred_masks_b = pred_masks[b][keep]
            if pred_scores_b.numel() == 0:
                self.fn += int(gt_boxes_list[b].shape[0])
                continue

            order = torch.argsort(pred_scores_b, descending=True)
            pred_boxes_b = pred_boxes_b[order]
            pred_masks_b = pred_masks_b[order].sigmoid() >= self.mask_threshold
            gt_boxes_b = gt_boxes_list[b]
            gt_masks_b = gt_masks_list[b].bool()
            if gt_masks_b.numel() == 0:
                self.fp += int(pred_boxes_b.shape[0])
                continue
            if pred_masks_b.shape[-2:] != gt_masks_b.shape[-2:]:
                pred_masks_b = F.interpolate(pred_masks_b[:, None].float(), size=gt_masks_b.shape[-2:], mode='nearest')[:, 0] > 0.5

            mask_iou = self._pairwise_mask_iou(pred_masks_b, gt_masks_b)
            box_iou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes_b), box_cxcywh_to_xyxy(gt_boxes_b))

            matched_gt = set()
            tp = 0
            fp = 0
            for i in range(pred_boxes_b.shape[0]):
                best_iou, best_j = mask_iou[i].max(dim=0)
                j = int(best_j.item())
                if float(best_iou.item()) >= self.match_iou_threshold and j not in matched_gt:
                    matched_gt.add(j)
                    tp += 1
                    self.sum_mask_iou += float(best_iou.item())
                    self.sum_box_iou += float(box_iou[i, j].item())
                    self.num_matches += 1
                else:
                    fp += 1
            fn = int(gt_boxes_b.shape[0]) - len(matched_gt)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def compute(self) -> Dict[str, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        mean_mask_iou = self.sum_mask_iou / max(self.num_matches, 1)
        mean_box_iou = self.sum_box_iou / max(self.num_matches, 1)
        return {
            'instance.precision': precision,
            'instance.recall': recall,
            'instance.f1': f1,
            'instance.mean_mask_iou': mean_mask_iou,
            'instance.mean_box_iou': mean_box_iou,
        }


class HybridEvaluator:
    def __init__(self, instance_evaluator: Optional[QueryMaskInstanceEvaluator] = None, semantic_evaluator: Optional[BinarySemanticEvaluator] = None):
        self.instance_evaluator = instance_evaluator or QueryMaskInstanceEvaluator()
        self.semantic_evaluator = semantic_evaluator or BinarySemanticEvaluator()

    def reset(self):
        self.instance_evaluator.reset()
        self.semantic_evaluator.reset()

    def update(self, outputs: Dict[str, Dict[str, torch.Tensor]], instance_targets: TensorDict, semantic_targets: TensorDict):
        self.instance_evaluator.update(outputs['instance_outputs'], instance_targets)
        self.semantic_evaluator.update(outputs['semantic_outputs'], semantic_targets)

    def compute(self) -> Dict[str, float]:
        out = {}
        out.update(self.instance_evaluator.compute())
        out.update(self.semantic_evaluator.compute())
        return out


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    task: str,
    device: torch.device | str = 'cuda',
    visualizer: Optional[VisualizationManager] = None,
    epoch: Optional[int] = None,
    stage: str = 'val',
) -> Dict[str, float]:
    device = torch.device(device)
    model.eval()
    if task == 'instance':
        evaluator = QueryMaskInstanceEvaluator()
    elif task == 'semantic':
        evaluator = BinarySemanticEvaluator()
    elif task == 'hybrid':
        evaluator = HybridEvaluator()
    else:
        raise ValueError(f'Unsupported task: {task}')

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if task == 'instance':
            outputs = model(batch, mode='instance')
            targets = TargetConverter.from_batch(batch, task='instance')
            evaluator.update(outputs['instance_outputs'], targets)
            if visualizer is not None:
                visualizer.save_instance_batch(batch, outputs['instance_outputs'], targets, epoch=epoch, stage=stage)
        elif task == 'semantic':
            outputs = model(batch, mode='semantic')
            targets = TargetConverter.from_batch(batch, task='semantic')
            evaluator.update(outputs['semantic_outputs'], targets)
            if visualizer is not None:
                visualizer.save_semantic_batch(batch, outputs['semantic_outputs'], targets, epoch=epoch, stage=stage)
        else:
            outputs = model(batch, mode='hybrid')
            instance_targets = TargetConverter.from_batch(batch, task='instance')
            semantic_targets = TargetConverter.from_batch(batch, task='semantic')
            evaluator.update(outputs, instance_targets, semantic_targets)
            if visualizer is not None:
                visualizer.save_semantic_batch(batch, outputs['semantic_outputs'], semantic_targets, epoch=epoch, stage=stage)
    return evaluator.compute()
