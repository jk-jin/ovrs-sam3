from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..models.task_modes import OUTPUT_KEYS

TensorDict = Dict[str, torch.Tensor]


def build_eval_semantic_pred(
    outputs: Dict[str, torch.Tensor],
    metadata,
    prob_thd: Optional[float] = None,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, int, list[str]]:
    """
    Shared post-processing for evaluator, visualizer, and TTA.

    Returns:
        pred_eval: [B, H, W] prediction mapped to eval class space
        eval_num_classes: number of eval classes (includes background if enabled)
        eval_class_names: eval class names
    """
    score_map = outputs.get(OUTPUT_KEYS.final_score_map, None)

    background_enabled = bool(getattr(metadata, "background_enabled", False))
    background_class_id = int(getattr(metadata, "background_class_id", 0))
    exclude_from_forward = bool(
        getattr(metadata, "background_exclude_from_forward", False)
    )

    eval_num_classes = int(
        getattr(metadata, "eval_num_classes", None)
        or getattr(metadata, "num_classes", 0)
    )
    eval_class_names = list(
        getattr(metadata, "eval_class_names", None)
        or getattr(metadata, "class_names", [])
    )

    if score_map is not None:
        forward_channels = int(score_map.shape[1])
        max_score, pred_model = score_map.max(dim=1)
    else:
        pred_model = outputs[OUTPUT_KEYS.final_pred].long()
        forward_channels = int(pred_model.max().item()) + 1 if pred_model.numel() > 0 else 1
        max_score = None

    # Shape invariant checks.
    if background_enabled and exclude_from_forward:
        if forward_channels != eval_num_classes - 1:
            raise ValueError(
                "Shape invariant violation: background excluded from forward, so "
                f"forward channels ({forward_channels}) must equal "
                f"eval_num_classes ({eval_num_classes}) - 1."
            )
    else:
        if forward_channels != eval_num_classes:
            raise ValueError(
                "Shape invariant violation: forward channels ({forward_channels}) "
                f"must equal eval_num_classes ({eval_num_classes})."
            )

    if not background_enabled:
        return pred_model.long(), eval_num_classes, eval_class_names

    if not exclude_from_forward:
        pred_eval = pred_model.long()

        if prob_thd is not None and max_score is not None:
            pred_eval = pred_eval.clone()
            pred_eval[max_score < float(prob_thd)] = background_class_id

        return pred_eval, eval_num_classes, eval_class_names

    pred_eval = pred_model.long().clone()
    pred_eval[pred_eval >= background_class_id] += 1

    if prob_thd is not None and max_score is not None:
        pred_eval[max_score < float(prob_thd)] = background_class_id

    return pred_eval, eval_num_classes, eval_class_names


class MulticlassSemanticEvaluator:
    def __init__(
        self,
        ignore_index: int = 255,
        prob_thd: Optional[float] = None,
        **kwargs,
    ):
        for forbidden_key in ("bg_idx", "use_score_map"):
            if forbidden_key in kwargs:
                raise ValueError(
                    f"eval_cfg.{forbidden_key} is removed. "
                    "Background id and behavior are now controlled by dataset.background_cfg."
                )

        self.num_classes: Optional[int] = None
        self.ignore_index = int(ignore_index)
        self.prob_thd = prob_thd
        self.reset()

    def reset(self):
        self.intersection = None
        self.union = None
        self.target_count = None
        self.correct = 0.0
        self.total = 0.0

    def _ensure_buffers(self, num_classes: int, device: torch.device):
        if self.num_classes is None:
            self.num_classes = num_classes
            self.intersection = torch.zeros(num_classes, dtype=torch.float64, device=device)
            self.union = torch.zeros(num_classes, dtype=torch.float64, device=device)
            self.target_count = torch.zeros(num_classes, dtype=torch.float64, device=device)
        elif self.num_classes != num_classes:
            raise ValueError(
                f"Number of classes changed during evaluation: "
                f"{self.num_classes} -> {num_classes}"
            )

    def _prepare_target(
        self,
        label_map: torch.Tensor,
        out_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    f"Expected label_map [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}"
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                f"Expected label_map [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}"
            )

        label_map = label_map.long().to(device)
        if tuple(label_map.shape[-2:]) != tuple(out_hw):
            label_map = F.interpolate(
                label_map[:, None].float(),
                size=out_hw,
                mode="nearest",
            )[:, 0].long()
        return label_map

    def update(self, outputs: TensorDict, targets: TensorDict):
        if OUTPUT_KEYS.final_pred not in outputs and OUTPUT_KEYS.final_score_map not in outputs:
            raise ValueError(
                f"Semantic outputs must contain '{OUTPUT_KEYS.final_pred}' "
                f"or '{OUTPUT_KEYS.final_score_map}'."
            )
        if "label_map" not in targets:
            raise ValueError("label_map is required in semantic targets.")

        metadata = targets.get("metadata", None)

        pred_eval, eval_num_classes, _ = build_eval_semantic_pred(
            outputs=outputs,
            metadata=metadata,
            prob_thd=self.prob_thd,
            ignore_index=self.ignore_index,
        )

        out_hw = tuple(pred_eval.shape[-2:])
        device = pred_eval.device

        target = self._prepare_target(
            label_map=targets["label_map"],
            out_hw=out_hw,
            device=device,
        )

        self._ensure_buffers(num_classes=eval_num_classes, device=device)

        valid = target != self.ignore_index
        self.correct += float(((pred_eval == target) & valid).sum().item())
        self.total += float(valid.sum().item())

        for cls_id in range(eval_num_classes):
            pred_c = (pred_eval == cls_id) & valid
            target_c = (target == cls_id) & valid

            inter = (pred_c & target_c).sum()
            union = (pred_c | target_c).sum()
            tgt_count = target_c.sum()

            self.intersection[cls_id] += inter.double()
            self.union[cls_id] += union.double()
            self.target_count[cls_id] += tgt_count.double()

    def compute(self) -> Dict[str, float]:
        if self.num_classes is None:
            return {}

        per_class_iou = self.intersection / self.union.clamp(min=1.0)
        per_class_acc = self.intersection / self.target_count.clamp(min=1.0)

        valid_iou = self.union > 0
        valid_acc = self.target_count > 0

        miou = per_class_iou[valid_iou].mean().item() if valid_iou.any() else 0.0
        macc = per_class_acc[valid_acc].mean().item() if valid_acc.any() else 0.0
        pixel_acc = self.correct / max(self.total, 1.0)

        out = {
            "semantic.miou": float(miou),
            "semantic.macc": float(macc),
            "semantic.pixel_acc": float(pixel_acc),
        }

        for i in range(self.num_classes):
            out[f"semantic.iou_class_{i}"] = float(per_class_iou[i].item())
            out[f"semantic.acc_class_{i}"] = float(per_class_acc[i].item())

        return out


def _round_up(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)


def _flip_image_batch(img_batch: torch.Tensor, flip_mode: str) -> torch.Tensor:
    if flip_mode == "none":
        return img_batch
    if flip_mode == "h":
        return torch.flip(img_batch, dims=[-1])
    if flip_mode == "v":
        return torch.flip(img_batch, dims=[-2])
    if flip_mode == "hv":
        return torch.flip(img_batch, dims=[-2, -1])
    raise ValueError(f"Unknown flip_mode: {flip_mode}")


def _flip_raw_images(
    raw_images: Optional[list[torch.Tensor]],
    flip_mode: str,
) -> Optional[list[torch.Tensor]]:
    if raw_images is None:
        return None
    return [
        _flip_image_batch(image, flip_mode)
        for image in raw_images
    ]


def _deaugment_logits(logits: torch.Tensor, target_hw: tuple[int, int], flip_mode: str) -> torch.Tensor:
    if flip_mode == "h":
        logits = torch.flip(logits, dims=[-1])
    elif flip_mode == "v":
        logits = torch.flip(logits, dims=[-2])
    elif flip_mode == "hv":
        logits = torch.flip(logits, dims=[-2, -1])
    elif flip_mode != "none":
        raise ValueError(f"Unknown flip_mode: {flip_mode}")

    if tuple(logits.shape[-2:]) != tuple(target_hw):
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def inference_with_tta(
    model,
    batch,
    tta_cfg: Optional[Dict],
):
    if tta_cfg is None or not bool(tta_cfg.get("enabled", False)):
        return model(batch)

    scales = [float(x) for x in tta_cfg.get("scales", [1.0])]
    flip_modes = list(tta_cfg.get("flip_modes", ["none"]))
    size_divisor = int(tta_cfg.get("size_divisor", 14))

    unsupported_scales = [
        scale for scale in scales
        if abs(float(scale) - 1.0) > 1e-8
    ]
    if unsupported_scales:
        raise ValueError(
            "Current OVRS-SAM3 refiner requires fixed 1008×1008 SAM3 input "
            "with 72×72 encoder and 36×36 refiner features. "
            f"Only TTA scale=1.0 is supported, got {unsupported_scales}."
        )

    base_img_batch = batch.img_batch
    target_hw = tuple(base_img_batch.shape[-2:])

    sum_4d: Dict[str, torch.Tensor] = {}
    sum_2d: Dict[str, torch.Tensor] = {}
    num_views = 0
    last_outputs = None

    for scale in scales:
        if scale <= 0:
            raise ValueError(f"Invalid TTA scale: {scale}")

        scaled_h = max(1, int(round(target_hw[0] * scale)))
        scaled_w = max(1, int(round(target_hw[1] * scale)))
        if size_divisor > 1:
            scaled_h = _round_up(scaled_h, size_divisor)
            scaled_w = _round_up(scaled_w, size_divisor)

        resized_img_batch = F.interpolate(
            base_img_batch,
            size=(scaled_h, scaled_w),
            mode="bilinear",
            align_corners=False,
        )

        for flip_mode in flip_modes:
            aug_batch = copy.deepcopy(batch)
            aug_batch.img_batch = _flip_image_batch(resized_img_batch, flip_mode)
            aug_batch.raw_images = _flip_raw_images(batch.raw_images, flip_mode)

            outputs = model(aug_batch)
            last_outputs = outputs

            for key, value in outputs.items():
                if not torch.is_tensor(value):
                    continue

                if value.dim() == 4:
                    deaug = _deaugment_logits(value, target_hw=target_hw, flip_mode=flip_mode)
                    if key not in sum_4d:
                        sum_4d[key] = deaug
                    else:
                        sum_4d[key] = sum_4d[key] + deaug

                elif value.dim() == 2:
                    if key not in sum_2d:
                        sum_2d[key] = value
                    else:
                        sum_2d[key] = sum_2d[key] + value

            num_views += 1

    if last_outputs is None or num_views == 0:
        raise RuntimeError("TTA produced no outputs.")

    merged_outputs = dict(last_outputs)
    for key, value in sum_4d.items():
        merged_outputs[key] = value / float(num_views)
    for key, value in sum_2d.items():
        merged_outputs[key] = value / float(num_views)

    if OUTPUT_KEYS.final_score_map in merged_outputs:
        score_map = merged_outputs[OUTPUT_KEYS.final_score_map]
        merged_outputs[OUTPUT_KEYS.final_pred] = score_map.argmax(dim=1).long()

    return merged_outputs


def _format_ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""

    all_rows = [headers] + rows
    col_widths = []
    for col_id in range(len(headers)):
        width = max(len(str(row[col_id])) for row in all_rows)
        col_widths.append(width)

    def _format_row(row: list[str]) -> str:
        cells = []
        for i, cell in enumerate(row):
            cells.append(f" {str(cell).ljust(col_widths[i])} ")
        return "|" + "|".join(cells) + "|"

    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [border, _format_row(headers), border]
    for row in rows:
        lines.append(_format_row(row))
    lines.append(border)
    return "\n".join(lines)


def _collect_semantic_metric_rows(
    metric_stats: Dict[str, float],
    class_names: Optional[Sequence[str]] = None,
) -> tuple[list[list[str]], list[list[str]]]:
    summary_rows: list[list[str]] = []
    per_class_rows: list[list[str]] = []

    if "semantic.miou" in metric_stats:
        summary_rows.append(["mIoU", f"{metric_stats['semantic.miou'] * 100.0:.2f}"])
    if "semantic.macc" in metric_stats:
        summary_rows.append(["mAcc", f"{metric_stats['semantic.macc'] * 100.0:.2f}"])
    if "semantic.pixel_acc" in metric_stats:
        summary_rows.append(["aAcc", f"{metric_stats['semantic.pixel_acc'] * 100.0:.2f}"])

    class_ids = []
    for key in metric_stats.keys():
        if key.startswith("semantic.iou_class_"):
            cls_id = int(key.rsplit("_", 1)[-1])
            class_ids.append(cls_id)
    class_ids = sorted(set(class_ids))

    for cls_id in class_ids:
        class_name = f"class_{cls_id}"
        if class_names is not None and cls_id < len(class_names):
            class_name = str(class_names[cls_id])

        iou_key = f"semantic.iou_class_{cls_id}"
        acc_key = f"semantic.acc_class_{cls_id}"

        iou = metric_stats.get(iou_key, float("nan"))
        acc = metric_stats.get(acc_key, float("nan"))

        per_class_rows.append([
            str(cls_id),
            class_name,
            f"{iou * 100.0:.2f}",
            f"{acc * 100.0:.2f}",
        ])

    return summary_rows, per_class_rows


def format_semantic_metric_tables(
    metric_stats: Dict[str, float],
    class_names: Optional[Sequence[str]] = None,
) -> tuple[str, str]:
    summary_rows, per_class_rows = _collect_semantic_metric_rows(
        metric_stats=metric_stats,
        class_names=class_names,
    )

    summary_table = ""
    per_class_table = ""

    if summary_rows:
        summary_table = _format_ascii_table(
            headers=["Metric", "Value"],
            rows=summary_rows,
        )

    if per_class_rows:
        per_class_table = _format_ascii_table(
            headers=["Class ID", "Class Name", "IoU", "Acc"],
            rows=per_class_rows,
        )

    return summary_table, per_class_table


def extract_semantic_targets_from_batch(batch) -> Dict[str, torch.Tensor]:
    target = batch.find_targets[0]
    label_map = getattr(target, "semantic_eval_label_map", None)
    if label_map is None:
        label_map = target.semantic_label_map

    return {
        "label_map": label_map,
        "metadata": batch.find_metadatas[0],
    }


def extract_class_names_from_batch(batch) -> Optional[List[str]]:
    try:
        meta = batch.find_metadatas[0]
        if hasattr(meta, "eval_class_names"):
            return [str(x) for x in meta.eval_class_names]
        return [str(x) for x in meta.class_names]
    except Exception:
        return None


def update_evaluator_with_batch(
    evaluator: MulticlassSemanticEvaluator,
    outputs: Dict[str, torch.Tensor],
    batch,
) -> Dict[str, torch.Tensor]:
    targets = extract_semantic_targets_from_batch(batch)
    evaluator.update(outputs, targets)
    return targets