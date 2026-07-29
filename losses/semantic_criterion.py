from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_dataclasses import SemanticCriterionConfig
from ..models.task_modes import OUTPUT_KEYS


TensorDict = Dict[str, torch.Tensor]


class SemanticCriterion(nn.Module):
    """Semantic segmentation criterion.

    Losses:
        1. 全像素等权 BCE on final_logits
           - 所有图像、所有类别、所有像素统一计算 BCE 均值
           - 标签 255 的所有类别通道目标为 0，作为负样本参与监督

        2. 可选 Dice on final_logits
           - 只对图像中存在的类别计算
           - 默认 final_dice_weight=0.0

        3. 可选 SAM3 teacher mask distillation BCE
           - 冻结 SAM3 semantic head logits 作为软目标
           - 只监督存在类别和非 255 像素
           - 默认 sam3_mask_distill_weight=0.0
    """

    def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
        super().__init__()
        self.cfg = cfg or SemanticCriterionConfig()

    def forward(
        self,
        outputs: TensorDict,
        targets: TensorDict,
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
    ) -> TensorDict:
        del chunk_class_ids

        if reduction != "mean":
            raise ValueError(
                f"SemanticCriterion only supports reduction='mean', got {reduction!r}."
            )

        if OUTPUT_KEYS.final_logits not in outputs:
            raise ValueError(
                f"SemanticCriterion requires outputs[{OUTPUT_KEYS.final_logits!r}]."
            )

        return self._forward_final(outputs=outputs, targets=targets)

    def _sam3_mask_distill_loss(
        self,
        final_logits: torch.Tensor,
        sam3_teacher_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        presence_target: torch.Tensor,
    ) -> torch.Tensor:
        """Distillation BCE: frozen SAM3 teacher → trainable student mask.

        Teacher logits are sigmoided to produce soft probability targets.
        Student stays in raw logit space for BCEWithLogits.
        Only present image-class pairs and valid pixels participate.
        """
        if final_logits.ndim != 4:
            raise ValueError(
                f"final_logits must be [B, C, H, W], got {tuple(final_logits.shape)}."
            )
        if sam3_teacher_logits.ndim != 4:
            raise ValueError(
                f"sam3_teacher_logits must be [B, C, H, W], "
                f"got {tuple(sam3_teacher_logits.shape)}."
            )

        if tuple(final_logits.shape) != tuple(sam3_teacher_logits.shape):
            raise ValueError(
                "Shape mismatch between final_logits "
                f"{tuple(final_logits.shape)} and "
                f"sam3_teacher_logits {tuple(sam3_teacher_logits.shape)}."
            )

        if tuple(final_logits.shape[-2:]) != (288, 288):
            raise ValueError(
                f"Both logits must be 288×288, "
                f"got {tuple(final_logits.shape[-2:])}."
            )

        teacher_prob = sam3_teacher_logits.detach().sigmoid()

        pixel_ce = F.binary_cross_entropy_with_logits(
            final_logits,
            teacher_prob,
            reduction="none",
        )

        present = presence_target > 0.5
        if not present.any().item():
            return final_logits.sum() * 0.0

        pair_pixel_mask = (
            present.to(device=pixel_ce.device, dtype=torch.bool)[:, :, None, None]
            & valid_mask.to(device=pixel_ce.device, dtype=torch.bool)[:, None, :, :]
        )
        pixel_weight = pair_pixel_mask.to(dtype=pixel_ce.dtype)

        denom = pixel_weight.sum().clamp_min(float(self.cfg.eps))
        return (pixel_ce * pixel_weight).sum() / denom

    def _forward_final(
        self,
        outputs: TensorDict,
        targets: TensorDict,
    ) -> TensorDict:
        final_logits = self._extract_4d_tensor(
            outputs,
            OUTPUT_KEYS.final_logits,
            "[B, C, H, W]",
        )

        B, C, H, W = final_logits.shape

        label_map = self._extract_label_map(targets)
        label_map = self._resize_label_map_to_hw(
            label_map=label_map,
            target_hw=(H, W),
        )

        class_ids = list(range(C))

        target, valid_mask = self._build_binary_targets(
            label_map=label_map,
            class_ids=class_ids,
            num_channels=C,
            dtype=final_logits.dtype,
        )

        presence_target = self._build_presence_target(
            label_map=label_map,
            valid_mask=valid_mask,
            class_ids=class_ids,
            dtype=final_logits.dtype,
        )

        zero = self._zero_loss(final_logits)

        # 主 BCE：所有图像、类别、像素等权计算
        loss_final_bce = F.binary_cross_entropy_with_logits(
            final_logits,
            target,
            reduction="mean",
        )

        if float(self.cfg.final_dice_weight) > 0.0 and bool(
            presence_target.bool().any().item()
        ):
            loss_final_dice = self._dice_loss_present_mean_from_logits(
                logits=final_logits,
                target=target,
                presence_target=presence_target,
            )
        else:
            loss_final_dice = zero

        distill_weight = float(self.cfg.sam3_mask_distill_weight)
        if not math.isfinite(distill_weight) or distill_weight < 0.0:
            raise ValueError(
                "criterion_cfg.sam3_mask_distill_weight must be finite and "
                f"non-negative, got {self.cfg.sam3_mask_distill_weight!r}."
            )

        if distill_weight > 0.0:
            if OUTPUT_KEYS.sam3_teacher_logits not in outputs:
                raise ValueError(
                    "criterion_cfg.sam3_mask_distill_weight > 0 "
                    f"({distill_weight}), but "
                    f"{OUTPUT_KEYS.sam3_teacher_logits!r} is missing from "
                    "outputs. Ensure the model returns teacher logits "
                    "in training mode."
                )

            loss_sam3_mask_distill_bce = self._sam3_mask_distill_loss(
                final_logits=final_logits,
                sam3_teacher_logits=outputs[OUTPUT_KEYS.sam3_teacher_logits],
                valid_mask=valid_mask,
                presence_target=presence_target,
            )
        else:
            loss_sam3_mask_distill_bce = zero

        total_loss = (
            float(self.cfg.final_bce_weight) * loss_final_bce
            + float(self.cfg.final_dice_weight) * loss_final_dice
            + distill_weight * loss_sam3_mask_distill_bce
        )

        return {
            "loss_final_bce": loss_final_bce,
            "loss_final_dice": loss_final_dice,
            "loss_sam3_mask_distill_bce": loss_sam3_mask_distill_bce,
            "total_loss": total_loss,
        }

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _extract_4d_tensor(
        outputs: TensorDict,
        key: str,
        shape_name: str,
    ) -> torch.Tensor:
        tensor = outputs.get(key, None)
        if tensor is None:
            raise ValueError(f"SemanticCriterion expects outputs[{key!r}].")
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected {key} as {shape_name}, got {tuple(tensor.shape)}."
            )
        return tensor

    def _extract_label_map(self, targets: TensorDict) -> torch.Tensor:
        if "label_map" not in targets:
            raise ValueError("SemanticCriterion expects targets['label_map'].")

        label_map = targets["label_map"]

        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    "Expected label_map as [B, 1, H, W] or [B, H, W]."
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                "Expected label_map as [B, H, W] or [B, 1, H, W]."
            )

        return label_map.long()

    @staticmethod
    def _resize_label_map_to_hw(
        label_map: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tuple(label_map.shape[-2:]) == tuple(target_hw):
            return label_map

        return F.interpolate(
            label_map[:, None].float(),
            size=target_hw,
            mode="nearest",
        )[:, 0].long()

    def _build_binary_targets(
        self,
        label_map: torch.Tensor,
        class_ids: Sequence[int],
        num_channels: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构建二值目标，行为与 RSKT-Seg 一致。

        标签 0..C-1 → 对应类别通道为 1，其余通道为 0。
        标签 255 → 所有类别通道均为 0，作为全类别负样本参与 BCE。
        """
        B, H, W = label_map.shape
        ignore_index = int(self.cfg.ignore_index)
        valid_mask = label_map != ignore_index

        # 先构造 [B, H, W, C]，valid_mask 索引后再 permute 到 [B, C, H, W]
        target = torch.zeros(
            (B, H, W, int(num_channels)),
            dtype=dtype,
            device=label_map.device,
        )

        if valid_mask.any():
            valid_labels = label_map[valid_mask]
            if valid_labels.min().item() < 0 or valid_labels.max().item() >= num_channels:
                raise ValueError(
                    f"Valid labels out of range: min={valid_labels.min().item()}, "
                    f"max={valid_labels.max().item()}, num_classes={num_channels}. "
                    "All non-ignore labels must be in [0, num_classes)."
                )
            one_hot = F.one_hot(valid_labels, num_classes=int(num_channels)).to(dtype=dtype)
            target[valid_mask] = one_hot

        target = target.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]

        return target, valid_mask

    @staticmethod
    def _build_presence_target(
        label_map: torch.Tensor,
        valid_mask: torch.Tensor,
        class_ids: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        B = int(label_map.shape[0])
        C = len(class_ids)

        presence_target = torch.zeros(
            (B, C),
            dtype=dtype,
            device=label_map.device,
        )

        for channel_idx, class_id in enumerate(class_ids):
            appears = ((label_map == int(class_id)) & valid_mask).flatten(1).any(dim=1)
            presence_target[:, channel_idx] = appears.to(dtype=dtype)

        return presence_target

    def _dice_loss_present_mean_from_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        presence_target: torch.Tensor,
    ) -> torch.Tensor:
        prob = logits.sigmoid()

        prob = prob.flatten(2)
        target = target.flatten(2)

        intersection = (prob * target).sum(dim=2)
        denominator = prob.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + float(self.cfg.eps)) / (
            denominator + float(self.cfg.eps)
        )
        dice_loss = 1.0 - dice

        pair_weight = presence_target.to(dtype=dice_loss.dtype)

        weight_sum = pair_weight.sum()
        if bool(weight_sum.detach().le(0).item()):
            return logits.sum() * 0.0

        return (dice_loss * pair_weight).sum() / weight_sum.clamp_min(float(self.cfg.eps))


class HybridCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HybridCriterion is not implemented yet.")
