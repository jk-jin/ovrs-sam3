from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


@dataclass
class SemanticLossWeights:
    semantic_bce: float = 1.0
    semantic_dice: float = 1.0
    instance_bce: float = 1.0
    instance_dice: float = 1.0
    presence_bce: float = 0.25


class SemanticCriterion(nn.Module):
    """
    训练目标：
    1. semantic_branch_logits: 每个类别一张二值 mask 的原始 logits
    2. instance_branch_logits: 每个类别一张二值 mask 的原始 logits
    3. presence_logits: 每个类别是否存在的原始 logits

    目标标签来自单张 label_map:
    - label_map: [B, H, W] 或 [B, 1, H, W]

    在 chunk 训练下：
    - chunk_class_ids 给出当前 chunk 覆盖的全局类别 id
    - criterion 会把 label_map 转成当前 chunk 的多通道二值目标
    """

    def __init__(
        self,
        weights: SemanticLossWeights | None = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.weights = weights or SemanticLossWeights()
        self.ignore_index = int(ignore_index)

    def _prepare_label_map(
        self,
        targets: TensorDict,
        out_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if "label_map" not in targets:
            raise ValueError("label_map is required in targets.")

        label_map = targets["label_map"]
        if label_map is None:
            raise ValueError("label_map is None.")

        if label_map.dim() == 4:
            if label_map.shape[1] != 1:
                raise ValueError(
                    f"Expected label_map as [B, H, W] or [B, 1, H, W], got {tuple(label_map.shape)}"
                )
            label_map = label_map[:, 0]
        elif label_map.dim() != 3:
            raise ValueError(
                f"Expected label_map as [B, H, W] or [B, 1, H, W], got {tuple(label_map.shape)}"
            )

        label_map = label_map.long().to(device)

        if tuple(label_map.shape[-2:]) != tuple(out_hw):
            label_map = F.interpolate(
                label_map[:, None].float(),
                size=out_hw,
                mode="nearest",
            )[:, 0].long()

        return label_map

    @staticmethod
    def _resolve_class_ids(
        num_classes: int,
        chunk_class_ids: Optional[Sequence[int]],
    ) -> list[int]:
        if chunk_class_ids is None:
            return list(range(num_classes))

        class_ids = [int(x) for x in chunk_class_ids]
        if len(class_ids) != num_classes:
            raise ValueError(
                f"Class count mismatch: logits has {num_classes} channels, "
                f"but chunk_class_ids has length {len(class_ids)}."
            )

        if len(set(class_ids)) != len(class_ids):
            raise ValueError(f"Duplicate class ids found in chunk_class_ids: {class_ids}")

        return class_ids

    def _build_multilabel_targets(
        self,
        label_map: torch.Tensor,                  # [B, H, W]
        num_classes: int,
        chunk_class_ids: Optional[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
        - binary_targets: [B, C, H, W]
        - valid_mask: [B, 1, H, W]
        - presence_targets: [B, C]
        """
        class_ids = self._resolve_class_ids(
            num_classes=num_classes,
            chunk_class_ids=chunk_class_ids,
        )

        b, h, w = label_map.shape
        device = label_map.device

        valid_mask = (label_map != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
        binary_targets = torch.zeros(
            (b, num_classes, h, w),
            dtype=torch.float32,
            device=device,
        )

        for local_id, global_id in enumerate(class_ids):
            binary_targets[:, local_id] = (label_map == global_id).to(torch.float32)

        binary_targets = binary_targets * valid_mask.to(binary_targets.dtype)
        presence_targets = binary_targets.flatten(2).amax(dim=2)  # [B, C]

        return binary_targets, valid_mask, presence_targets

    @staticmethod
    def _masked_multilabel_bce_mean(
        logits: torch.Tensor,        # [B, C, H, W]
        targets: torch.Tensor,       # [B, C, H, W]
        valid_mask: torch.Tensor,    # [B, 1, H, W]
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
            )

        if valid_mask.dim() != 4 or valid_mask.shape[1] != 1:
            raise ValueError(
                f"Expected valid_mask as [B, 1, H, W], got {tuple(valid_mask.shape)}"
            )

        mask = valid_mask.expand_as(logits).to(logits.dtype)  # [B, C, H, W]

        loss_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        denom = mask.sum().clamp(min=1.0)
        return (loss_map * mask).sum() / denom

    def _multilabel_dice_mean(
        self,
        logits: torch.Tensor,        # [B, C, H, W]
        targets: torch.Tensor,       # [B, C, H, W]
        valid_mask: torch.Tensor,    # [B, 1, H, W]
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
            )

        if valid_mask.dim() != 4 or valid_mask.shape[1] != 1:
            raise ValueError(
                f"Expected valid_mask as [B, 1, H, W], got {tuple(valid_mask.shape)}"
            )

        probs = logits.sigmoid()
        mask = valid_mask.to(logits.dtype)

        probs = probs * mask
        targets = targets * mask

        # 对每个 batch、每个类别分别算 dice
        dims = (2, 3)
        intersection = (probs * targets).sum(dim=dims)        # [B, C]
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)  # [B, C]

        # 只在该类别有正样本时计算 dice
        positive_class_mask = targets.sum(dim=dims) > 0  # [B, C]

        if positive_class_mask.any():
            dice = (2.0 * intersection[positive_class_mask] + 1.0) / (
                denominator[positive_class_mask] + 1.0
            )
            return 1.0 - dice.mean()

        return logits.sum() * 0.0

    @staticmethod
    def _presence_bce_mean(
        presence_logits: torch.Tensor,   # [B, C]
        presence_targets: torch.Tensor,  # [B, C]
    ) -> torch.Tensor:
        if presence_logits.shape != presence_targets.shape:
            raise ValueError(
                f"presence_logits and presence_targets shape mismatch: "
                f"{tuple(presence_logits.shape)} vs {tuple(presence_targets.shape)}"
            )

        return F.binary_cross_entropy_with_logits(
            presence_logits,
            presence_targets,
            reduction="mean",
        )

    @staticmethod
    def _scale_loss_for_reduction(
        base_loss: torch.Tensor,
        num_valid_pixels: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        if reduction == "mean":
            return base_loss

        if reduction == "sum":
            return base_loss * num_valid_pixels.to(dtype=base_loss.dtype)

        raise ValueError(f"Unsupported reduction={reduction}. Expected 'mean' or 'sum'.")

    @staticmethod
    def _select_output_hw(outputs: TensorDict) -> tuple[int, int]:
        for key in ("semantic_branch_logits", "instance_branch_logits"):
            value = outputs.get(key, None)
            if value is not None:
                if value.dim() != 4:
                    raise ValueError(
                        f"Expected {key} as [B, C, H, W], got {tuple(value.shape)}"
                    )
                return tuple(value.shape[-2:])

        raise ValueError(
            "At least one of semantic_branch_logits or instance_branch_logits must be present."
        )

    def forward(
        self,
        outputs: TensorDict,
        targets: TensorDict,
        chunk_class_ids: Optional[Sequence[int]] = None,
        reduction: str = "mean",
    ) -> TensorDict:
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction={reduction}. Expected 'mean' or 'sum'.")

        semantic_branch_logits = outputs.get("semantic_branch_logits", None)
        instance_branch_logits = outputs.get("instance_branch_logits", None)
        presence_logits = outputs.get("presence_logits", None)

        if semantic_branch_logits is None and instance_branch_logits is None:
            raise ValueError(
                "At least one of semantic_branch_logits or instance_branch_logits must be present."
            )

        out_hw = self._select_output_hw(outputs)

        device = None
        num_classes = None

        if semantic_branch_logits is not None:
            device = semantic_branch_logits.device
            num_classes = int(semantic_branch_logits.shape[1])

        if instance_branch_logits is not None:
            if device is None:
                device = instance_branch_logits.device
                num_classes = int(instance_branch_logits.shape[1])
            else:
                if instance_branch_logits.device != device:
                    raise ValueError("semantic_branch_logits and instance_branch_logits are on different devices.")
                if int(instance_branch_logits.shape[1]) != int(num_classes):
                    raise ValueError(
                        "semantic_branch_logits and instance_branch_logits must have the same class count."
                    )

        if device is None or num_classes is None:
            raise RuntimeError("Failed to infer device or num_classes from outputs.")

        label_map = self._prepare_label_map(
            targets=targets,
            out_hw=out_hw,
            device=device,
        )

        binary_targets, valid_mask, presence_targets = self._build_multilabel_targets(
            label_map=label_map,
            num_classes=num_classes,
            chunk_class_ids=chunk_class_ids,
        )

        num_valid_pixels = valid_mask.sum()
        if int(num_valid_pixels.item()) == 0:
            zero = (
                (semantic_branch_logits.sum() if semantic_branch_logits is not None else 0.0)
                if isinstance(semantic_branch_logits, torch.Tensor)
                else 0.0
            )
            if not torch.is_tensor(zero):
                zero = instance_branch_logits.sum() * 0.0

            return {
                "loss_semantic_bce": zero,
                "loss_semantic_dice": zero,
                "loss_instance_bce": zero,
                "loss_instance_dice": zero,
                "loss_presence_bce": zero,
                "total_loss": zero,
                "num_valid": num_valid_pixels,
            }

        if semantic_branch_logits is not None:
            semantic_bce_base = self._masked_multilabel_bce_mean(
                logits=semantic_branch_logits,
                targets=binary_targets,
                valid_mask=valid_mask,
            )
            semantic_dice_base = self._multilabel_dice_mean(
                logits=semantic_branch_logits,
                targets=binary_targets,
                valid_mask=valid_mask,
            )

            loss_semantic_bce = self._scale_loss_for_reduction(
                base_loss=semantic_bce_base,
                num_valid_pixels=num_valid_pixels,
                reduction=reduction,
            )
            loss_semantic_dice = self._scale_loss_for_reduction(
                base_loss=semantic_dice_base,
                num_valid_pixels=num_valid_pixels,
                reduction=reduction,
            )
        else:
            ref = instance_branch_logits
            loss_semantic_bce = ref.sum() * 0.0
            loss_semantic_dice = ref.sum() * 0.0

        if instance_branch_logits is not None:
            instance_bce_base = self._masked_multilabel_bce_mean(
                logits=instance_branch_logits,
                targets=binary_targets,
                valid_mask=valid_mask,
            )
            instance_dice_base = self._multilabel_dice_mean(
                logits=instance_branch_logits,
                targets=binary_targets,
                valid_mask=valid_mask,
            )

            loss_instance_bce = self._scale_loss_for_reduction(
                base_loss=instance_bce_base,
                num_valid_pixels=num_valid_pixels,
                reduction=reduction,
            )
            loss_instance_dice = self._scale_loss_for_reduction(
                base_loss=instance_dice_base,
                num_valid_pixels=num_valid_pixels,
                reduction=reduction,
            )
        else:
            ref = semantic_branch_logits
            loss_instance_bce = ref.sum() * 0.0
            loss_instance_dice = ref.sum() * 0.0

        if presence_logits is not None:
            if presence_logits.dim() != 2:
                raise ValueError(
                    f"Expected presence_logits as [B, C], got {tuple(presence_logits.shape)}"
                )
            if tuple(presence_logits.shape) != tuple(presence_targets.shape):
                raise ValueError(
                    f"presence_logits shape mismatch: expected {tuple(presence_targets.shape)}, "
                    f"got {tuple(presence_logits.shape)}"
                )

            presence_bce_base = self._presence_bce_mean(
                presence_logits=presence_logits,
                presence_targets=presence_targets,
            )
            loss_presence_bce = self._scale_loss_for_reduction(
                base_loss=presence_bce_base,
                num_valid_pixels=num_valid_pixels,
                reduction=reduction,
            )
        else:
            ref = semantic_branch_logits if semantic_branch_logits is not None else instance_branch_logits
            loss_presence_bce = ref.sum() * 0.0

        total_loss = (
            self.weights.semantic_bce * loss_semantic_bce
            + self.weights.semantic_dice * loss_semantic_dice
            + self.weights.instance_bce * loss_instance_bce
            + self.weights.instance_dice * loss_instance_dice
            + self.weights.presence_bce * loss_presence_bce
        )

        return {
            "loss_semantic_bce": loss_semantic_bce,
            "loss_semantic_dice": loss_semantic_dice,
            "loss_instance_bce": loss_instance_bce,
            "loss_instance_dice": loss_instance_dice,
            "loss_presence_bce": loss_presence_bce,
            "total_loss": total_loss,
            "num_valid": num_valid_pixels,
        }