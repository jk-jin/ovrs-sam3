from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.task_modes import OUTPUT_KEYS


@dataclass
class SemanticCriterionConfig:
	ignore_index: int = 255

	bce_weight: float = 1.0
	dice_weight: float = 1.0
	presence_bce_weight: float = 1.0
	final_bce_weight: float = 0.4
	final_dice_weight: float = 0.5
	final_ce_weight: float = 1.0

	eps: float = 1e-6

	bce_class_balance_clamp_min: float = 0.2
	bce_class_balance_clamp_max: float = 5.0

	presence_pos_weight: float = 1.0


class SemanticCriterion(nn.Module):
	def __init__(self, cfg: Optional[SemanticCriterionConfig] = None):
		super().__init__()
		self.cfg = cfg or SemanticCriterionConfig()

	def _extract_semantic_logits(
		self,
		outputs: Dict[str, torch.Tensor],
	) -> torch.Tensor:
		if OUTPUT_KEYS.semantic_logits not in outputs:
			raise ValueError(
				f"SemanticCriterion expects outputs['{OUTPUT_KEYS.semantic_logits}']."
			)

		logits = outputs[OUTPUT_KEYS.semantic_logits]
		if logits.dim() != 4:
			raise ValueError(
				f"Expected semantic_logits as [B, C, H, W], got {tuple(logits.shape)}"
			)
		return logits

	def _extract_final_logits(
			self,
			outputs: Dict[str, torch.Tensor],
	) -> torch.Tensor:
		if OUTPUT_KEYS.final_logits not in outputs:
			raise ValueError(
				f"SemanticCriterion expects outputs['{OUTPUT_KEYS.final_logits}']."
			)

		logits = outputs[OUTPUT_KEYS.final_logits]
		if logits.dim() != 4:
			raise ValueError(
				f"Expected final_logits as [B, C, H, W], got {tuple(logits.shape)}"
			)
		return logits

	def _extract_presence_logits(
		self,
		outputs: Dict[str, torch.Tensor],
	) -> torch.Tensor:
		if OUTPUT_KEYS.presence_logits not in outputs:
			raise ValueError(
				f"SemanticCriterion expects outputs['{OUTPUT_KEYS.presence_logits}']."
			)

		logits = outputs[OUTPUT_KEYS.presence_logits]
		if logits.dim() == 3:
			if logits.shape[-1] != 1:
				raise ValueError(
					f"Expected presence_logits as [B, C, 1], got {tuple(logits.shape)}"
				)
			logits = logits[..., 0]
		elif logits.dim() != 2:
			raise ValueError(
				f"Expected presence_logits as [B, C] or [B, C, 1], got {tuple(logits.shape)}"
			)

		return logits

	def _extract_label_map(
		self,
		targets: Dict[str, torch.Tensor],
	) -> torch.Tensor:
		if "label_map" not in targets:
			raise ValueError("SemanticCriterion expects targets['label_map'].")

		label_map = targets["label_map"]
		if label_map.dim() == 4:
			if label_map.shape[1] != 1:
				raise ValueError(
					f"Expected label_map as [B,1,H,W] or [B,H,W], got {tuple(label_map.shape)}"
				)
			label_map = label_map[:, 0]
		elif label_map.dim() != 3:
			raise ValueError(
				f"Expected label_map as [B,H,W] or [B,1,H,W], got {tuple(label_map.shape)}"
			)

		return label_map.long()

	def _resize_label_map_to_logits(
		self,
		label_map: torch.Tensor,
		target_hw: tuple[int, int],
	) -> torch.Tensor:
		if tuple(label_map.shape[-2:]) == tuple(target_hw):
			return label_map

		resized = F.interpolate(
			label_map[:, None].float(),
			size=target_hw,
			mode="nearest",
		)[:, 0]
		return resized.long()

	def _build_chunk_targets(
		self,
		label_map: torch.Tensor,
		chunk_class_ids: Sequence[int],
		num_channels: int,
	) -> tuple[torch.Tensor, torch.Tensor]:
		if len(chunk_class_ids) != num_channels:
			raise ValueError(
				f"chunk_class_ids length mismatch: expected {num_channels}, got {len(chunk_class_ids)}"
			)

		bsz, h, w = label_map.shape
		device = label_map.device

		valid_mask = label_map != int(self.cfg.ignore_index)
		target = torch.zeros((bsz, num_channels, h, w), dtype=torch.float32, device=device)

		for ch, class_id in enumerate(chunk_class_ids):
			target[:, ch] = (label_map == int(class_id)).to(torch.float32)

		valid_mask_4d = valid_mask[:, None].expand(bsz, num_channels, h, w)
		return target, valid_mask_4d

	def _build_chunk_ce_label_map(
			self,
			label_map: torch.Tensor,
			chunk_class_ids: Sequence[int],
	) -> torch.Tensor:
		"""
		把全局类别 id 的 label_map 映射成当前 chunk 内部的局部类别 id。
		不属于当前 chunk 的像素，以及原始 ignore_index，都设成 ignore_index。

		Args:
			label_map: [B, H, W], 值是全局类别 id
			chunk_class_ids: 当前 chunk 覆盖的全局类别 id 列表，长度为 C

		Returns:
			ce_label_map: [B, H, W], 值域是 [0, C-1] 或 ignore_index
		"""
		if label_map.dim() != 3:
			raise ValueError(
				f"Expected label_map as [B, H, W], got {tuple(label_map.shape)}"
			)

		ce_label_map = torch.full_like(label_map, fill_value=int(self.cfg.ignore_index))

		for local_idx, global_class_id in enumerate(chunk_class_ids):
			ce_label_map[label_map == int(global_class_id)] = int(local_idx)

		return ce_label_map.long()

	def _build_present_pair_mask(
		self,
		target: torch.Tensor,
		valid_mask: torch.Tensor,
	) -> torch.Tensor:
		"""
		Args:
			target:     [B, C, H, W]
			valid_mask: [B, C, H, W]
		Returns:
			present_pair_mask: [B, C]
				True 表示该图该类在有效区域内存在前景像素
		"""
		target_valid = target * valid_mask.to(target.dtype)
		fg_pixels_per_pair = target_valid.flatten(2).sum(dim=2)  # [B, C]
		present_pair_mask = fg_pixels_per_pair > 0
		return present_pair_mask

	def _build_presence_targets(
		self,
		present_pair_mask: torch.Tensor,
	) -> torch.Tensor:
		"""
		Args:
			present_pair_mask: [B, C], bool
		Returns:
			presence_target: [B, C], float
		"""
		if present_pair_mask.dim() != 2:
			raise ValueError(
				f"Expected present_pair_mask as [B, C], got {tuple(present_pair_mask.shape)}"
			)
		return present_pair_mask.to(torch.float32)

	def _build_dynamic_class_weights(
		self,
		target: torch.Tensor,
		valid_mask: torch.Tensor,
		present_pair_mask: torch.Tensor,
	) -> torch.Tensor:
		"""
		只对 present 的类别对构造动态类别权重。
		权重和该类在图上的像素数成反比，像素越少权重越大。

		Args:
			target:            [B, C, H, W]
			valid_mask:        [B, C, H, W]
			present_pair_mask: [B, C]

		Returns:
			class_weights: [B, C]
		"""
		target_valid = target * valid_mask.to(target.dtype)              # [B, C, H, W]
		fg_pixels = target_valid.flatten(2).sum(dim=2)                   # [B, C]

		class_weights = torch.zeros_like(fg_pixels, dtype=target.dtype)  # [B, C]

		if present_pair_mask.any():
			present_fg = fg_pixels[present_pair_mask]                    # [N_present]
			mean_fg = present_fg.mean().clamp_min(1.0)
			class_weights[present_pair_mask] = mean_fg / fg_pixels[present_pair_mask].clamp_min(1.0)

		class_weights = class_weights.clamp(
			min=float(self.cfg.bce_class_balance_clamp_min),
			max=float(self.cfg.bce_class_balance_clamp_max),
		)

		return class_weights

	def _binary_cross_entropy_present_balanced_mean(
		self,
		logits: torch.Tensor,
		target: torch.Tensor,
		valid_mask: torch.Tensor,
		present_pair_mask: torch.Tensor,
		class_weights: torch.Tensor,
	) -> torch.Tensor:
		"""
		只对图上存在的类别通道计算 BCE。
		同时按类别像素数动态加权，防止背景类压制小类。

		Args:
			logits:            [B, C, H, W]
			target:            [B, C, H, W]
			valid_mask:        [B, C, H, W]
			present_pair_mask: [B, C]
			class_weights:     [B, C]
		"""
		pair_mask_4d = present_pair_mask[:, :, None, None]               # [B, C, 1, 1]
		effective_mask = valid_mask & pair_mask_4d                       # [B, C, H, W]

		per_elem = F.binary_cross_entropy_with_logits(
			logits,
			target,
			reduction="none",
		)                                                                # [B, C, H, W]

		weight_4d = class_weights[:, :, None, None]                      # [B, C, 1, 1]
		per_elem = per_elem * weight_4d
		per_elem = per_elem * effective_mask.to(per_elem.dtype)

		denom = (effective_mask.to(per_elem.dtype) * weight_4d).sum().clamp_min(1.0)
		return per_elem.sum() / denom

	def _dice_loss_present_mean_from_logits(
		self,
		logits: torch.Tensor,
		target: torch.Tensor,
		valid_mask: torch.Tensor,
		present_pair_mask: torch.Tensor,
	) -> torch.Tensor:
		"""
		只对图上存在的类别通道计算 Dice loss，并求平均。
		输入是 logits。
		"""
		prob = logits.sigmoid()
		return self._dice_loss_present_mean_from_probs(
			prob=prob,
			target=target,
			valid_mask=valid_mask,
			present_pair_mask=present_pair_mask,
		)

	def _dice_loss_present_mean_from_probs(
		self,
		prob: torch.Tensor,
		target: torch.Tensor,
		valid_mask: torch.Tensor,
		present_pair_mask: torch.Tensor,
	) -> torch.Tensor:
		"""
		只对图上存在的类别通道计算 Dice loss，并求平均。
		输入是概率图 prob，范围应在 [0, 1]。

		Args:
			prob:              [B, C, H, W]
			target:            [B, C, H, W]
			valid_mask:        [B, C, H, W]
			present_pair_mask: [B, C]
		"""
		if prob.dim() != 4:
			raise ValueError(f"Expected prob as [B, C, H, W], got {tuple(prob.shape)}")
		if target.shape != prob.shape:
			raise ValueError(
				f"Shape mismatch between prob and target: {tuple(prob.shape)} vs {tuple(target.shape)}"
			)
		if valid_mask.shape != prob.shape:
			raise ValueError(
				f"Shape mismatch between prob and valid_mask: {tuple(prob.shape)} vs {tuple(valid_mask.shape)}"
			)
		if present_pair_mask.shape != prob.shape[:2]:
			raise ValueError(
				f"Shape mismatch between prob and present_pair_mask: {tuple(prob.shape[:2])} vs {tuple(present_pair_mask.shape)}"
			)

		prob = prob * valid_mask.to(prob.dtype)
		target = target * valid_mask.to(target.dtype)

		prob = prob.flatten(2)    # [B, C, H*W]
		target = target.flatten(2)

		intersection = (prob * target).sum(dim=2)           # [B, C]
		denominator = prob.sum(dim=2) + target.sum(dim=2)   # [B, C]

		dice = (2.0 * intersection + self.cfg.eps) / (denominator + self.cfg.eps)
		dice_loss = 1.0 - dice                              # [B, C]

		pair_weight = present_pair_mask.to(dice_loss.dtype)
		valid_pair_count = pair_weight.sum().clamp_min(1.0)

		return (dice_loss * pair_weight).sum() / valid_pair_count

	def _presence_bce_loss(
		self,
		presence_logits: torch.Tensor,
		presence_target: torch.Tensor,
	) -> torch.Tensor:
		"""
		Args:
			presence_logits: [B, C]
			presence_target: [B, C]
		"""
		if presence_logits.dim() != 2:
			raise ValueError(
				f"Expected presence_logits as [B, C], got {tuple(presence_logits.shape)}"
			)
		if presence_target.dim() != 2:
			raise ValueError(
				f"Expected presence_target as [B, C], got {tuple(presence_target.shape)}"
			)
		if presence_logits.shape != presence_target.shape:
			raise ValueError(
				f"Shape mismatch between presence_logits and presence_target: "
				f"{tuple(presence_logits.shape)} vs {tuple(presence_target.shape)}"
			)

		pos_weight = torch.as_tensor(
			float(self.cfg.presence_pos_weight),
			dtype=presence_logits.dtype,
			device=presence_logits.device,
		)

		return F.binary_cross_entropy_with_logits(
			presence_logits,
			presence_target,
			pos_weight=pos_weight,
			reduction="mean",
		)

	def _final_cross_entropy_loss(
			self,
			final_logits: torch.Tensor,
			ce_label_map: torch.Tensor,
	) -> tuple[torch.Tensor, int]:
		"""
		对当前 chunk 的 final_logits 计算标准多类交叉熵。
		只在当前 chunk 覆盖到的类别像素上计算，其余像素为 ignore_index。

		Args:
			final_logits: [B, C, H, W]
			ce_label_map: [B, H, W], 值域是 [0, C-1] 或 ignore_index

		Returns:
			loss_final_ce: 标量
			num_ce_valid: 当前 chunk 中参与 CE 的有效像素数
		"""
		if final_logits.dim() != 4:
			raise ValueError(
				f"Expected final_logits as [B, C, H, W], got {tuple(final_logits.shape)}"
			)
		if ce_label_map.dim() != 3:
			raise ValueError(
				f"Expected ce_label_map as [B, H, W], got {tuple(ce_label_map.shape)}"
			)
		if final_logits.shape[0] != ce_label_map.shape[0] or final_logits.shape[-2:] != ce_label_map.shape[-2:]:
			raise ValueError(
				f"Shape mismatch between final_logits and ce_label_map: "
				f"{tuple(final_logits.shape)} vs {tuple(ce_label_map.shape)}"
			)

		num_ce_valid = int((ce_label_map != int(self.cfg.ignore_index)).sum().item())
		if num_ce_valid <= 0:
			zero = final_logits.sum() * 0.0
			return zero, 0

		loss_final_ce = F.cross_entropy(
			final_logits,
			ce_label_map,
			ignore_index=int(self.cfg.ignore_index),
			reduction="mean",
		)
		return loss_final_ce, num_ce_valid
	
	def forward(
		self,
		outputs: Dict[str, torch.Tensor],
		targets: Dict[str, torch.Tensor],
		chunk_class_ids: Optional[Sequence[int]] = None,
		reduction: str = "mean",
	) -> Dict[str, torch.Tensor]:
		if reduction != "mean":
			raise ValueError(
				f"SemanticCriterion only supports reduction='mean', got {reduction!r}"
			)

		semantic_logits = self._extract_semantic_logits(outputs)
		presence_logits = self._extract_presence_logits(outputs)
		final_logits = self._extract_final_logits(outputs)

		if semantic_logits.shape[:2] != presence_logits.shape:
			raise ValueError(
				"Shape mismatch between semantic_logits and presence_logits: "
				f"semantic_logits.shape[:2]={tuple(semantic_logits.shape[:2])}, "
				f"presence_logits.shape={tuple(presence_logits.shape)}"
			)

		if final_logits.shape != semantic_logits.shape:
			raise ValueError(
				"Shape mismatch between final_logits and semantic_logits: "
				f"final_logits.shape={tuple(final_logits.shape)}, "
				f"semantic_logits.shape={tuple(semantic_logits.shape)}"
			)

		label_map = self._extract_label_map(targets)
		label_map = self._resize_label_map_to_logits(
			label_map,
			target_hw=tuple(semantic_logits.shape[-2:]),
		)

		num_channels = int(semantic_logits.shape[1])
		if chunk_class_ids is None:
			raise ValueError(
				"SemanticCriterion requires chunk_class_ids for chunk-wise semantic training."
			)

		target, valid_mask = self._build_chunk_targets(
			label_map=label_map,
			chunk_class_ids=chunk_class_ids,
			num_channels=num_channels,
		)

		ce_label_map = self._build_chunk_ce_label_map(
			label_map=label_map,
			chunk_class_ids=chunk_class_ids,
		)

		num_valid_pixels = int((label_map != int(self.cfg.ignore_index)).sum().item())
		if num_valid_pixels <= 0:
			zero = semantic_logits.sum() * 0.0
			return {
				"loss_semantic_bce": zero,
				"loss_semantic_dice": zero,
				"loss_presence_bce": zero,
				"loss_final_bce": zero,
				"loss_final_dice": zero,
				"loss_final_ce": zero,
				"total_loss": zero,
				"num_valid": torch.tensor(0, device=semantic_logits.device, dtype=torch.long),
			}

		present_pair_mask = self._build_present_pair_mask(
			target=target,
			valid_mask=valid_mask,
		)  # [B, C]

		presence_target = self._build_presence_targets(
			present_pair_mask=present_pair_mask,
		)  # [B, C]

		loss_presence_bce = self._presence_bce_loss(
			presence_logits=presence_logits,
			presence_target=presence_target,
		)

		num_present_pairs = int(present_pair_mask.sum().item())
		if num_present_pairs <= 0:
			zero = semantic_logits.sum() * 0.0
			loss_final_ce, num_ce_valid = self._final_cross_entropy_loss(
				final_logits=final_logits,
				ce_label_map=ce_label_map,
			)

			total_loss = (
					float(self.cfg.presence_bce_weight) * loss_presence_bce
					+ float(self.cfg.final_ce_weight) * loss_final_ce
			)

			num_valid_for_backward = max(num_valid_pixels, num_ce_valid)

			return {
				"loss_semantic_bce": zero,
				"loss_semantic_dice": zero,
				"loss_presence_bce": loss_presence_bce,
				"loss_final_bce": zero,
				"loss_final_dice": zero,
				"loss_final_ce": loss_final_ce,
				"total_loss": total_loss,
				"num_valid": torch.tensor(
					num_valid_for_backward,
					device=semantic_logits.device,
					dtype=torch.long,
				),
			}

		class_weights = self._build_dynamic_class_weights(
			target=target,
			valid_mask=valid_mask,
			present_pair_mask=present_pair_mask,
		)  # [B, C]

		loss_semantic_bce = self._binary_cross_entropy_present_balanced_mean(
			logits=semantic_logits,
			target=target,
			valid_mask=valid_mask,
			present_pair_mask=present_pair_mask,
			class_weights=class_weights,
		)

		loss_semantic_dice = self._dice_loss_present_mean_from_logits(
			logits=semantic_logits,
			target=target,
			valid_mask=valid_mask,
			present_pair_mask=present_pair_mask,
		)

		loss_final_bce = self._binary_cross_entropy_present_balanced_mean(
			logits=final_logits,
			target=target,
			valid_mask=valid_mask,
			present_pair_mask=present_pair_mask,
			class_weights=class_weights,
		)

		loss_final_dice = self._dice_loss_present_mean_from_logits(
			logits=final_logits,
			target=target,
			valid_mask=valid_mask,
			present_pair_mask=present_pair_mask,
		)

		loss_final_ce, _ = self._final_cross_entropy_loss(
			final_logits=final_logits,
			ce_label_map=ce_label_map,
		)

		total_loss = (
				float(self.cfg.bce_weight) * loss_semantic_bce
				+ float(self.cfg.dice_weight) * loss_semantic_dice
				+ float(self.cfg.presence_bce_weight) * loss_presence_bce
				+ float(self.cfg.final_bce_weight) * loss_final_bce
				+ float(self.cfg.final_dice_weight) * loss_final_dice
				+ float(self.cfg.final_ce_weight) * loss_final_ce
		)

		return {
			"loss_semantic_bce": loss_semantic_bce,
			"loss_semantic_dice": loss_semantic_dice,
			"loss_presence_bce": loss_presence_bce,
			"loss_final_bce": loss_final_bce,
			"loss_final_dice": loss_final_dice,
			"loss_final_ce": loss_final_ce,
			"total_loss": total_loss,
			"num_valid": torch.tensor(
				num_valid_pixels,
				device=semantic_logits.device,
				dtype=torch.long,
			),
		}


class HybridCriterion(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(
		self,
		outputs: Dict[str, torch.Tensor],
		targets: Dict[str, torch.Tensor],
		chunk_class_ids: Optional[Sequence[int]] = None,
		reduction: str = "sum",
	) -> Dict[str, torch.Tensor]:
		raise NotImplementedError("HybridCriterion is not implemented yet.")