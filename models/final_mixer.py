from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_class_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


class BoundedLearnableScale(nn.Module):
    def __init__(
        self,
        init: float,
        min: float,
        max: float,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()

        self.min_value = float(min)
        self.max_value = float(max)
        self.temperature = float(temperature)

        if self.max_value <= self.min_value:
            raise ValueError(
                f"Scale max must be greater than min, got min={min}, max={max}."
            )
        if self.temperature <= 0:
            raise ValueError(
                f"Scale temperature must be positive, got {temperature}."
            )
        if not (self.min_value < float(init) < self.max_value):
            raise ValueError(
                "Scale init must be inside (min, max), got "
                f"init={init}, min={min}, max={max}."
            )

        ratio = (float(init) - self.min_value) / (
            self.max_value - self.min_value
        )

        if ratio < 1e-6:
            ratio = 1e-6
        elif ratio > 1.0 - 1e-6:
            ratio = 1.0 - 1e-6

        raw_init = self.temperature * math.log(ratio / (1.0 - ratio))

        self.raw_scale = nn.Parameter(
            torch.tensor(raw_init, dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        ratio = torch.sigmoid(self.raw_scale / self.temperature)
        return self.min_value + (self.max_value - self.min_value) * ratio


class HashRandomClassCodeBuilder(nn.Module):
    """
    Build deterministic class codes from class names.

    Rule:
        class name -> normalized string -> SHA256 hash -> fixed random seed
        -> random vector -> L2 normalize.

    Output:
        class_codes_unit: [C, D]
    """

    def __init__(
        self,
        dim: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.normalize = bool(normalize)

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")

    @staticmethod
    def _seed_from_name(name: str) -> int:
        normalized = _normalize_class_name(name)
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        # Keep the seed in signed 63-bit range for torch.Generator.
        return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)

    def _code_for_name(self, name: str) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed_from_name(name))
        code = torch.randn(self.dim, generator=generator, dtype=torch.float32)
        if self.normalize:
            code = F.normalize(code[None], dim=1, eps=1e-12)[0]
        return code

    def forward(
        self,
        class_names: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(class_names) == 0:
            raise ValueError("class_names is empty.")

        codes = [self._code_for_name(name) for name in class_names]
        class_codes = torch.stack(codes, dim=0)
        return class_codes.to(device=device, dtype=dtype, non_blocking=True)


class ClassTokenFusionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        presence_enabled: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.presence_enabled = bool(presence_enabled)

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, got "
                f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )

        self.slot_inter_class_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.slot_inter_class_norm = nn.LayerNorm(self.hidden_dim)

        self.intra_class_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.intra_class_norm = nn.LayerNorm(self.hidden_dim)

        self.presence_head = nn.Linear(self.hidden_dim, 1)

        self.sam3_low_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.sam3_low_norm = nn.LayerNorm(self.hidden_dim)

        self.clip_sam_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.clip_sam_norm = nn.LayerNorm(self.hidden_dim)

        self.dropout = nn.Dropout(float(dropout))

    def _slot_wise_inter_class_self_attn(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape

        x = class_tokens.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * num_tokens, num_classes, dim)

        delta, _ = self.slot_inter_class_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )
        x = self.slot_inter_class_norm(x + self.dropout(delta))

        x = x.reshape(batch_size, num_tokens, num_classes, dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _intra_class_self_attn(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape

        x = class_tokens.reshape(batch_size * num_classes, num_tokens, dim)
        delta, _ = self.intra_class_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )
        x = self.intra_class_norm(x + self.dropout(delta))

        return x.reshape(batch_size, num_classes, num_tokens, dim).contiguous()

    def _build_presence_logits(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        class_summary = class_tokens.mean(dim=2)
        presence_logits = self.presence_head(class_summary).squeeze(-1)
        return presence_logits.contiguous()

    @staticmethod
    def _flatten_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(
                "feature_map must be [B, D, H, W], "
                f"got {tuple(feature_map.shape)}."
            )
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def _attend_feature_tokens(
        self,
        class_tokens: torch.Tensor,
        class_codes_unit: torch.Tensor,
        feature_tokens: torch.Tensor,
        code_tokens: torch.Tensor,
        token_scale: torch.Tensor,
        feature_scale: torch.Tensor,
        attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape
        feature_batch, num_feature_tokens, feature_dim = feature_tokens.shape

        if int(feature_batch) != batch_size:
            raise ValueError(
                "feature_tokens batch mismatch: "
                f"{feature_batch} vs {batch_size}."
            )
        if int(feature_dim) != dim:
            raise ValueError(
                f"feature_tokens dim mismatch: expected {dim}, got {feature_dim}."
            )
        if tuple(code_tokens.shape) != tuple(feature_tokens.shape):
            raise ValueError(
                "code_tokens must have same shape as feature_tokens, got "
                f"{tuple(code_tokens.shape)} vs {tuple(feature_tokens.shape)}."
            )

        query = class_tokens + token_scale * class_codes_unit[None, :, None, :]
        key = feature_tokens + feature_scale * code_tokens
        value = feature_tokens

        query = query.reshape(batch_size * num_classes, num_tokens, dim)

        key = key[:, None].expand(
            batch_size,
            num_classes,
            num_feature_tokens,
            dim,
        )
        key = key.reshape(batch_size * num_classes, num_feature_tokens, dim)

        value = value[:, None].expand(
            batch_size,
            num_classes,
            num_feature_tokens,
            dim,
        )
        value = value.reshape(batch_size * num_classes, num_feature_tokens, dim)

        residual = class_tokens.reshape(batch_size * num_classes, num_tokens, dim)

        attn_out, _ = attn(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )
        out = norm(residual + self.dropout(attn_out))

        return out.reshape(batch_size, num_classes, num_tokens, dim).contiguous()

    def forward(
        self,
        class_tokens: torch.Tensor,
        semantic_logits: torch.Tensor,
        class_codes_unit: torch.Tensor,
        sam3_feature_low: torch.Tensor,
        shared_clip_feature: torch.Tensor,
        token_scale: torch.Tensor,
        feature_low_scale: torch.Tensor,
        clip_feature_scale: torch.Tensor,
        clip_grid_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if sam3_feature_low.dim() != 4:
            raise ValueError(
                "sam3_feature_low must be [B, D, Hc, Wc], "
                f"got {tuple(sam3_feature_low.shape)}."
            )
        if shared_clip_feature.dim() != 3:
            raise ValueError(
                "shared_clip_feature must be [B, Hc*Wc, D], "
                f"got {tuple(shared_clip_feature.shape)}."
            )

        batch_size, num_classes, _, dim = class_tokens.shape
        clip_h, clip_w = int(clip_grid_hw[0]), int(clip_grid_hw[1])
        num_clip_tokens = clip_h * clip_w

        if tuple(semantic_logits.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "semantic_logits batch/class mismatch: "
                f"{tuple(semantic_logits.shape[:2])} vs {(batch_size, num_classes)}."
            )
        if tuple(class_codes_unit.shape) != (num_classes, dim):
            raise ValueError(
                "class_codes_unit must be [C, D], got "
                f"{tuple(class_codes_unit.shape)}, expected {(num_classes, dim)}."
            )
        if tuple(sam3_feature_low.shape) != (batch_size, dim, clip_h, clip_w):
            raise ValueError(
                "sam3_feature_low shape mismatch: expected "
                f"{(batch_size, dim, clip_h, clip_w)}, "
                f"got {tuple(sam3_feature_low.shape)}."
            )
        if tuple(shared_clip_feature.shape) != (batch_size, num_clip_tokens, dim):
            raise ValueError(
                "shared_clip_feature shape mismatch: expected "
                f"{(batch_size, num_clip_tokens, dim)}, "
                f"got {tuple(shared_clip_feature.shape)}."
            )

        class_tokens = self._slot_wise_inter_class_self_attn(class_tokens)
        class_tokens = self._intra_class_self_attn(class_tokens)

        if self.presence_enabled:
            presence_logits = self._build_presence_logits(class_tokens)
            presence_for_prior = presence_logits.sigmoid()
        else:
            presence_logits = semantic_logits.new_zeros(batch_size, num_classes)
            presence_for_prior = semantic_logits.new_ones(batch_size, num_classes)

        semantic_prob = semantic_logits.detach().sigmoid()
        mask_prior = presence_for_prior[:, :, None, None] * semantic_prob

        mask_prior_low = F.interpolate(
            mask_prior,
            size=(clip_h, clip_w),
            mode="bilinear",
            align_corners=False,
        )

        code_map_low = torch.einsum(
            "bchw,cd->bdhw",
            mask_prior_low,
            class_codes_unit,
        ).contiguous()

        sam3_low_tokens = self._flatten_feature_map(sam3_feature_low)
        code_low_tokens = self._flatten_feature_map(code_map_low)

        class_tokens = self._attend_feature_tokens(
            class_tokens=class_tokens,
            class_codes_unit=class_codes_unit,
            feature_tokens=sam3_low_tokens,
            code_tokens=code_low_tokens,
            token_scale=token_scale,
            feature_scale=feature_low_scale,
            attn=self.sam3_low_attn,
            norm=self.sam3_low_norm,
        )

        class_tokens = self._attend_feature_tokens(
            class_tokens=class_tokens,
            class_codes_unit=class_codes_unit,
            feature_tokens=shared_clip_feature,
            code_tokens=code_low_tokens,
            token_scale=token_scale,
            feature_scale=clip_feature_scale,
            attn=self.clip_sam_attn,
            norm=self.clip_sam_norm,
        )

        return class_tokens.contiguous(), presence_logits.contiguous()


class ClassTokenSemanticFinalMixer(nn.Module):
    """
    New final mixer.

    Inputs:
        semantic_logits:       [B, C, H, W]
        class_tokens:          [B, C, Q, D]
        shared_clip_feature:   [B, Hc * Wc, D]
        sam3_feature_high:     [B, D, H, W]
        class_names:           list[str], length C
        clip_grid_hw:          (Hc, Wc)

    Outputs:
        final_logits:           [B, C, H, W]
        presence_logits:        [B, C]
        presence_score:         [B, C]
        presence_logits_layers: [L, B, C]

    Symbol meanings:
        B means batch size.
        C means class count.
        Q means class token count per class.
        D means hidden feature dimension, currently 256.
        H and W mean high-resolution mask feature size.
        Hc and Wc mean CLIP feature grid height and width.
        L means final mixer layer count.
    """

    def __init__(
        self,
        sam_dim: int = 256,
        num_heads: int = 8,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        presence_enabled: bool = True,
        class_code_cfg: Optional[Dict] = None,
        presence_cfg: Optional[Dict] = None,
        mask_head_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.num_heads = int(num_heads)
        self.fusion_layers = int(fusion_layers)
        self.presence_enabled = bool(presence_enabled)

        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.fusion_layers <= 0:
            raise ValueError(
                f"fusion_layers must be positive, got {fusion_layers}."
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.sam_dim % self.num_heads != 0:
            raise ValueError(
                "sam_dim must be divisible by num_heads, got "
                f"sam_dim={self.sam_dim}, num_heads={self.num_heads}."
            )

        class_code_cfg = dict(class_code_cfg or {})
        presence_cfg = dict(presence_cfg or {})
        mask_head_cfg = dict(mask_head_cfg or {})

        class_code_dim = int(class_code_cfg.get("dim", self.sam_dim))
        if class_code_dim != self.sam_dim:
            raise ValueError(
                "class_code_cfg.dim must match sam_dim for direct addition, "
                f"got {class_code_dim} and {self.sam_dim}."
            )

        self.class_code_builder = HashRandomClassCodeBuilder(
            dim=class_code_dim,
            normalize=bool(class_code_cfg.get("normalize", True)),
        )

        def _scale_cfg(name: str, default: Dict[str, float]) -> Dict[str, float]:
            value = class_code_cfg.get(name, None)
            return dict(default if value is None else value)

        self.token_scale = BoundedLearnableScale(
            **_scale_cfg(
                "token_scale",
                dict(init=8.0, min=2.0, max=16.0, temperature=0.5),
            )
        )
        self.feature_low_scale = BoundedLearnableScale(
            **_scale_cfg(
                "feature_low_scale",
                dict(init=8.0, min=2.0, max=16.0, temperature=0.5),
            )
        )
        self.clip_feature_scale = BoundedLearnableScale(
            **_scale_cfg(
                "clip_feature_scale",
                dict(init=8.0, min=2.0, max=16.0, temperature=0.5),
            )
        )
        self.feature_high_scale = BoundedLearnableScale(
            **_scale_cfg(
                "feature_high_scale",
                dict(init=6.0, min=1.0, max=12.0, temperature=0.5),
            )
        )

        self.layers = nn.ModuleList(
            [
                ClassTokenFusionLayer(
                    hidden_dim=self.sam_dim,
                    num_heads=self.num_heads,
                    dropout=float(dropout),
                    presence_enabled=self.presence_enabled,
                )
                for _ in range(self.fusion_layers)
            ]
        )

        self.supervise_all_presence_layers = bool(
            presence_cfg.get("supervise_all_layers", True)
        )

        self.train_token_pooling = str(
            mask_head_cfg.get("train_token_pooling", "logsumexp")
        )
        self.infer_token_pooling = str(
            mask_head_cfg.get("infer_token_pooling", "max")
        )
        self.logsumexp_tau = float(mask_head_cfg.get("logsumexp_tau", 0.2))
        if self.logsumexp_tau <= 0:
            raise ValueError(
                f"logsumexp_tau must be positive, got {self.logsumexp_tau}."
            )

        if self.train_token_pooling not in {"logsumexp", "max"}:
            raise ValueError(
                "train_token_pooling must be 'logsumexp' or 'max', got "
                f"{self.train_token_pooling!r}."
            )
        if self.infer_token_pooling not in {"logsumexp", "max"}:
            raise ValueError(
                "infer_token_pooling must be 'logsumexp' or 'max', got "
                f"{self.infer_token_pooling!r}."
            )

    @staticmethod
    def _check_clip_grid(
        shared_clip_feature: torch.Tensor,
        clip_grid_hw: Tuple[int, int],
    ) -> None:
        clip_h, clip_w = int(clip_grid_hw[0]), int(clip_grid_hw[1])
        if clip_h <= 0 or clip_w <= 0:
            raise ValueError(f"clip_grid_hw must be positive, got {clip_grid_hw}.")
        if clip_h * clip_w != int(shared_clip_feature.shape[1]):
            raise ValueError(
                "clip_grid_hw does not match shared_clip_feature token count: "
                f"clip_grid_hw={clip_grid_hw}, product={clip_h * clip_w}, "
                f"N_clip={shared_clip_feature.shape[1]}."
            )

    @staticmethod
    def _pool_token_logits(
        token_logits: torch.Tensor,
        mode: str,
        tau: float,
    ) -> torch.Tensor:
        if mode == "max":
            return token_logits.max(dim=2).values
        if mode == "logsumexp":
            return tau * torch.logsumexp(token_logits / tau, dim=2)
        raise ValueError(f"Unknown token pooling mode: {mode!r}.")

    def _build_high_res_code_map(
        self,
        semantic_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        class_codes_unit: torch.Tensor,
    ) -> torch.Tensor:
        if self.presence_enabled:
            presence = presence_logits.sigmoid()
        else:
            presence = semantic_logits.new_ones(semantic_logits.shape[:2])

        semantic_prob = semantic_logits.detach().sigmoid()
        mask_prior = presence[:, :, None, None] * semantic_prob

        code_map_high = torch.einsum(
            "bchw,cd->bdhw",
            mask_prior,
            class_codes_unit,
        )
        return code_map_high.contiguous()

    def _build_mask_logits(
        self,
        class_tokens: torch.Tensor,
        sam3_feature_high: torch.Tensor,
        class_codes_unit: torch.Tensor,
        code_map_high: torch.Tensor,
    ) -> torch.Tensor:
        token_scale = self.token_scale().to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )
        feature_high_scale = self.feature_high_scale().to(
            device=sam3_feature_high.device,
            dtype=sam3_feature_high.dtype,
        )

        class_tokens_for_mask = (
            class_tokens + token_scale * class_codes_unit[None, :, None, :]
        )
        sam3_feature_for_mask = sam3_feature_high + feature_high_scale * code_map_high

        token_logits = torch.einsum(
            "bcqd,bdhw->bcqhw",
            class_tokens_for_mask,
            sam3_feature_for_mask,
        )

        pooling_mode = self.train_token_pooling if self.training else self.infer_token_pooling
        return self._pool_token_logits(
            token_logits=token_logits,
            mode=pooling_mode,
            tau=self.logsumexp_tau,
        ).contiguous()

    def _scale_debug_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "token_scale": self.token_scale(),
            "feature_low_scale": self.feature_low_scale(),
            "clip_feature_scale": self.clip_feature_scale(),
            "feature_high_scale": self.feature_high_scale(),
        }

    def forward(
        self,
        semantic_logits: torch.Tensor,
        class_tokens: torch.Tensor,
        shared_clip_feature: torch.Tensor,
        sam3_feature_high: torch.Tensor,
        class_names: Sequence[str],
        clip_grid_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )
        if shared_clip_feature.dim() != 3:
            raise ValueError(
                "shared_clip_feature must be [B, Hc*Wc, D], "
                f"got {tuple(shared_clip_feature.shape)}."
            )
        if sam3_feature_high.dim() != 4:
            raise ValueError(
                "sam3_feature_high must be [B, D, H, W], "
                f"got {tuple(sam3_feature_high.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape
        _, class_token_classes, _, token_dim = class_tokens.shape

        if int(token_dim) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, got {token_dim}."
            )
        if tuple(class_tokens.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens batch/class mismatch: "
                f"{tuple(class_tokens.shape[:2])} vs {(batch_size, num_classes)}."
            )
        if tuple(shared_clip_feature.shape[:1]) != (batch_size,):
            raise ValueError(
                "shared_clip_feature batch mismatch: "
                f"{shared_clip_feature.shape[0]} vs {batch_size}."
            )
        if int(shared_clip_feature.shape[-1]) != self.sam_dim:
            raise ValueError(
                "shared_clip_feature channel mismatch: expected "
                f"{self.sam_dim}, got {shared_clip_feature.shape[-1]}."
            )
        if tuple(sam3_feature_high.shape) != (
            batch_size,
            self.sam_dim,
            height,
            width,
        ):
            raise ValueError(
                "sam3_feature_high shape mismatch: expected "
                f"{(batch_size, self.sam_dim, height, width)}, "
                f"got {tuple(sam3_feature_high.shape)}."
            )
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names length must match C={num_classes}, got {len(class_names)}."
            )

        self._check_clip_grid(
            shared_clip_feature=shared_clip_feature,
            clip_grid_hw=clip_grid_hw,
        )

        device = class_tokens.device
        dtype = class_tokens.dtype

        semantic_logits = semantic_logits.to(device=device, dtype=dtype)
        shared_clip_feature = shared_clip_feature.to(device=device, dtype=dtype)
        sam3_feature_high = sam3_feature_high.to(device=device, dtype=dtype)

        class_codes_unit = self.class_code_builder(
            class_names=class_names,
            device=device,
            dtype=dtype,
        )

        clip_h, clip_w = int(clip_grid_hw[0]), int(clip_grid_hw[1])
        sam3_feature_low = F.adaptive_avg_pool2d(
            sam3_feature_high,
            output_size=(clip_h, clip_w),
        )

        presence_logits_layers = []

        token_scale = self.token_scale().to(device=device, dtype=dtype)
        feature_low_scale = self.feature_low_scale().to(device=device, dtype=dtype)
        clip_feature_scale = self.clip_feature_scale().to(device=device, dtype=dtype)

        for layer in self.layers:
            class_tokens, presence_logits_l = layer(
                class_tokens=class_tokens,
                semantic_logits=semantic_logits,
                class_codes_unit=class_codes_unit,
                sam3_feature_low=sam3_feature_low,
                shared_clip_feature=shared_clip_feature,
                token_scale=token_scale,
                feature_low_scale=feature_low_scale,
                clip_feature_scale=clip_feature_scale,
                clip_grid_hw=clip_grid_hw,
            )
            presence_logits_layers.append(presence_logits_l)

        presence_logits_layers_tensor = torch.stack(
            presence_logits_layers,
            dim=0,
        )

        presence_logits_last = presence_logits_layers_tensor[-1]
        if self.presence_enabled:
            presence_score = presence_logits_last.sigmoid()
        else:
            presence_score = semantic_logits.new_ones(batch_size, num_classes)

        code_map_high = self._build_high_res_code_map(
            semantic_logits=semantic_logits,
            presence_logits=presence_logits_last,
            class_codes_unit=class_codes_unit,
        )

        final_logits = self._build_mask_logits(
            class_tokens=class_tokens,
            sam3_feature_high=sam3_feature_high,
            class_codes_unit=class_codes_unit,
            code_map_high=code_map_high,
        )

        return {
            "final_logits": final_logits,
            "presence_logits": presence_logits_last.contiguous(),
            "presence_score": presence_score.contiguous(),
            "presence_logits_layers": presence_logits_layers_tensor.contiguous(),
            "class_code_scales": self._scale_debug_dict(),
        }