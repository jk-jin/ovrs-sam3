from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .task_modes import OUTPUT_KEYS
from .shifted_window_attention import ShiftedWindowAttention2D

class DynamicTextAlignedMaskFusionLayer(nn.Module):
    """
    One layer of the new final mixer.

    Input:
        class_tokens:          [B, C, Q, D]
        source_logits:         [B, C, H, W]
        shared_clip_feature_high: [B, H*W, D]
        sam3_text_tokens_full: [M, C, D]
        sam3_text_mask_full:   [C, M]

    Output:
        class_tokens:          [B, C, Q, D]
        presence_logits:       [B, C]
        dynamic_class_code:    [B, C, D]
        mask_logits:           [B, C, H, W]

    Symbol meanings:
        B means batch size.
        C means class count.
        Q means class token count per class.
        D means hidden feature dimension.
        M means SAM3 text token count.
        H and W mean mask height and width.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        presence_enabled: bool = True,
        window_size: int = 8,
        shift_size: int = 0,
        multiply_presence: bool = True,
        class_feature_pool_stride: int = 4,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.presence_enabled = bool(presence_enabled)
        self.multiply_presence = bool(multiply_presence)
        self.class_feature_pool_stride = int(class_feature_pool_stride)
        if self.class_feature_pool_stride <= 0:
            raise ValueError(
                "class_feature_pool_stride must be positive, "
                f"got {class_feature_pool_stride}."
            )

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
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

        self.presence_query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.presence_query, std=0.02)

        self.presence_token_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.presence_token_norm = nn.LayerNorm(self.hidden_dim)

        self.presence_summary_norm = nn.LayerNorm(self.hidden_dim * 3)
        self.presence_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

        self.code_class_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        self.code_residual_norm = nn.LayerNorm(self.hidden_dim)
        self.code_output_norm = nn.LayerNorm(self.hidden_dim)

        self.mask_embed_norm = nn.LayerNorm(self.hidden_dim)

        self.mask_feature_attn = ShiftedWindowAttention2D(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            window_size=int(window_size),
            shift_size=int(shift_size),
            dropout=float(dropout),
            value_preserving=True,
            residual_source="value",
            use_residual_norm=True,
            use_rel_pos_bias=True,
        )

        self.class_to_feature_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.class_to_feature_norm = nn.LayerNorm(self.hidden_dim)

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
        return x.permute(0, 2, 1, 3).contiguous()

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

    def _build_text_query(
        self,
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if sam3_text_tokens_full.dim() != 3:
            raise ValueError(
                "sam3_text_tokens_full must be [M, C, D], "
                f"got {tuple(sam3_text_tokens_full.shape)}."
            )
        if sam3_text_mask_full.dim() != 2:
            raise ValueError(
                "sam3_text_mask_full must be [C, M], "
                f"got {tuple(sam3_text_mask_full.shape)}."
            )

        text_len, num_classes, dim = sam3_text_tokens_full.shape
        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"SAM3 text dim mismatch: expected {self.hidden_dim}, got {dim}."
            )
        if tuple(sam3_text_mask_full.shape) != (num_classes, text_len):
            raise ValueError(
                "sam3_text_mask_full shape mismatch: expected "
                f"{(num_classes, text_len)}, got {tuple(sam3_text_mask_full.shape)}."
            )

        text_tokens = sam3_text_tokens_full.to(device=device, dtype=dtype)
        text_mask = sam3_text_mask_full.to(device=device).bool()

        # [M, C, D] -> [C, M, D]
        text_tokens = text_tokens.permute(1, 0, 2).contiguous()

        valid = (~text_mask).to(dtype=dtype)

        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        text_query = (text_tokens * valid[:, :, None]).sum(dim=1) / denom
        # [C, D]

        text_query = text_query[None].expand(
            batch_size,
            num_classes,
            dim,
        )
        # [B, C, D]

        return text_query.contiguous()

    def _build_presence_logits(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, num_classes, num_tokens, dim = class_tokens.shape
        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"class token dim mismatch: expected {self.hidden_dim}, got {dim}."
            )

        x = class_tokens.reshape(
            batch_size * num_classes,
            num_tokens,
            dim,
        )

        query = self.presence_query.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )
        query = query.expand(batch_size * num_classes, 1, dim)

        attn_out, _ = self.presence_token_attn(
            query=query,
            key=x,
            value=x,
            need_weights=False,
        )
        attn_summary = self.presence_token_norm(
            query + self.dropout(attn_out)
        ).squeeze(1)

        mean_summary = x.mean(dim=1)
        max_summary = x.max(dim=1).values

        summary = torch.cat(
            [
                attn_summary,
                mean_summary,
                max_summary,
            ],
            dim=-1,
        )
        summary = self.presence_summary_norm(summary)

        presence_logits = self.presence_head(summary).squeeze(-1)
        return presence_logits.reshape(batch_size, num_classes).contiguous()

    def _build_dynamic_class_code(
        self,
        class_tokens: torch.Tensor,
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
    ) -> torch.Tensor:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, num_classes, num_tokens, dim = class_tokens.shape
        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"class token dim mismatch: expected {self.hidden_dim}, got {dim}."
            )

        text_query = self._build_text_query(
            sam3_text_tokens_full=sam3_text_tokens_full,
            sam3_text_mask_full=sam3_text_mask_full,
            batch_size=batch_size,
            dtype=class_tokens.dtype,
            device=class_tokens.device,
        )

        if int(text_query.shape[1]) != num_classes:
            raise ValueError(
                "SAM3 text class count mismatch: "
                f"{text_query.shape[1]} vs {num_classes}."
            )

        query = text_query.reshape(batch_size * num_classes, 1, dim)

        class_tokens_flat = class_tokens.reshape(
            batch_size * num_classes,
            num_tokens,
            dim,
        )

        attn_out, _ = self.code_class_attn(
            query=query,
            key=class_tokens_flat,
            value=class_tokens_flat,
            need_weights=False,
        )

        code = self.code_residual_norm(query + self.dropout(attn_out))
        code = code.squeeze(1).reshape(batch_size, num_classes, dim).contiguous()
        code = self.code_output_norm(code)

        return code.contiguous()

    def _build_mask_embedding(
        self,
        source_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        dynamic_class_code: torch.Tensor,
    ) -> torch.Tensor:
        mask_prob = torch.softmax(source_logits, dim=1)

        if self.presence_enabled and self.multiply_presence:
            presence_score = torch.sigmoid(presence_logits)
        else:
            presence_score = source_logits.new_ones(source_logits.shape[:2])

        mask_weight = mask_prob * presence_score[:, :, None, None]

        mask_embed = torch.einsum(
            "bchw,bcd->bdhw",
            mask_weight,
            dynamic_class_code,
        ).contiguous()

        mask_embed_dtype = mask_embed.dtype
        mask_embed = mask_embed.float().permute(0, 2, 3, 1).contiguous()
        mask_embed = self.mask_embed_norm(mask_embed)
        mask_embed = mask_embed.permute(0, 3, 1, 2).contiguous()
        mask_embed = mask_embed.to(dtype=mask_embed_dtype)

        return mask_embed

    def _pool_feature_for_class_attention(
        self,
        attn_feature: torch.Tensor,
    ) -> torch.Tensor:
        if attn_feature.dim() != 4:
            raise ValueError(
                "attn_feature must be [B, D, H, W], "
                f"got {tuple(attn_feature.shape)}."
            )

        stride = int(self.class_feature_pool_stride)
        if stride <= 1:
            return attn_feature

        return F.avg_pool2d(
            attn_feature,
            kernel_size=stride,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False,
        )

    def _attend_feature_with_class_tokens(
        self,
        class_tokens: torch.Tensor,
        attn_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape

        pooled_feature = self._pool_feature_for_class_attention(attn_feature)

        attn_tokens = pooled_feature.flatten(2).transpose(1, 2).contiguous()
        num_pixels = int(attn_tokens.shape[1])

        query = class_tokens.reshape(batch_size * num_classes, num_tokens, dim)

        key = attn_tokens[:, None].expand(
            batch_size,
            num_classes,
            num_pixels,
            dim,
        )
        key = key.reshape(batch_size * num_classes, num_pixels, dim)

        value = key

        attn_out, _ = self.class_to_feature_attn(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )
        out = self.class_to_feature_norm(query + self.dropout(attn_out))

        return out.reshape(batch_size, num_classes, num_tokens, dim).contiguous()

    def _build_mask_logits(
        self,
        attn_feature: torch.Tensor,
        dynamic_class_code: torch.Tensor,
        logit_temperature: float,
    ) -> torch.Tensor:
        if logit_temperature <= 0:
            raise ValueError(
                f"logit_temperature must be positive, got {logit_temperature}."
            )

        batch_size, dim, height, width = attn_feature.shape

        attn_tokens = attn_feature.flatten(2).transpose(1, 2).contiguous()

        raw_mask_logits = torch.einsum(
            "bnd,bcd->bcn",
            attn_tokens,
            dynamic_class_code,
        )

        mask_logits = raw_mask_logits / float(logit_temperature)

        return mask_logits.reshape(
            batch_size,
            dynamic_class_code.shape[1],
            height,
            width,
        ).contiguous()

    def forward(
        self,
        class_tokens: torch.Tensor,
        source_logits: torch.Tensor,
        shared_clip_feature_high: torch.Tensor,
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
        tau_mask: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D], "
                f"got {tuple(class_tokens.shape)}."
            )
        if source_logits.dim() != 4:
            raise ValueError(
                "source_logits must be [B, C, H, W], "
                f"got {tuple(source_logits.shape)}."
            )
        if shared_clip_feature_high.dim() != 3:
            raise ValueError(
                "shared_clip_feature_high must be [B, H*W, D], "
                f"got {tuple(shared_clip_feature_high.shape)}."
            )

        batch_size, num_classes, height, width = source_logits.shape
        token_batch, token_classes, _, dim = class_tokens.shape

        if (token_batch, token_classes) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens and source_logits batch/class mismatch: "
                f"{tuple(class_tokens.shape[:2])} vs {(batch_size, num_classes)}."
            )
        if int(shared_clip_feature_high.shape[0]) != batch_size:
            raise ValueError(
                "shared_clip_feature_high batch mismatch: "
                f"{shared_clip_feature_high.shape[0]} vs {batch_size}."
            )
        if int(shared_clip_feature_high.shape[1]) != height * width:
            raise ValueError(
                "shared_clip_feature_high token count must equal H*W: "
                f"{shared_clip_feature_high.shape[1]} vs {height * width}."
            )
        if int(shared_clip_feature_high.shape[2]) != dim:
            raise ValueError(
                "shared_clip_feature_high dim mismatch: "
                f"{shared_clip_feature_high.shape[2]} vs {dim}."
            )
        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"class token dim mismatch: expected {self.hidden_dim}, got {dim}."
            )

        source_logits = source_logits.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )
        shared_clip_feature_high = shared_clip_feature_high.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )

        class_tokens = self._slot_wise_inter_class_self_attn(class_tokens)
        class_tokens = self._intra_class_self_attn(class_tokens)

        if self.presence_enabled:
            presence_logits = self._build_presence_logits(class_tokens)
        else:
            presence_logits = source_logits.new_zeros(batch_size, num_classes)

        dynamic_class_code = self._build_dynamic_class_code(
            class_tokens=class_tokens,
            sam3_text_tokens_full=sam3_text_tokens_full,
            sam3_text_mask_full=sam3_text_mask_full,
        )

        mask_embed = self._build_mask_embedding(
            source_logits=source_logits,
            presence_logits=presence_logits,
            dynamic_class_code=dynamic_class_code,
        )

        clip_map = shared_clip_feature_high.transpose(1, 2).reshape(
            batch_size,
            dim,
            height,
            width,
        )

        attn_feature = self.mask_feature_attn(
            query_map=clip_map,
            key_map=mask_embed,
            value_map=mask_embed,
        )

        class_tokens = self._attend_feature_with_class_tokens(
            class_tokens=class_tokens,
            attn_feature=attn_feature,
        )

        mask_logits = self._build_mask_logits(
            attn_feature=attn_feature,
            dynamic_class_code=dynamic_class_code,
            logit_temperature=float(tau_mask),
        )

        return (
            class_tokens.contiguous(),
            presence_logits.contiguous(),
            dynamic_class_code.contiguous(),
            mask_logits.contiguous(),
        )


class ClassTokenSemanticFinalMixer(nn.Module):
    """
    Dynamic text-aligned final mixer.

    Input:
        semantic_logits:           [B, C, H, W]
        class_tokens:              [B, C, Q, D]
        shared_clip_feature_high:  [B, H*W, D]
        sam3_text_tokens_full:     [M, C, D]
        sam3_text_mask_full:       [C, M]

    Output:
        final_logits:              [B, C, H, W]
        mask_logits_layers:        [L, B, C, H, W]
        presence_logits:           [B, C]
        presence_score:            [B, C]
        presence_logits_layers:    [L, B, C]

    Symbol meanings:
        B means batch size.
        C means class count.
        Q means class token count per class.
        D means hidden feature dimension.
        M means SAM3 text token count.
        H and W mean mask height and width.
        L means fusion layer count.
    """

    def __init__(
        self,
        sam_dim: int = 256,
        num_heads: int = 8,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        presence_enabled: bool = True,
        dynamic_code_cfg: Optional[Dict] = None,
        mask_prior_cfg: Optional[Dict] = None,
        window_attention_cfg: Optional[Dict] = None,
        mask_head_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.num_heads = int(num_heads)
        self.fusion_layers = int(fusion_layers)
        self.presence_enabled = bool(presence_enabled)

        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.fusion_layers <= 0:
            raise ValueError(f"fusion_layers must be positive, got {fusion_layers}.")
        if self.sam_dim % self.num_heads != 0:
            raise ValueError(
                "sam_dim must be divisible by num_heads, "
                f"got sam_dim={self.sam_dim}, num_heads={self.num_heads}."
            )

        dynamic_code_cfg = dict(dynamic_code_cfg or {})
        mask_prior_cfg = dict(mask_prior_cfg or {})
        window_attention_cfg = dict(window_attention_cfg or {})
        mask_head_cfg = dict(mask_head_cfg or {})

        self.class_feature_pool_stride = int(
            mask_head_cfg.get("class_feature_pool_stride", 4)
        )
        if self.class_feature_pool_stride <= 0:
            raise ValueError(
                "class_feature_pool_stride must be positive, "
                f"got {self.class_feature_pool_stride}."
            )

        if str(dynamic_code_cfg.get("source", "class_token_to_sam3_text")) != "class_token_to_sam3_text":
            raise ValueError(
                "dynamic_code_cfg.source must be 'class_token_to_sam3_text'."
            )

        if str(mask_prior_cfg.get("type", "softmax")) != "softmax":
            raise ValueError("mask_prior_cfg.type must be 'softmax'.")

        if str(mask_head_cfg.get("type", "attn_feature_dot_dynamic_code")) != "attn_feature_dot_dynamic_code":
            raise ValueError(
                "mask_head_cfg.type must be 'attn_feature_dot_dynamic_code'."
            )

        self.tau_mask = float(mask_prior_cfg.get("tau", 64.0))
        self.multiply_presence = bool(mask_prior_cfg.get("multiply_presence", True))
        self.window_size = int(window_attention_cfg.get("window_size", 8))
        self.shift_size = int(window_attention_cfg.get("shift_size", self.window_size // 2))
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                "window_attention_cfg.shift_size must satisfy "
                "0 <= shift_size < window_size, "
                f"got shift_size={self.shift_size}, window_size={self.window_size}."
            )
        self.window_dropout = float(window_attention_cfg.get("dropout", dropout))

        if self.tau_mask <= 0:
            raise ValueError(f"tau_mask must be positive, got {self.tau_mask}.")

        layers = []
        for layer_idx in range(self.fusion_layers):
            layer_shift_size = 0 if layer_idx % 2 == 0 else self.shift_size

            layers.append(
                DynamicTextAlignedMaskFusionLayer(
                    hidden_dim=self.sam_dim,
                    num_heads=self.num_heads,
                    dropout=float(dropout),
                    presence_enabled=self.presence_enabled,
                    window_size=self.window_size,
                    shift_size=layer_shift_size,
                    multiply_presence=self.multiply_presence,
                    class_feature_pool_stride=self.class_feature_pool_stride,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        semantic_logits: torch.Tensor,
        class_tokens: torch.Tensor,
        shared_clip_feature_high: torch.Tensor,
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
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
        if shared_clip_feature_high.dim() != 3:
            raise ValueError(
                "shared_clip_feature_high must be [B, H*W, D], "
                f"got {tuple(shared_clip_feature_high.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape
        _, token_classes, _, token_dim = class_tokens.shape

        if int(token_dim) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, got {token_dim}."
            )
        if int(token_classes) != num_classes:
            raise ValueError(
                f"class count mismatch: class_tokens has {token_classes}, "
                f"semantic_logits has {num_classes}."
            )
        if tuple(class_tokens.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens batch/class mismatch: "
                f"{tuple(class_tokens.shape[:2])} vs {(batch_size, num_classes)}."
            )
        if tuple(shared_clip_feature_high.shape) != (
            batch_size,
            height * width,
            self.sam_dim,
        ):
            raise ValueError(
                "shared_clip_feature_high shape mismatch: expected "
                f"{(batch_size, height * width, self.sam_dim)}, "
                f"got {tuple(shared_clip_feature_high.shape)}."
            )

        device = class_tokens.device
        dtype = class_tokens.dtype

        source_logits = semantic_logits.detach().to(device=device, dtype=dtype)
        source_logits = source_logits
        shared_clip_feature_high = shared_clip_feature_high.to(
            device=device,
            dtype=dtype,
        )
        sam3_text_tokens_full = sam3_text_tokens_full.detach().to(
            device=device,
            dtype=dtype,
        )
        sam3_text_mask_full = sam3_text_mask_full.detach().to(device=device)

        mask_logits_layers = []
        presence_logits_layers = []

        for layer in self.layers:
            (
                class_tokens,
                presence_logits,
                _dynamic_class_code,
                mask_logits,
            ) = layer(
                class_tokens=class_tokens,
                source_logits=source_logits,
                shared_clip_feature_high=shared_clip_feature_high,
                sam3_text_tokens_full=sam3_text_tokens_full,
                sam3_text_mask_full=sam3_text_mask_full,
                tau_mask=self.tau_mask,
            )

            mask_logits_layers.append(mask_logits)
            presence_logits_layers.append(presence_logits)
            source_logits = mask_logits

        mask_logits_layers_tensor = torch.stack(mask_logits_layers, dim=0)
        presence_logits_layers_tensor = torch.stack(presence_logits_layers, dim=0)

        final_logits = mask_logits_layers_tensor[-1]
        presence_logits_last = presence_logits_layers_tensor[-1]

        if self.presence_enabled:
            presence_score = torch.sigmoid(presence_logits_last)
        else:
            presence_score = final_logits.new_ones(batch_size, num_classes)

        return {
            OUTPUT_KEYS.final_logits: final_logits.contiguous(),
            OUTPUT_KEYS.presence_logits: presence_logits_last.contiguous(),
            OUTPUT_KEYS.presence_score: presence_score.contiguous(),
            OUTPUT_KEYS.presence_logits_layers: presence_logits_layers_tensor.contiguous(),
            OUTPUT_KEYS.mask_logits_layers: mask_logits_layers_tensor.contiguous(),
        }