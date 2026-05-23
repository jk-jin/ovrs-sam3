from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shifted_window_attention import ShiftedWindowAttention2D


class ClipSamFeatureInitializer(nn.Module):
    """
    Build low-resolution aligned CLIP-SAM feature inside the final mixer.

    New design:
        1. Reuse final mixer's class-token query weights.
        2. Project class-token query from SAM dimension to CLIP dimension.
        3. Do not project CLIP text tokens.
        4. Use projected class-token query to attend each class's CLIP template tokens.
        5. Use the attended CLIP-space tokens as attention keys.
        6. Use current class tokens directly as attention values.
        7. Use CLIP image feature tokens as attention queries.
        8. Do not add CLIP image residual.

    Input shapes:
        clip_image_feat_map_native: [B, D_clip, Hc, Wc]
        clip_text_tokens_native:    [C, K, D_clip]
        class_token_query_embed:    [1, Q, D_sam]
        class_tokens:               [B, C, Q, D_sam]

    Output:
        aligned_clip_sam_feature_low: [B, Hc*Wc, D_sam]

    Symbol meanings:
        B means batch size.
        C means class count.
        K means CLIP prompt-template count per class.
        Q means class-token count per class.
        D_clip means CLIP feature dimension.
        D_sam means SAM3 hidden dimension.
        Hc and Wc mean CLIP feature grid height and width.
    """

    def __init__(
        self,
        clip_dim: int,
        sam_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.sam_dim = int(sam_dim)
        self.num_heads = int(num_heads)

        if self.clip_dim <= 0:
            raise ValueError(f"clip_dim must be positive, got {clip_dim}.")
        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.clip_dim % self.num_heads != 0:
            raise ValueError(
                "clip_dim must be divisible by num_heads, "
                f"got clip_dim={self.clip_dim}, num_heads={self.num_heads}."
            )
        if self.sam_dim % self.num_heads != 0:
            raise ValueError(
                "sam_dim must be divisible by num_heads, "
                f"got sam_dim={self.sam_dim}, num_heads={self.num_heads}."
            )

        self.clip_head_dim = self.clip_dim // self.num_heads
        self.sam_head_dim = self.sam_dim // self.num_heads

        # Only query is projected into CLIP text space.
        # CLIP text tokens are intentionally not projected.
        self.class_query_to_clip = nn.Linear(self.sam_dim, self.clip_dim)
        self.clip_template_norm = nn.LayerNorm(self.clip_dim)

        self.dropout = nn.Dropout(float(dropout))

    def _build_clip_keys_from_class_queries(
        self,
        clip_text_tokens_native: torch.Tensor,
        class_token_query_embed: torch.Tensor,
    ) -> torch.Tensor:
        if clip_text_tokens_native.dim() != 3:
            raise ValueError(
                "clip_text_tokens_native must be [C, K, D_clip], "
                f"got {tuple(clip_text_tokens_native.shape)}."
            )
        if class_token_query_embed.dim() != 3:
            raise ValueError(
                "class_token_query_embed must be [1, Q, D_sam], "
                f"got {tuple(class_token_query_embed.shape)}."
            )

        num_classes, num_templates, text_dim = clip_text_tokens_native.shape
        query_batch, num_queries, query_dim = class_token_query_embed.shape

        if int(text_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, "
                f"got {text_dim}."
            )
        if int(query_batch) != 1:
            raise ValueError(
                "class_token_query_embed first dim must be 1, "
                f"got {query_batch}."
            )
        if int(query_dim) != self.sam_dim:
            raise ValueError(
                f"class_token_query_embed dim mismatch: expected {self.sam_dim}, "
                f"got {query_dim}."
            )

        query = class_token_query_embed.to(
            device=clip_text_tokens_native.device,
            dtype=clip_text_tokens_native.dtype,
        )
        query = query.expand(num_classes, -1, -1)

        # The only learned projection here:
        # class-token query: SAM space -> CLIP text space.
        query = self.class_query_to_clip(query)

        # Do not project CLIP text tokens.
        key = clip_text_tokens_native
        value = clip_text_tokens_native

        query_heads = query.reshape(
            num_classes,
            num_queries,
            self.num_heads,
            self.clip_head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        key_heads = key.reshape(
            num_classes,
            num_templates,
            self.num_heads,
            self.clip_head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        value_heads = value.reshape(
            num_classes,
            num_templates,
            self.num_heads,
            self.clip_head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        attn_logits = torch.einsum(
            "chqd,chkd->chqk",
            query_heads,
            key_heads,
        )
        attn_logits = attn_logits / math.sqrt(float(self.clip_head_dim))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        attn_out = torch.einsum(
            "chqk,chkd->chqd",
            attn,
            value_heads,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.reshape(num_classes, num_queries, self.clip_dim)

        clip_keys = self.clip_template_norm(query + attn_out)
        return clip_keys.contiguous()

    def _image_tokens_attend_class_keys_values(
        self,
        image_tokens: torch.Tensor,
        clip_keys: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if image_tokens.dim() != 3:
            raise ValueError(
                "image_tokens must be [B, N, D_clip], "
                f"got {tuple(image_tokens.shape)}."
            )
        if clip_keys.dim() != 3:
            raise ValueError(
                "clip_keys must be [C, Q, D_clip], "
                f"got {tuple(clip_keys.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D_sam], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, num_image_tokens, image_dim = image_tokens.shape
        num_classes, num_class_tokens, key_dim = clip_keys.shape
        token_batch, token_classes, token_count, token_dim = class_tokens.shape

        if int(image_dim) != self.clip_dim:
            raise ValueError(
                f"image_tokens dim mismatch: expected {self.clip_dim}, "
                f"got {image_dim}."
            )
        if int(key_dim) != self.clip_dim:
            raise ValueError(
                f"clip_keys dim mismatch: expected {self.clip_dim}, got {key_dim}."
            )
        if int(token_batch) != int(batch_size):
            raise ValueError(
                "class_tokens batch mismatch: "
                f"{token_batch} vs {batch_size}."
            )
        if (int(token_classes), int(token_count)) != (
            int(num_classes),
            int(num_class_tokens),
        ):
            raise ValueError(
                "class_tokens class/token shape mismatch: expected "
                f"{(num_classes, num_class_tokens)}, "
                f"got {(token_classes, token_count)}."
            )
        if int(token_dim) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, "
                f"got {token_dim}."
            )

        # Query/key are in CLIP space.
        query = image_tokens.reshape(
            batch_size,
            num_image_tokens,
            self.num_heads,
            self.clip_head_dim,
        )
        query = query.permute(0, 2, 1, 3).contiguous()

        key = clip_keys.reshape(
            num_classes * num_class_tokens,
            self.num_heads,
            self.clip_head_dim,
        )
        key = key.permute(1, 0, 2).contiguous()

        # Value is directly class_tokens in SAM space.
        value = class_tokens.reshape(
            batch_size,
            num_classes * num_class_tokens,
            self.num_heads,
            self.sam_head_dim,
        )
        value = value.permute(0, 2, 1, 3).contiguous()

        attn_logits = torch.einsum("bhnd,hkd->bhnk", query, key)
        attn_logits = attn_logits / math.sqrt(float(self.clip_head_dim))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        aligned = torch.einsum("bhnk,bhkd->bhnd", attn, value)
        aligned = aligned.permute(0, 2, 1, 3).contiguous()
        aligned = aligned.reshape(batch_size, num_image_tokens, self.sam_dim)

        return aligned.contiguous()

    def forward(
        self,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        class_token_query_embed: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if clip_image_feat_map_native.dim() != 4:
            raise ValueError(
                "clip_image_feat_map_native must be [B, D_clip, Hc, Wc], "
                f"got {tuple(clip_image_feat_map_native.shape)}."
            )

        batch_size, image_dim, grid_h, grid_w = clip_image_feat_map_native.shape
        if int(image_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP image dim mismatch: expected {self.clip_dim}, "
                f"got {image_dim}."
            )

        dtype = clip_image_feat_map_native.dtype
        device = clip_image_feat_map_native.device

        clip_text_tokens_native = clip_text_tokens_native.to(
            device=device,
            dtype=dtype,
        )
        class_tokens = class_tokens.to(device=device, dtype=dtype)

        image_tokens = clip_image_feat_map_native.flatten(2).transpose(1, 2)
        image_tokens = image_tokens.contiguous()

        clip_keys = self._build_clip_keys_from_class_queries(
            clip_text_tokens_native=clip_text_tokens_native,
            class_token_query_embed=class_token_query_embed,
        )

        aligned = self._image_tokens_attend_class_keys_values(
            image_tokens=image_tokens,
            clip_keys=clip_keys,
            class_tokens=class_tokens,
        )

        expected_shape = (
            int(batch_size),
            int(grid_h) * int(grid_w),
            self.sam_dim,
        )
        if tuple(aligned.shape) != expected_shape:
            raise RuntimeError(
                "aligned CLIP-SAM feature shape mismatch: expected "
                f"{expected_shape}, got {tuple(aligned.shape)}."
            )

        # No CLIP image residual here.
        return aligned.contiguous()


class CrossGuidedClipSamUpsampler(nn.Module):
    """
    Upsample aligned low-resolution CLIP-SAM feature with SAM3 high-res feature.

    New design:
        query = SAM3 high-res feature
        key   = aligned CLIP-SAM high-res feature
        value = aligned CLIP-SAM high-res feature

    There is no learnable gamma. Residual connection is handled by the window
    attention blocks with residual_source='value'.

    Input:
        aligned_clip_sam_feature_low: [B, Hc*Wc, D]
        sam3_feature_high:            [B, D, H, W]
        clip_grid_hw:                 (Hc, Wc)

    Output:
        clip_sam_feature_high:         [B, H*W, D]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = (
            self.window_size // 2
            if shift_size is None
            else int(shift_size)
        )

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                "shift_size must satisfy 0 <= shift_size < window_size, "
                f"got shift_size={self.shift_size}, window_size={self.window_size}."
            )

        self.sam_norm = nn.LayerNorm(self.hidden_dim)
        self.clip_norm = nn.LayerNorm(self.hidden_dim)

        self.window_attn = ShiftedWindowAttention2D(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=0,
            dropout=float(dropout),
            value_preserving=False,
            residual_source="value",
            use_residual_norm=True,
            use_rel_pos_bias=True,
        )

        self.shifted_window_attn = ShiftedWindowAttention2D(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            dropout=float(dropout),
            value_preserving=False,
            residual_source="value",
            use_residual_norm=True,
            use_rel_pos_bias=True,
        )

        self.out_norm = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def _map_layer_norm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim, height, width = x.shape
        x_tokens = x.flatten(2).transpose(1, 2).contiguous()
        x_tokens = norm(x_tokens)
        return x_tokens.transpose(1, 2).reshape(
            batch_size,
            dim,
            height,
            width,
        ).contiguous()

    def forward(
        self,
        aligned_clip_sam_feature_low: torch.Tensor,
        sam3_feature_high: torch.Tensor,
        clip_grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        if aligned_clip_sam_feature_low.dim() != 3:
            raise ValueError(
                "aligned_clip_sam_feature_low must be [B, Hc*Wc, D], "
                f"got {tuple(aligned_clip_sam_feature_low.shape)}."
            )
        if sam3_feature_high.dim() != 4:
            raise ValueError(
                "sam3_feature_high must be [B, D, H, W], "
                f"got {tuple(sam3_feature_high.shape)}."
            )

        batch_size, num_low_tokens, dim = aligned_clip_sam_feature_low.shape
        sam_batch, sam_dim, high_h, high_w = sam3_feature_high.shape

        if int(dim) != self.hidden_dim:
            raise ValueError(
                f"aligned_clip_sam_feature_low dim mismatch: "
                f"expected {self.hidden_dim}, got {dim}."
            )
        if int(sam_dim) != self.hidden_dim:
            raise ValueError(
                f"sam3_feature_high dim mismatch: expected {self.hidden_dim}, "
                f"got {sam_dim}."
            )
        if int(sam_batch) != int(batch_size):
            raise ValueError(
                "Batch mismatch between aligned feature and SAM3 feature: "
                f"{batch_size} vs {sam_batch}."
            )

        clip_h, clip_w = int(clip_grid_hw[0]), int(clip_grid_hw[1])
        if clip_h * clip_w != int(num_low_tokens):
            raise ValueError(
                "clip_grid_hw does not match aligned feature token count: "
                f"{clip_h} * {clip_w} != {num_low_tokens}."
            )

        clip_low_map = aligned_clip_sam_feature_low.transpose(1, 2).reshape(
            batch_size,
            self.hidden_dim,
            clip_h,
            clip_w,
        )

        clip_high_base = F.interpolate(
            clip_low_map,
            size=(int(high_h), int(high_w)),
            mode="bilinear",
            align_corners=False,
        )

        sam_map = self._map_layer_norm(self.sam_norm, sam3_feature_high)
        clip_high_base = self._map_layer_norm(self.clip_norm, clip_high_base)

        guided = self.window_attn(
            query_map=sam_map,
            key_map=clip_high_base,
            value_map=clip_high_base,
        )

        guided = self.shifted_window_attn(
            query_map=sam_map,
            key_map=clip_high_base,
            value_map=guided,
        )

        clip_sam_feature = self._map_layer_norm(self.out_norm, guided)
        return clip_sam_feature.flatten(2).transpose(1, 2).contiguous()


class ClipCoarseMaskEmbedder(nn.Module):
    """
    Build coarse CLIP segmentation embedding and add it to CLIP-SAM feature.

    New design:
        1. Average all CLIP template tokens for each class.
        2. Upsample CLIP image feature map to high resolution.
        3. Compute image-text similarity.
        4. Take per-pixel argmax class.
        5. Convert coarse class map to embedding map with class_code.
        6. Add coarse embedding to CLIP-SAM feature.
        7. Normalize.

    Input:
        clip_image_feat_map_native: [B, D_clip, Hc, Wc]
        clip_text_tokens_native:    [C, K, D_clip]
        class_code:                 [B, C, D_sam]
        clip_sam_feature_high:      [B, H*W, D_sam]
        output_hw:                  (H, W)

    Output:
        clip_sam_feature_high:      [B, H*W, D_sam]
        clip_coarse_logits:         [B, C, H, W]
        clip_coarse_pred:           [B, H, W]
    """

    def __init__(
        self,
        clip_dim: int,
        sam_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.sam_dim = int(sam_dim)
        self.eps = float(eps)

        if self.clip_dim <= 0:
            raise ValueError(f"clip_dim must be positive, got {clip_dim}.")
        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")

        self.out_norm = nn.LayerNorm(self.sam_dim)

    def _build_clip_coarse_logits(
        self,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        if clip_image_feat_map_native.dim() != 4:
            raise ValueError(
                "clip_image_feat_map_native must be [B, D_clip, Hc, Wc], "
                f"got {tuple(clip_image_feat_map_native.shape)}."
            )
        if clip_text_tokens_native.dim() != 3:
            raise ValueError(
                "clip_text_tokens_native must be [C, K, D_clip], "
                f"got {tuple(clip_text_tokens_native.shape)}."
            )

        batch_size, image_dim, _, _ = clip_image_feat_map_native.shape
        num_classes, _, text_dim = clip_text_tokens_native.shape

        if int(image_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP image dim mismatch: expected {self.clip_dim}, "
                f"got {image_dim}."
            )
        if int(text_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, "
                f"got {text_dim}."
            )

        high_h, high_w = int(output_hw[0]), int(output_hw[1])
        clip_high = F.interpolate(
            clip_image_feat_map_native,
            size=(high_h, high_w),
            mode="bilinear",
            align_corners=False,
        )

        image_tokens = clip_high.flatten(2).transpose(1, 2).contiguous()
        image_tokens = F.normalize(image_tokens, dim=-1, eps=self.eps)

        text_tokens = F.normalize(
            clip_text_tokens_native.to(
                device=clip_high.device,
                dtype=clip_high.dtype,
            ),
            dim=-1,
            eps=self.eps,
        )
        text_proto = text_tokens.mean(dim=1)
        text_proto = F.normalize(text_proto, dim=-1, eps=self.eps)

        clip_score = torch.einsum(
            "bnd,cd->bcn",
            image_tokens,
            text_proto,
        )

        return clip_score.reshape(
            int(batch_size),
            int(num_classes),
            high_h,
            high_w,
        ).contiguous()

    @staticmethod
    def _class_map_to_embedding(
        coarse_pred: torch.Tensor,
        class_code: torch.Tensor,
    ) -> torch.Tensor:
        if coarse_pred.dim() != 3:
            raise ValueError(
                "coarse_pred must be [B, H, W], "
                f"got {tuple(coarse_pred.shape)}."
            )
        if class_code.dim() != 3:
            raise ValueError(
                "class_code must be [B, C, D], "
                f"got {tuple(class_code.shape)}."
            )

        batch_size, height, width = coarse_pred.shape
        code_batch, num_classes, dim = class_code.shape

        if int(code_batch) != int(batch_size):
            raise ValueError(
                f"class_code batch mismatch: {code_batch} vs {batch_size}."
            )

        if coarse_pred.min().item() < 0 or coarse_pred.max().item() >= num_classes:
            raise ValueError(
                "coarse_pred contains class index outside class_code range."
            )

        flat_index = coarse_pred.reshape(batch_size, height * width)
        gather_index = flat_index.unsqueeze(-1).expand(batch_size, height * width, dim)

        coarse_embed = torch.gather(
            class_code,
            dim=1,
            index=gather_index,
        )

        return coarse_embed.transpose(1, 2).reshape(
            batch_size,
            dim,
            height,
            width,
        ).contiguous()

    def forward(
        self,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        class_code: torch.Tensor,
        clip_sam_feature_high: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if clip_sam_feature_high.dim() != 3:
            raise ValueError(
                "clip_sam_feature_high must be [B, H*W, D_sam], "
                f"got {tuple(clip_sam_feature_high.shape)}."
            )
        if class_code.dim() != 3:
            raise ValueError(
                "class_code must be [B, C, D_sam], "
                f"got {tuple(class_code.shape)}."
            )

        batch_size, num_tokens, dim = clip_sam_feature_high.shape
        code_batch, _, code_dim = class_code.shape
        high_h, high_w = int(output_hw[0]), int(output_hw[1])

        if int(num_tokens) != high_h * high_w:
            raise ValueError(
                "clip_sam_feature_high token count mismatch: "
                f"{num_tokens} vs {high_h}*{high_w}."
            )
        if int(dim) != self.sam_dim:
            raise ValueError(
                f"clip_sam_feature_high dim mismatch: expected {self.sam_dim}, "
                f"got {dim}."
            )
        if int(code_batch) != int(batch_size):
            raise ValueError(
                f"class_code batch mismatch: {code_batch} vs {batch_size}."
            )
        if int(code_dim) != self.sam_dim:
            raise ValueError(
                f"class_code dim mismatch: expected {self.sam_dim}, "
                f"got {code_dim}."
            )

        clip_coarse_logits = self._build_clip_coarse_logits(
            clip_image_feat_map_native=clip_image_feat_map_native,
            clip_text_tokens_native=clip_text_tokens_native,
            output_hw=(high_h, high_w),
        )
        clip_coarse_pred = clip_coarse_logits.argmax(dim=1).long()

        coarse_embed = self._class_map_to_embedding(
            coarse_pred=clip_coarse_pred,
            class_code=class_code,
        )

        clip_sam_map = clip_sam_feature_high.transpose(1, 2).reshape(
            batch_size,
            self.sam_dim,
            high_h,
            high_w,
        )

        fused = clip_sam_map + coarse_embed
        fused = fused.flatten(2).transpose(1, 2).contiguous()
        fused = self.out_norm(fused)

        return (
            fused.contiguous(),
            clip_coarse_logits.contiguous(),
            clip_coarse_pred.contiguous(),
        )