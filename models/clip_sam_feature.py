from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalClipSamFeatureBuilder(nn.Module):
    """
    Build one shared full-class CLIP-SAM feature memory for the whole batch.

    Input shapes:
        clip_image_feat_map_native: [B, D_clip, Hc, Wc]
        clip_text_tokens_native:    [C, K, D_clip]
        sam3_text_tokens_full:      [M, C, D_sam]
        sam3_text_mask_full:        [C, M]

    Output shape:
        shared_clip_feature: [B, Hc * Wc, D_out]

    Symbol meanings:
        B means batch size.
        C means full class count.
        K means CLIP prompt template count per class.
        M means SAM3 text token count per class.
        D_clip means native CLIP feature dimension.
        D_sam means SAM3 hidden dimension.
        Hc and Wc mean CLIP image patch grid height and width.
        D_out is clip_feature_dim. In the current config it should be 256.
    """

    def __init__(
        self,
        clip_dim: int,
        sam_dim: int,
        clip_feature_dim: int = 256,
        attn_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_text_latents: int = 32,
    ) -> None:
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.sam_dim = int(sam_dim)
        self.clip_feature_dim = int(clip_feature_dim)
        self.attn_dim = int(attn_dim) if attn_dim is not None else int(clip_dim)
        self.num_heads = int(num_heads)
        self.num_text_latents = int(num_text_latents)

        if self.clip_dim <= 0:
            raise ValueError(f"clip_dim must be positive, got {clip_dim}.")
        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.clip_feature_dim <= 0:
            raise ValueError(
                f"clip_feature_dim must be positive, got {clip_feature_dim}."
            )
        if self.attn_dim <= 0:
            raise ValueError(f"attn_dim must be positive, got {attn_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.num_text_latents <= 0:
            raise ValueError(
                f"num_text_latents must be positive, got {num_text_latents}."
            )
        if self.attn_dim % self.num_heads != 0:
            raise ValueError(
                "attn_dim must be divisible by num_heads, "
                f"got attn_dim={self.attn_dim}, num_heads={self.num_heads}."
            )
        if self.sam_dim % self.num_heads != 0:
            raise ValueError(
                "sam_dim must be divisible by num_heads, "
                f"got sam_dim={self.sam_dim}, num_heads={self.num_heads}."
            )
        if self.clip_feature_dim % self.num_heads != 0:
            raise ValueError(
                "clip_feature_dim must be divisible by num_heads, "
                f"got clip_feature_dim={self.clip_feature_dim}, "
                f"num_heads={self.num_heads}."
            )

        self.qk_head_dim = self.attn_dim // self.num_heads
        self.v_head_dim = self.clip_feature_dim // self.num_heads

        # Step 1:
        # Per-class learnable slots attend CLIP prompt template tokens.
        self.clip_template_queries = nn.Parameter(
            torch.zeros(1, self.num_text_latents, self.clip_dim)
        )
        nn.init.normal_(self.clip_template_queries, std=0.02)

        self.clip_template_attn = nn.MultiheadAttention(
            embed_dim=self.clip_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.clip_template_norm = nn.LayerNorm(self.clip_dim)

        # Step 2:
        # CLIP latents query valid SAM3 text tokens.
        self.clip_latent_to_sam_query = nn.Linear(self.clip_dim, self.sam_dim)
        self.sam_text_attn = nn.MultiheadAttention(
            embed_dim=self.sam_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.sam_aligned_norm = nn.LayerNorm(self.sam_dim)

        # Step 3:
        # CLIP image tokens attend all class-latent keys and SAM-aligned values.
        self.image_query_proj = nn.Linear(self.clip_dim, self.attn_dim)
        self.latent_key_proj = nn.Linear(self.clip_dim, self.attn_dim)
        self.aligned_value_proj = nn.Linear(self.sam_dim, self.clip_feature_dim)

        self.clip_image_to_sam_proj = nn.Linear(
            self.clip_dim,
            self.clip_feature_dim,
        )

        self.dropout = nn.Dropout(float(dropout))
        self.out_norm = nn.LayerNorm(self.clip_feature_dim)

    @staticmethod
    def _check_sam3_text_inputs(
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
        num_classes: int,
    ) -> None:
        if sam3_text_tokens_full.dim() != 3:
            raise ValueError(
                "sam3_text_tokens_full must be [M, C, D_sam], "
                f"got {tuple(sam3_text_tokens_full.shape)}."
            )
        if sam3_text_mask_full.dim() != 2:
            raise ValueError(
                "sam3_text_mask_full must be [C, M], "
                f"got {tuple(sam3_text_mask_full.shape)}."
            )

        seq_len, sam_num_classes, _ = sam3_text_tokens_full.shape
        if int(sam_num_classes) != int(num_classes):
            raise ValueError(
                "Class count mismatch between CLIP text and SAM3 text: "
                f"{num_classes} vs {sam_num_classes}."
            )
        if tuple(sam3_text_mask_full.shape) != (num_classes, seq_len):
            raise ValueError(
                "SAM3 text mask shape mismatch: expected "
                f"[C, M] = [{num_classes}, {seq_len}], "
                f"got {tuple(sam3_text_mask_full.shape)}."
            )

    @staticmethod
    def _sanitize_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        MultiheadAttention returns NaN if one row is fully masked.
        This should not happen for valid SAM3 text, but this guard makes the
        module robust to empty class names or unexpected tokenizer output.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()

        fully_masked = mask.all(dim=1)
        if fully_masked.any():
            mask = mask.clone()
            mask[fully_masked, 0] = False
        return mask

    def _build_clip_latents(
        self,
        clip_text_tokens_native: torch.Tensor,
    ) -> torch.Tensor:
        if clip_text_tokens_native.dim() != 3:
            raise ValueError(
                "clip_text_tokens_native must be [C, K, D_clip], "
                f"got {tuple(clip_text_tokens_native.shape)}."
            )

        num_classes, _, text_dim = clip_text_tokens_native.shape
        if int(text_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, got {text_dim}."
            )

        query = self.clip_template_queries.to(
            device=clip_text_tokens_native.device,
            dtype=clip_text_tokens_native.dtype,
        )
        query = query.expand(
            num_classes,
            self.num_text_latents,
            self.clip_dim,
        )

        attn_out, _ = self.clip_template_attn(
            query=query,
            key=clip_text_tokens_native,
            value=clip_text_tokens_native,
            need_weights=False,
        )

        clip_latents = self.clip_template_norm(query + attn_out)
        return clip_latents

    def _build_sam3_aligned_values(
        self,
        clip_latents: torch.Tensor,
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
    ) -> torch.Tensor:
        if clip_latents.dim() != 3:
            raise ValueError(
                "clip_latents must be [C, L, D_clip], "
                f"got {tuple(clip_latents.shape)}."
            )

        num_classes, _, latent_dim = clip_latents.shape
        if int(latent_dim) != self.clip_dim:
            raise ValueError(
                f"clip_latents dim mismatch: expected {self.clip_dim}, "
                f"got {latent_dim}."
            )

        self._check_sam3_text_inputs(
            sam3_text_tokens_full=sam3_text_tokens_full,
            sam3_text_mask_full=sam3_text_mask_full,
            num_classes=num_classes,
        )

        sam3_tokens = sam3_text_tokens_full.permute(1, 0, 2).contiguous()
        sam3_mask = self._sanitize_key_padding_mask(sam3_text_mask_full)

        query = self.clip_latent_to_sam_query(clip_latents)

        attn_out, _ = self.sam_text_attn(
            query=query,
            key=sam3_tokens,
            value=sam3_tokens,
            key_padding_mask=sam3_mask,
            need_weights=False,
        )

        sam3_aligned_values = self.sam_aligned_norm(query + attn_out)
        return sam3_aligned_values

    def _image_tokens_attend_global_latents(
        self,
        image_tokens: torch.Tensor,
        clip_latents: torch.Tensor,
        sam3_aligned_values: torch.Tensor,
    ) -> torch.Tensor:
        if image_tokens.dim() != 3:
            raise ValueError(
                "image_tokens must be [B, N_clip, D_clip], "
                f"got {tuple(image_tokens.shape)}."
            )
        if clip_latents.dim() != 3:
            raise ValueError(
                "clip_latents must be [C, L, D_clip], "
                f"got {tuple(clip_latents.shape)}."
            )
        if sam3_aligned_values.dim() != 3:
            raise ValueError(
                "sam3_aligned_values must be [C, L, D_sam], "
                f"got {tuple(sam3_aligned_values.shape)}."
            )

        batch_size, num_clip_tokens, image_dim = image_tokens.shape
        num_classes, num_latents, latent_dim = clip_latents.shape

        if int(image_dim) != self.clip_dim:
            raise ValueError(
                f"image_tokens dim mismatch: expected {self.clip_dim}, got {image_dim}."
            )
        if int(latent_dim) != self.clip_dim:
            raise ValueError(
                f"clip_latents dim mismatch: expected {self.clip_dim}, got {latent_dim}."
            )
        if tuple(sam3_aligned_values.shape[:2]) != (num_classes, num_latents):
            raise ValueError(
                "sam3_aligned_values shape mismatch: expected first dims "
                f"{(num_classes, num_latents)}, got "
                f"{tuple(sam3_aligned_values.shape[:2])}."
            )
        if int(sam3_aligned_values.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"sam3_aligned_values dim mismatch: expected {self.sam_dim}, "
                f"got {sam3_aligned_values.shape[-1]}."
            )

        query = self.image_query_proj(image_tokens)
        key = self.latent_key_proj(clip_latents)
        value = self.aligned_value_proj(sam3_aligned_values)

        key = key.reshape(
            num_classes * num_latents,
            self.attn_dim,
        )
        value = value.reshape(
            num_classes * num_latents,
            self.clip_feature_dim,
        )

        query = query.reshape(
            batch_size,
            num_clip_tokens,
            self.num_heads,
            self.qk_head_dim,
        )
        query = query.permute(0, 2, 1, 3).contiguous()

        key = key.reshape(
            num_classes * num_latents,
            self.num_heads,
            self.qk_head_dim,
        )
        key = key.permute(1, 0, 2).contiguous()

        value = value.reshape(
            num_classes * num_latents,
            self.num_heads,
            self.v_head_dim,
        )
        value = value.permute(1, 0, 2).contiguous()

        attn_logits = torch.einsum("bhnd,hkd->bhnk", query, key)
        attn_logits = attn_logits / math.sqrt(float(self.qk_head_dim))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        attention_out = torch.einsum("bhnk,hkd->bhnd", attn, value)
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
        attention_out = attention_out.reshape(
            batch_size,
            num_clip_tokens,
            self.clip_feature_dim,
        )
        return attention_out

    def forward(
        self,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        sam3_text_tokens_full: torch.Tensor,
        sam3_text_mask_full: torch.Tensor,
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

        batch_size, image_dim, grid_h, grid_w = clip_image_feat_map_native.shape
        num_classes, _, text_dim = clip_text_tokens_native.shape

        if int(image_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP image dim mismatch: expected {self.clip_dim}, got {image_dim}."
            )
        if int(text_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, got {text_dim}."
            )
        if int(sam3_text_tokens_full.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"SAM3 text dim mismatch: expected {self.sam_dim}, "
                f"got {sam3_text_tokens_full.shape[-1]}."
            )

        device = clip_image_feat_map_native.device
        dtype = clip_image_feat_map_native.dtype

        clip_text_tokens_native = clip_text_tokens_native.to(
            device=device,
            dtype=dtype,
        )
        sam3_text_tokens_full = sam3_text_tokens_full.to(
            device=device,
            dtype=dtype,
        )
        sam3_text_mask_full = sam3_text_mask_full.to(device=device)

        image_tokens = clip_image_feat_map_native.flatten(2).transpose(1, 2)
        image_tokens = image_tokens.contiguous()

        clip_latents = self._build_clip_latents(
            clip_text_tokens_native=clip_text_tokens_native,
        )
        sam3_aligned_values = self._build_sam3_aligned_values(
            clip_latents=clip_latents,
            sam3_text_tokens_full=sam3_text_tokens_full,
            sam3_text_mask_full=sam3_text_mask_full,
        )

        attention_out = self._image_tokens_attend_global_latents(
            image_tokens=image_tokens,
            clip_latents=clip_latents,
            sam3_aligned_values=sam3_aligned_values,
        )

        image_residual = self.clip_image_to_sam_proj(image_tokens)

        # Direct residual add. No learnable alpha.
        shared_clip_feature = image_residual + attention_out
        shared_clip_feature = self.out_norm(shared_clip_feature)

        expected_tokens = int(grid_h) * int(grid_w)
        expected_shape = (
            int(batch_size),
            expected_tokens,
            self.clip_feature_dim,
        )
        if tuple(shared_clip_feature.shape) != expected_shape:
            raise RuntimeError(
                "shared_clip_feature shape mismatch: expected "
                f"{expected_shape}, got {tuple(shared_clip_feature.shape)}."
            )

        return shared_clip_feature.contiguous()