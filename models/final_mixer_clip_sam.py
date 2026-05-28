from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipSamFeatureInitializer(nn.Module):
    """
    Build low-resolution aligned CLIP-SAM feature inside the final mixer.

    New design:
        1. Use current class tokens as q1.
        2. Map class tokens from SAM space to CLIP space with a lightweight
           channel adapter.
        3. Do not use learnable Q/K/V projections inside attention.
        4. q1 attends the frozen CLIP text template tokens with parameter-free
           multi-head scaled dot-product attention to generate k1.
        5. CLIP image tokens attend k1.
        6. Values are still the current class tokens in SAM space.
        7. No gate, no residual, no CLIP image residual.

    Input shapes:
        clip_image_feat_map_native: [B, D_clip, Hc, Wc]
        clip_text_tokens_native:    [C, K, D_clip]
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

        # This is a channel adapter, not an attention Q/K/V projection.
        # bias=False avoids adding a learned global shift in CLIP space.
        self.class_token_norm = nn.LayerNorm(self.sam_dim)
        self.class_token_to_clip = nn.Linear(
            self.sam_dim,
            self.clip_dim,
            bias=False,
        )

        self.attn_dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-6)

    def _build_clip_keys_from_class_tokens(
        self,
        clip_text_tokens_native: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build k1 by letting current class tokens attend the class-specific
        CLIP template tokens.

        Attention has no learnable Q/K/V projections.

        Args:
            clip_text_tokens_native: [C, K, D_clip]
            class_tokens:            [B, C, Q, D_sam]

        Returns:
            clip_keys: [B, C, Q, D_clip]
        """
        if clip_text_tokens_native.dim() != 3:
            raise ValueError(
                "clip_text_tokens_native must be [C, K, D_clip], "
                f"got {tuple(clip_text_tokens_native.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D_sam], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, num_classes, num_class_tokens, token_dim = class_tokens.shape
        text_classes, num_templates, text_dim = clip_text_tokens_native.shape

        if int(token_dim) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, "
                f"got {token_dim}."
            )
        if int(text_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP text dim mismatch: expected {self.clip_dim}, "
                f"got {text_dim}."
            )
        if int(text_classes) != int(num_classes):
            raise ValueError(
                "CLIP text class count mismatch: "
                f"{text_classes} vs {num_classes}."
            )

        clip_text_tokens_native = clip_text_tokens_native.to(
            device=class_tokens.device,
            dtype=class_tokens.dtype,
        )

        # q1: current image-conditioned class tokens in CLIP space.
        query = self.class_token_norm(class_tokens)
        query = self.class_token_to_clip(query)
        query = self._l2_normalize(query)

        # Frozen CLIP text tokens stay in CLIP space. No projection.
        text = self._l2_normalize(clip_text_tokens_native)

        query_heads = query.reshape(
            batch_size,
            num_classes,
            num_class_tokens,
            self.num_heads,
            self.clip_head_dim,
        ).permute(0, 1, 3, 2, 4).contiguous()
        # [B, C, heads, Q, D_head]

        text_heads = text.reshape(
            num_classes,
            num_templates,
            self.num_heads,
            self.clip_head_dim,
        ).permute(0, 2, 1, 3).contiguous()
        # [C, heads, K, D_head]

        attn_logits = torch.einsum(
            "bchqd,chkd->bchqk",
            query_heads,
            text_heads,
        )
        attn_logits = attn_logits * 20.0

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        attn_out = torch.einsum(
            "bchqk,chkd->bchqd",
            attn,
            text_heads,
        )
        attn_out = attn_out.permute(0, 1, 3, 2, 4).contiguous()
        clip_keys = attn_out.reshape(
            batch_size,
            num_classes,
            num_class_tokens,
            self.clip_dim,
        )

        # k1 is kept on the CLIP unit sphere for the following image attention.
        clip_keys = self._l2_normalize(clip_keys)
        return clip_keys.contiguous()

    def _image_tokens_attend_class_keys_values(
        self,
        image_tokens: torch.Tensor,
        clip_keys: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Let CLIP image tokens attend class-conditioned CLIP keys. Values are
        class tokens in SAM space.

        Attention has no learnable Q/K/V/out projections.

        Args:
            image_tokens: [B, N, D_clip]
            clip_keys:    [B, C, Q, D_clip]
            class_tokens: [B, C, Q, D_sam]

        Returns:
            aligned: [B, N, D_sam]
        """
        if image_tokens.dim() != 3:
            raise ValueError(
                "image_tokens must be [B, N, D_clip], "
                f"got {tuple(image_tokens.shape)}."
            )
        if clip_keys.dim() != 4:
            raise ValueError(
                "clip_keys must be [B, C, Q, D_clip], "
                f"got {tuple(clip_keys.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D_sam], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, num_image_tokens, image_dim = image_tokens.shape
        key_batch, num_classes, num_class_tokens, key_dim = clip_keys.shape
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
        if int(token_dim) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, "
                f"got {token_dim}."
            )
        if int(key_batch) != int(batch_size):
            raise ValueError(
                f"clip_keys batch mismatch: {key_batch} vs {batch_size}."
            )
        if int(token_batch) != int(batch_size):
            raise ValueError(
                f"class_tokens batch mismatch: {token_batch} vs {batch_size}."
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

        image_tokens = self._l2_normalize(image_tokens)
        clip_keys = self._l2_normalize(clip_keys)

        query = image_tokens.reshape(
            batch_size,
            num_image_tokens,
            self.num_heads,
            self.clip_head_dim,
        ).permute(0, 2, 1, 3).contiguous()
        # [B, heads, N, D_clip_head]

        key = clip_keys.reshape(
            batch_size,
            num_classes * num_class_tokens,
            self.num_heads,
            self.clip_head_dim,
        ).permute(0, 2, 1, 3).contiguous()
        # [B, heads, C*Q, D_clip_head]

        value = class_tokens.reshape(
            batch_size,
            num_classes * num_class_tokens,
            self.num_heads,
            self.sam_head_dim,
        ).permute(0, 2, 1, 3).contiguous()
        # [B, heads, C*Q, D_sam_head]

        attn_logits = torch.matmul(query, key.transpose(-2, -1))
        attn_logits = attn_logits * 20.0

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        aligned = torch.matmul(attn, value)
        aligned = aligned.permute(0, 2, 1, 3).contiguous()
        aligned = aligned.reshape(batch_size, num_image_tokens, self.sam_dim)

        return aligned.contiguous()

    def forward(
        self,
        clip_image_feat_map_native: torch.Tensor,
        clip_text_tokens_native: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if clip_image_feat_map_native.dim() != 4:
            raise ValueError(
                "clip_image_feat_map_native must be [B, D_clip, Hc, Wc], "
                f"got {tuple(clip_image_feat_map_native.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, D_sam], "
                f"got {tuple(class_tokens.shape)}."
            )

        batch_size, image_dim, grid_h, grid_w = clip_image_feat_map_native.shape
        token_batch, _, _, token_dim = class_tokens.shape

        if int(image_dim) != self.clip_dim:
            raise ValueError(
                f"CLIP image dim mismatch: expected {self.clip_dim}, "
                f"got {image_dim}."
            )
        if int(token_batch) != int(batch_size):
            raise ValueError(
                f"class_tokens batch mismatch: {token_batch} vs {batch_size}."
            )
        if int(token_dim) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, "
                f"got {token_dim}."
            )

        dtype = class_tokens.dtype
        device = class_tokens.device

        clip_image_feat_map_native = clip_image_feat_map_native.to(
            device=device,
            dtype=dtype,
        )
        clip_text_tokens_native = clip_text_tokens_native.to(
            device=device,
            dtype=dtype,
        )

        image_tokens = clip_image_feat_map_native.flatten(2).transpose(1, 2)
        image_tokens = image_tokens.contiguous()

        clip_keys = self._build_clip_keys_from_class_tokens(
            clip_text_tokens_native=clip_text_tokens_native,
            class_tokens=class_tokens,
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

        return aligned.contiguous()