from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SingleTokenClipPromptEncoder(nn.Module):
    """
    DenseCLIP-style CLIP text token builder.

    Flow:
        class_names + template
        → OpenCLIP text encoder
        → base_clip_text: [C, D_clip]

        base_clip_text (repeated Q times)
        + class_query_tokens (projected to D_clip)
        → channel concat
        → Linear → D_clip
        + projected_class_tokens (residual)
        → cross-attention to clip_image_feat_map
        + repeated_base_text (residual)
        → dynamic_clip_text: [B, C, Q, D_clip]

    Important:
        - Do NOT insert dynamic tokens into CLIP text encoder input.
        - Only static class prompts go through the CLIP text encoder.
    """

    def __init__(
        self,
        clip_text_encoder,
        prompt_template: str = "a remote sensing image of {}.",
        sam_dim: int = 256,
        normalize_label: bool = True,
        use_checkpoint: bool = True,
        num_attention_heads: int = 8,
    ):
        super().__init__()

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.prompt_template = str(prompt_template)
        self.sam_dim = int(sam_dim)
        self.normalize_label = bool(normalize_label)
        self.use_checkpoint = bool(use_checkpoint)

        if "{}" not in self.prompt_template:
            raise ValueError(
                f"prompt_template must contain '{{}}', got {self.prompt_template!r}"
            )

        self.clip_output_dim = int(clip_text_encoder.output_dim)
        self._context_length = int(clip_text_encoder.context_length)
        self._tokenizer = clip_text_encoder.tokenizer

        if self.clip_output_dim % int(num_attention_heads) != 0:
            raise ValueError(
                f"clip_output_dim={self.clip_output_dim} must be divisible by "
                f"num_attention_heads={num_attention_heads}"
            )

        self.class_token_to_clip = nn.Sequential(
            nn.LayerNorm(self.sam_dim),
            nn.Linear(self.sam_dim, self.clip_output_dim),
        )

        self.text_class_fusion = nn.Linear(
            self.clip_output_dim * 2,
            self.clip_output_dim,
        )
        self.fusion_norm = nn.LayerNorm(self.clip_output_dim)

        self.query_norm = nn.LayerNorm(self.clip_output_dim)
        self.visual_norm = nn.LayerNorm(self.clip_output_dim)

        self.clip_image_attn = nn.MultiheadAttention(
            embed_dim=self.clip_output_dim,
            num_heads=int(num_attention_heads),
            batch_first=True,
        )

        self._base_text_cache_key: Optional[Tuple[Tuple[str, ...], str]] = None
        self._base_text_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        name = str(name).strip()
        name = name.replace("_", " ").replace("-", " ")
        return " ".join(name.split())

    def _build_prompt_texts(self, class_names: List[str]) -> List[str]:
        texts = []
        for name in class_names:
            label = self._normalize_class_name(name) if self.normalize_label else str(name)
            texts.append(self.prompt_template.format(label))
        return texts

    def _has_trainable_clip_text_params(self) -> bool:
        return any(p.requires_grad for p in self.clip_text_encoder.parameters())

    def _should_use_text_cache(self) -> bool:
        return (not self.training) or (not self._has_trainable_clip_text_params())

    def clear_text_cache(self) -> None:
        self._base_text_cache_key = None
        self._base_text_cache = None

    def _cache_key(self, class_names: List[str], device: torch.device):
        return (tuple(self._build_prompt_texts(class_names)), str(device))

    # ------------------------------------------------------------------
    # Base text encoding
    # ------------------------------------------------------------------

    def _encode_base_text_features(
        self,
        class_names: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        texts = self._build_prompt_texts(class_names)
        tokenized = self._tokenizer(
            texts,
            context_length=self._context_length,
        ).to(device)

        def _encode(tokens):
            return self.clip_text_encoder.encode_tokenized(
                tokenized=tokens,
                normalize=True,
                detach_output=False,
            )

        if (
            self.training
            and self.use_checkpoint
            and self._has_trainable_clip_text_params()
        ):
            return checkpoint(
                _encode, tokenized, use_reentrant=False,
            )

        return _encode(tokenized)

    def _get_base_text_features(
        self,
        class_names: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        if not self._should_use_text_cache():
            return self._encode_base_text_features(class_names, device)

        key = self._cache_key(class_names, device)
        if self._base_text_cache is not None and self._base_text_cache_key == key:
            return self._base_text_cache.to(device=device)

        with torch.no_grad():
            base_text = self._encode_base_text_features(class_names, device)

        self._base_text_cache_key = key
        self._base_text_cache = base_text.detach().contiguous()
        return self._base_text_cache.to(device=device)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_text_and_class_tokens(
        self,
        class_query_tokens: torch.Tensor,
        base_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            class_query_tokens: [B, C, Q, D]
            base_text:          [C, D_clip]

        Returns:
            fused: [B, C, Q, D_clip]
        """
        B, C, Q, _ = class_query_tokens.shape

        projected_class_tokens = self.class_token_to_clip(class_query_tokens)
        base_text = base_text.to(
            device=projected_class_tokens.device,
            dtype=projected_class_tokens.dtype,
        )

        repeated_base_text = base_text[None, :, None, :].expand(
            B, C, Q, self.clip_output_dim
        )

        x = torch.cat([repeated_base_text, projected_class_tokens], dim=-1)
        fused = self.text_class_fusion(x)
        fused = self.fusion_norm(fused + projected_class_tokens)
        return fused

    def _attend_to_clip_image(
        self,
        fused_tokens: torch.Tensor,
        base_text: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_tokens:        [B, C, Q, D_clip]
            base_text:           [C, D_clip]
            clip_image_feat_map: [B, D_clip, Hc, Wc]

        Returns:
            dynamic_clip_text: [B, C, Q, D_clip]
        """
        B, C, Q, D_clip = fused_tokens.shape

        if clip_image_feat_map.shape[0] != B:
            raise ValueError(
                f"clip_image_feat_map batch mismatch: "
                f"expected {B}, got {clip_image_feat_map.shape[0]}"
            )
        if clip_image_feat_map.shape[1] != D_clip:
            raise ValueError(
                f"clip_image_feat_map dim mismatch: "
                f"expected {D_clip}, got {clip_image_feat_map.shape[1]}"
            )

        query = self.query_norm(fused_tokens).reshape(B * C, Q, D_clip)

        visual = clip_image_feat_map.flatten(2).transpose(1, 2)
        visual = visual.to(device=query.device, dtype=query.dtype)
        visual = self.visual_norm(visual)

        visual = (
            visual[:, None]
            .expand(B, C, visual.shape[1], D_clip)
            .reshape(B * C, visual.shape[1], D_clip)
        )

        attn_out, _ = self.clip_image_attn(
            query=query,
            key=visual,
            value=visual,
            need_weights=False,
        )
        attn_out = attn_out.reshape(B, C, Q, D_clip)

        base_text = base_text.to(device=attn_out.device, dtype=attn_out.dtype)
        repeated_base_text = base_text[None, :, None, :].expand(B, C, Q, D_clip)

        dynamic_clip_text = attn_out + repeated_base_text
        return dynamic_clip_text.contiguous()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        class_query_tokens: torch.Tensor,
        class_names: List[str],
        clip_image_feat_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            class_query_tokens:  [B, C, Q, D]
            class_names:         list[str], length C
            clip_image_feat_map: [B, D_clip, Hc, Wc]

        Returns:
            dynamic_clip_text: [B, C, Q, D_clip]
        """
        B, C, Q, _ = class_query_tokens.shape

        if len(class_names) != C:
            raise ValueError(
                f"class_names length mismatch: expected {C}, got {len(class_names)}"
            )

        base_text = self._get_base_text_features(
            class_names=class_names,
            device=class_query_tokens.device,
        )

        fused = self._fuse_text_and_class_tokens(
            class_query_tokens=class_query_tokens,
            base_text=base_text,
        )

        dynamic_clip_text = self._attend_to_clip_image(
            fused_tokens=fused,
            base_text=base_text,
            clip_image_feat_map=clip_image_feat_map,
        )

        return dynamic_clip_text
