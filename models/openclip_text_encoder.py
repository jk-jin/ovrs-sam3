from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class OpenCLIPTextEncoder(nn.Module):
    """
    OpenCLIP text wrapper.

    Responsibilities:
    1. Hold tokenizer and raw OpenCLIP text tower modules.
    2. Return text outputs in the same contract as VETextEncoder.
    3. Internally project token-level encoded text features to d_model.

    Notes:
    - We only project encoded token features (text_memory).
    - We keep input_embeds in the original OpenCLIP width, matching VETextEncoder behavior.
    """

    def __init__(
        self,
        tokenizer: Callable,
        token_embedding: nn.Module,
        positional_embedding: torch.Tensor,
        transformer: nn.Module,
        ln_final: nn.Module,
        attn_mask: Optional[torch.Tensor],
        context_length: int,
        width: int,
        d_model: int = 256,
        use_ln_post: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.token_embedding = token_embedding
        self.transformer = transformer
        self.ln_final = ln_final if use_ln_post else nn.Identity()
        self.context_length = int(context_length)
        self.width = int(width)
        self.d_model = int(d_model)

        self.positional_embedding = positional_embedding
        self.resizer = nn.Linear(self.width, self.d_model)

        self.register_buffer(
            "_attn_mask_buffer",
            attn_mask if attn_mask is not None else torch.empty(0),
            persistent=False,
        )

    def _get_attn_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self._attn_mask_buffer.numel() == 0:
            return None

        attn_mask = self._attn_mask_buffer[:seq_len, :seq_len].to(device=device)
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(dtype=dtype)
        return attn_mask

    def _encode_token_features(
            self,
            tokenized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token_features: [B, L, C_text]
            input_embeds:   [B, L, C_text]
        """
        seq_len = tokenized.shape[1]

        input_embeds = self.token_embedding(tokenized)  # [B, L, C_text]
        x = input_embeds + self.positional_embedding[:seq_len].to(
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        attn_mask = self._get_attn_mask(
            seq_len=seq_len,
            device=x.device,
            dtype=x.dtype,
        )

        x = self.transformer(x, attn_mask=attn_mask)  # [B, L, C_text]
        x = self.ln_final(x)

        return x, input_embeds

    def encode_text(
        self,
        text: List[str],
        device: Optional[torch.device] = None,
        output_mode: str = "token_features",
    ):
        tokenized = self.tokenizer(text, context_length=self.context_length)
        if device is not None:
            tokenized = tokenized.to(device)

        token_features, input_embeds = self._encode_token_features(tokenized)
        token_features_resized = self.resizer(token_features)

        if output_mode == "token_features":
            return tokenized, token_features_resized, input_embeds

        if output_mode == "pooled":
            pooled = token_features_resized[
                torch.arange(token_features_resized.shape[0], device=token_features_resized.device),
                tokenized.argmax(dim=-1),
            ]
            return tokenized, pooled, input_embeds

        if output_mode == "all":
            pooled = token_features_resized[
                torch.arange(token_features_resized.shape[0], device=token_features_resized.device),
                tokenized.argmax(dim=-1),
            ]
            return {
                "tokenized": tokenized,
                "token_features": token_features_resized,
                "input_embeds": input_embeds,
                "pooled": pooled,
            }

        raise ValueError(
            f"Unknown output_mode={output_mode}. "
            "Supported modes are: token_features, pooled, all."
        )

    def encode_prompt_templates(
            self,
            class_names: List[str],
            templates: List[str],
            device: Optional[torch.device] = None,
            normalize_label: bool = True,
    ) -> torch.Tensor:
        if len(class_names) == 0:
            raise ValueError("class_names is empty.")
        if len(templates) == 0:
            raise ValueError("templates is empty.")

        def normalize_name(x: str) -> str:
            x = x.strip()
            if normalize_label:
                x = x.replace("_", " ").replace("-", " ")
                x = " ".join(x.split())
            return x

        flat_texts = []
        for name in class_names:
            name = normalize_name(name)
            for tpl in templates:
                flat_texts.append(tpl.format(name))

        _, pooled, _ = self.encode_text(
            text=flat_texts,
            device=device,
            output_mode="pooled",
        )

        num_classes = len(class_names)
        num_templates = len(templates)
        pooled = pooled.view(num_classes, num_templates, self.d_model)  # [B, K, 256]
        return pooled

    def forward(
        self,
        text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
        input_boxes: Optional[List] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_boxes is not None and len(input_boxes) > 0:
            raise NotImplementedError(
                "OpenCLIPTextEncoder currently does not support box replacement inside text."
            )

        if not isinstance(text, list) or len(text) == 0 or not isinstance(text[0], str):
            raise TypeError(
                "OpenCLIPTextEncoder currently expects a non-empty List[str]."
            )

        tokenized, token_features_resized, input_embeds = self.encode_text(
            text=text,
            device=device,
            output_mode="token_features",
        )

        # Keep the same convention as VETextEncoder:
        # True means padding, False means valid token
        text_attention_mask = tokenized.eq(0)  # [B, L]

        # Downstream expects sequence-first
        text_memory = token_features_resized.transpose(0, 1)  # [L, B, d_model]
        text_embeds = input_embeds.transpose(0, 1)            # [L, B, C_text]

        return text_attention_mask, text_memory, text_embeds