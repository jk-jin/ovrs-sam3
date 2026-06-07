from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenCLIPTextEncoder(nn.Module):
    """
    Frozen OpenCLIP text wrapper.

    输出规则：
    1. token_features 仍然是 transformer + ln_final 后的 token 序列特征。
    2. pooled / prompt template features 会经过 OpenCLIP 原始 text_projection。
    3. output_dim 表示投影后的 CLIP 图文对齐空间维度。
    """

    def __init__(
        self,
        tokenizer: Callable,
        token_embedding: nn.Module,
        positional_embedding: torch.Tensor,
        transformer: nn.Module,
        ln_final: nn.Module,
        text_projection: nn.Module | torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        context_length: int,
        width: int,
        use_ln_post: bool = True,
    ) -> None:
        super().__init__()

        if text_projection is None:
            raise ValueError("OpenCLIPTextEncoder requires text_projection.")

        self.tokenizer = tokenizer
        self.token_embedding = token_embedding
        self.transformer = transformer
        self.ln_final = ln_final if use_ln_post else nn.Identity()
        self.context_length = int(context_length)
        self.width = int(width)

        if isinstance(text_projection, nn.Linear):
            self.text_projection = text_projection
            self.output_dim = int(text_projection.out_features)
        else:
            proj = torch.as_tensor(text_projection).detach().clone()
            if proj.ndim != 2:
                raise ValueError(
                    f"Expected text_projection as 2D matrix, got {tuple(proj.shape)}"
                )
            self.text_projection = nn.Parameter(proj)
            self.output_dim = int(proj.shape[1])

        self.positional_embedding = nn.Parameter(
            positional_embedding.detach().clone(),
            requires_grad=False,
        )

        self.register_buffer(
            "_attn_mask_buffer",
            attn_mask.detach().clone() if attn_mask is not None else torch.empty(0),
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

    def _apply_text_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, width]

        Returns:
            projected: [N, output_dim]
        """
        if isinstance(self.text_projection, nn.Linear):
            return self.text_projection(x)

        proj = self.text_projection.to(device=x.device, dtype=x.dtype)
        return x @ proj

    def encode_text(
        self,
        text: List[str],
        device: Optional[torch.device] = None,
        output_mode: str = "pooled",
        normalize: bool = False,
    ):
        tokenized = self.tokenizer(text, context_length=self.context_length)
        if device is not None:
            tokenized = tokenized.to(device)

        seq_len = tokenized.shape[1]

        with torch.no_grad():
            input_embeds = self.token_embedding(tokenized)

            x = input_embeds + self.positional_embedding[:seq_len].to(
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            )

            attn_mask = self._get_attn_mask(
                seq_len=seq_len,
                device=x.device,
                dtype=x.dtype,
            )

            x = self.transformer(x, attn_mask=attn_mask)
            token_features = self.ln_final(x)

            pooled = token_features[
                torch.arange(token_features.shape[0], device=token_features.device),
                tokenized.argmax(dim=-1),
            ]

            pooled = self._apply_text_projection(pooled)

            if normalize:
                pooled = F.normalize(pooled, dim=-1)

        token_features = token_features.detach()
        input_embeds = input_embeds.detach()
        pooled = pooled.detach()

        if output_mode == "token_features":
            return tokenized, token_features, input_embeds

        if output_mode == "pooled":
            return tokenized, pooled, input_embeds

        if output_mode == "all":
            return {
                "tokenized": tokenized,
                "token_features": token_features,
                "input_embeds": input_embeds,
                "pooled": pooled,
            }

        raise ValueError(
            f"Unknown output_mode={output_mode}. "
            "Supported modes are: token_features, pooled, all."
        )

    def encode_embeds(
        self,
        input_embeds: torch.Tensor,
        tokenized: torch.Tensor,
        normalize: bool = True,
        detach_output: bool = False,
    ) -> torch.Tensor:
        """
        Forward text embeddings through the frozen CLIP text transformer.

        Unlike encode_text(), this method does NOT wrap the forward pass in
        torch.no_grad().  CLIP text encoder parameters are frozen
        (requires_grad=False), but autograd can still propagate gradients
        back to input_embeds, which is required for dynamic prompt training.

        Args:
            input_embeds: [N, L, width]  token embeddings
            tokenized:    [N, L]          token ids, used to locate EOT
            normalize:    whether to L2-normalize pooled output
            detach_output: if True, detach pooled before returning

        Returns:
            pooled: [N, output_dim]
        """
        seq_len = input_embeds.shape[1]
        x = input_embeds + self.positional_embedding[:seq_len].to(
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )
        attn_mask = self._get_attn_mask(seq_len=seq_len, device=x.device, dtype=x.dtype)
        x = self.transformer(x, attn_mask=attn_mask)
        token_features = self.ln_final(x)

        pooled = token_features[
            torch.arange(token_features.shape[0], device=token_features.device),
            tokenized.argmax(dim=-1),
        ]
        pooled = self._apply_text_projection(pooled)
        if normalize:
            pooled = F.normalize(pooled, dim=-1)
        if detach_output:
            pooled = pooled.detach()
        return pooled

    def encode_prompt_templates(
        self,
        class_names: List[str],
        templates: List[str],
        device: Optional[torch.device] = None,
        normalize_label: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            class_names: 类别名列表，长度为 C
            templates: prompt 模板列表，长度为 K
            normalize_label: 是否把类别名里的 '_'、'-' 替换成空格
            normalize: 是否对投影后的文本向量做 L2 normalize

        Returns:
            pooled: [C, K, output_dim]

        符号说明：
            C 表示类别数。
            K 表示每个类别使用的模板数量。
            output_dim 表示 OpenCLIP 图文对齐空间维度。
        """
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
            normalize=normalize,
        )

        num_classes = len(class_names)
        num_templates = len(templates)

        pooled = pooled.view(num_classes, num_templates, self.output_dim)
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
                "OpenCLIPTextEncoder expects a non-empty List[str]."
            )

        tokenized, token_features, input_embeds = self.encode_text(
            text=text,
            device=device,
            output_mode="token_features",
            normalize=False,
        )

        text_attention_mask = tokenized.eq(0)          # [B, L]
        text_memory = token_features.transpose(0, 1)   # [L, B, width]
        text_embeds = input_embeds.transpose(0, 1)     # [L, B, width]

        return text_attention_mask, text_memory, text_embeds