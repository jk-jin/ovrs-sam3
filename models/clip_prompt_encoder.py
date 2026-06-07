from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SingleTokenClipPromptEncoder(nn.Module):
    """
    Single-template dynamic CLIP prompt encoder.

    For each class, each of the Q query tokens is individually projected to
    CLIP token space and inserted into the single prompt template.

    Layout:
        [SOS] prefix dynamic_token class_name suffix [EOS] [PAD...]

    Input:
        class_query_tokens: [B, C, Q, D_sam]
        class_names: list of C strings

    Output:
        dynamic_clip_text: [B, C, Q, D_clip]
    """

    def __init__(
        self,
        clip_text_encoder,
        prompt_template: str = "a remote sensing image of {}.",
        sam_dim: int = 256,
        normalize_label: bool = True,
    ):
        super().__init__()

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.prompt_template = str(prompt_template)
        self.sam_dim = int(sam_dim)
        self.normalize_label = bool(normalize_label)

        if "{}" not in self.prompt_template:
            raise ValueError(
                f"Prompt template must contain '{{}}' placeholder, "
                f"got {self.prompt_template!r}"
            )

        self.text_width = int(clip_text_encoder.width)
        self.clip_output_dim = int(clip_text_encoder.output_dim)
        self._context_length = int(clip_text_encoder.context_length)

        self.token_to_clip = nn.Sequential(
            nn.LayerNorm(self.sam_dim),
            nn.Linear(self.sam_dim, self.text_width),
        )

        self.dynamic_norm = nn.LayerNorm(self.text_width)
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))

        object.__setattr__(
            self, "_frozen_token_embedding", clip_text_encoder.token_embedding
        )
        self._tokenizer = clip_text_encoder.tokenizer

        # Pre-compute template parts.
        prefix_str, suffix_str = self.prompt_template.split("{}", 1)
        self._prefix_ids = self._tokenize_and_strip(prefix_str)
        self._suffix_ids = self._tokenize_and_strip(suffix_str)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        name = name.strip()
        name = name.replace("_", " ").replace("-", " ")
        return " ".join(name.split())

    def _tokenize_single(self, text: str) -> torch.Tensor:
        tokens = self._tokenizer(text, context_length=self._context_length)
        return tokens[0]

    def _tokenize_and_strip(self, text: str) -> torch.Tensor:
        ids = self._tokenize_single(text)
        ids = ids[ids != 0]
        if ids.numel() >= 2:
            ids = ids[1:-1]
        elif ids.numel() == 1:
            ids = ids[0:0]
        return ids

    def _get_embed(self, ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        if ids.numel() == 0:
            return torch.empty(0, self.text_width, device=device)
        with torch.no_grad():
            emb = self._frozen_token_embedding(ids.unsqueeze(0).to(device))[0]
        return emb

    # ------------------------------------------------------------------
    # Dynamic embedding construction
    # ------------------------------------------------------------------

    def _build_dynamic_embeds(
        self, class_query_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Project each query token to CLIP token-embedding space.

        Args:
            class_query_tokens: [B, C, Q, D_sam]

        Returns:
            dynamic_embeds: [B, C, Q, text_width]
        """
        x = self.token_to_clip(class_query_tokens)
        x = self.dynamic_norm(x)
        x = x * self.dynamic_scale
        return x

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _assemble_one_prompt(
        self,
        class_name: str,
        dynamic_embed: torch.Tensor,  # [text_width]
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build prompt for one class × one query token.

        Layout:
            [SOS] prefix dynamic_token class_name suffix [EOS] [PAD...]
        """
        name_str = (
            self._normalize_class_name(class_name)
            if self.normalize_label
            else class_name
        )
        name_ids = self._tokenize_and_strip(name_str)

        sos_ids = self._tokenize_single("")[0:1]
        eos_ids = self._tokenize_single("")
        eos_ids = eos_ids[eos_ids != 0]
        if eos_ids.numel() >= 2:
            eos_ids = eos_ids[-1:]

        sos_emb = self._get_embed(sos_ids, device)
        prefix_emb = self._get_embed(self._prefix_ids, device)
        suffix_emb = self._get_embed(self._suffix_ids, device)
        name_emb = self._get_embed(name_ids, device)
        eos_emb = self._get_embed(eos_ids, device)

        all_embeds = torch.cat(
            [
                sos_emb,
                prefix_emb,
                dynamic_embed.unsqueeze(0),
                name_emb,
                suffix_emb,
                eos_emb,
            ],
            dim=0,
        )

        all_tokens = torch.cat(
            [
                sos_ids.to(device),
                self._prefix_ids.to(device),
                torch.zeros(1, dtype=torch.long, device=device),
                name_ids.to(device),
                self._suffix_ids.to(device),
                eos_ids.to(device),
            ],
            dim=0,
        )

        actual_len = all_embeds.shape[0]
        L = self._context_length

        if actual_len > L:
            raise ValueError(
                f"Assembled prompt length ({actual_len}) exceeds "
                f"context_length ({L}) for class={class_name!r}."
            )

        if actual_len < L:
            pad_emb = torch.zeros(L - actual_len, self.text_width, device=device)
            pad_tok = torch.zeros(L - actual_len, dtype=torch.long, device=device)
            all_embeds = torch.cat([all_embeds, pad_emb], dim=0)
            all_tokens = torch.cat([all_tokens, pad_tok], dim=0)

        return all_embeds, all_tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        class_query_tokens: torch.Tensor,
        class_names: List[str],
    ) -> torch.Tensor:
        """
        Args:
            class_query_tokens: [B, C, Q, D_sam]
            class_names: list of C strings

        Returns:
            dynamic_clip_text: [B, C, Q, D_clip]
        """
        B, C, Q, _ = class_query_tokens.shape
        device = class_query_tokens.device

        if len(class_names) != C:
            raise ValueError(
                f"class_names length ({len(class_names)}) != C ({C})"
            )

        dynamic_embeds = self._build_dynamic_embeds(class_query_tokens)

        L = self._context_length
        all_embeds_list = []
        all_tokens_list = []

        for b in range(B):
            for c in range(C):
                name = class_names[c]
                for q in range(Q):
                    dyn = dynamic_embeds[b, c, q]
                    emb, tok = self._assemble_one_prompt(name, dyn, device)
                    all_embeds_list.append(emb)
                    all_tokens_list.append(tok)

        merged_embeds = torch.stack(all_embeds_list, dim=0)  # [B*C*Q, L, text_width]
        merged_tokens = torch.stack(all_tokens_list, dim=0)  # [B*C*Q, L]

        pooled = self.clip_text_encoder.encode_embeds(
            input_embeds=merged_embeds,
            tokenized=merged_tokens,
            normalize=True,
            detach_output=False,
        )

        return pooled.reshape(B, C, Q, self.clip_output_dim).contiguous()
