from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DynamicClipPromptEncoder(nn.Module):
    """
    Generate dynamic CLIP text features from class_tokens.

    Constructs prompts explicitly instead of searching for class-name spans:

        template = "an overhead view of {}."
        → prefix = "an overhead view of " , suffix = "."
        → tokenize each part, strip SOS/EOS/PAD
        → assemble: [SOS] prefix dynamic class_name suffix [EOS] [PAD...]
        → trim to context_length

    Flow:
        class_tokens [B, C, G×T, D_sam]
        → split into G groups of T tokens
        → project each token SAM→text_width:  [B, C, G, T, text_width]
        → assemble per-template prompt embeddings
        → call clip_text_encoder.encode_embeds(detach_output=False)
        → dynamic_text_features [B, C, G, output_dim]

    where:
        G = len(prompt_templates)       (e.g. 4)
        T = tokens_per_template         (e.g. 4)
        Total class_tokens = G × T      (e.g. 16)

    CLIP text encoder params are frozen, but autograd propagates gradients
    back through encode_embeds to the projected dynamic tokens.
    """

    def __init__(
        self,
        clip_text_encoder,
        prompt_templates: list[str],
        sam_dim: int = 256,
        tokens_per_template: int = 4,
        normalize_label: bool = True,
    ):
        super().__init__()

        if not prompt_templates:
            raise ValueError("prompt_templates must not be empty")

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.prompt_templates = list(prompt_templates)
        self.sam_dim = int(sam_dim)
        self.num_groups = len(self.prompt_templates)
        self.tokens_per_template = int(tokens_per_template)
        self.normalize_label = bool(normalize_label)

        if self.tokens_per_template <= 0:
            raise ValueError("tokens_per_template must be positive")

        # Split dimensions: text_width for token embeddings,
        # output_dim for the pooled / projected CLIP feature.
        self.text_width = int(clip_text_encoder.width)
        self.clip_output_dim = int(clip_text_encoder.output_dim)
        self._context_length = int(clip_text_encoder.context_length)

        self.num_class_tokens = self.num_groups * self.tokens_per_template

        # Per-token projection from SAM space to CLIP token-embedding space.
        # Shared across all groups.
        self.token_to_clip = nn.Sequential(
            nn.LayerNorm(self.sam_dim),
            nn.Linear(self.sam_dim, self.text_width),
        )

        # Optional: scale / norm to keep dynamic tokens numerically compatible
        # with frozen token embeddings (which are NOT unit-normalised).
        self.dynamic_norm = nn.LayerNorm(self.text_width)
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))

        object.__setattr__(self, "_frozen_token_embedding", clip_text_encoder.token_embedding)
        self._tokenizer = clip_text_encoder.tokenizer

        # Pre-compute tokenized template parts (no class name).
        self._template_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        for tpl in self.prompt_templates:
            if "{}" not in tpl:
                raise ValueError(
                    f"Prompt template must contain '{{}}' placeholder, got {tpl!r}"
                )
            prefix_str, suffix_str = tpl.split("{}", 1)
            prefix_ids = self._tokenize_and_strip(prefix_str)
            suffix_ids = self._tokenize_and_strip(suffix_str)
            self._template_parts.append((prefix_ids, suffix_ids))

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        name = name.strip()
        name = name.replace("_", " ").replace("-", " ")
        return " ".join(name.split())

    def _tokenize_single(self, text: str) -> torch.Tensor:
        """Return 1D long tensor of token ids (includes SOS/EOS/PAD)."""
        tokens = self._tokenizer(text, context_length=self._context_length)
        return tokens[0]

    def _tokenize_and_strip(self, text: str) -> torch.Tensor:
        """
        Tokenize text and return only the "core" tokens:
        strip SOS, EOS, and PAD.
        """
        ids = self._tokenize_single(text)
        ids = ids[ids != 0]               # remove PAD
        if ids.numel() == 0:
            return ids
        # Remove SOS (first token) and EOS (last token) if present.
        # CLIP SOS token is typically 49406; we detect by position.
        if ids.numel() >= 2:
            ids = ids[1:-1]
        elif ids.numel() == 1:
            ids = ids[0:0]  # only SOS+EOS -> empty core
        return ids

    def _get_embed(self, ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Frozen token embedding lookup."""
        if ids.numel() == 0:
            return torch.empty(0, self.text_width, device=device)
        with torch.no_grad():
            emb = self._frozen_token_embedding(ids.unsqueeze(0).to(device))[0]
        return emb

    # ------------------------------------------------------------------
    # Dynamic embedding construction
    # ------------------------------------------------------------------

    def _build_dynamic_embeds(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Split class_tokens into groups and project each token individually.

        Args:
            class_tokens: [B, C, G×T, D_sam]

        Returns:
            dynamic_embeds: [B, C, G, T, text_width]
        """
        B, C, total_tokens, _ = class_tokens.shape
        G = self.num_groups
        T = self.tokens_per_template

        if total_tokens != G * T:
            raise ValueError(
                f"class_tokens count {total_tokens} != G×T = {G}×{T}"
            )

        # [B, C, G×T, D_sam] → [B, C, G, T, D_sam]
        x = class_tokens.reshape(B, C, G, T, self.sam_dim)

        # Project → [B, C, G, T, text_width]
        x = self.token_to_clip(x)

        # Light normalisation for numerical stability, with learnable scale.
        # Do NOT use F.normalize — CLIP token embeddings are not unit vectors.
        x = self.dynamic_norm(x)
        x = x * self.dynamic_scale

        return x

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _assemble_one_prompt(
        self,
        class_name: str,
        group_idx: int,
        dynamic_embeds_bc: torch.Tensor,  # [T, text_width]
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the full prompt for one class × one template × one batch element.

        Layout:
            [SOS] prefix dynamic suffix class_name [EOS] [PAD...]

        Dynamic tokens are inserted BETWEEN prefix and class_name,
        i.e. right before the class name placeholder in the template.

        Returns:
            embeds: [L, text_width]
            tokens: [L]
        """
        prefix_ids, suffix_ids = self._template_parts[group_idx]

        name_str = self._normalize_class_name(class_name) if self.normalize_label else class_name
        name_ids = self._tokenize_and_strip(name_str)

        # Frozen embeddings for static parts.
        sos_ids = self._tokenize_single("")[0:1]  # SOS token
        eos_ids = self._tokenize_single("")
        eos_ids = eos_ids[eos_ids != 0]
        if eos_ids.numel() >= 2:
            eos_ids = eos_ids[-1:]  # last token before padding is EOS

        sos_emb = self._get_embed(sos_ids, device)
        prefix_emb = self._get_embed(prefix_ids, device)
        suffix_emb = self._get_embed(suffix_ids, device)
        name_emb = self._get_embed(name_ids, device)
        eos_emb = self._get_embed(eos_ids, device)

        # Concatenate: [SOS] prefix dynamic name suffix [EOS]
        all_embeds = torch.cat([
            sos_emb,
            prefix_emb,
            dynamic_embeds_bc,
            name_emb,
            suffix_emb,
            eos_emb,
        ], dim=0)  # [actual_len, text_width]

        all_tokens = torch.cat([
            sos_ids.to(device),
            prefix_ids.to(device),
            torch.zeros(self.tokens_per_template, dtype=torch.long, device=device),
            name_ids.to(device),
            suffix_ids.to(device),
            eos_ids.to(device),
        ], dim=0)

        actual_len = all_embeds.shape[0]
        L = self._context_length

        if actual_len > L:
            raise ValueError(
                f"Assembled prompt length ({actual_len}) exceeds "
                f"context_length ({L}) for class={class_name!r}, "
                f"template group={group_idx}. "
                "Reduce tokens_per_template or use shorter class names."
            )

        # Pad to context_length
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
        class_tokens: torch.Tensor,
        class_names: List[str],
    ) -> torch.Tensor:
        """
        Args:
            class_tokens: [B, C, G×T, D_sam]
            class_names:  list of C strings

        Returns:
            dynamic_text_features: [B, C, G, clip_output_dim]
        """
        B, C, total_tokens, _ = class_tokens.shape
        device = class_tokens.device
        G = self.num_groups
        T = self.tokens_per_template

        if len(class_names) != C:
            raise ValueError(
                f"class_names length ({len(class_names)}) != C ({C})"
            )
        if total_tokens != G * T:
            raise ValueError(
                f"class_tokens count {total_tokens} != G×T = {G}×{T}"
            )

        # Build dynamic embeddings: [B, C, G, T, text_width]
        dynamic_embeds = self._build_dynamic_embeds(class_tokens)

        # Assemble prompts and run through CLIP text encoder.
        # Loop order: B → C → G so that stacked order matches (B, C, G).
        L = self._context_length
        all_embeds_list = []
        all_tokens_list = []

        for b in range(B):
            for c in range(C):
                name = class_names[c]
                for g in range(G):
                    dyn = dynamic_embeds[b, c, g]  # [T, text_width]
                    emb, tok = self._assemble_one_prompt(name, g, dyn, device)
                    all_embeds_list.append(emb)
                    all_tokens_list.append(tok)

        # [B*C*G, L, text_width] and [B*C*G, L]
        merged_embeds = torch.stack(all_embeds_list, dim=0)
        merged_tokens = torch.stack(all_tokens_list, dim=0)

        # Run through CLIP text transformer WITH gradient.
        pooled = self.clip_text_encoder.encode_embeds(
            input_embeds=merged_embeds,
            tokenized=merged_tokens,
            normalize=True,
            detach_output=False,
        )
        # pooled: [B*C*G, clip_output_dim]

        pooled = pooled.reshape(B, C, G, self.clip_output_dim)
        return pooled.contiguous()