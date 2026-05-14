from __future__ import annotations

import math
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_group_norm_groups(num_channels: int, preferred_groups: int = 8) -> int:
    for groups in (preferred_groups, 4, 2, 1):
        if groups <= num_channels and num_channels % groups == 0:
            return groups
    return 1


class SemanticScoreEmbedding(nn.Module):
    """
    Convert semantic logits into a small-channel spatial score embedding.

    Input:
        semantic_logits: [B, C, H, W]

    Output:
        score_embed: [B, C, score_dim, H, W]
    """

    def __init__(
        self,
        score_dim: int = 32,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        self.score_dim = int(score_dim)
        if self.score_dim <= 0:
            raise ValueError(f"score_dim must be positive, got {score_dim}.")

        groups = _choose_group_norm_groups(
            num_channels=self.score_dim,
            preferred_groups=int(norm_groups),
        )

        self.net = nn.Sequential(
            nn.Conv2d(
                1,
                self.score_dim,
                kernel_size=5,
                padding=2,
                padding_mode="replicate",
            ),
            nn.GroupNorm(groups, self.score_dim),
            nn.GELU(),
            nn.Conv2d(
                self.score_dim,
                self.score_dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GroupNorm(groups, self.score_dim),
            nn.GELU(),
        )

    def forward(self, semantic_logits: torch.Tensor) -> torch.Tensor:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape

        x = semantic_logits.reshape(batch_size * num_classes, 1, height, width)
        x = self.net(x)

        return x.reshape(
            batch_size,
            num_classes,
            self.score_dim,
            height,
            width,
        ).contiguous()


class ClassPixelAttentionFusionLayer(nn.Module):
    """
    One class-token final-mixer fusion layer.

    Inputs:
        score_embed:         [B, C, score_dim, H, W]
        class_tokens:        [B, C, Q, sam_dim]
        shared_clip_feature: [B, N_clip, sam_dim]
        clip_grid_hw:        (Hc, Wc), and Hc * Wc == N_clip

    Outputs:
        updated_score_embed:         [B, C, score_dim, H, W]
        updated_class_tokens:        [B, C, Q, sam_dim]
        updated_shared_clip_feature: [B, N_clip, sam_dim]
        class_clip_feature:          [B, C, class_dim, H, W]
    """

    def __init__(
        self,
        sam_dim: int = 256,
        score_dim: int = 32,
        class_dim: int = 128,
        attn_dim: int = 160,
        num_heads: int = 8,
        dropout: float = 0.1,
        norm_groups: int = 8,
        class_token_self_attn_mode: Literal["axial"] = "axial",
        use_ffn: bool = True,
    ) -> None:
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.score_dim = int(score_dim)
        self.class_dim = int(class_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.use_ffn = bool(use_ffn)
        self.class_token_self_attn_mode = str(class_token_self_attn_mode)

        if self.class_token_self_attn_mode != "axial":
            raise ValueError(
                "Only class_token_self_attn_mode='axial' is supported, "
                f"got {class_token_self_attn_mode!r}."
            )

        if self.sam_dim <= 0:
            raise ValueError(f"sam_dim must be positive, got {sam_dim}.")
        if self.score_dim <= 0:
            raise ValueError(f"score_dim must be positive, got {score_dim}.")
        if self.class_dim <= 0:
            raise ValueError(f"class_dim must be positive, got {class_dim}.")
        if self.attn_dim <= 0:
            raise ValueError(f"attn_dim must be positive, got {attn_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.sam_dim % self.num_heads != 0:
            raise ValueError(
                "sam_dim must be divisible by num_heads, "
                f"got sam_dim={self.sam_dim}, num_heads={self.num_heads}."
            )
        if self.class_dim % self.num_heads != 0:
            raise ValueError(
                "class_dim must be divisible by num_heads for presence pooling, "
                f"got class_dim={self.class_dim}, num_heads={self.num_heads}."
            )
        if self.attn_dim % self.num_heads != 0:
            raise ValueError(
                "attn_dim must be divisible by num_heads, "
                f"got attn_dim={self.attn_dim}, num_heads={self.num_heads}."
            )
        if self.score_dim % self.num_heads != 0:
            raise ValueError(
                "score_dim must be divisible by num_heads for value heads, "
                f"got score_dim={self.score_dim}, num_heads={self.num_heads}."
            )

        self.class_token_intra_attn = nn.MultiheadAttention(
            embed_dim=self.sam_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.class_token_intra_norm = nn.LayerNorm(self.sam_dim)

        self.class_token_inter_attn = nn.MultiheadAttention(
            embed_dim=self.sam_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.class_token_inter_norm = nn.LayerNorm(self.sam_dim)

        self.clip_to_class_attn = nn.MultiheadAttention(
            embed_dim=self.sam_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.clip_to_class_norm = nn.LayerNorm(self.sam_dim)

        self.class_clip_proj = nn.Linear(self.sam_dim, self.class_dim)
        self.shared_update_proj = nn.Linear(self.class_dim, self.sam_dim)
        self.shared_update_norm = nn.LayerNorm(self.sam_dim)

        pixel_in_dim = self.score_dim + self.class_dim
        self.q_proj = nn.Linear(pixel_in_dim, self.attn_dim)
        self.k_proj = nn.Linear(pixel_in_dim, self.attn_dim)
        self.v_proj = nn.Linear(self.score_dim, self.score_dim)
        self.out_proj = nn.Linear(self.score_dim, self.score_dim)

        self.dropout = nn.Dropout(float(dropout))

        groups = _choose_group_norm_groups(
            num_channels=self.score_dim,
            preferred_groups=int(norm_groups),
        )

        self.score_attn_norm = nn.GroupNorm(groups, self.score_dim)
        self.score_ffn_norm = nn.GroupNorm(groups, self.score_dim)

        if self.use_ffn:
            self.score_ffn = nn.Sequential(
                nn.Conv2d(self.score_dim, self.score_dim * 4, kernel_size=1),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Conv2d(self.score_dim * 4, self.score_dim, kernel_size=1),
            )
        else:
            self.score_ffn = None

    def _apply_score_norm(
        self,
        score_embed: torch.Tensor,
        norm: nn.GroupNorm,
    ) -> torch.Tensor:
        batch_size, num_classes, score_dim, height, width = score_embed.shape

        x = score_embed.reshape(
            batch_size * num_classes,
            score_dim,
            height,
            width,
        )
        x = norm(x)

        return x.reshape(
            batch_size,
            num_classes,
            score_dim,
            height,
            width,
        ).contiguous()

    def _run_axial_class_token_self_attention(
        self,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape

        token_state = class_tokens.reshape(
            batch_size * num_classes,
            num_tokens,
            dim,
        )
        token_delta, _ = self.class_token_intra_attn(
            query=token_state,
            key=token_state,
            value=token_state,
            need_weights=False,
        )
        token_state = self.class_token_intra_norm(
            token_state + self.dropout(token_delta)
        )
        class_tokens = token_state.reshape(
            batch_size,
            num_classes,
            num_tokens,
            dim,
        ).contiguous()

        class_summary = class_tokens.mean(dim=2)
        class_delta, _ = self.class_token_inter_attn(
            query=class_summary,
            key=class_summary,
            value=class_summary,
            need_weights=False,
        )
        class_summary = self.class_token_inter_norm(
            class_summary + self.dropout(class_delta)
        )

        class_tokens = class_tokens + class_summary[:, :, None, :]
        return class_tokens.contiguous()

    def _build_class_clip_feature(
        self,
        class_tokens: torch.Tensor,
        shared_clip_feature: torch.Tensor,
        clip_grid_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_classes, num_tokens, dim = class_tokens.shape
        _, num_clip_tokens, _ = shared_clip_feature.shape

        clip_h, clip_w = int(clip_grid_hw[0]), int(clip_grid_hw[1])
        if clip_h <= 0 or clip_w <= 0:
            raise ValueError(f"clip_grid_hw must be positive, got {clip_grid_hw}.")
        if clip_h * clip_w != int(num_clip_tokens):
            raise ValueError(
                "clip_grid_hw does not match shared_clip_feature token count: "
                f"clip_grid_hw={clip_grid_hw}, product={clip_h * clip_w}, "
                f"N_clip={num_clip_tokens}."
            )

        clip_query = shared_clip_feature[:, None].expand(
            batch_size,
            num_classes,
            num_clip_tokens,
            dim,
        )
        clip_query = clip_query.reshape(
            batch_size * num_classes,
            num_clip_tokens,
            dim,
        ).contiguous()

        class_kv = class_tokens.reshape(
            batch_size * num_classes,
            num_tokens,
            dim,
        ).contiguous()

        attn_out, _ = self.clip_to_class_attn(
            query=clip_query,
            key=class_kv,
            value=class_kv,
            need_weights=False,
        )
        clip_class_state = self.clip_to_class_norm(
            clip_query + self.dropout(attn_out)
        )

        class_clip_tokens = self.class_clip_proj(clip_class_state)
        class_clip_tokens = class_clip_tokens.reshape(
            batch_size,
            num_classes,
            num_clip_tokens,
            self.class_dim,
        ).contiguous()

        shared_delta = class_clip_tokens.mean(dim=1)
        shared_delta = self.shared_update_proj(shared_delta)
        shared_clip_feature = self.shared_update_norm(
            shared_clip_feature + self.dropout(shared_delta)
        )

        class_clip_grid = class_clip_tokens.reshape(
            batch_size,
            num_classes,
            clip_h,
            clip_w,
            self.class_dim,
        )
        class_clip_grid = class_clip_grid.permute(0, 1, 4, 2, 3).contiguous()
        class_clip_grid = class_clip_grid.reshape(
            batch_size * num_classes,
            self.class_dim,
            clip_h,
            clip_w,
        )

        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        class_clip_grid = F.interpolate(
            class_clip_grid,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        class_clip_feature = class_clip_grid.reshape(
            batch_size,
            num_classes,
            self.class_dim,
            target_h,
            target_w,
        ).contiguous()

        return class_clip_feature, shared_clip_feature

    def _run_class_pixel_attention(
        self,
        score_embed: torch.Tensor,
        class_clip_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_classes, score_dim, height, width = score_embed.shape

        pixel_state = torch.cat([score_embed, class_clip_feature], dim=2)

        pixel_state = pixel_state.permute(0, 3, 4, 1, 2).contiguous()
        pixel_state = pixel_state.reshape(
            batch_size * height * width,
            num_classes,
            self.score_dim + self.class_dim,
        )

        score_state = score_embed.permute(0, 3, 4, 1, 2).contiguous()
        score_state = score_state.reshape(
            batch_size * height * width,
            num_classes,
            score_dim,
        )

        q = self.q_proj(pixel_state)
        k = self.k_proj(pixel_state)
        v = self.v_proj(score_state)

        q_head_dim = self.attn_dim // self.num_heads
        v_head_dim = self.score_dim // self.num_heads

        q = q.reshape(
            batch_size * height * width,
            num_classes,
            self.num_heads,
            q_head_dim,
        ).permute(0, 2, 1, 3)

        k = k.reshape(
            batch_size * height * width,
            num_classes,
            self.num_heads,
            q_head_dim,
        ).permute(0, 2, 1, 3)

        v = v.reshape(
            batch_size * height * width,
            num_classes,
            self.num_heads,
            v_head_dim,
        ).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(float(q_head_dim))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(
            batch_size * height * width,
            num_classes,
            self.score_dim,
        )

        out = self.out_proj(out)

        out = out.reshape(
            batch_size,
            height,
            width,
            num_classes,
            self.score_dim,
        )
        out = out.permute(0, 3, 4, 1, 2).contiguous()

        return out

    def forward(
        self,
        score_embed: torch.Tensor,
        class_tokens: torch.Tensor,
        shared_clip_feature: torch.Tensor,
        clip_grid_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if score_embed.dim() != 5:
            raise ValueError(
                "score_embed must be [B, C, score_dim, H, W], "
                f"got {tuple(score_embed.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, sam_dim], "
                f"got {tuple(class_tokens.shape)}."
            )
        if shared_clip_feature.dim() != 3:
            raise ValueError(
                "shared_clip_feature must be [B, N_clip, sam_dim], "
                f"got {tuple(shared_clip_feature.shape)}."
            )

        batch_size, num_classes, score_dim, height, width = score_embed.shape

        if int(score_dim) != self.score_dim:
            raise ValueError(
                f"score_embed channel mismatch: expected {self.score_dim}, "
                f"got {score_dim}."
            )
        if tuple(class_tokens.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens batch/class shape mismatch: "
                f"{tuple(class_tokens.shape[:2])} vs "
                f"{(batch_size, num_classes)}."
            )
        if int(class_tokens.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, "
                f"got {class_tokens.shape[-1]}."
            )
        if int(shared_clip_feature.shape[0]) != batch_size:
            raise ValueError(
                "shared_clip_feature batch shape mismatch: "
                f"{shared_clip_feature.shape[0]} vs {batch_size}."
            )
        if int(shared_clip_feature.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"shared_clip_feature dim mismatch: expected {self.sam_dim}, "
                f"got {shared_clip_feature.shape[-1]}."
            )

        class_tokens = self._run_axial_class_token_self_attention(class_tokens)

        class_clip_feature, shared_clip_feature = self._build_class_clip_feature(
            class_tokens=class_tokens,
            shared_clip_feature=shared_clip_feature,
            clip_grid_hw=clip_grid_hw,
            target_hw=(height, width),
        )

        score_delta = self._run_class_pixel_attention(
            score_embed=score_embed,
            class_clip_feature=class_clip_feature,
        )
        score_embed = score_embed + self.dropout(score_delta)
        score_embed = self._apply_score_norm(
            score_embed=score_embed,
            norm=self.score_attn_norm,
        )

        if self.score_ffn is not None:
            ffn_in = self._apply_score_norm(
                score_embed=score_embed,
                norm=self.score_ffn_norm,
            )
            ffn_in = ffn_in.reshape(
                batch_size * num_classes,
                score_dim,
                height,
                width,
            )
            ffn_out = self.score_ffn(ffn_in)
            ffn_out = ffn_out.reshape(
                batch_size,
                num_classes,
                score_dim,
                height,
                width,
            )
            score_embed = score_embed + self.dropout(ffn_out)

        return (
            score_embed.contiguous(),
            class_tokens.contiguous(),
            shared_clip_feature.contiguous(),
            class_clip_feature.contiguous(),
        )


class ClassTokenSemanticFinalMixer(nn.Module):
    """
    Class-token final mixer.

    Inputs:
        semantic_logits:     [B, C, H, W]
        class_tokens:        [B, C, Q, sam_dim]
        shared_clip_feature: [B, N_clip, sam_dim]
        clip_grid_hw:        (Hc, Wc), and Hc * Wc == N_clip

    Outputs:
        {
            "final_logits":    [B, C, H, W],
            "presence_logits": [B, C],
            "presence_score":  [B, C],
        }

    Final formula:
        final_logits = semantic_logits + presence_score * delta_logits
    """

    def __init__(
        self,
        sam_dim: int = 256,
        score_dim: int = 32,
        class_dim: int = 128,
        attn_dim: int = 160,
        num_heads: int = 8,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        class_token_self_attn_mode: Literal["axial"] = "axial",
        presence_enabled: bool = True,
    ) -> None:
        super().__init__()

        self.sam_dim = int(sam_dim)
        self.score_dim = int(score_dim)
        self.class_dim = int(class_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.fusion_layers = int(fusion_layers)
        self.class_token_self_attn_mode = str(class_token_self_attn_mode)
        self.presence_enabled = bool(presence_enabled)

        if self.fusion_layers <= 0:
            raise ValueError(
                f"fusion_layers must be positive, got {fusion_layers}."
            )
        if self.class_dim % self.num_heads != 0:
            raise ValueError(
                "class_dim must be divisible by num_heads, "
                f"got class_dim={self.class_dim}, num_heads={self.num_heads}."
            )

        self.score_embedding = SemanticScoreEmbedding(score_dim=self.score_dim)

        self.layers = nn.ModuleList(
            [
                ClassPixelAttentionFusionLayer(
                    sam_dim=self.sam_dim,
                    score_dim=self.score_dim,
                    class_dim=self.class_dim,
                    attn_dim=self.attn_dim,
                    num_heads=self.num_heads,
                    dropout=float(dropout),
                    class_token_self_attn_mode=self.class_token_self_attn_mode,
                )
                for _ in range(self.fusion_layers)
            ]
        )

        groups = _choose_group_norm_groups(self.score_dim, preferred_groups=8)
        self.score_head = nn.Sequential(
            nn.Conv2d(
                self.score_dim,
                self.score_dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GroupNorm(groups, self.score_dim),
            nn.GELU(),
            nn.Conv2d(
                self.score_dim,
                1,
                kernel_size=1,
                padding_mode="replicate",
            ),
        )

        self.presence_query = nn.Parameter(torch.zeros(1, 1, self.class_dim))
        nn.init.normal_(self.presence_query, std=0.02)

        self.presence_attn = nn.MultiheadAttention(
            embed_dim=self.class_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.presence_norm = nn.LayerNorm(self.class_dim)
        self.presence_head = nn.Linear(self.class_dim, 1)

    def _build_presence_logits(
        self,
        class_clip_feature: torch.Tensor,
    ) -> torch.Tensor:
        if class_clip_feature.dim() != 5:
            raise ValueError(
                "class_clip_feature must be [B, C, class_dim, H, W], "
                f"got {tuple(class_clip_feature.shape)}."
            )

        batch_size, num_classes, class_dim, height, width = class_clip_feature.shape
        if int(class_dim) != self.class_dim:
            raise ValueError(
                f"class_clip_feature dim mismatch: expected {self.class_dim}, "
                f"got {class_dim}."
            )

        spatial_tokens = class_clip_feature.permute(0, 1, 3, 4, 2).contiguous()
        spatial_tokens = spatial_tokens.reshape(
            batch_size * num_classes,
            height * width,
            class_dim,
        )

        query = self.presence_query.to(
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        query = query.expand(batch_size * num_classes, 1, class_dim)

        pooled, _ = self.presence_attn(
            query=query,
            key=spatial_tokens,
            value=spatial_tokens,
            need_weights=False,
        )
        pooled = self.presence_norm(query + pooled)
        presence_logits = self.presence_head(pooled[:, 0])
        return presence_logits.reshape(batch_size, num_classes).contiguous()

    def forward(
        self,
        semantic_logits: torch.Tensor,
        class_tokens: torch.Tensor,
        shared_clip_feature: torch.Tensor,
        clip_grid_hw: Tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )
        if class_tokens.dim() != 4:
            raise ValueError(
                "class_tokens must be [B, C, Q, sam_dim], "
                f"got {tuple(class_tokens.shape)}."
            )
        if shared_clip_feature.dim() != 3:
            raise ValueError(
                "shared_clip_feature must be [B, N_clip, sam_dim], "
                f"got {tuple(shared_clip_feature.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape

        if tuple(class_tokens.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens batch/class shape mismatch: "
                f"{tuple(class_tokens.shape[:2])} vs "
                f"{(batch_size, num_classes)}."
            )
        if int(class_tokens.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.sam_dim}, "
                f"got {class_tokens.shape[-1]}."
            )
        if int(shared_clip_feature.shape[0]) != batch_size:
            raise ValueError(
                "shared_clip_feature batch size mismatch: "
                f"{shared_clip_feature.shape[0]} vs {batch_size}."
            )
        if int(shared_clip_feature.shape[-1]) != self.sam_dim:
            raise ValueError(
                f"shared_clip_feature dim mismatch: expected {self.sam_dim}, "
                f"got {shared_clip_feature.shape[-1]}."
            )

        score_embed = self.score_embedding(semantic_logits)
        last_class_clip_feature = None

        for layer in self.layers:
            (
                score_embed,
                class_tokens,
                shared_clip_feature,
                last_class_clip_feature,
            ) = layer(
                score_embed=score_embed,
                class_tokens=class_tokens,
                shared_clip_feature=shared_clip_feature,
                clip_grid_hw=clip_grid_hw,
            )

        score_state = score_embed.reshape(
            batch_size * num_classes,
            self.score_dim,
            height,
            width,
        )
        delta_logits = self.score_head(score_state)
        delta_logits = delta_logits.reshape(
            batch_size,
            num_classes,
            height,
            width,
        )

        if last_class_clip_feature is None:
            raise RuntimeError("Final mixer did not produce class_clip_feature.")

        if self.presence_enabled:
            presence_logits = self._build_presence_logits(last_class_clip_feature)
            presence_score = presence_logits.sigmoid()
        else:
            presence_logits = semantic_logits.new_zeros(batch_size, num_classes)
            presence_score = semantic_logits.new_ones(batch_size, num_classes)

        modulated_semantic_logits = presence_score[:, :, None, None] * semantic_logits
        final_logits = modulated_semantic_logits * delta_logits

        return {
            "delta_logits": delta_logits.contiguous(),
            "modulated_semantic_logits": modulated_semantic_logits.contiguous(),
            "final_logits": final_logits.contiguous(),
            "presence_logits": presence_logits.contiguous(),
            "presence_score": presence_score.contiguous(),
        }