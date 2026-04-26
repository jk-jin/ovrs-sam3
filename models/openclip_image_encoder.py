from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenCLIPImageEncoder(nn.Module):
    """
    MaskCLIP-style OpenCLIP ViT dense image encoder.

    输出:
        feat_map: [B, D_align, Hc, Wc]

    符号说明:
        B 表示 batch size。
        D_align 表示 CLIP 图文对齐空间维度。
        Hc, Wc 表示 CLIP patch 网格高和宽。

    实现逻辑对齐官方 MaskCLIP ViT:
        1. 正常执行到最后一个 transformer block 之前；
        2. 在最后一个 block 中只走 V 分支；
        3. 对 V 做 attn.out_proj；
        4. 做 attention residual；
        5. 做最后 block 的 FFN；
        6. 做 visual.ln_post；
        7. 做 visual.proj；
        8. reshape 成 dense feature map。
    """

    def __init__(
        self,
        visual: nn.Module,
        default_output: str = "feat_map",
        maskclip_skip_last_layers: int = 1,
    ) -> None:
        super().__init__()

        self.visual = visual
        self.default_output = str(default_output)
        self.maskclip_skip_last_layers = int(maskclip_skip_last_layers)

        if self.maskclip_skip_last_layers <= 0:
            raise ValueError(
                f"maskclip_skip_last_layers must be positive, got {self.maskclip_skip_last_layers}"
            )

        self.native_dim = self._infer_native_feature_dim(visual)
        self.output_dim = self._infer_projected_feature_dim(visual)
        self.channel_list = [self.output_dim]

        self.visual.eval()
        for param in self.visual.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _infer_native_feature_dim(visual: nn.Module) -> int:
        candidates = [
            getattr(visual, "width", None),
            getattr(getattr(visual, "transformer", None), "width", None),
            getattr(visual, "num_features", None),
            getattr(visual, "embed_dim", None),
        ]

        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value

        raise AttributeError(
            "Cannot infer OpenCLIP visual native feature dimension."
        )

    @staticmethod
    def _infer_projected_feature_dim(visual: nn.Module) -> int:
        output_dim = getattr(visual, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim

        proj = getattr(visual, "proj", None)
        if proj is None:
            raise AttributeError(
                "OpenCLIP visual.proj is missing, cannot infer projected feature dimension."
            )

        if isinstance(proj, nn.Linear):
            return int(proj.out_features)

        if isinstance(proj, (torch.Tensor, nn.Parameter)):
            if proj.ndim != 2:
                raise ValueError(
                    f"Expected visual.proj as 2D matrix, got {tuple(proj.shape)}"
                )
            return int(proj.shape[1])

        raise TypeError(f"Unsupported visual.proj type: {type(proj)}")

    @staticmethod
    def _to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
        if isinstance(x, int):
            return x, x
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return int(x[0]), int(x[1])
        raise TypeError(f"Cannot convert to 2-tuple: {x!r}")

    def _is_openclip_vit_like(self) -> bool:
        required_attrs = [
            "conv1",
            "class_embedding",
            "positional_embedding",
            "patch_dropout",
            "ln_pre",
            "ln_post",
            "proj",
            "transformer",
        ]
        return all(hasattr(self.visual, name) for name in required_attrs)

    def _get_resblocks(self) -> list[nn.Module]:
        transformer = getattr(self.visual, "transformer", None)
        if transformer is None or not hasattr(transformer, "resblocks"):
            raise AttributeError(
                "OpenCLIP visual.transformer.resblocks is required for MaskCLIP-style output."
            )

        blocks = list(transformer.resblocks)
        if len(blocks) == 0:
            raise ValueError("visual.transformer.resblocks is empty.")

        return blocks

    def _get_base_grid_size(self) -> Tuple[int, int]:
        pos_embed = getattr(self.visual, "positional_embedding", None)
        if pos_embed is None:
            raise AttributeError("visual.positional_embedding is missing.")

        num_prefix_tokens = 1
        num_patch_tokens = int(pos_embed.shape[0]) - num_prefix_tokens
        if num_patch_tokens <= 0:
            raise ValueError(
                f"Invalid positional embedding shape: {tuple(pos_embed.shape)}"
            )

        grid_size = getattr(self.visual, "grid_size", None)
        if grid_size is not None:
            grid_h, grid_w = self._to_2tuple(grid_size)
            if grid_h * grid_w == num_patch_tokens:
                return grid_h, grid_w

        side = int(round(math.sqrt(num_patch_tokens)))
        if side * side != num_patch_tokens:
            raise ValueError(
                "Cannot infer a square base patch grid from positional embedding. "
                f"num_patch_tokens={num_patch_tokens}"
            )

        return side, side

    @staticmethod
    def _expand_class_token(token: torch.Tensor, batch_size: int) -> torch.Tensor:
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def _interpolate_positional_embedding(
        self,
        target_grid_hw: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            target_grid_hw: (Hc, Wc)

        Returns:
            pos_embed_resized: [1 + Hc * Wc, native_dim]
        """
        pos_embed = self.visual.positional_embedding
        if pos_embed.ndim != 2:
            raise ValueError(
                "Expected visual.positional_embedding to have shape [L, C], "
                f"but got {tuple(pos_embed.shape)}"
            )

        target_h, target_w = int(target_grid_hw[0]), int(target_grid_hw[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid target grid size: {target_grid_hw}")

        base_h, base_w = self._get_base_grid_size()
        num_prefix_tokens = 1

        cls_pos = pos_embed[:num_prefix_tokens]
        patch_pos = pos_embed[num_prefix_tokens:]
        embed_dim = int(patch_pos.shape[-1])

        if base_h == target_h and base_w == target_w:
            return pos_embed.to(device=device, dtype=dtype)

        patch_pos = patch_pos.reshape(base_h, base_w, embed_dim)
        patch_pos = patch_pos.permute(2, 0, 1).unsqueeze(0)

        patch_pos = F.interpolate(
            patch_pos,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos = patch_pos.squeeze(0).permute(1, 2, 0).reshape(
            target_h * target_w,
            embed_dim,
        )

        pos_embed_resized = torch.cat([cls_pos, patch_pos], dim=0)
        return pos_embed_resized.to(device=device, dtype=dtype)

    @staticmethod
    def _call_resblock(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = block(x)
        if isinstance(out, tuple):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(
                f"Expected transformer block to return Tensor or tuple(Tensor, ...), got {type(out)}"
            )
        return out

    @staticmethod
    def _extract_qkv(
        attn: nn.Module,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            attn: OpenCLIP attention module
            x: [B, N, C]

        Returns:
            q, k, v: each [B, N, C]
        """
        if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
            qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
            return qkv.chunk(3, dim=-1)

        if hasattr(attn, "qkv"):
            qkv = attn.qkv(x)
            return qkv.chunk(3, dim=-1)

        raise RuntimeError(
            "Unsupported OpenCLIP attention qkv structure. "
            "Expected attn.in_proj_weight or attn.qkv."
        )

    @staticmethod
    def _apply_attention_out_proj(
        attn: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]

        Returns:
            out_proj(x): [B, N, C]
        """
        if not hasattr(attn, "out_proj"):
            raise AttributeError("attention module must contain out_proj.")

        x = attn.out_proj(x)

        if hasattr(attn, "out_drop"):
            x = attn.out_drop(x)

        return x

    @staticmethod
    def _apply_first_residual(
        block: nn.Module,
        x_in: torch.Tensor,
        attn_branch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Official MaskCLIP ViT idea:
            replace the attention output by V branch,
            then keep residual structure.

        Args:
            x_in: [B, N, C]
            attn_branch: [B, N, C]

        Returns:
            x: [B, N, C]
        """
        if hasattr(block, "ls_1"):
            attn_branch = block.ls_1(attn_branch)

        return x_in + attn_branch

    @staticmethod
    def _apply_second_residual(
        block: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the block FFN/MLP after V-branch residual.

        Args:
            x: [B, N, C]

        Returns:
            x: [B, N, C]
        """
        if not hasattr(block, "ln_2") or not hasattr(block, "mlp"):
            raise AttributeError(
                "MaskCLIP-style V branch requires transformer block with ln_2 and mlp."
            )

        mlp_out = block.mlp(block.ln_2(x))

        if hasattr(block, "ls_2"):
            mlp_out = block.ls_2(mlp_out)

        return x + mlp_out

    def _apply_visual_ln_post_and_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, native_dim]

        Returns:
            projected: [B, N, output_dim]
        """
        x = self.visual.ln_post(x)

        proj = self.visual.proj
        if isinstance(proj, nn.Linear):
            return proj(x)

        proj = proj.to(device=x.device, dtype=x.dtype)
        return x @ proj

    def _forward_maskclip_dense_tokens(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            dense_tokens: [B, Hc * Wc, output_dim]
            grid_hw: (Hc, Wc)
        """
        if not self._is_openclip_vit_like():
            raise NotImplementedError(
                "MaskCLIP-style dense output expects an OpenCLIP ViT-like visual tower."
            )

        blocks = self._get_resblocks()
        block_index = len(blocks) - self.maskclip_skip_last_layers

        if block_index < 0:
            raise ValueError(
                "maskclip_skip_last_layers is larger than the number of transformer blocks: "
                f"skip={self.maskclip_skip_last_layers}, num_blocks={len(blocks)}"
            )

        x = self.visual.conv1(images)
        if x.ndim != 4:
            raise ValueError(
                f"Expected conv1 output as [B, C, Hc, Wc], got {tuple(x.shape)}"
            )

        batch_size, width, grid_h, grid_w = x.shape

        x = x.reshape(batch_size, width, grid_h * grid_w).permute(0, 2, 1)

        cls_token = self._expand_class_token(
            self.visual.class_embedding.to(dtype=x.dtype, device=x.device),
            batch_size=batch_size,
        )
        x = torch.cat([cls_token, x], dim=1)

        pos_embed = self._interpolate_positional_embedding(
            target_grid_hw=(grid_h, grid_w),
            dtype=x.dtype,
            device=x.device,
        )
        x = x + pos_embed.unsqueeze(0)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        for block in blocks[:block_index]:
            x = self._call_resblock(block, x)

        maskclip_block = blocks[block_index]

        if not hasattr(maskclip_block, "ln_1") or not hasattr(maskclip_block, "attn"):
            raise AttributeError(
                "MaskCLIP-style output requires transformer block with ln_1 and attn."
            )

        x_norm = maskclip_block.ln_1(x)

        _, _, v = self._extract_qkv(
            attn=maskclip_block.attn,
            x=x_norm,
        )

        v = self._apply_attention_out_proj(
            attn=maskclip_block.attn,
            x=v,
        )

        v = self._apply_first_residual(
            block=maskclip_block,
            x_in=x,
            attn_branch=v,
        )

        v = self._apply_second_residual(
            block=maskclip_block,
            x=v,
        )

        patch_tokens = v[:, 1:].contiguous()

        expected_num_tokens = int(grid_h) * int(grid_w)
        if patch_tokens.shape[1] != expected_num_tokens:
            raise ValueError(
                "Patch token count mismatch: "
                f"expected {expected_num_tokens}, got {patch_tokens.shape[1]}"
            )

        dense_tokens = self._apply_visual_ln_post_and_projection(patch_tokens)
        return dense_tokens, (int(grid_h), int(grid_w))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.visual.eval()

        with torch.no_grad():
            dense_tokens, (grid_h, grid_w) = self._forward_maskclip_dense_tokens(images)

        feat_map = dense_tokens.reshape(
            images.shape[0],
            grid_h,
            grid_w,
            self.output_dim,
        ).permute(0, 3, 1, 2).contiguous()

        return feat_map

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)