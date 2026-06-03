from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenCLIPImageEncoder(nn.Module):
    def __init__(
        self,
        visual: nn.Module,
        default_output: str = "feat_map",
        image_encoder_mode: str = "maskclip",
        maskclip_skip_last_layers: int = 1,
    ) -> None:
        super().__init__()

        self.visual = visual
        self.default_output = str(default_output)
        self.image_encoder_mode = str(image_encoder_mode).strip().lower()
        self.maskclip_skip_last_layers = int(maskclip_skip_last_layers)

        valid_modes = {"maskclip", "full_vit_dense"}
        if self.image_encoder_mode not in valid_modes:
            raise ValueError(
                f"Unknown image_encoder_mode={image_encoder_mode!r}. "
                f"Supported modes are: {sorted(valid_modes)}"
            )

        if self.image_encoder_mode == "maskclip" and self.maskclip_skip_last_layers <= 0:
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

    def get_native_image_size(self) -> tuple[int, int]:
        image_size = getattr(self.visual, "image_size", None)
        if image_size is not None:
            return self._to_2tuple(image_size)

        input_resolution = getattr(self.visual, "input_resolution", None)
        if input_resolution is not None:
            return self._to_2tuple(input_resolution)

        grid_h, grid_w = self._get_base_grid_size()
        patch_h, patch_w = self.get_patch_size()
        return grid_h * patch_h, grid_w * patch_w

    def get_patch_size(self) -> tuple[int, int]:
        patch_size = getattr(self.visual, "patch_size", None)
        if patch_size is not None:
            return self._to_2tuple(patch_size)

        conv1 = getattr(self.visual, "conv1", None)
        if conv1 is None:
            raise AttributeError("Cannot infer OpenCLIP patch size.")
        return self._to_2tuple(conv1.kernel_size)

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

    def _prepare_vit_tokens(
            self,
            images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if not self._is_openclip_vit_like():
            raise NotImplementedError(
                "Dense OpenCLIP image output expects an OpenCLIP ViT-like visual tower."
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

        base_h, base_w = self._get_base_grid_size()
        if (grid_h, grid_w) != (base_h, base_w):
            raise ValueError(
                "OpenCLIP dense image encoder now requires native grid size. "
                f"Got {(grid_h, grid_w)}, expected {(base_h, base_w)}. "
                "Resize input images to clip_image_encoder.get_native_image_size() before calling."
            )

        pos_embed = self.visual.positional_embedding.to(device=x.device, dtype=x.dtype)
        x = x + pos_embed.unsqueeze(0)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        return x, (int(grid_h), int(grid_w))

    def _forward_maskclip_dense_tokens(
            self,
            images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        blocks = self._get_resblocks()
        block_index = len(blocks) - self.maskclip_skip_last_layers

        if block_index < 0:
            raise ValueError(
                "maskclip_skip_last_layers is larger than the number of transformer blocks: "
                f"skip={self.maskclip_skip_last_layers}, num_blocks={len(blocks)}"
            )

        x, (grid_h, grid_w) = self._prepare_vit_tokens(images)

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

    def _forward_full_vit_dense_tokens(
            self,
            images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        blocks = self._get_resblocks()
        x, (grid_h, grid_w) = self._prepare_vit_tokens(images)

        for block in blocks:
            x = self._call_resblock(block, x)

        patch_tokens = x[:, 1:].contiguous()

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
            if self.image_encoder_mode == "maskclip":
                dense_tokens, (grid_h, grid_w) = self._forward_maskclip_dense_tokens(images)
            elif self.image_encoder_mode == "full_vit_dense":
                dense_tokens, (grid_h, grid_w) = self._forward_full_vit_dense_tokens(images)
            else:
                raise RuntimeError(
                    f"Unexpected image_encoder_mode={self.image_encoder_mode!r}"
                )

        feat_map = dense_tokens.reshape(
            images.shape[0],
            grid_h,
            grid_w,
            self.output_dim,
        ).permute(0, 3, 1, 2).contiguous()

        return feat_map

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)