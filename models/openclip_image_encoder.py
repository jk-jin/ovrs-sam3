from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


class OpenCLIPImageEncoder(nn.Module):
    """
    RemoteCLIP dense ViT image encoder with 36×36 output.

    Image input is fixed to 504×504 so that ViT-L/14 produces a 36×36 patch grid.
    Positional embeddings are bicubic-interpolated from the pretrained grid.
    The last transformer block uses a dense value-branch forward (no QK attention).

    Output:
        feat_map:      [B, D_clip, 36, 36]
        mid_features:  List[[B, D_native, 36, 36]]
    """

    def __init__(
        self,
        visual: nn.Module,
        default_output: str = "feat_map",
        intermediate_layers: Optional[Sequence[int]] = None,
        image_size: int = 504,
    ) -> None:
        super().__init__()

        self.visual = visual
        self.default_output = str(default_output)
        self.image_size = int(image_size)

        if self.default_output != "feat_map":
            raise ValueError(
                "OpenCLIPImageEncoder only supports default_output='feat_map'. "
                f"Got {self.default_output!r}."
            )

        self.native_dim = self._infer_native_feature_dim(visual)
        self.output_dim = self._infer_projected_feature_dim(visual)
        self.channel_list = [self.output_dim]

        blocks = self._get_resblocks()
        self.num_visual_blocks = len(blocks)

        if intermediate_layers is None:
            if self.num_visual_blocks >= 16:
                intermediate_layers = (7, 15)
            else:
                intermediate_layers = (3, 7)

        self.intermediate_layers = tuple(int(i) for i in intermediate_layers)
        for layer_idx in self.intermediate_layers:
            if layer_idx < 0 or layer_idx >= self.num_visual_blocks:
                raise ValueError(
                    f"Invalid CLIP intermediate layer index {layer_idx}; "
                    f"visual transformer has {self.num_visual_blocks} blocks."
                )

        self.enable_grad = False

        self.visual.eval()
        for param in self.visual.parameters():
            param.requires_grad_(False)

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Dimension inference
    # ------------------------------------------------------------------

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

    def set_enable_grad(self, enable: bool) -> None:
        self.enable_grad = bool(enable)

    def has_trainable_params(self) -> bool:
        return any(p.requires_grad for p in self.visual.parameters())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
        if isinstance(x, int):
            return x, x
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return int(x[0]), int(x[1])
        raise TypeError(f"Cannot convert to 2-tuple: {x!r}")

    def _is_openclip_vit_like(self) -> bool:
        required_attrs = [
            "conv1", "class_embedding", "positional_embedding",
            "patch_dropout", "ln_pre", "ln_post", "proj", "transformer",
        ]
        return all(hasattr(self.visual, name) for name in required_attrs)

    def _get_resblocks(self) -> list[nn.Module]:
        transformer = getattr(self.visual, "transformer", None)
        if transformer is None or not hasattr(transformer, "resblocks"):
            raise AttributeError(
                "OpenCLIP visual.transformer.resblocks is required."
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

    def get_patch_size(self) -> tuple[int, int]:
        patch_size = getattr(self.visual, "patch_size", None)
        if patch_size is not None:
            return self._to_2tuple(patch_size)
        conv1 = getattr(self.visual, "conv1", None)
        if conv1 is None:
            raise AttributeError("Cannot infer OpenCLIP patch size.")
        return self._to_2tuple(conv1.kernel_size)

    @staticmethod
    def _expand_class_token(token: torch.Tensor, batch_size: int) -> torch.Tensor:
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    @staticmethod
    def _call_resblock(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = block(x)
        if isinstance(out, tuple):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(
                f"Expected transformer block to return Tensor or tuple, got {type(out)}"
            )
        return out

    # ------------------------------------------------------------------
    # Positional embedding interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def resize_positional_embedding(
        positional_embedding: torch.Tensor,
        target_grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Bicubic-interpolate patch positional embeddings to a new grid size.

        Args:
            positional_embedding: [1 + old_h*old_w, D]
            target_grid_hw: (new_h, new_w), fixed to (36, 36) in this project.

        Returns:
            interpolated: [1 + new_h*new_w, D]
        """
        cls_pos = positional_embedding[:1]          # [1, D]
        patch_pos = positional_embedding[1:]        # [old_h*old_w, D]

        old_num = patch_pos.shape[0]
        old_side = int(old_num ** 0.5)
        if old_side * old_side != old_num:
            raise ValueError(
                "Only square CLIP positional embedding is supported. "
                f"Got {old_num} patch tokens."
            )

        D = patch_pos.shape[-1]
        target_h, target_w = target_grid_hw

        patch_pos = patch_pos.reshape(1, old_side, old_side, D)
        patch_pos = patch_pos.permute(0, 3, 1, 2)   # [1, D, old_h, old_w]

        patch_pos = F.interpolate(
            patch_pos,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos = patch_pos.squeeze(0).permute(1, 2, 0)  # [new_h, new_w, D]
        patch_pos = patch_pos.reshape(target_h * target_w, D)

        return torch.cat([cls_pos, patch_pos], dim=0)

    # ------------------------------------------------------------------
    # Dense value-branch last block
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_optional_layer_scale(module, x: torch.Tensor) -> torch.Tensor:
        """Apply optional LayerScale (ls_1 / ls_2) if present."""
        if module is None:
            return x
        if isinstance(module, nn.Identity):
            return x
        if callable(module):
            return module(x)
        if torch.is_tensor(module):
            return x * module
        return x

    @staticmethod
    def _forward_dense_last_block(
        block: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the last ViT block in dense value-branch mode.

        Instead of standard QK attention aggregation, this extracts the value
        branch of the last attention layer, adds class token information,
        and passes through the MLP.

        Handles:
          - Fused qkv (in_proj_weight) and split q/k/v weights.
          - Optional in_proj_bias (may be None).
          - Optional LayerScale (ls_1 / ls_2).

        Args:
            block: OpenCLIP ResBlock with ln_1, attn, ln_2, mlp.
            x: [B, N, D]

        Returns:
            v: [B, N, D]
        """
        # Convert to [N, B, D] for internal operations.
        x = x.permute(1, 0, 2).contiguous()  # [N, B, D]

        y = block.ln_1(x)                     # [N, B, D]
        attn = block.attn

        # Project q/k/v from the fused or split weight.
        bias = getattr(attn, "in_proj_bias", None)
        if getattr(attn, "in_proj_weight", None) is not None:
            y = F.linear(y, attn.in_proj_weight, bias)
        else:
            qkv_weight = torch.cat(
                [attn.q_proj_weight, attn.k_proj_weight, attn.v_proj_weight],
                dim=0,
            )
            y = F.linear(y, qkv_weight, bias)

        # y: [N, B, 3*D]
        if y.shape[-1] % 3 != 0:
            raise ValueError(
                f"Expected qkv projection dim divisible by 3, got {y.shape[-1]}."
            )

        N, B, three_D = y.shape
        D = three_D // 3

        y = y.reshape(N, B, 3, D)
        v = y[:, :, 2, :]                    # [N, B, D]

        # Value branch output projection.
        v = F.linear(v, attn.out_proj.weight, attn.out_proj.bias)

        # Optional LayerScale after attention.
        v = OpenCLIPImageEncoder._apply_optional_layer_scale(
            getattr(block, "ls_1", None), v
        )

        # Inject class token information into all spatial tokens.
        v = v + x[:1]                         # [N, B, D]

        # MLP residual with optional LayerScale.
        mlp_out = block.mlp(block.ln_2(v))
        mlp_out = OpenCLIPImageEncoder._apply_optional_layer_scale(
            getattr(block, "ls_2", None), mlp_out
        )
        v = v + mlp_out                       # [N, B, D]

        # Convert back to [B, N, D].
        v = v.permute(1, 0, 2).contiguous()   # [B, N, D]

        return v

    # ------------------------------------------------------------------
    # Token preparation
    # ------------------------------------------------------------------

    def _prepare_vit_tokens(
        self,
        images: torch.Tensor,
        target_grid_hw: Tuple[int, int] = (36, 36),
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepare ViT input tokens with interpolated positional embedding.

        Unlike the old code, this allows arbitrary grid sizes by interpolating
        the positional embedding to match the actual patch grid.

        Args:
            images: [B, 3, H_img, W_img]  — already resized to target size.
            target_grid_hw: expected (grid_h, grid_w).

        Returns:
            x: [B, 1 + grid_h*grid_w, D]
            (grid_h, grid_w)
        """
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
        expected_h, expected_w = target_grid_hw

        if (grid_h, grid_w) != (expected_h, expected_w):
            raise ValueError(
                f"Patch grid mismatch: expected {(expected_h, expected_w)} "
                f"for input size {self.image_size}, got {(grid_h, grid_w)}. "
                f"Input images must be {self.image_size}×{self.image_size}."
            )

        x = x.reshape(batch_size, width, grid_h * grid_w).permute(0, 2, 1)

        cls_token = self._expand_class_token(
            self.visual.class_embedding.to(dtype=x.dtype, device=x.device),
            batch_size=batch_size,
        )
        x = torch.cat([cls_token, x], dim=1)  # [B, 1 + N, D]

        # Interpolate positional embedding to target grid.
        pos_embed = self.visual.positional_embedding.to(
            device=x.device, dtype=x.dtype
        )
        pos_embed = self.resize_positional_embedding(
            pos_embed, target_grid_hw=target_grid_hw
        )
        x = x + pos_embed.unsqueeze(0)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        return x, (int(grid_h), int(grid_w))

    # ------------------------------------------------------------------
    # Full ViT forward
    # ------------------------------------------------------------------

    def _apply_visual_ln_post_and_projection(
        self, x: torch.Tensor
    ) -> torch.Tensor:
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

    @staticmethod
    def _tokens_to_feature_map(
        tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, Hc*Wc, D]
        Returns:
            feat_map: [B, D, Hc, Wc]
        """
        B, N, D = tokens.shape
        expected = int(grid_h) * int(grid_w)
        if N != expected:
            raise ValueError(
                f"Token count mismatch: expected {expected}, got {N}."
            )
        return tokens.reshape(B, int(grid_h), int(grid_w), D).permute(
            0, 3, 1, 2
        ).contiguous()

    def _forward_full_vit_dense_tokens(
        self,
        images: torch.Tensor,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int], list[torch.Tensor]]:
        """
        Full ViT forward where the last block uses dense value-branch.

        First L-1 blocks run normally (producing intermediate features).
        The last block uses _forward_dense_last_block.
        Output tokens are ln_post + proj applied, cls token removed.

        Returns:
            dense_tokens:  [B, Hc*Wc, output_dim]
            grid_hw:       (Hc, Wc) = (36, 36)
            mid_features:  List[[B, D_native, Hc, Wc]]
        """
        blocks = self._get_resblocks()
        target_grid_hw = (36, 36)
        x, (grid_h, grid_w) = self._prepare_vit_tokens(
            images, target_grid_hw=target_grid_hw
        )

        num_blocks = len(blocks)
        last_idx = num_blocks - 1

        mid_features: list[torch.Tensor] = []
        target_layers = (
            set(self.intermediate_layers) if return_intermediate else set()
        )

        for layer_idx in range(last_idx):
            block = blocks[layer_idx]
            x = self._call_resblock(block, x)

            if layer_idx in target_layers:
                patch_tokens_mid = x[:, 1:].contiguous()
                mid_features.append(
                    self._tokens_to_feature_map(
                        patch_tokens_mid,
                        grid_h=grid_h,
                        grid_w=grid_w,
                    )
                )

        # Last block: dense value-branch.
        x = self._forward_dense_last_block(blocks[last_idx], x)

        # Apply ln_post + proj to all tokens, then remove cls token.
        projected = self._apply_visual_ln_post_and_projection(x)
        patch_tokens = projected[:, 1:].contiguous()

        expected_num_tokens = int(grid_h) * int(grid_w)
        if patch_tokens.shape[1] != expected_num_tokens:
            raise ValueError(
                "Patch token count mismatch: "
                f"expected {expected_num_tokens}, got {patch_tokens.shape[1]}"
            )

        return patch_tokens, (int(grid_h), int(grid_w)), mid_features

    # ------------------------------------------------------------------
    # Public encode methods
    # ------------------------------------------------------------------

    def encode_image_with_intermediate(self, images: torch.Tensor) -> dict:
        """
        Returns:
            {
                "feat_map":          [B, D_clip, 36, 36],
                "mid_features":      List[[B, D_native, 36, 36]],
                "mid_layer_indices": tuple[int, ...],
            }
        """
        self.visual.eval()

        grad_enabled = bool(
            torch.is_grad_enabled()
            and self.enable_grad
            and self.has_trainable_params()
        )

        with torch.set_grad_enabled(grad_enabled):
            dense_tokens, (grid_h, grid_w), mid_features = (
                self._forward_full_vit_dense_tokens(
                    images,
                    return_intermediate=True,
                )
            )

        feat_map = self._tokens_to_feature_map(
            dense_tokens, grid_h=grid_h, grid_w=grid_w
        )

        mid_features_out = [x.detach().contiguous() for x in mid_features]

        return {
            "feat_map": feat_map,
            "mid_features": mid_features_out,
            "mid_layer_indices": self.intermediate_layers,
        }

    # ------------------------------------------------------------------
    # Raw image preprocessing
    # ------------------------------------------------------------------

    def preprocess_raw_images(
        self,
        raw_images: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Resize raw images to 504×504 and normalize with CLIP mean/std.

        Args:
            raw_images: list of [3, H, W] tensors.
            device: target device.

        Returns:
            images: [B, 3, 504, 504], OpenCLIP-normalized.
        """
        if len(raw_images) == 0:
            raise ValueError("raw_images is empty.")

        target_size = (self.image_size, self.image_size)

        processed = []
        for i, x in enumerate(raw_images):
            if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 3:
                raise ValueError(
                    f"raw_images[{i}] must be [3, H, W], got "
                    f"{None if not isinstance(x, torch.Tensor) else tuple(x.shape)}"
                )
            x = x.to(device=device, dtype=torch.float32).unsqueeze(0)
            x = F.interpolate(
                x, size=target_size,
                mode="bilinear", align_corners=False,
            )
            processed.append(x.squeeze(0))

        batch = torch.stack(processed, dim=0)
        return (batch - self.image_mean) / self.image_std

    def encode_raw_images(
        self,
        raw_images: List[torch.Tensor],
        device: torch.device,
    ) -> dict:
        """
        Full raw-image-to-dense-CLIP-features pipeline.

        Returns:
            {
                "feat_map":          [B, D_clip, 36, 36],
                "mid_features":      List[[B, D_native, 36, 36]],
                "mid_layer_indices": tuple[int, ...],
            }
        """
        images = self.preprocess_raw_images(raw_images=raw_images, device=device)
        return self.encode_image_with_intermediate(images)

    def forward(self, images: torch.Tensor) -> dict:
        return self.encode_image_with_intermediate(images)
