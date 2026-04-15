from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenCLIPImageEncoder(nn.Module):
    """
    Pure OpenCLIP vision wrapper with positional-embedding interpolation support
    for larger input resolutions (for example, 1008x1008).

    Responsibilities:
    1. Hold a loaded OpenCLIP visual tower.
    2. Expose a trunk-like interface for downstream use.
    3. Default to returning the last low-resolution feature map in NCHW format.
    4. For OpenCLIP ViT towers, bypass the original `_embeds()` path and
       interpolate learnable positional embeddings when input resolution changes.

    Non-responsibilities:
    - No channel projection
    - No external task-specific adaptation
    - No FPN / neck logic
    """

    def __init__(
        self,
        visual: nn.Module,
        default_output: str = "feat_map",
    ) -> None:
        super().__init__()
        self.visual = visual
        self.default_output = default_output

        feature_dim = self._infer_feature_dim(visual)
        self.channel_list = [feature_dim]

    @staticmethod
    def _infer_feature_dim(visual: nn.Module) -> int:
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
            "Cannot infer OpenCLIP visual feature dimension. "
            "Please inspect the visual tower and add a new rule here."
        )

    @staticmethod
    def _to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return (int(x[0]), int(x[1]))
        raise TypeError(f"Cannot convert to 2-tuple: {x!r}")

    def _is_openclip_vit_like(self) -> bool:
        """
        Heuristically detect whether `self.visual` looks like OpenCLIP's
        VisionTransformer implementation.
        """
        required_attrs = [
            "conv1",
            "class_embedding",
            "positional_embedding",
            "patch_dropout",
            "ln_pre",
            "transformer",
        ]
        return all(hasattr(self.visual, name) for name in required_attrs)

    def _get_base_grid_size(self) -> Tuple[int, int]:
        """
        Infer the original patch grid size used by the learnable positional embedding.

        Priority:
        1. visual.grid_size if available and valid
        2. infer from positional_embedding length assuming 1 cls token
        """
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
        """
        Expand a [C] class token parameter into [B, 1, C].
        """
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def _interpolate_positional_embedding(
        self,
        target_grid_hw: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Resize OpenCLIP learnable positional embedding from its original patch grid
        to `target_grid_hw`.

        Returns:
            pos_embed_resized: [1 + target_h * target_w, C]
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

        cls_pos = pos_embed[:num_prefix_tokens]           # [1, C]
        patch_pos = pos_embed[num_prefix_tokens:]         # [base_h * base_w, C]
        embed_dim = int(patch_pos.shape[-1])

        if base_h == target_h and base_w == target_w:
            return pos_embed.to(device=device, dtype=dtype)

        patch_pos = patch_pos.reshape(base_h, base_w, embed_dim)   # [H0, W0, C]
        patch_pos = patch_pos.permute(2, 0, 1).unsqueeze(0)        # [1, C, H0, W0]

        patch_pos = F.interpolate(
            patch_pos,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos = patch_pos.squeeze(0).permute(1, 2, 0).reshape(
            target_h * target_w, embed_dim
        )  # [H1 * W1, C]

        pos_embed_resized = torch.cat([cls_pos, patch_pos], dim=0)
        return pos_embed_resized.to(device=device, dtype=dtype)

    def _forward_intermediates_vit(
        self,
        images: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        stop_early: bool = False,
        normalize_intermediates: bool = False,
        intermediates_only: bool = True,
        output_fmt: str = "NCHW",
        output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Custom forward_intermediates for OpenCLIP VisionTransformer that supports
        positional-embedding interpolation.

        This path mirrors OpenCLIP VisionTransformer.forward_intermediates():
        - patch embed
        - add class token
        - add resized positional embedding
        - patch dropout
        - ln_pre
        - transformer.forward_intermediates
        - split cls token and spatial tokens
        - optionally reshape to NCHW
        """
        if output_fmt not in ("NCHW", "NLC"):
            raise ValueError("output_fmt must be one of {'NCHW', 'NLC'}.")

        x = self.visual.conv1(images)  # [B, C, Gh, Gw]
        if x.ndim != 4:
            raise ValueError(
                f"Expected conv1 output as [B, C, H, W], got shape={tuple(x.shape)}"
            )

        batch_size, width, grid_h, grid_w = x.shape

        x = x.reshape(batch_size, width, grid_h * grid_w).permute(0, 2, 1)
        # [B, Gh * Gw, C]

        cls_token = self._expand_class_token(
            self.visual.class_embedding.to(dtype=x.dtype, device=x.device),
            batch_size=batch_size,
        )
        x = torch.cat([cls_token, x], dim=1)  # [B, 1 + Gh * Gw, C]

        pos_embed = self._interpolate_positional_embedding(
            target_grid_hw=(grid_h, grid_w),
            dtype=x.dtype,
            device=x.device,
        )
        x = x + pos_embed.unsqueeze(0)  # [B, 1 + Gh * Gw, C]

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        if not hasattr(self.visual.transformer, "forward_intermediates"):
            raise RuntimeError(
                "The loaded OpenCLIP transformer does not provide "
                "transformer.forward_intermediates()."
            )

        x, intermediates = self.visual.transformer.forward_intermediates(
            x,
            indices=indices,
            stop_early=stop_early,
        )

        if normalize_intermediates:
            if not hasattr(self.visual, "ln_post"):
                raise AttributeError(
                    "visual.ln_post is missing, but normalize_intermediates=True."
                )
            intermediates = [self.visual.ln_post(t) for t in intermediates]

        num_prefix_tokens = 1
        if num_prefix_tokens > 0:
            prefix_tokens = [t[:, :num_prefix_tokens] for t in intermediates]
            intermediates = [t[:, num_prefix_tokens:] for t in intermediates]
        else:
            prefix_tokens = None

        if output_fmt == "NCHW":
            intermediates = [
                t.reshape(batch_size, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
                for t in intermediates
            ]

        output: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            "image_intermediates": intermediates
        }

        if prefix_tokens is not None and output_extra_tokens:
            output["image_intermediates_prefix"] = prefix_tokens

        if intermediates_only:
            return output

        # Optional pooled image feature path; kept here for completeness.
        if not hasattr(self.visual, "_pool"):
            raise AttributeError(
                "visual._pool is missing, cannot produce pooled image_features."
            )

        pooled, _ = self.visual._pool(x)
        if getattr(self.visual, "proj", None) is not None:
            pooled = pooled @ self.visual.proj

        output["image_features"] = pooled
        return output

    def _extract_last_tensor(self, obj: Any) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise ValueError("Received empty list/tuple while parsing image intermediates.")
            return self._extract_last_tensor(obj[-1])

        if isinstance(obj, dict):
            preferred_keys = (
                "image_intermediates",
                "intermediates",
                "features",
                "feature_maps",
                "x",
            )
            for key in preferred_keys:
                if key in obj:
                    return self._extract_last_tensor(obj[key])
            raise KeyError(
                f"Cannot find a known feature key in forward_intermediates output: {list(obj.keys())}"
            )

        raise TypeError(f"Unsupported intermediate output type: {type(obj)}")

    def _forward_intermediates(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return the last low-resolution feature map in NCHW format.
        """
        if self._is_openclip_vit_like():
            out = self._forward_intermediates_vit(
                images=images,
                indices=[-1],
                stop_early=False,
                normalize_intermediates=False,
                intermediates_only=True,
                output_fmt="NCHW",
                output_extra_tokens=False,
            )
            feat_map = self._extract_last_tensor(out)
        else:
            if not hasattr(self.visual, "forward_intermediates"):
                raise RuntimeError(
                    "The loaded OpenCLIP visual tower does not provide forward_intermediates()."
                )

            try:
                out = self.visual.forward_intermediates(
                    images,
                    indices=[-1],
                    output_fmt="NCHW",
                    intermediates_only=True,
                )
            except TypeError:
                out = self.visual.forward_intermediates(images)

            feat_map = self._extract_last_tensor(out)

        if feat_map.ndim != 4:
            raise ValueError(
                f"Expected a 4D NCHW feature map, but got shape={tuple(feat_map.shape)}"
            )

        return feat_map

    def encode_image(
        self,
        images: torch.Tensor,
        output_mode: Optional[str] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        mode = output_mode or self.default_output

        feat_map = self._forward_intermediates(images)

        if mode == "feat_map":
            return feat_map

        if mode == "tokens":
            # [B, C, H, W] -> [B, H * W, C]
            return feat_map.flatten(2).transpose(1, 2)

        if mode == "all":
            return {
                "feat_map": feat_map,
                "tokens": feat_map.flatten(2).transpose(1, 2),
            }

        raise ValueError(
            f"Unknown output_mode={mode}. "
            "Supported modes are: feat_map, tokens, all."
        )

    def forward(self, images: torch.Tensor):
        return self.encode_image(images, output_mode=self.default_output)