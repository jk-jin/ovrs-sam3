from __future__ import annotations

from typing import Dict, Optional, Iterator, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vl_combiner import SAM3VLBackbone
from .data_misc import BatchedDatapoint, FindStage
from .geometry_encoders import Prompt
from .task_modes import TASK_MODE_SEMANTIC, normalize_task_mode

class ClipSamImageFusionEncoder(nn.Module):
    """
    Fuse SAM3 image feature, CLIP image feature, and CLIP dense score map.

    Args:
        clip_image_dim:
            CLIP image feature channel after visual projection.
        hidden_dim:
            SAM3 hidden dimension. 当前项目里通常是 256。
        score_prior_dim:
            CLIP score map 分支压缩后的通道数。

    Input:
        sam3_image_feat: [B, hidden_dim, Hs, Ws]
        clip_image_feat: [B, clip_image_dim, Hc, Wc]
        clip_score_map:  [B, C, Hc, Wc]

    Output:
        fused_image_feat: [B, hidden_dim, Hs, Ws]

    符号说明:
        B 表示 batch size。
        C 表示所有类别数量。
        Hs, Ws 表示 SAM3 图像特征图高宽。
        Hc, Wc 表示 CLIP 图像特征图高宽。
        hidden_dim 表示 SAM3 token 维度，当前是 256。
    """

    def __init__(
        self,
        clip_image_dim: int,
        hidden_dim: int = 256,
        score_prior_dim: int = 64,
        init_fusion_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.clip_image_dim = int(clip_image_dim)
        self.hidden_dim = int(hidden_dim)
        self.score_prior_dim = int(score_prior_dim)

        self.sam3_branch = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=self.hidden_dim),
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=16, num_channels=self.hidden_dim),
            nn.GELU(),
        )

        self.clip_branch = nn.Sequential(
            nn.Conv2d(
                in_channels=self.clip_image_dim,
                out_channels=self.hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=16, num_channels=self.hidden_dim),
            nn.GELU(),
        )

        self.score_branch_3d = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),

            nn.Conv3d(
                in_channels=32,
                out_channels=self.score_prior_dim,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=self.score_prior_dim),
            nn.GELU(),
        )

        self.score_branch_2d = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=self.score_prior_dim),
            nn.Conv2d(
                in_channels=self.score_prior_dim,
                out_channels=self.score_prior_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=self.score_prior_dim),
            nn.GELU(),
        )

        fusion_in_channels = self.hidden_dim * 2 + self.score_prior_dim

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=fusion_in_channels,
                out_channels=self.hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=16, num_channels=self.hidden_dim),
            nn.GELU(),

            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=16, num_channels=self.hidden_dim),
            nn.GELU(),

            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=16, num_channels=self.hidden_dim),
            nn.GELU(),
        )

        self.fusion_scale = nn.Parameter(
            torch.tensor(float(init_fusion_scale), dtype=torch.float32)
        )

    @staticmethod
    def _resize_to(
        x: torch.Tensor,
        size: tuple[int, int],
        mode: str = "bilinear",
    ) -> torch.Tensor:
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        return F.interpolate(
            x,
            size=size,
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )

    def _encode_score_map(
        self,
        clip_score_map: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        if clip_score_map.ndim != 4:
            raise ValueError(
                f"Expected clip_score_map as [B, C, Hc, Wc], got {tuple(clip_score_map.shape)}"
            )

        x = clip_score_map.unsqueeze(1)     # [B, 1, C, Hc, Wc]
        x = self.score_branch_3d(x)         # [B, score_prior_dim, C, Hc, Wc]
        x = x.mean(dim=2)                   # [B, score_prior_dim, Hc, Wc]
        x = self._resize_to(x, target_hw)
        x = self.score_branch_2d(x)         # [B, score_prior_dim, Hs, Ws]
        return x

    def forward(
        self,
        sam3_image_feat: torch.Tensor,
        clip_image_feat: torch.Tensor,
        clip_score_map: torch.Tensor,
    ) -> torch.Tensor:
        if sam3_image_feat.ndim != 4:
            raise ValueError(
                f"Expected sam3_image_feat as [B, D, Hs, Ws], got {tuple(sam3_image_feat.shape)}"
            )
        if clip_image_feat.ndim != 4:
            raise ValueError(
                f"Expected clip_image_feat as [B, D_clip, Hc, Wc], got {tuple(clip_image_feat.shape)}"
            )
        if clip_score_map.ndim != 4:
            raise ValueError(
                f"Expected clip_score_map as [B, C, Hc, Wc], got {tuple(clip_score_map.shape)}"
            )

        batch_size, sam3_dim, target_h, target_w = sam3_image_feat.shape
        if sam3_dim != self.hidden_dim:
            raise ValueError(
                f"SAM3 image feature dim mismatch: expected {self.hidden_dim}, got {sam3_dim}"
            )

        if clip_image_feat.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between sam3_image_feat and clip_image_feat: "
                f"{batch_size} vs {clip_image_feat.shape[0]}"
            )

        if clip_score_map.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between sam3_image_feat and clip_score_map: "
                f"{batch_size} vs {clip_score_map.shape[0]}"
            )

        target_hw = (int(target_h), int(target_w))

        sam3_feat = self.sam3_branch(sam3_image_feat)

        clip_feat = self._resize_to(clip_image_feat, target_hw)
        clip_feat = self.clip_branch(clip_feat)

        score_feat = self._encode_score_map(
            clip_score_map=clip_score_map,
            target_hw=target_hw,
        )

        fused_input = torch.cat(
            [sam3_feat, clip_feat, score_feat],
            dim=1,
        )

        fusion_delta = self.fusion(fused_input)

        fused_image_feat = sam3_image_feat + self.fusion_scale * fusion_delta
        return fused_image_feat

class Sam3Image(torch.nn.Module):
    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,
        detach_presence_in_joint_score: bool = False,
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        clip_image_encoder=None,
        clip_text_encoder=None,
        openclip_cfg=None,
        task_mode: str = TASK_MODE_SEMANTIC,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        self.o2m_mask_predict = o2m_mask_predict
        self.dot_prod_scoring = dot_prod_scoring
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher = matcher

        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring
        self.clip_image_encoder = clip_image_encoder
        self.clip_text_encoder = clip_text_encoder

        self.register_buffer(
            "openclip_image_mean",
            torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.register_buffer(
            "openclip_image_std",
            torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.task_mode = normalize_task_mode(task_mode)
        if self.task_mode != TASK_MODE_SEMANTIC:
            raise NotImplementedError(
                "Sam3Image currently only supports semantic task mode."
            )

        self.clip_extra_token_templates = []
        self.num_clip_extra_tokens = 0
        self.normalize_label_for_clip = True
        self.clip_token_global_scale = 0

        if openclip_cfg is not None:
            self.clip_extra_token_templates = list(
                getattr(openclip_cfg, "extra_token_templates", [])
            )
            self.num_clip_extra_tokens = int(
                getattr(openclip_cfg, "num_extra_tokens", len(self.clip_extra_token_templates))
            )
            self.clip_extra_token_templates = self.clip_extra_token_templates[:self.num_clip_extra_tokens]
            self.normalize_label_for_clip = bool(
                getattr(openclip_cfg, "normalize_label_for_clip", True)
            )
            self.clip_token_global_scale = float(
                getattr(openclip_cfg, "clip_token_global_scale", 0)
            )

        self.clip_text_token_norm = nn.LayerNorm(self.hidden_dim)

        self.clip_dynamic_gate = nn.Linear(self.hidden_dim * 2, 1)
        nn.init.zeros_(self.clip_dynamic_gate.weight)
        nn.init.zeros_(self.clip_dynamic_gate.bias)

        self.presence_query_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        nn.init.xavier_uniform_(self.presence_query_proj.weight)
        nn.init.zeros_(self.presence_query_proj.bias)

        self.presence_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.presence_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        self.presence_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1),
        )

        self.clip_text_dim = None
        self.clip_text_proj = None
        if self.clip_text_encoder is not None:
            self.clip_text_dim = self._infer_clip_text_dim()
            self.clip_text_proj = nn.Linear(self.clip_text_dim, self.hidden_dim)

        self.clip_image_dim = None
        if self.clip_image_encoder is not None:
            self.clip_image_dim = self._infer_clip_image_dim()

        self.clip_align_dim = None
        if self.clip_text_dim is not None and self.clip_image_dim is not None:
            if self.clip_text_dim != self.clip_image_dim:
                raise ValueError(
                    "Projected OpenCLIP text/image dimensions must match for dense similarity. "
                    f"Got text_dim={self.clip_text_dim}, image_dim={self.clip_image_dim}."
                )
            self.clip_align_dim = self.clip_text_dim

        self.clip_sam_image_fusion = None
        if self.clip_image_encoder is not None and self.clip_text_encoder is not None:
            if self.clip_image_dim is None:
                raise RuntimeError(
                    "clip_image_dim must be initialized before clip_sam_image_fusion."
                )

            self.clip_sam_image_fusion = ClipSamImageFusionEncoder(
                clip_image_dim=self.clip_image_dim,
                hidden_dim=self.hidden_dim,
                score_prior_dim=64,
                init_fusion_scale=0.1,
            )

        self.clip_text_to_fused_image_attn = None
        self.clip_text_to_fused_image_norm = None
        self.clip_to_sam3_text_attn = None
        self.clip_to_sam3_text_norm = None

        if self.clip_text_encoder is not None and self.clip_image_encoder is not None:
            self.clip_text_to_fused_image_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.clip_text_to_fused_image_norm = nn.LayerNorm(self.hidden_dim)

            self.clip_to_sam3_text_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.clip_to_sam3_text_norm = nn.LayerNorm(self.hidden_dim)

        self.prompt_chunk_size = None

        self._text_cache: Optional[Dict[str, torch.Tensor]] = None
        self._text_cache_key: Optional[Tuple[str, ...]] = None
        self._text_cache_device: Optional[str] = None

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        self.clear_text_cache()
        return super().to(*args, **kwargs)

    @staticmethod
    def _normalize_text_cache_key(class_texts: List[str]) -> Tuple[str, ...]:
        return tuple(str(x) for x in class_texts)

    def clear_text_cache(self) -> None:
        self._text_cache = None
        self._text_cache_key = None
        self._text_cache_device = None

    def prepare_text_cache(
        self,
        class_texts: List[str],
        device: Optional[torch.device] = None,
        force: bool = False,
    ) -> None:
        if len(class_texts) == 0:
            raise ValueError("class_texts is empty, cannot build text cache.")

        device = torch.device(device) if device is not None else self.device
        cache_key = self._normalize_text_cache_key(class_texts)
        cache_device = str(device)

        if (
            not force
            and self._text_cache is not None
            and self._text_cache_key == cache_key
            and self._text_cache_device == cache_device
        ):
            return

        with torch.no_grad():
            text_backbone_out = self.backbone.forward_text(
                class_texts,
                device=device,
            )

        text_backbone_out = self._detach_tree(text_backbone_out)

        cache: Dict[str, torch.Tensor] = {
            "language_features": text_backbone_out["language_features"].contiguous(),
            "language_mask": text_backbone_out["language_mask"].contiguous(),
        }

        if "language_embeds" in text_backbone_out and text_backbone_out["language_embeds"] is not None:
            cache["language_embeds"] = text_backbone_out["language_embeds"].contiguous()

        if self.clip_text_encoder is not None and len(self.clip_extra_token_templates) > 0:
            with torch.no_grad():
                clip_text_tokens_native = self.clip_text_encoder.encode_prompt_templates(
                    class_names=class_texts,
                    templates=self.clip_extra_token_templates,
                    device=device,
                    normalize_label=self.normalize_label_for_clip,
                )  # [C, K, C_text]

            cache["clip_text_tokens_native"] = clip_text_tokens_native.detach().contiguous()

        self._text_cache = cache
        self._text_cache_key = cache_key
        self._text_cache_device = cache_device

    def ensure_text_cache(
        self,
        class_texts: List[str],
        device: Optional[torch.device] = None,
    ) -> None:
        self.prepare_text_cache(class_texts=class_texts, device=device, force=False)

    def _slice_text_cache(
        self,
        start: int,
        end: int,
    ) -> Dict[str, torch.Tensor]:
        if self._text_cache is None:
            raise RuntimeError("Text cache is not prepared.")

        out: Dict[str, torch.Tensor] = {
            "language_features": self._text_cache["language_features"][:, start:end].contiguous(),
            "language_mask": self._text_cache["language_mask"][start:end].contiguous(),
        }

        if "language_embeds" in self._text_cache:
            out["language_embeds"] = self._text_cache["language_embeds"][:, start:end].contiguous()

        if "clip_text_tokens_native" in self._text_cache:
            out["clip_text_tokens_native"] = self._text_cache["clip_text_tokens_native"][start:end].contiguous()

        return out

    def _get_prompt_chunk_size(self, num_classes: int) -> int:
        chunk_size = getattr(self, "prompt_chunk_size", None)
        if chunk_size is None:
            return num_classes
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            return num_classes
        return min(chunk_size, num_classes)

    def _detach_tree(self, obj: Any):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        if isinstance(obj, dict):
            return {k: self._detach_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._detach_tree(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._detach_tree(v) for v in obj)
        return obj

    def _infer_clip_text_dim(self) -> int:
        if self.clip_text_encoder is None:
            raise ValueError("clip_text_encoder is None.")

        output_dim = getattr(self.clip_text_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim

        raise AttributeError(
            "clip_text_encoder must expose a positive integer `output_dim`. "
            "This project expects OpenCLIP text features after text_projection."
        )

    def _infer_clip_image_dim(self) -> int:
        if self.clip_image_encoder is None:
            raise ValueError("clip_image_encoder is None.")

        output_dim = getattr(self.clip_image_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim

        raise AttributeError(
            "clip_image_encoder must expose a positive integer `output_dim`. "
            "This project expects OpenCLIP image features after visual.proj."
        )

    def _get_openclip_patch_size(self) -> Tuple[int, int]:
        if self.clip_image_encoder is None:
            return (1, 1)

        visual = self.clip_image_encoder.visual
        patch_size = getattr(visual, "patch_size", None)
        if isinstance(patch_size, int):
            return (patch_size, patch_size)
        if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            return (int(patch_size[0]), int(patch_size[1]))

        conv1 = getattr(visual, "conv1", None)
        if conv1 is not None:
            kernel_size = getattr(conv1, "kernel_size", None)
            if isinstance(kernel_size, int):
                return (kernel_size, kernel_size)
            if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
                return (int(kernel_size[0]), int(kernel_size[1]))

        raise AttributeError(
            "Cannot infer OpenCLIP patch size from visual.patch_size or visual.conv1.kernel_size."
        )

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 1:
            return int(value)
        return ((int(value) + multiple - 1) // multiple) * multiple

    @staticmethod
    def _pad_chw_image(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected raw image as [C, H, W], got {tuple(x.shape)}")
        h, w = int(x.shape[-2]), int(x.shape[-1])
        pad_h = max(0, out_h - h)
        pad_w = max(0, out_w - w)
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    def _prepare_openclip_image_batch(
        self,
        raw_images: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if len(raw_images) == 0:
            raise ValueError("raw_images is empty.")

        processed = []
        for i, x in enumerate(raw_images):
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Each raw image must be a torch.Tensor, got index={i}, type={type(x)}"
                )
            if x.ndim != 3:
                raise ValueError(
                    f"Each raw image must have shape [C, H, W], got index={i}, shape={tuple(x.shape)}"
                )
            if x.shape[0] != 3:
                raise ValueError(
                    f"Each raw image must have 3 channels, got index={i}, shape={tuple(x.shape)}"
                )
            x = x.to(device=device, dtype=torch.float32)
            processed.append(x)

        patch_h, patch_w = self._get_openclip_patch_size()
        max_h = max(int(x.shape[-2]) for x in processed)
        max_w = max(int(x.shape[-1]) for x in processed)

        max_h = self._round_up_to_multiple(max_h, patch_h)
        max_w = self._round_up_to_multiple(max_w, patch_w)

        batch = torch.stack(
            [self._pad_chw_image(x, max_h, max_w) for x in processed],
            dim=0,
        )

        batch = (batch - self.openclip_image_mean) / self.openclip_image_std
        return batch

    def _build_clip_image_cache(
            self,
            input: BatchedDatapoint,
            device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.clip_image_encoder is None:
            return None

        if input.raw_images is None:
            raise ValueError(
                "clip_image_encoder is enabled, but BatchedDatapoint.raw_images is None."
            )

        clip_img_batch = self._prepare_openclip_image_batch(
            raw_images=input.raw_images,
            device=device,
        )

        with torch.no_grad():
            clip_feat_map_native = self.clip_image_encoder(clip_img_batch)

        if not isinstance(clip_feat_map_native, torch.Tensor):
            raise TypeError(
                "clip_image_encoder must return a torch.Tensor in this stage."
            )
        if clip_feat_map_native.ndim != 4:
            raise ValueError(
                "Expected clip image feature map as [B, C, H, W], "
                f"got {tuple(clip_feat_map_native.shape)}"
            )

        clip_feat_map_native = clip_feat_map_native.detach().contiguous()

        grid_h = int(clip_feat_map_native.shape[-2])
        grid_w = int(clip_feat_map_native.shape[-1])

        return {
            "clip_image_feat_map_native": clip_feat_map_native,
            "clip_image_grid_hw": (grid_h, grid_w),
        }

    @staticmethod
    def _average_clip_template_tokens(
            clip_text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clip_text_tokens: [C, K, D]

        Returns:
            class_text_tokens: [C, D]

        符号说明：
            C 表示类别数。
            K 表示每个类别的 prompt 模板数量。
            D 表示 OpenCLIP 图文对齐空间维度。
        """
        if clip_text_tokens.ndim != 3:
            raise ValueError(
                f"Expected clip_text_tokens as [C, K, D], got {tuple(clip_text_tokens.shape)}"
            )

        clip_text_tokens = F.normalize(clip_text_tokens, dim=-1)
        class_text_tokens = clip_text_tokens.mean(dim=1)
        class_text_tokens = F.normalize(class_text_tokens, dim=-1)
        return class_text_tokens

    def _build_clip_dense_score_map(
            self,
            clip_image_cache: Optional[Dict[str, torch.Tensor]],
            clip_text_tokens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Args:
            clip_image_cache:
                must contain 'clip_image_feat_map_native', shape [B, D, Hc, Wc]
            clip_text_tokens:
                projected OpenCLIP text tokens, shape [C, K, D]

        Returns:
            clip_score_map: [B, C, Hc, Wc]

        符号说明：
            B 表示 batch size。
            C 表示所有类别数。
            K 表示每类模板数。
            D 表示 OpenCLIP 图文对齐空间维度。
            Hc, Wc 表示 OpenCLIP patch feature map 的高和宽。
        """
        if clip_image_cache is None or clip_text_tokens is None:
            return None

        if "clip_image_feat_map_native" not in clip_image_cache:
            raise KeyError("clip_image_cache must contain 'clip_image_feat_map_native'.")

        clip_image_feat_map = clip_image_cache["clip_image_feat_map_native"]
        if clip_image_feat_map.ndim != 4:
            raise ValueError(
                "Expected clip_image_feat_map_native as [B, D, Hc, Wc], "
                f"got {tuple(clip_image_feat_map.shape)}"
            )

        if clip_text_tokens.ndim != 3:
            raise ValueError(
                f"Expected clip_text_tokens as [C, K, D], got {tuple(clip_text_tokens.shape)}"
            )

        image_dim = int(clip_image_feat_map.shape[1])
        text_dim = int(clip_text_tokens.shape[-1])
        if image_dim != text_dim:
            raise ValueError(
                "Projected OpenCLIP image/text dimensions must match before dense similarity. "
                f"Got image_dim={image_dim}, text_dim={text_dim}."
            )

        clip_image_feat_map = F.normalize(clip_image_feat_map, dim=1)
        class_text_tokens = self._average_clip_template_tokens(
            clip_text_tokens=clip_text_tokens.to(
                device=clip_image_feat_map.device,
                dtype=clip_image_feat_map.dtype,
            )
        )

        clip_score_map = torch.einsum(
            "bdhw,cd->bchw",
            clip_image_feat_map,
            class_text_tokens,
        )

        return clip_score_map.contiguous()

    def _expand_clip_text_tokens_to_pairs(
        self,
        clip_text_tokens: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clip_text_tokens: [C, K, D]
        Returns:
            pair_tokens: [K, B*C, D]
            pair_mask:   [B*C, K]
        """
        if clip_text_tokens.ndim != 3:
            raise ValueError(
                f"Expected clip_text_tokens as [C, K, D], got {tuple(clip_text_tokens.shape)}"
            )

        num_classes, num_tokens, dim = clip_text_tokens.shape
        x = clip_text_tokens.unsqueeze(0).expand(batch_size, num_classes, num_tokens, dim)
        x = x.reshape(batch_size * num_classes, num_tokens, dim).contiguous()
        x = x.transpose(0, 1).contiguous()

        mask = torch.zeros(
            (batch_size * num_classes, num_tokens),
            dtype=torch.bool,
            device=device,
        )
        return x, mask

    def _expand_sam3_text_to_pairs(
        self,
        sam3_text_feats: torch.Tensor,
        sam3_text_mask: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sam3_text_feats: [L, C, D]
            sam3_text_mask:  [C, L]
        Returns:
            pair_feats: [B*C, L, D]
            pair_mask:  [B*C, L]
        """
        if sam3_text_feats.ndim != 3:
            raise ValueError(
                f"Expected sam3_text_feats as [L, C, D], got {tuple(sam3_text_feats.shape)}"
            )
        if sam3_text_mask.ndim != 2:
            raise ValueError(
                f"Expected sam3_text_mask as [C, L], got {tuple(sam3_text_mask.shape)}"
            )

        seq_len, num_classes, dim = sam3_text_feats.shape
        if sam3_text_mask.shape != (num_classes, seq_len):
            raise ValueError(
                f"sam3_text_mask shape mismatch: expected {(num_classes, seq_len)}, "
                f"got {tuple(sam3_text_mask.shape)}"
            )

        feats = sam3_text_feats.permute(1, 0, 2).contiguous()  # [C, L, D]
        feats = feats.unsqueeze(0).expand(batch_size, num_classes, seq_len, dim)
        feats = feats.reshape(batch_size * num_classes, seq_len, dim).contiguous()  # [B*C, L, D]

        mask = sam3_text_mask.unsqueeze(0).expand(batch_size, num_classes, seq_len)
        mask = mask.reshape(batch_size * num_classes, seq_len).contiguous()  # [B*C, L]

        return feats, mask

    def _get_sam3_image_feature_map(
            self,
            backbone_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract the SAM3 image feature map used as the spatial reference for fusion.

        Args:
            backbone_out:
                must contain "backbone_fpn"

        Returns:
            sam3_image_feat: [B, hidden_dim, Hs, Ws]

        符号说明:
            B 表示 batch size。
            hidden_dim 表示 SAM3 图像特征通道数，当前通常是 256。
            Hs, Ws 表示 SAM3 图像特征图高宽。
        """
        if "backbone_fpn" not in backbone_out:
            raise KeyError("backbone_out must contain 'backbone_fpn'.")

        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        if len(vis_feats) != 1:
            raise ValueError(
                f"Current implementation expects exactly one feature level, got {len(vis_feats)}"
            )

        sam3_image_feat = vis_feats[0]
        if sam3_image_feat.ndim != 4:
            raise ValueError(
                "Expected SAM3 image feature as [B, D, Hs, Ws], "
                f"got {tuple(sam3_image_feat.shape)}"
            )

        if sam3_image_feat.shape[1] != self.hidden_dim:
            raise ValueError(
                "SAM3 image feature channel mismatch: "
                f"expected {self.hidden_dim}, got {sam3_image_feat.shape[1]}"
            )

        return sam3_image_feat

    def _build_fused_image_tokens(
            self,
            backbone_out: Dict[str, torch.Tensor],
            clip_image_cache: Optional[Dict[str, torch.Tensor]],
            clip_score_map: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build fused image tokens from SAM3 image feature, CLIP image feature, and CLIP score map.

        Args:
            backbone_out:
                SAM3 image backbone output.
            clip_image_cache:
                must contain "clip_image_feat_map_native", shape [B, D_clip, Hc, Wc]
            clip_score_map:
                CLIP dense score map, shape [B, C, Hc, Wc]

        Returns:
            fused_image_tokens: [B, N, hidden_dim]
            fused_image_mask:   [B, N]

        符号说明:
            B 表示 batch size。
            C 表示类别数。
            D_clip 表示 CLIP 图像特征通道数。
            Hc, Wc 表示 CLIP 图像特征图高宽。
            N = Hs * Ws，表示 SAM3 图像特征图展平后的 token 数。
            hidden_dim 表示 SAM3 token 维度，当前通常是 256。
        """
        if self.clip_sam_image_fusion is None:
            raise RuntimeError("clip_sam_image_fusion is not initialized.")

        if clip_image_cache is None:
            raise ValueError("clip_image_cache is None, cannot build fused image tokens.")

        if clip_score_map is None:
            raise ValueError("clip_score_map is None, cannot build fused image tokens.")

        if "clip_image_feat_map_native" not in clip_image_cache:
            raise KeyError("clip_image_cache must contain 'clip_image_feat_map_native'.")

        sam3_image_feat = self._get_sam3_image_feature_map(backbone_out)
        clip_image_feat = clip_image_cache["clip_image_feat_map_native"]

        if clip_image_feat.ndim != 4:
            raise ValueError(
                "Expected clip_image_feat_map_native as [B, D_clip, Hc, Wc], "
                f"got {tuple(clip_image_feat.shape)}"
            )

        if clip_score_map.ndim != 4:
            raise ValueError(
                f"Expected clip_score_map as [B, C, Hc, Wc], got {tuple(clip_score_map.shape)}"
            )

        batch_size = int(sam3_image_feat.shape[0])

        if clip_image_feat.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between SAM3 image feature and CLIP image feature: "
                f"{batch_size} vs {clip_image_feat.shape[0]}"
            )

        if clip_score_map.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between SAM3 image feature and CLIP score map: "
                f"{batch_size} vs {clip_score_map.shape[0]}"
            )

        fused_image_feat = self.clip_sam_image_fusion(
            sam3_image_feat=sam3_image_feat,
            clip_image_feat=clip_image_feat,
            clip_score_map=clip_score_map,
        )  # [B, hidden_dim, Hs, Ws]

        if fused_image_feat.ndim != 4:
            raise ValueError(
                f"Expected fused_image_feat as [B, hidden_dim, Hs, Ws], got {tuple(fused_image_feat.shape)}"
            )

        if fused_image_feat.shape[1] != self.hidden_dim:
            raise ValueError(
                "Fused image feature channel mismatch: "
                f"expected {self.hidden_dim}, got {fused_image_feat.shape[1]}"
            )

        fused_image_tokens = fused_image_feat.flatten(2).transpose(1, 2).contiguous()
        # [B, N, hidden_dim]

        fused_image_mask = torch.zeros(
            (batch_size, fused_image_tokens.shape[1]),
            dtype=torch.bool,
            device=fused_image_tokens.device,
        )
        # [B, N]

        return fused_image_tokens, fused_image_mask

    def _fuse_clip_text_tokens_with_fused_image(
            self,
            clip_text_tokens: torch.Tensor,
            fused_image_tokens: torch.Tensor,
            fused_image_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Let CLIP text tokens attend to fused image tokens.

        Args:
            clip_text_tokens:
                [C, K, D]
            fused_image_tokens:
                [B, N, D]
            fused_image_mask:
                [B, N]

        Returns:
            pair_tokens:
                [K, B * C, D]
            pair_mask:
                [B * C, K]

        符号说明:
            B 表示 batch size。
            C 表示当前 chunk 的类别数。
            K 表示每个类别的 CLIP prompt token 数量。
            N 表示融合图像 token 数量，也就是 Hs * Ws。
            D 表示 token 维度，当前是 256。
        """
        if self.clip_text_to_fused_image_attn is None:
            raise RuntimeError("clip_text_to_fused_image_attn is not initialized.")
        if self.clip_text_to_fused_image_norm is None:
            raise RuntimeError("clip_text_to_fused_image_norm is not initialized.")

        if clip_text_tokens.ndim != 3:
            raise ValueError(
                f"Expected clip_text_tokens as [C, K, D], got {tuple(clip_text_tokens.shape)}"
            )
        if fused_image_tokens.ndim != 3:
            raise ValueError(
                f"Expected fused_image_tokens as [B, N, D], got {tuple(fused_image_tokens.shape)}"
            )
        if fused_image_mask.ndim != 2:
            raise ValueError(
                f"Expected fused_image_mask as [B, N], got {tuple(fused_image_mask.shape)}"
            )

        num_classes, num_tokens, dim = clip_text_tokens.shape
        batch_size, num_image_tokens, image_dim = fused_image_tokens.shape

        if image_dim != dim:
            raise ValueError(
                "Dimension mismatch between clip_text_tokens and fused_image_tokens: "
                f"{dim} vs {image_dim}"
            )

        if fused_image_mask.shape != (batch_size, num_image_tokens):
            raise ValueError(
                "fused_image_mask shape mismatch: "
                f"expected {(batch_size, num_image_tokens)}, got {tuple(fused_image_mask.shape)}"
            )

        query = clip_text_tokens.unsqueeze(0).expand(
            batch_size,
            num_classes,
            num_tokens,
            dim,
        )
        query = query.reshape(
            batch_size * num_classes,
            num_tokens,
            dim,
        ).contiguous()
        # [B * C, K, D]

        key_value = fused_image_tokens.unsqueeze(1).expand(
            batch_size,
            num_classes,
            num_image_tokens,
            dim,
        )
        key_value = key_value.reshape(
            batch_size * num_classes,
            num_image_tokens,
            dim,
        ).contiguous()
        # [B * C, N, D]

        key_padding_mask = fused_image_mask.unsqueeze(1).expand(
            batch_size,
            num_classes,
            num_image_tokens,
        )
        key_padding_mask = key_padding_mask.reshape(
            batch_size * num_classes,
            num_image_tokens,
        ).contiguous()
        # [B * C, N]

        attn_out, _ = self.clip_text_to_fused_image_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        fused = self.clip_text_to_fused_image_norm(query + attn_out)
        # [B * C, K, D]

        pair_tokens = fused.transpose(0, 1).contiguous()
        # [K, B * C, D]

        pair_mask = torch.zeros(
            (batch_size * num_classes, num_tokens),
            dtype=torch.bool,
            device=fused.device,
        )
        # [B * C, K]

        return pair_tokens, pair_mask

    def _align_clip_pair_tokens_with_sam3_text(
        self,
        clip_pair_tokens: torch.Tensor,
        clip_pair_mask: torch.Tensor,
        sam3_pair_text_feats: torch.Tensor,
        sam3_pair_text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.clip_to_sam3_text_attn is None:
            raise RuntimeError("clip_to_sam3_text_attn is not initialized.")
        if self.clip_to_sam3_text_norm is None:
            raise RuntimeError("clip_to_sam3_text_norm is not initialized.")

        if clip_pair_tokens.ndim != 3:
            raise ValueError(
                f"Expected clip_pair_tokens as [K, B*C, D], got {tuple(clip_pair_tokens.shape)}"
            )
        if clip_pair_mask.ndim != 2:
            raise ValueError(
                f"Expected clip_pair_mask as [B*C, K], got {tuple(clip_pair_mask.shape)}"
            )
        if sam3_pair_text_feats.ndim != 3:
            raise ValueError(
                f"Expected sam3_pair_text_feats as [B*C, L, D], got {tuple(sam3_pair_text_feats.shape)}"
            )
        if sam3_pair_text_mask.ndim != 2:
            raise ValueError(
                f"Expected sam3_pair_text_mask as [B*C, L], got {tuple(sam3_pair_text_mask.shape)}"
            )

        num_tokens, num_pairs, dim = clip_pair_tokens.shape
        if clip_pair_mask.shape != (num_pairs, num_tokens):
            raise ValueError(
                f"clip_pair_mask shape mismatch: expected {(num_pairs, num_tokens)}, "
                f"got {tuple(clip_pair_mask.shape)}"
            )
        if sam3_pair_text_feats.shape[0] != num_pairs:
            raise ValueError(
                f"sam3_pair_text_feats first dim mismatch: expected {num_pairs}, "
                f"got {sam3_pair_text_feats.shape[0]}"
            )
        if sam3_pair_text_feats.shape[2] != dim:
            raise ValueError(
                f"Feature dim mismatch: clip_pair_tokens dim={dim}, "
                f"sam3_pair_text_feats dim={sam3_pair_text_feats.shape[2]}"
            )
        if sam3_pair_text_mask.shape[0] != num_pairs:
            raise ValueError(
                f"sam3_pair_text_mask first dim mismatch: expected {num_pairs}, "
                f"got {sam3_pair_text_mask.shape[0]}"
            )

        query = clip_pair_tokens.transpose(0, 1).contiguous()  # [B*C, K, D]

        attn_out, _ = self.clip_to_sam3_text_attn(
            query=query,
            key=sam3_pair_text_feats,
            value=sam3_pair_text_feats,
            key_padding_mask=sam3_pair_text_mask,
            need_weights=False,
        )

        aligned = self.clip_to_sam3_text_norm(query + attn_out)  # [B*C, K, D]
        aligned = aligned.transpose(0, 1).contiguous()  # [K, B*C, D]

        return aligned, clip_pair_mask

    @staticmethod
    def _masked_mean_pool(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T], True 表示 padding / invalid
        Returns:
            pooled: [B, D]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x as [B, T, D], got {tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask as [B, T], got {tuple(mask.shape)}")
        if x.shape[:2] != mask.shape:
            raise ValueError(
                f"Shape mismatch between x and mask: x.shape[:2]={x.shape[:2]}, "
                f"mask.shape={mask.shape}"
            )

        valid = (~mask).to(dtype=x.dtype).unsqueeze(-1)  # [B, T, 1]
        denom = valid.sum(dim=1).clamp_min(1.0)          # [B, 1]
        pooled = (x * valid).sum(dim=1) / denom
        return pooled

    def _build_pair_summaries(
        self,
        clip_pair_tokens: torch.Tensor,
        clip_pair_mask: torch.Tensor,
        sam3_pair_text_feats: torch.Tensor,
        sam3_pair_text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clip_pair_tokens: [K, B*C, D]
            clip_pair_mask: [B*C, K]
            sam3_pair_text_feats: [B*C, L, D]
            sam3_pair_text_mask: [B*C, L]
        Returns:
            clip_summary: [B*C, D]
            sam3_summary: [B*C, D]
        """
        if clip_pair_tokens.ndim != 3:
            raise ValueError(
                f"Expected clip_pair_tokens as [K, B*C, D], got {tuple(clip_pair_tokens.shape)}"
            )
        if clip_pair_mask.ndim != 2:
            raise ValueError(
                f"Expected clip_pair_mask as [B*C, K], got {tuple(clip_pair_mask.shape)}"
            )
        if sam3_pair_text_feats.ndim != 3:
            raise ValueError(
                f"Expected sam3_pair_text_feats as [B*C, L, D], got {tuple(sam3_pair_text_feats.shape)}"
            )
        if sam3_pair_text_mask.ndim != 2:
            raise ValueError(
                f"Expected sam3_pair_text_mask as [B*C, L], got {tuple(sam3_pair_text_mask.shape)}"
            )

        num_clip_tokens, num_pairs, dim = clip_pair_tokens.shape
        if clip_pair_mask.shape != (num_pairs, num_clip_tokens):
            raise ValueError(
                f"clip_pair_mask shape mismatch: expected {(num_pairs, num_clip_tokens)}, "
                f"got {tuple(clip_pair_mask.shape)}"
            )
        if sam3_pair_text_feats.shape[0] != num_pairs:
            raise ValueError(
                f"sam3_pair_text_feats first dim mismatch: expected {num_pairs}, "
                f"got {sam3_pair_text_feats.shape[0]}"
            )
        if sam3_pair_text_feats.shape[2] != dim:
            raise ValueError(
                f"Feature dim mismatch: clip_pair_tokens dim={dim}, "
                f"sam3_pair_text_feats dim={sam3_pair_text_feats.shape[2]}"
            )
        if sam3_pair_text_mask.shape[0] != num_pairs:
            raise ValueError(
                f"sam3_pair_text_mask first dim mismatch: expected {num_pairs}, "
                f"got {sam3_pair_text_mask.shape[0]}"
            )

        clip_feats = clip_pair_tokens.transpose(0, 1).contiguous()  # [B*C, K, D]
        clip_summary = self._masked_mean_pool(clip_feats, clip_pair_mask)  # [B*C, D]
        sam3_summary = self._masked_mean_pool(sam3_pair_text_feats, sam3_pair_text_mask)  # [B*C, D]

        return clip_summary, sam3_summary

    def _build_presence_query(
            self,
            clip_summary: torch.Tensor,
            sam3_summary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clip_summary: [B*C, D]
            sam3_summary: [B*C, D]
        Returns:
            presence_query: [B*C, D]
        """
        if clip_summary.ndim != 2:
            raise ValueError(
                f"Expected clip_summary as [B*C, D], got {tuple(clip_summary.shape)}"
            )
        if sam3_summary.ndim != 2:
            raise ValueError(
                f"Expected sam3_summary as [B*C, D], got {tuple(sam3_summary.shape)}"
            )
        if clip_summary.shape != sam3_summary.shape:
            raise ValueError(
                f"Shape mismatch between clip_summary and sam3_summary: "
                f"{tuple(clip_summary.shape)} vs {tuple(sam3_summary.shape)}"
            )

        clip_summary_detached = clip_summary.detach()
        sam3_summary_detached = sam3_summary.detach()

        pair_summary = torch.cat(
            [clip_summary_detached, sam3_summary_detached],
            dim=-1,
        )  # [B*C, 2D]
        presence_query = self.presence_query_proj(pair_summary)  # [B*C, D]
        return presence_query

    def _prepare_encoder_tokens_for_presence(
        self,
        encoder_hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        num_pairs: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            encoder_hidden_states:
                either [N, B*C, D] or [B*C, N, D]
            padding_mask:
                either None or [B*C, N]
            num_pairs:
                B*C
        Returns:
            encoder_tokens: [B*C, N, D]
            encoder_padding_mask: None or [B*C, N]
        """
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "Expected encoder_hidden_states to have 3 dims, "
                f"got {tuple(encoder_hidden_states.shape)}"
            )

        if encoder_hidden_states.shape[0] == num_pairs:
            encoder_tokens = encoder_hidden_states.contiguous()
        elif encoder_hidden_states.shape[1] == num_pairs:
            encoder_tokens = encoder_hidden_states.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "Cannot infer encoder token layout for presence branch: "
                f"encoder_hidden_states.shape={tuple(encoder_hidden_states.shape)}, "
                f"num_pairs={num_pairs}"
            )

        encoder_padding_mask = None
        if padding_mask is not None:
            if padding_mask.ndim != 2:
                raise ValueError(
                    f"Expected padding_mask as [B*C, N], got {tuple(padding_mask.shape)}"
                )
            if padding_mask.shape[0] != num_pairs:
                raise ValueError(
                    f"padding_mask first dim mismatch: expected {num_pairs}, "
                    f"got {padding_mask.shape[0]}"
                )
            if padding_mask.shape[1] != encoder_tokens.shape[1]:
                raise ValueError(
                    f"padding_mask second dim mismatch: expected {encoder_tokens.shape[1]}, "
                    f"got {padding_mask.shape[1]}"
                )
            encoder_padding_mask = padding_mask.contiguous()

        return encoder_tokens, encoder_padding_mask

    def _run_presence_head(
        self,
        presence_query: torch.Tensor,
        encoder_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            presence_query: [B*C, D]
            encoder_out:
                must contain encoder_hidden_states, may contain padding_mask
        Returns:
            presence_logits: [B*C]
        """
        if presence_query.ndim != 2:
            raise ValueError(
                f"Expected presence_query as [B*C, D], got {tuple(presence_query.shape)}"
            )

        num_pairs, dim = presence_query.shape
        if dim != self.hidden_dim:
            raise ValueError(
                f"presence_query dim mismatch: expected {self.hidden_dim}, got {dim}"
            )

        encoder_hidden_states = encoder_out["encoder_hidden_states"]
        padding_mask = encoder_out.get("padding_mask", None)

        encoder_tokens, encoder_padding_mask = self._prepare_encoder_tokens_for_presence(
            encoder_hidden_states=encoder_hidden_states,
            padding_mask=padding_mask,
            num_pairs=num_pairs,
        )  # [B*C, N, D]

        query = presence_query.unsqueeze(1)  # [B*C, 1, D]

        attn_out, _ = self.presence_cross_attn(
            query=query,
            key=encoder_tokens,
            value=encoder_tokens,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )

        presence_context = self.presence_cross_attn_norm(query + attn_out).squeeze(1)  # [B*C, D]
        presence_logits = self.presence_head(presence_context).squeeze(-1)  # [B*C]

        return presence_logits

    def _apply_clip_dynamic_gate(
            self,
            clip_pair_tokens: torch.Tensor,
            clip_pair_mask: torch.Tensor,
            sam3_pair_text_feats: torch.Tensor,
            sam3_pair_text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clip_pair_tokens: [K, B*C, D]
            clip_pair_mask: [B*C, K]
            sam3_pair_text_feats: [B*C, L, D]
            sam3_pair_text_mask: [B*C, L]
        Returns:
            gated_clip_pair_tokens: [K, B*C, D]
        """
        if clip_pair_tokens.ndim != 3:
            raise ValueError(
                f"Expected clip_pair_tokens as [K, B*C, D], got {tuple(clip_pair_tokens.shape)}"
            )
        if clip_pair_mask.ndim != 2:
            raise ValueError(
                f"Expected clip_pair_mask as [B*C, K], got {tuple(clip_pair_mask.shape)}"
            )
        if sam3_pair_text_feats.ndim != 3:
            raise ValueError(
                f"Expected sam3_pair_text_feats as [B*C, L, D], got {tuple(sam3_pair_text_feats.shape)}"
            )
        if sam3_pair_text_mask.ndim != 2:
            raise ValueError(
                f"Expected sam3_pair_text_mask as [B*C, L], got {tuple(sam3_pair_text_mask.shape)}"
            )

        num_clip_tokens, num_pairs, dim = clip_pair_tokens.shape
        if clip_pair_mask.shape != (num_pairs, num_clip_tokens):
            raise ValueError(
                f"clip_pair_mask shape mismatch: expected {(num_pairs, num_clip_tokens)}, "
                f"got {tuple(clip_pair_mask.shape)}"
            )
        if sam3_pair_text_feats.shape[0] != num_pairs:
            raise ValueError(
                f"sam3_pair_text_feats first dim mismatch: expected {num_pairs}, "
                f"got {sam3_pair_text_feats.shape[0]}"
            )
        if sam3_pair_text_feats.shape[2] != dim:
            raise ValueError(
                f"Feature dim mismatch: clip_pair_tokens dim={dim}, "
                f"sam3_pair_text_feats dim={sam3_pair_text_feats.shape[2]}"
            )
        if sam3_pair_text_mask.shape[0] != num_pairs:
            raise ValueError(
                f"sam3_pair_text_mask first dim mismatch: expected {num_pairs}, "
                f"got {sam3_pair_text_mask.shape[0]}"
            )

        clip_summary, sam3_summary = self._build_pair_summaries(
            clip_pair_tokens=clip_pair_tokens,
            clip_pair_mask=clip_pair_mask,
            sam3_pair_text_feats=sam3_pair_text_feats,
            sam3_pair_text_mask=sam3_pair_text_mask,
        )

        gate_input = torch.cat([clip_summary, sam3_summary], dim=-1)  # [B*C, 2D]
        dynamic_gate = torch.sigmoid(self.clip_dynamic_gate(gate_input))  # [B*C, 1]

        clip_feats = clip_pair_tokens.transpose(0, 1).contiguous()  # [B*C, K, D]
        scale = self.clip_token_global_scale * dynamic_gate  # [B*C, 1]
        clip_feats = clip_feats * scale.unsqueeze(-1)  # [B*C, K, D]

        return clip_feats.transpose(0, 1).contiguous()  # [K, B*C, D]

    def iter_chunk_raw_outputs(
        self,
        input: BatchedDatapoint,
    ) -> Iterator[Dict[str, Any]]:
        device = self.device

        if len(input.find_inputs) != 1:
            raise ValueError(
                "Current semantic-only pipeline assumes exactly one find stage per batch."
            )

        base_find_input = input.find_inputs[0]

        class_texts = list(input.find_text_batch)
        if len(class_texts) == 0:
            raise ValueError(
                "find_text_batch is empty. It should contain the shared class vocabulary."
            )

        self.ensure_text_cache(class_texts=class_texts, device=device)

        batch_size = int(input.img_batch.shape[0])
        num_classes = len(class_texts)
        chunk_size = self._get_prompt_chunk_size(num_classes)

        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)
        image_backbone_out = self._detach_tree(image_backbone_out)

        clip_image_cache = self._build_clip_image_cache(
            input=input,
            device=device,
        )

        clip_text_tokens_for_dense_score = None
        if self._text_cache is not None and "clip_text_tokens_native" in self._text_cache:
            clip_text_tokens_for_dense_score = self._text_cache["clip_text_tokens_native"]

        clip_score_map = self._build_clip_dense_score_map(
            clip_image_cache=clip_image_cache,
            clip_text_tokens=clip_text_tokens_for_dense_score,
        )

        if clip_image_cache is not None and clip_score_map is not None:
            clip_image_cache["clip_score_map"] = clip_score_map

        for start in range(0, num_classes, chunk_size):
            end = min(start + chunk_size, num_classes)
            chunk_texts = class_texts[start:end]
            num_chunk_classes = len(chunk_texts)
            chunk_class_ids = list(range(start, end))

            chunk_text_cache = self._slice_text_cache(start=start, end=end)

            chunk_backbone_out = dict(image_backbone_out)
            chunk_backbone_out["language_features"] = chunk_text_cache["language_features"]
            chunk_backbone_out["language_mask"] = chunk_text_cache["language_mask"]
            if "language_embeds" in chunk_text_cache:
                chunk_backbone_out["language_embeds"] = chunk_text_cache["language_embeds"]

            chunk_find_input = self._build_prompt_expanded_find_stage(
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
                device=device,
                base_find_input=base_find_input,
            )

            if "clip_text_tokens_native" in chunk_text_cache:
                if self.clip_text_proj is None:
                    raise RuntimeError(
                        "clip_text_encoder is enabled, but clip_text_proj is not initialized."
                    )

                clip_text_tokens = self.clip_text_proj(
                    chunk_text_cache["clip_text_tokens_native"]
                )
                clip_text_tokens = self.clip_text_token_norm(clip_text_tokens)
                # [C_chunk, K, 256]

                fused_image_tokens, fused_image_mask = self._build_fused_image_tokens(
                    backbone_out=chunk_backbone_out,
                    clip_image_cache=clip_image_cache,
                    clip_score_map=clip_score_map,
                )
                # fused_image_tokens: [B, N, 256]
                # fused_image_mask:   [B, N]

                pair_tokens, pair_mask = self._fuse_clip_text_tokens_with_fused_image(
                    clip_text_tokens=clip_text_tokens,
                    fused_image_tokens=fused_image_tokens,
                    fused_image_mask=fused_image_mask,
                )
                # pair_tokens: [K, B * C_chunk, 256]
                # pair_mask:   [B * C_chunk, K]

                sam3_pair_feats, sam3_pair_mask = self._expand_sam3_text_to_pairs(
                    sam3_text_feats=chunk_backbone_out["language_features"],
                    sam3_text_mask=chunk_backbone_out["language_mask"],
                    batch_size=batch_size,
                )

                pair_tokens, pair_mask = self._align_clip_pair_tokens_with_sam3_text(
                    clip_pair_tokens=pair_tokens,
                    clip_pair_mask=pair_mask,
                    sam3_pair_text_feats=sam3_pair_feats,
                    sam3_pair_text_mask=sam3_pair_mask,
                )

                clip_summary, sam3_summary = self._build_pair_summaries(
                    clip_pair_tokens=pair_tokens,
                    clip_pair_mask=pair_mask,
                    sam3_pair_text_feats=sam3_pair_feats,
                    sam3_pair_text_mask=sam3_pair_mask,
                )

                presence_query = self._build_presence_query(
                    clip_summary=clip_summary,
                    sam3_summary=sam3_summary,
                )

                pair_tokens = self._apply_clip_dynamic_gate(
                    clip_pair_tokens=pair_tokens,
                    clip_pair_mask=pair_mask,
                    sam3_pair_text_feats=sam3_pair_feats,
                    sam3_pair_text_mask=sam3_pair_mask,
                )

                chunk_backbone_out["clip_language_features_pair"] = pair_tokens
                chunk_backbone_out["clip_language_mask_pair"] = pair_mask
                chunk_backbone_out["presence_query_pair"] = presence_query

            geometric_prompt = Prompt(
                box_embeddings=chunk_find_input.input_boxes,
                box_mask=chunk_find_input.input_boxes_mask,
                box_labels=chunk_find_input.input_boxes_label,
            )

            chunk_raw_outputs = self.forward_grounding_raw(
                backbone_out=chunk_backbone_out,
                find_input=chunk_find_input,
                geometric_prompt=geometric_prompt,
            )

            chunk_out = self._extract_and_reshape_chunk_outputs(
                raw_outputs=chunk_raw_outputs,
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
            )

            yield {
                "chunk_start": start,
                "chunk_end": end,
                "chunk_class_ids": chunk_class_ids,
                "chunk_class_names": chunk_texts,
                "raw_outputs": chunk_out,
            }

    @staticmethod
    def _has_nonempty_geometric_prompt(find_input: Optional[FindStage]) -> bool:
        if find_input is None:
            return False

        tensor_fields = [
            getattr(find_input, "input_boxes", None),
            getattr(find_input, "input_points", None),
        ]
        for x in tensor_fields:
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                return True
        return False

    def _build_prompt_expanded_find_stage(
        self,
        batch_size: int,
        num_chunk_classes: int,
        device: torch.device,
        base_find_input: Optional[FindStage] = None,
    ) -> FindStage:
        if self._has_nonempty_geometric_prompt(base_find_input):
            raise NotImplementedError(
                "Current stage-1 internal chunking only supports semantic-only batches "
                "without non-empty geometric prompts."
            )

        num_pairs = batch_size * num_chunk_classes

        img_ids = torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(num_chunk_classes)
        text_ids = torch.arange(num_chunk_classes, device=device, dtype=torch.long).repeat(batch_size)

        return FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=torch.zeros((0, num_pairs, 4), dtype=torch.float32, device=device),
            input_boxes_mask=torch.zeros((num_pairs, 0), dtype=torch.bool, device=device),
            input_boxes_label=torch.zeros((0, num_pairs), dtype=torch.long, device=device),
            input_points=torch.zeros((0, num_pairs, 2), dtype=torch.float32, device=device),
            input_points_mask=torch.zeros((num_pairs, 0), dtype=torch.bool, device=device),
        )

    @staticmethod
    def _reshape_prompt_first_tensor(
        x: Optional[torch.Tensor],
        batch_size: int,
        num_chunk_classes: int,
        key: str,
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None

        expected = batch_size * num_chunk_classes
        if x.shape[0] != expected:
            raise ValueError(
                f"Cannot reshape key={key}: expected first dim = {expected}, got {tuple(x.shape)}"
            )

        return x.reshape(batch_size, num_chunk_classes, *x.shape[1:])

    def _extract_and_reshape_chunk_outputs(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        batch_size: int,
        num_chunk_classes: int,
    ) -> Dict[str, torch.Tensor]:
        keep_keys = [
            "semantic_logits",
            "presence_logits",
        ]

        out = {}
        for key in keep_keys:
            if key in raw_outputs and raw_outputs[key] is not None:
                out[key] = self._reshape_prompt_first_tensor(
                    raw_outputs[key],
                    batch_size=batch_size,
                    num_chunk_classes=num_chunk_classes,
                    key=key,
                )
        return out

    @staticmethod
    def _merge_chunk_outputs(chunk_outputs: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(chunk_outputs) == 0:
            raise ValueError("chunk_outputs is empty.")

        all_keys = set()
        for chunk_out in chunk_outputs:
            all_keys.update(chunk_out.keys())

        merged = {}
        for key in all_keys:
            values = [chunk_out[key] for chunk_out in chunk_outputs if key in chunk_out]
            if len(values) == 0:
                continue
            merged[key] = torch.cat(values, dim=1)

        return merged

    def _get_img_feats(self, backbone_out, img_ids):
        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]

        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]

        return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
    ):
        txt_feats = backbone_out["language_features"][:, find_input.text_ids]
        txt_masks = backbone_out["language_mask"][find_input.text_ids]

        clip_txt_feats = None
        clip_txt_masks = None
        if "clip_language_features_pair" in backbone_out:
            clip_txt_feats = backbone_out["clip_language_features_pair"]
            clip_txt_masks = backbone_out["clip_language_mask_pair"]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )

        if encode_text:
            prompt_list = [txt_feats]
            prompt_mask_list = [txt_masks]

            if clip_txt_feats is not None:
                prompt_list.append(clip_txt_feats)
                prompt_mask_list.append(clip_txt_masks)

            prompt_list.extend([geo_feats, visual_prompt_embed])
            prompt_mask_list.extend([geo_masks, visual_prompt_mask])

            prompt = torch.cat(prompt_list, dim=0)
            prompt_mask = torch.cat(prompt_mask_list, dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)

        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        prompt_pos_embed = torch.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _run_semantic_segmentation_head(
        self,
        backbone_out,
        find_input,
        encoder_out,
        prompt,
        prompt_mask,
    ) -> Dict[str, torch.Tensor]:
        if self.segmentation_head is None:
            raise ValueError("segmentation_head is None in semantic mode.")

        seg_outputs = self.segmentation_head(
            backbone_feats=backbone_out["backbone_fpn"],
            obj_queries=torch.empty(0, device=prompt.device),
            image_ids=find_input.img_ids,
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        if "semantic_seg" not in seg_outputs or seg_outputs["semantic_seg"] is None:
            raise ValueError(
                "segmentation_head did not return 'semantic_seg' in semantic mode."
            )

        return {
            "semantic_logits": seg_outputs["semantic_seg"],
        }

    def forward_grounding_raw(
            self,
            backbone_out: Dict[str, torch.Tensor],
            find_input,
            geometric_prompt: Prompt,
    ) -> Dict[str, torch.Tensor]:
        with torch.profiler.record_function("Sam3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )

        with torch.profiler.record_function("Sam3Image._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )

        with torch.profiler.record_function("Sam3Image._run_semantic_segmentation_head"):
            out = self._run_semantic_segmentation_head(
                backbone_out=backbone_out,
                find_input=find_input,
                encoder_out=encoder_out,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )

        if "presence_query_pair" not in backbone_out:
            raise ValueError(
                "presence_query_pair is missing in backbone_out. "
                "Current presence design requires CLIP-enhanced pair features."
            )

        with torch.profiler.record_function("Sam3Image._run_presence_head"):
            presence_logits = self._run_presence_head(
                presence_query=backbone_out["presence_query_pair"],
                encoder_out=encoder_out,
            )

        out["presence_logits"] = presence_logits
        return out

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        chunk_outputs = []
        for chunk in self.iter_chunk_raw_outputs(input):
            chunk_outputs.append(chunk["raw_outputs"])

        merged_outputs = self._merge_chunk_outputs(chunk_outputs)
        return merged_outputs