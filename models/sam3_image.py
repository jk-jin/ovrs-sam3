from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_sam_feature import GlobalClipSamFeatureBuilder, SamGuidedClipSamUpsampler
from .data_misc import BatchedDatapoint, FindStage
from .final_mixer import ClassTokenSemanticFinalMixer
from .geometry_encoders import Prompt
from .task_modes import OUTPUT_KEYS, TASK_MODE_SEMANTIC, normalize_task_mode
from .vl_combiner import SAM3VLBackbone

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
        final_mixer_cfg=None,
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

        # Kept for build/config compatibility.
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
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "openclip_image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.task_mode = normalize_task_mode(task_mode)
        if self.task_mode != TASK_MODE_SEMANTIC:
            raise NotImplementedError("Sam3Image currently only supports semantic task mode.")

        self.clip_prompt_templates: List[str] = []
        self.num_clip_prompt_templates = 0
        self.num_clip_text_latents = 32
        self.normalize_label_for_clip = True

        if openclip_cfg is not None:
            self.clip_prompt_templates = list(
                getattr(
                    openclip_cfg,
                    "prompt_templates",
                    getattr(openclip_cfg, "extra_token_templates", []),
                )
            )
            self.num_clip_prompt_templates = int(
                getattr(
                    openclip_cfg,
                    "num_prompt_templates",
                    getattr(openclip_cfg, "num_extra_tokens", len(self.clip_prompt_templates)),
                )
            )
            self.clip_prompt_templates = self.clip_prompt_templates[
                : self.num_clip_prompt_templates
            ]

            self.num_clip_text_latents = int(
                getattr(openclip_cfg, "num_clip_text_latents", 32)
            )
            if self.num_clip_text_latents <= 0:
                raise ValueError(
                    "num_clip_text_latents must be positive, "
                    f"got {self.num_clip_text_latents}."
                )

            self.normalize_label_for_clip = bool(
                getattr(openclip_cfg, "normalize_label_for_clip", True)
            )

        if (self.clip_text_encoder is None) != (self.clip_image_encoder is None):
            raise RuntimeError(
                "OpenCLIP is partially initialized: clip_text_encoder and "
                "clip_image_encoder must either both exist or both be None."
            )

        self.clip_text_dim = self._infer_clip_text_dim() if self.clip_text_encoder is not None else None
        self.clip_image_dim = self._infer_clip_image_dim() if self.clip_image_encoder is not None else None
        self.clip_align_dim = None

        if self.clip_text_dim is not None and self.clip_image_dim is not None:
            if self.clip_text_dim != self.clip_image_dim:
                raise ValueError(
                    "Projected OpenCLIP text/image dimensions must match for native CLIP attention. "
                    f"Got text_dim={self.clip_text_dim}, image_dim={self.clip_image_dim}."
                )
            self.clip_align_dim = self.clip_text_dim

        self.final_mixer_dropout = float(
            getattr(final_mixer_cfg, "dropout", 0.1)
        )
        self.final_mixer_num_heads = int(
            getattr(final_mixer_cfg, "num_heads", 8)
        )
        self.final_mixer_fusion_layers = int(
            getattr(final_mixer_cfg, "fusion_layers", 2)
        )
        clip_sam_upsample_cfg = getattr(
            final_mixer_cfg,
            "clip_sam_upsample_cfg",
            None,
        )
        if clip_sam_upsample_cfg is None:
            clip_sam_upsample_cfg = {}

        self.clip_sam_upsample_enabled = bool(
            clip_sam_upsample_cfg.get("enabled", True)
        )
        self.clip_sam_upsample_window_size = int(
            clip_sam_upsample_cfg.get("window_size", 8)
        )
        self.clip_sam_upsample_dropout = float(
            clip_sam_upsample_cfg.get("dropout", self.final_mixer_dropout)
        )
        self.clip_sam_upsample_gamma_init = float(
            clip_sam_upsample_cfg.get("gamma_init", 0.0)
        )
        self.clip_sam_upsample_gamma_max = float(
            clip_sam_upsample_cfg.get("gamma_max", 0.5)
        )

        self.global_clip_sam_feature_builder = None
        if self.clip_align_dim is not None:
            self.global_clip_sam_feature_builder = GlobalClipSamFeatureBuilder(
                clip_dim=self.clip_align_dim,
                sam_dim=self.hidden_dim,
                clip_feature_dim=self.hidden_dim,
                attn_dim=self.clip_align_dim,
                num_heads=self.final_mixer_num_heads,
                dropout=self.final_mixer_dropout,
                num_text_latents=self.num_clip_text_latents,
            )

        self.clip_sam_upsampler = None
        if self.clip_sam_upsample_enabled:
            self.clip_sam_upsampler = SamGuidedClipSamUpsampler(
                hidden_dim=self.hidden_dim,
                num_heads=self.final_mixer_num_heads,
                window_size=self.clip_sam_upsample_window_size,
                dropout=self.clip_sam_upsample_dropout,
                gamma_init=self.clip_sam_upsample_gamma_init,
                gamma_max=self.clip_sam_upsample_gamma_max,
            )
        self.num_class_tokens = int(
            getattr(final_mixer_cfg, "num_class_tokens", 32)
        )
        if self.num_class_tokens <= 0:
            raise ValueError(
                f"num_class_tokens must be positive, got {self.num_class_tokens}."
            )
        self.presence_enabled = bool(
            getattr(final_mixer_cfg, "presence_enabled", True)
        )

        self.class_token_query_embed = nn.Parameter(
            torch.zeros(1, self.num_class_tokens, self.hidden_dim)
        )
        nn.init.normal_(self.class_token_query_embed, std=0.02)

        self.class_token_text_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.class_token_text_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        self.class_token_encoder_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.class_token_encoder_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        self.final_mixer = ClassTokenSemanticFinalMixer(
            sam_dim=self.hidden_dim,
            num_heads=self.final_mixer_num_heads,
            fusion_layers=self.final_mixer_fusion_layers,
            dropout=self.final_mixer_dropout,
            presence_enabled=self.presence_enabled,
            dynamic_code_cfg=getattr(final_mixer_cfg, "dynamic_code_cfg", None),
            mask_prior_cfg=getattr(final_mixer_cfg, "mask_prior_cfg", None),
            window_attention_cfg=getattr(final_mixer_cfg, "window_attention_cfg", None),
            mask_head_cfg=getattr(final_mixer_cfg, "mask_head_cfg", None),
        )

        self.prompt_chunk_size = None
        self._text_cache: Optional[Dict[str, torch.Tensor]] = None
        self._text_cache_key: Optional[Tuple[str, ...]] = None
        self._text_cache_device: Optional[str] = None
        self._last_clip_grid_hw: Optional[Tuple[int, int]] = None

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
            text_out = self.backbone.forward_text(class_texts, device=device)
        text_out = self._detach_tree(text_out)

        cache: Dict[str, torch.Tensor] = {
            "language_features": text_out["language_features"].contiguous(),
            "language_mask": text_out["language_mask"].contiguous(),
        }
        if text_out.get("language_embeds") is not None:
            cache["language_embeds"] = text_out["language_embeds"].contiguous()

        if self.clip_text_encoder is not None and len(self.clip_prompt_templates) > 0:
            with torch.no_grad():
                clip_text_tokens = self.clip_text_encoder.encode_prompt_templates(
                    class_names=class_texts,
                    templates=self.clip_prompt_templates,
                    device=device,
                    normalize_label=self.normalize_label_for_clip,
                )
            cache["clip_text_tokens_native"] = clip_text_tokens.detach().contiguous()

        self._text_cache = cache
        self._text_cache_key = cache_key
        self._text_cache_device = cache_device

    def ensure_text_cache(self, class_texts: List[str], device: Optional[torch.device] = None) -> None:
        self.prepare_text_cache(class_texts=class_texts, device=device, force=False)

    def _slice_text_cache(self, start: int, end: int) -> Dict[str, torch.Tensor]:
        if self._text_cache is None:
            raise RuntimeError("Text cache is not prepared.")

        out = {
            "language_features": self._text_cache["language_features"][:, start:end].contiguous(),
            "language_mask": self._text_cache["language_mask"][start:end].contiguous(),
        }
        for key in ("language_embeds", "clip_text_tokens_native"):
            if key in self._text_cache:
                out[key] = self._text_cache[key][start:end].contiguous() if key == "clip_text_tokens_native" else self._text_cache[key][:, start:end].contiguous()
        return out

    def _get_prompt_chunk_size(self, num_classes: int) -> int:
        chunk_size = getattr(self, "prompt_chunk_size", None)
        if chunk_size is None or int(chunk_size) <= 0:
            return num_classes
        return min(int(chunk_size), num_classes)

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
        output_dim = getattr(self.clip_text_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim
        raise AttributeError("clip_text_encoder must expose a positive integer `output_dim`.")

    def _infer_clip_image_dim(self) -> int:
        output_dim = getattr(self.clip_image_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim
        raise AttributeError("clip_image_encoder must expose a positive integer `output_dim`.")

    def _get_openclip_patch_size(self) -> Tuple[int, int]:
        visual = self.clip_image_encoder.visual
        patch_size = getattr(visual, "patch_size", None)
        if isinstance(patch_size, int):
            return (patch_size, patch_size)
        if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            return (int(patch_size[0]), int(patch_size[1]))

        conv1 = getattr(visual, "conv1", None)
        kernel_size = getattr(conv1, "kernel_size", None) if conv1 is not None else None
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            return (int(kernel_size[0]), int(kernel_size[1]))

        raise AttributeError("Cannot infer OpenCLIP patch size.")

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        return int(value) if multiple <= 1 else ((int(value) + multiple - 1) // multiple) * multiple

    @staticmethod
    def _pad_chw_image(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        pad_h = max(0, int(out_h) - int(x.shape[-2]))
        pad_w = max(0, int(out_w) - int(x.shape[-1]))
        return x if pad_h == 0 and pad_w == 0 else F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    def _prepare_openclip_image_batch(self, raw_images: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        if len(raw_images) == 0:
            raise ValueError("raw_images is empty.")

        processed = []
        for i, x in enumerate(raw_images):
            if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 3:
                raise ValueError(
                    f"raw_images[{i}] must be a tensor with shape [3, H, W], got "
                    f"{None if not isinstance(x, torch.Tensor) else tuple(x.shape)}"
                )
            processed.append(x.to(device=device, dtype=torch.float32))

        patch_h, patch_w = self._get_openclip_patch_size()
        max_h = self._round_up_to_multiple(max(int(x.shape[-2]) for x in processed), patch_h)
        max_w = self._round_up_to_multiple(max(int(x.shape[-1]) for x in processed), patch_w)

        batch = torch.stack([self._pad_chw_image(x, max_h, max_w) for x in processed], dim=0)
        return (batch - self.openclip_image_mean) / self.openclip_image_std

    def _build_clip_image_cache(
        self,
        input: BatchedDatapoint,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.clip_image_encoder is None:
            return None
        if input.raw_images is None:
            raise ValueError("clip_image_encoder is enabled, but BatchedDatapoint.raw_images is None.")

        clip_img_batch = self._prepare_openclip_image_batch(raw_images=input.raw_images, device=device)
        with torch.no_grad():
            clip_feat_map = self.clip_image_encoder(clip_img_batch)

        if not isinstance(clip_feat_map, torch.Tensor) or clip_feat_map.ndim != 4:
            raise ValueError(
                "clip_image_encoder must return a tensor with shape [B, D_clip, Hc, Wc]."
            )

        clip_feat_map = clip_feat_map.detach().contiguous()
        return {
            "clip_image_feat_map_native": clip_feat_map,
            "clip_image_grid_hw": (int(clip_feat_map.shape[-2]), int(clip_feat_map.shape[-1])),
        }

    def build_shared_clip_feature(
        self,
        input: BatchedDatapoint,
    ) -> torch.Tensor:
        if self.global_clip_sam_feature_builder is None:
            raise RuntimeError(
                "global_clip_sam_feature_builder is not initialized. "
                "Check openclip_cfg.enabled and OpenCLIP encoder setup."
            )

        device = self.device
        class_texts = list(input.find_text_batch)
        if len(class_texts) == 0:
            raise ValueError("find_text_batch is empty.")

        self.ensure_text_cache(class_texts=class_texts, device=device)

        if self._text_cache is None:
            raise RuntimeError("Text cache is not prepared.")
        if "clip_text_tokens_native" not in self._text_cache:
            raise ValueError(
                "clip_text_tokens_native is missing. "
                "Check openclip_cfg.prompt_templates."
            )

        clip_image_cache = self._build_clip_image_cache(input=input, device=device)
        if clip_image_cache is None:
            raise ValueError(
                "clip_image_cache is None. "
                "Current semantic pipeline expects OpenCLIP image features."
            )

        self._last_clip_grid_hw = clip_image_cache["clip_image_grid_hw"]

        return self.global_clip_sam_feature_builder(
            clip_image_feat_map_native=clip_image_cache["clip_image_feat_map_native"],
            clip_text_tokens_native=self._text_cache["clip_text_tokens_native"],
            sam3_text_tokens_full=self._text_cache["language_features"],
            sam3_text_mask_full=self._text_cache["language_mask"],
        )

    def build_sam3_pixel_feature_from_backbone(
        self,
        backbone_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.segmentation_head is None:
            raise RuntimeError(
                "segmentation_head is None, cannot build SAM3 pixel feature."
            )

        pixel_decoder = getattr(self.segmentation_head, "pixel_decoder", None)
        if pixel_decoder is None:
            raise RuntimeError(
                "segmentation_head does not expose pixel_decoder, "
                "cannot build SAM3 pixel feature."
            )

        backbone_feats = backbone_out.get("backbone_fpn", None)
        if backbone_feats is None:
            raise ValueError(
                "backbone_out must contain 'backbone_fpn' to build SAM3 pixel feature."
            )
        if not isinstance(backbone_feats, (list, tuple)) or len(backbone_feats) == 0:
            raise ValueError(
                "backbone_out['backbone_fpn'] must be a non-empty list/tuple "
                f"of feature maps, got {type(backbone_feats)}."
            )

        model_device = (
            self.segmentation_head.device
            if hasattr(self.segmentation_head, "device")
            else self.device
        )

        with torch.no_grad():
            pixel_decoder_inputs = [
                feat.to(device=model_device) for feat in backbone_feats
            ]
            pixel_embed = pixel_decoder(pixel_decoder_inputs)

        if pixel_embed.dim() != 4:
            raise ValueError(
                "SAM3 pixel feature must be [B, D, H, W], "
                f"got {tuple(pixel_embed.shape)}."
            )

        if int(pixel_embed.shape[1]) != self.hidden_dim:
            raise ValueError(
                "SAM3 pixel feature channel mismatch: expected "
                f"{self.hidden_dim}, got {pixel_embed.shape[1]}."
            )

        return pixel_embed.detach().contiguous()

    def build_high_res_clip_sam_feature(
            self,
            shared_clip_feature: torch.Tensor,
            sam3_feature_high: torch.Tensor,
    ) -> torch.Tensor:
        if self.clip_sam_upsampler is None:
            raise RuntimeError(
                "clip_sam_upsampler is not initialized. "
                "Check final_mixer_cfg.clip_sam_upsample_cfg.enabled."
            )

        clip_grid_hw = getattr(self, "_last_clip_grid_hw", None)
        if clip_grid_hw is None:
            raise RuntimeError(
                "clip_grid_hw is not cached. Call build_shared_clip_feature() "
                "before build_high_res_clip_sam_feature()."
            )

        return self.clip_sam_upsampler(
            shared_clip_feature_low=shared_clip_feature,
            sam3_feature_high=sam3_feature_high.detach(),
            clip_grid_hw=clip_grid_hw,
        )

    def build_sam3_pixel_feature(
        self,
        input: BatchedDatapoint,
    ) -> torch.Tensor:
        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)

        image_backbone_out = self._detach_tree(image_backbone_out)

        return self.build_sam3_pixel_feature_from_backbone(
            backbone_out=image_backbone_out,
        )

    def _expand_sam3_text_to_pairs(
        self,
        sam3_text_feats: torch.Tensor,
        sam3_text_mask: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # sam3_text_feats: [M, C, D], sam3_text_mask: [C, M]
        seq_len, num_classes, dim = sam3_text_feats.shape
        feats = sam3_text_feats.permute(1, 0, 2).contiguous()
        feats = feats.unsqueeze(0).expand(batch_size, num_classes, seq_len, dim)
        feats = feats.reshape(batch_size * num_classes, seq_len, dim).contiguous()

        mask = sam3_text_mask.unsqueeze(0).expand(batch_size, num_classes, seq_len)
        mask = mask.reshape(batch_size * num_classes, seq_len).contiguous()
        return feats, mask

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).to(dtype=x.dtype).unsqueeze(-1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def _build_class_token_seed_from_sam3_text(
        self,
        sam3_pair_feats: torch.Tensor,
        sam3_pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sam3_pair_feats.dim() != 3:
            raise ValueError(
                "sam3_pair_feats must be [B*C_chunk, M, D_sam], "
                f"got {tuple(sam3_pair_feats.shape)}."
            )

        pair_count, _, feat_dim = sam3_pair_feats.shape
        if int(feat_dim) != self.hidden_dim:
            raise ValueError(
                f"sam3_pair_feats dim mismatch: expected {self.hidden_dim}, "
                f"got {feat_dim}."
            )

        if sam3_pair_mask is not None:
            if sam3_pair_mask.shape[:2] != sam3_pair_feats.shape[:2]:
                raise ValueError(
                    "sam3_pair_mask shape mismatch: expected "
                    f"{tuple(sam3_pair_feats.shape[:2])}, "
                    f"got {tuple(sam3_pair_mask.shape)}."
                )
            sam3_pair_mask = sam3_pair_mask.detach()

        query_embed = self.class_token_query_embed.to(
            device=sam3_pair_feats.device,
            dtype=sam3_pair_feats.dtype,
        )
        query_embed = query_embed.expand(
            pair_count,
            self.num_class_tokens,
            self.hidden_dim,
        )

        sam3_pair_feats = sam3_pair_feats.detach()

        attn_out, _ = self.class_token_text_cross_attn(
            query=query_embed,
            key=sam3_pair_feats,
            value=sam3_pair_feats,
            key_padding_mask=sam3_pair_mask,
            need_weights=False,
        )

        class_token_seed = self.class_token_text_cross_attn_norm(
            query_embed + attn_out
        )
        return class_token_seed

    @staticmethod
    def _prepare_encoder_tokens(
        encoder_hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        num_pairs: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states.shape[0] == num_pairs:
            encoder_tokens = encoder_hidden_states.contiguous()
        elif encoder_hidden_states.shape[1] == num_pairs:
            encoder_tokens = encoder_hidden_states.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "Cannot infer encoder token layout: "
                f"encoder_hidden_states.shape={tuple(encoder_hidden_states.shape)}, num_pairs={num_pairs}."
            )

        if padding_mask is not None:
            if padding_mask.shape[0] != num_pairs or padding_mask.shape[1] != encoder_tokens.shape[1]:
                raise ValueError(
                    f"padding_mask shape mismatch: expected "
                    f"({num_pairs}, {encoder_tokens.shape[1]}), got {tuple(padding_mask.shape)}."
                )
            padding_mask = padding_mask.contiguous()

        return encoder_tokens, padding_mask

    def _run_class_token_encoder_cross_attn(
        self,
        class_token_seed: torch.Tensor,
        encoder_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if class_token_seed.dim() != 3:
            raise ValueError(
                "class_token_seed must be [B*C_chunk, Q, D_sam], "
                f"got {tuple(class_token_seed.shape)}."
            )

        num_pairs = int(class_token_seed.shape[0])

        encoder_tokens, encoder_padding_mask = self._prepare_encoder_tokens(
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            padding_mask=encoder_out.get("padding_mask", None),
            num_pairs=num_pairs,
        )

        encoder_tokens = encoder_tokens.detach()
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.detach()

        attn_out, _ = self.class_token_encoder_cross_attn(
            query=class_token_seed,
            key=encoder_tokens,
            value=encoder_tokens,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )

        class_tokens = self.class_token_encoder_cross_attn_norm(
            class_token_seed + attn_out
        )
        return class_tokens

    def iter_chunk_raw_outputs(
        self,
        input: BatchedDatapoint,
        shared_clip_feature: Optional[torch.Tensor] = None,
    ) -> Iterator[Dict[str, Any]]:
        device = self.device

        if len(input.find_inputs) != 1:
            raise ValueError("Current semantic-only pipeline assumes exactly one find stage per batch.")

        base_find_input = input.find_inputs[0]
        class_texts = list(input.find_text_batch)
        if len(class_texts) == 0:
            raise ValueError("find_text_batch is empty.")

        self.ensure_text_cache(class_texts=class_texts, device=device)

        batch_size = int(input.img_batch.shape[0])
        num_classes = len(class_texts)
        chunk_size = self._get_prompt_chunk_size(num_classes)

        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)
        image_backbone_out = self._detach_tree(image_backbone_out)

        if shared_clip_feature is None:
            shared_clip_feature = self.build_shared_clip_feature(input)

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

            sam3_pair_feats, sam3_pair_mask = self._expand_sam3_text_to_pairs(
                sam3_text_feats=chunk_backbone_out["language_features"],
                sam3_text_mask=chunk_backbone_out["language_mask"],
                batch_size=batch_size,
            )

            chunk_backbone_out["class_token_seed_pair"] = (
                self._build_class_token_seed_from_sam3_text(
                    sam3_pair_feats=sam3_pair_feats,
                    sam3_pair_mask=sam3_pair_mask,
                )
            )

            geometric_prompt = Prompt(
                box_embeddings=chunk_find_input.input_boxes,
                box_mask=chunk_find_input.input_boxes_mask,
                box_labels=chunk_find_input.input_boxes_label,
            )

            raw_outputs = self.forward_grounding_raw(
                backbone_out=chunk_backbone_out,
                find_input=chunk_find_input,
                geometric_prompt=geometric_prompt,
            )

            chunk_outputs = self._extract_and_reshape_chunk_outputs(
                raw_outputs=raw_outputs,
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
            )

            semantic_logits = self._ensure_4d_logits(
                chunk_outputs[OUTPUT_KEYS.semantic_logits],
                OUTPUT_KEYS.semantic_logits,
            )

            chunk_outputs[OUTPUT_KEYS.semantic_logits] = semantic_logits

            yield {
                "chunk_start": start,
                "chunk_end": end,
                "chunk_class_ids": chunk_class_ids,
                "chunk_class_names": chunk_texts,
                "raw_outputs": chunk_outputs,
            }

    @staticmethod
    def _has_nonempty_geometric_prompt(find_input: Optional[FindStage]) -> bool:
        if find_input is None:
            return False
        for x in (getattr(find_input, "input_boxes", None), getattr(find_input, "input_points", None)):
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
        out = {}

        for key in (
            OUTPUT_KEYS.semantic_logits,
            OUTPUT_KEYS.class_tokens,
        ):
            if key not in raw_outputs or raw_outputs[key] is None:
                continue

            out[key] = self._reshape_prompt_first_tensor(
                raw_outputs[key],
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
                key=key,
            )

        if OUTPUT_KEYS.semantic_logits in out:
            out[OUTPUT_KEYS.semantic_logits] = self._ensure_4d_logits(
                out[OUTPUT_KEYS.semantic_logits],
                OUTPUT_KEYS.semantic_logits,
            )

        return out

    @staticmethod
    def _merge_chunk_outputs(chunk_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(chunk_outputs) == 0:
            raise ValueError("chunk_outputs is empty.")

        merged = {}
        for key in sorted({k for chunk_out in chunk_outputs for k in chunk_out.keys()}):
            values = [chunk_out[key] for chunk_out in chunk_outputs if key in chunk_out]
            if values:
                merged[key] = torch.cat(values, dim=1)
        return merged

    @staticmethod
    def _ensure_4d_logits(x: torch.Tensor, key: str) -> torch.Tensor:
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(
                    f"Expected {key} as [B, C, 1, H, W] when 5D, "
                    f"got {tuple(x.shape)}."
                )
            x = x[:, :, 0]

        if x.dim() != 4:
            raise ValueError(
                f"Expected {key} as [B, C, H, W], got {tuple(x.shape)}."
            )

        return x

    def run_final_mixer(
        self,
        semantic_logits: torch.Tensor,
        class_tokens: torch.Tensor,
        shared_clip_feature: torch.Tensor,
        sam3_feature_high: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._ensure_4d_logits(
            semantic_logits,
            OUTPUT_KEYS.semantic_logits,
        )

        if class_tokens.dim() != 4:
            raise ValueError(
                f"class_tokens must be [B, C, Q, D], got {tuple(class_tokens.shape)}."
            )
        if shared_clip_feature.dim() != 3:
            raise ValueError(
                "shared_clip_feature must be [B, N_clip, D], "
                f"got {tuple(shared_clip_feature.shape)}."
            )
        if sam3_feature_high.dim() != 4:
            raise ValueError(
                "sam3_feature_high must be [B, D, H, W], "
                f"got {tuple(sam3_feature_high.shape)}."
            )

        batch_size, num_classes, height, width = semantic_logits.shape

        if tuple(class_tokens.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "class_tokens shape mismatch: "
                f"class_tokens.shape[:2]={tuple(class_tokens.shape[:2])}, "
                f"expected={(batch_size, num_classes)}."
            )
        if int(class_tokens.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"class_tokens dim mismatch: expected {self.hidden_dim}, "
                f"got {class_tokens.shape[-1]}."
            )
        if int(shared_clip_feature.shape[0]) != batch_size:
            raise ValueError(
                "shared_clip_feature batch size mismatch: "
                f"{shared_clip_feature.shape[0]} vs {batch_size}."
            )
        if int(shared_clip_feature.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"shared_clip_feature dim mismatch: expected {self.hidden_dim}, "
                f"got {shared_clip_feature.shape[-1]}."
            )
        if tuple(sam3_feature_high.shape) != (
                batch_size,
                self.hidden_dim,
                height,
                width,
        ):
            raise ValueError(
                "sam3_feature_high shape mismatch: expected "
                f"{(batch_size, self.hidden_dim, height, width)}, "
                f"got {tuple(sam3_feature_high.shape)}."
            )

        clip_grid_hw = getattr(self, "_last_clip_grid_hw", None)
        if clip_grid_hw is None:
            raise RuntimeError(
                "clip_grid_hw is not cached. Call build_shared_clip_feature() "
                "before run_final_mixer()."
            )

        if self._text_cache is None:
            raise RuntimeError("Text cache is not prepared.")

        shared_clip_feature_high = self.build_high_res_clip_sam_feature(
            shared_clip_feature=shared_clip_feature,
            sam3_feature_high=sam3_feature_high,
        )

        sam3_text_tokens_full = self._text_cache["language_features"].detach()
        sam3_text_mask_full = self._text_cache["language_mask"].detach()

        mixer_outputs = self.final_mixer(
            semantic_logits=semantic_logits.detach(),
            class_tokens=class_tokens,
            shared_clip_feature_high=shared_clip_feature_high,
            sam3_text_tokens_full=sam3_text_tokens_full,
            sam3_text_mask_full=sam3_text_mask_full,
        )

        required_keys = (
            OUTPUT_KEYS.final_logits,
            OUTPUT_KEYS.presence_logits,
            OUTPUT_KEYS.presence_score,
            OUTPUT_KEYS.presence_logits_layers,
            OUTPUT_KEYS.mask_logits_layers,
        )
        for key in required_keys:
            if key not in mixer_outputs:
                raise ValueError(f"final_mixer output is missing key={key!r}.")

        out = {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.class_tokens: class_tokens,
            OUTPUT_KEYS.final_logits: mixer_outputs[OUTPUT_KEYS.final_logits],
            OUTPUT_KEYS.presence_logits: mixer_outputs[OUTPUT_KEYS.presence_logits],
            OUTPUT_KEYS.presence_score: mixer_outputs[OUTPUT_KEYS.presence_score],
            OUTPUT_KEYS.presence_logits_layers: mixer_outputs[
                OUTPUT_KEYS.presence_logits_layers
            ],
            OUTPUT_KEYS.mask_logits_layers: mixer_outputs[
                OUTPUT_KEYS.mask_logits_layers
            ],
        }

        return out

    def run_final_mixer_from_chunks(
        self,
        mixer_cache: List[Dict[str, torch.Tensor]],
        batch: Optional[BatchedDatapoint] = None,
        shared_clip_feature: Optional[torch.Tensor] = None,
        sam3_feature_high: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if len(mixer_cache) == 0:
            raise ValueError("mixer_cache is empty.")

        if batch is None:
            raise ValueError(
                "batch must be provided because the new final mixer needs class_names."
            )

        if shared_clip_feature is None:
            shared_clip_feature = self.build_shared_clip_feature(batch)

        if sam3_feature_high is None:
            sam3_feature_high = self.build_sam3_pixel_feature(batch)

        class_names = list(batch.find_text_batch)
        if len(class_names) == 0:
            raise ValueError("batch.find_text_batch is empty.")

        mixer_cache = sorted(
            mixer_cache,
            key=lambda item: int(item["chunk_class_ids"][0]),
        )

        semantic_logits = torch.cat(
            [item[OUTPUT_KEYS.semantic_logits] for item in mixer_cache],
            dim=1,
        )
        class_tokens = torch.cat(
            [item[OUTPUT_KEYS.class_tokens] for item in mixer_cache],
            dim=1,
        )

        merged_class_ids = []
        for item in mixer_cache:
            merged_class_ids.extend([int(x) for x in item["chunk_class_ids"]])

        expected_class_ids = list(range(len(merged_class_ids)))
        if merged_class_ids != expected_class_ids:
            raise ValueError(
                "mixer_cache chunks must cover classes in order without gaps. "
                f"Got {merged_class_ids}, expected {expected_class_ids}."
            )

        if len(class_names) != len(merged_class_ids):
            raise ValueError(
                "class_names length must match merged class count: "
                f"{len(class_names)} vs {len(merged_class_ids)}."
            )

        return self.run_final_mixer(
            semantic_logits=semantic_logits,
            class_tokens=class_tokens,
            shared_clip_feature=shared_clip_feature,
            sam3_feature_high=sam3_feature_high,
        )

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

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros((0, *geo_feats.shape[1:]), device=geo_feats.device)
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )

        if not encode_text:
            return (
                torch.cat([geo_feats, visual_prompt_embed], dim=0),
                torch.cat([geo_masks, visual_prompt_mask], dim=1),
                backbone_out,
            )

        prompt_list = [txt_feats, geo_feats, visual_prompt_embed]
        prompt_mask_list = [txt_masks, geo_masks, visual_prompt_mask]

        return torch.cat(prompt_list, dim=0), torch.cat(prompt_mask_list, dim=1), backbone_out

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

        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=torch.zeros_like(prompt),
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
        semantic_logits = seg_outputs.get("semantic_seg")
        if semantic_logits is None:
            raise ValueError("segmentation_head did not return 'semantic_seg' in semantic mode.")
        return {OUTPUT_KEYS.semantic_logits: semantic_logits}

    def forward_grounding_raw(
        self,
        backbone_out: Dict[str, torch.Tensor],
        find_input,
        geometric_prompt: Prompt,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            with torch.profiler.record_function("Sam3Image._encode_prompt"):
                prompt, prompt_mask, backbone_out = self._encode_prompt(
                    backbone_out,
                    find_input,
                    geometric_prompt,
                )

            with torch.profiler.record_function("Sam3Image._run_encoder"):
                backbone_out, encoder_out, _ = self._run_encoder(
                    backbone_out,
                    find_input,
                    prompt,
                    prompt_mask,
                )

            with torch.profiler.record_function("Sam3Image._run_semantic_segmentation_head"):
                out = self._run_semantic_segmentation_head(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    encoder_out=encoder_out,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )

        out[OUTPUT_KEYS.class_tokens] = self._run_class_token_encoder_cross_attn(
            class_token_seed=backbone_out["class_token_seed_pair"],
            encoder_out=encoder_out,
        )
        return out

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        shared_clip_feature = self.build_shared_clip_feature(input)
        sam3_feature_high = self.build_sam3_pixel_feature(input)

        mixer_cache = []

        for chunk in self.iter_chunk_raw_outputs(
                input,
                shared_clip_feature=shared_clip_feature,
        ):
            raw_outputs = chunk["raw_outputs"]

            required_keys = (
                OUTPUT_KEYS.semantic_logits,
                OUTPUT_KEYS.class_tokens,
            )
            for key in required_keys:
                if key not in raw_outputs:
                    raise ValueError(
                        f"Chunk raw_outputs must contain '{key}' for final mixer."
                    )

            mixer_cache.append(
                {
                    OUTPUT_KEYS.semantic_logits: raw_outputs[
                        OUTPUT_KEYS.semantic_logits
                    ].detach(),
                    OUTPUT_KEYS.class_tokens: raw_outputs[
                        OUTPUT_KEYS.class_tokens
                    ],
                    "chunk_class_ids": list(chunk["chunk_class_ids"]),
                }
            )

        return self.run_final_mixer_from_chunks(
            mixer_cache=mixer_cache,
            batch=input,
            shared_clip_feature=shared_clip_feature,
            sam3_feature_high=sam3_feature_high,
        )