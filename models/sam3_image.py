from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_misc import BatchedDatapoint, FindStage
from .class_text_guided_mask_prior_clip_fusion_mixer import (
    ClassTextGuidedMaskPriorClipFusionFinalMixer,
)
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
        clip_prompt_templates: Optional[List[str]] = None,
        num_clip_prompt_templates: int = 0,
        normalize_label_for_clip: bool = True,
        final_mixer_fusion_layers: int = 4,
        final_mixer_num_heads: int = 8,
        final_mixer_dropout: float = 0.1,
        lowres_hidden_dim: int = 256,
        lowres_score_embed_dim: int = 32,
        lowres_window_size: int = 8,
        lowres_shift_size: int = 4,
        dynamic_tokens_per_template: int = 4,
        upsampler_class_chunk_size: int = 4,
        upsampler_decoder_channels: Optional[List[int]] = None,
        upsampler_sam_guidance_channels: Optional[List[int]] = None,
        upsampler_clip_guidance_channels: Optional[List[int]] = None,
        upsampler_clip_guidance_stage_indices: Optional[List[int]] = None,
        upsampler_upsample_mode: str = "bilinear",
        upsampler_norm: str = "group_norm",
        upsampler_act: str = "gelu",
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

        self.clip_prompt_templates = list(clip_prompt_templates or [])
        self.num_clip_prompt_templates = int(num_clip_prompt_templates)
        self.clip_prompt_templates = self.clip_prompt_templates[
            : self.num_clip_prompt_templates
        ]

        self.normalize_label_for_clip = bool(normalize_label_for_clip)

        if (self.clip_text_encoder is None) != (self.clip_image_encoder is None):
            raise RuntimeError(
                "OpenCLIP is partially initialized: clip_text_encoder and "
                "clip_image_encoder must either both exist or both be None."
            )

        self.clip_text_dim = self._infer_clip_text_dim() if self.clip_text_encoder is not None else None
        self.clip_image_dim = self._infer_clip_image_dim() if self.clip_image_encoder is not None else None
        self.clip_image_native_dim = (
            int(getattr(self.clip_image_encoder, "native_dim", None))
            if self.clip_image_encoder is not None
            and hasattr(self.clip_image_encoder, "native_dim")
            else self.clip_image_dim
        )
        self.clip_align_dim = None

        if self.clip_text_dim is not None and self.clip_image_dim is not None:
            if self.clip_text_dim != self.clip_image_dim:
                raise ValueError(
                    "Projected OpenCLIP text/image dimensions must match for native CLIP attention. "
                    f"Got text_dim={self.clip_text_dim}, image_dim={self.clip_image_dim}."
                )
            self.clip_align_dim = self.clip_text_dim

        self.final_mixer_num_heads = int(final_mixer_num_heads)
        self.final_mixer_fusion_layers = int(final_mixer_fusion_layers)
        self.final_mixer_dropout = float(final_mixer_dropout)

        if self.clip_align_dim is None:
            raise RuntimeError(
                "OpenCLIP image/text encoders are required by the final mixer."
            )

        self.final_mixer = ClassTextGuidedMaskPriorClipFusionFinalMixer(
            sam_dim=self.hidden_dim,
            fusion_layers=self.final_mixer_fusion_layers,
            num_heads=self.final_mixer_num_heads,
            dropout=self.final_mixer_dropout,
            clip_text_encoder=self.clip_text_encoder,
            clip_prompt_templates=self.clip_prompt_templates,
            normalize_label_for_clip=self.normalize_label_for_clip,
            clip_image_native_dim=self.clip_image_native_dim,
            score_embed_dim=int(lowres_score_embed_dim),
            lowres_hidden_dim=int(lowres_hidden_dim),
            window_size=int(lowres_window_size),
            shift_size=int(lowres_shift_size),
            tokens_per_template=int(dynamic_tokens_per_template),
            upsampler_class_chunk_size=int(upsampler_class_chunk_size),
            upsampler_decoder_channels=(
                list(upsampler_decoder_channels)
                if upsampler_decoder_channels is not None
                else None
            ),
            upsampler_sam_guidance_channels=(
                list(upsampler_sam_guidance_channels)
                if upsampler_sam_guidance_channels is not None
                else None
            ),
            upsampler_clip_guidance_channels=(
                list(upsampler_clip_guidance_channels)
                if upsampler_clip_guidance_channels is not None
                else None
            ),
            upsampler_clip_guidance_stage_indices=(
                list(upsampler_clip_guidance_stage_indices)
                if upsampler_clip_guidance_stage_indices is not None
                else None
            ),
            upsampler_upsample_mode=str(upsampler_upsample_mode),
            upsampler_norm=str(upsampler_norm),
            upsampler_act=str(upsampler_act),
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

        if "language_embeds" in self._text_cache:
            out["language_embeds"] = self._text_cache["language_embeds"][:, start:end].contiguous()

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

        native_h, native_w = self.clip_image_encoder.get_native_image_size()

        processed = []
        for i, x in enumerate(raw_images):
            if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 3:
                raise ValueError(
                    f"raw_images[{i}] must be a tensor with shape [3, H, W], got "
                    f"{None if not isinstance(x, torch.Tensor) else tuple(x.shape)}"
                )
            x = x.to(device=device, dtype=torch.float32)
            x = x.unsqueeze(0)
            x = F.interpolate(x, size=(native_h, native_w), mode="bilinear", align_corners=False)
            processed.append(x.squeeze(0))

        batch = torch.stack(processed, dim=0)
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
            clip_out = self.clip_image_encoder.encode_image_with_intermediate(
                clip_img_batch
            )

        if not isinstance(clip_out, dict):
            raise TypeError(
                "clip_image_encoder must return a dict with keys "
                "'feat_map', 'mid_features', and 'mid_layer_indices'."
            )

        clip_feat_map = clip_out["feat_map"]
        clip_mid_features = clip_out["mid_features"]
        clip_mid_layer_indices = clip_out["mid_layer_indices"]

        if not isinstance(clip_feat_map, torch.Tensor) or clip_feat_map.ndim != 4:
            raise ValueError(
                "clip_out['feat_map'] must be [B, D_clip, Hc, Wc]."
            )

        if not isinstance(clip_mid_features, list):
            raise TypeError("clip_out['mid_features'] must be a list of tensors.")

        if len(clip_mid_features) != 2:
            raise ValueError(
                f"Expected exactly 2 CLIP middle features, got {len(clip_mid_features)}."
            )

        clip_feat_map = clip_feat_map.detach().contiguous()
        clip_grid_hw = (
            int(clip_feat_map.shape[-2]),
            int(clip_feat_map.shape[-1]),
        )

        clean_mid_features = []
        for i, feat in enumerate(clip_mid_features):
            if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                raise ValueError(
                    f"clip_mid_features[{i}] must be [B, D, Hc, Wc], "
                    f"got {None if not isinstance(feat, torch.Tensor) else tuple(feat.shape)}."
                )
            if int(feat.shape[0]) != int(clip_feat_map.shape[0]):
                raise ValueError(
                    f"clip_mid_features[{i}] batch mismatch: "
                    f"{feat.shape[0]} vs {clip_feat_map.shape[0]}."
                )
            if tuple(feat.shape[-2:]) != clip_grid_hw:
                raise ValueError(
                    f"clip_mid_features[{i}] spatial size mismatch: "
                    f"{tuple(feat.shape[-2:])} vs {clip_grid_hw}."
                )
            clean_mid_features.append(feat.detach().contiguous())

        return {
            "clip_image_feat_map_native": clip_feat_map,
            "clip_image_grid_hw": clip_grid_hw,
            OUTPUT_KEYS.clip_mid_features: clean_mid_features,
            "clip_mid_layer_indices": tuple(int(x) for x in clip_mid_layer_indices),
        }

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

    def build_final_mixer_cache(
        self,
        input: BatchedDatapoint,
    ) -> Dict[str, Any]:
        device = self.device

        if len(input.find_inputs) != 1:
            raise ValueError(
                "Current semantic-only pipeline assumes exactly one find stage per batch."
            )

        base_find_input = input.find_inputs[0]
        class_texts = list(input.find_text_batch)
        if len(class_texts) == 0:
            raise ValueError("find_text_batch is empty.")

        self.ensure_text_cache(class_texts=class_texts, device=device)

        batch_size = int(input.img_batch.shape[0])
        num_classes = len(class_texts)
        chunk_size = self._get_prompt_chunk_size(num_classes)

        # The only SAM3 image-backbone forward for this batch.
        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)
        image_backbone_out = self._detach_tree(image_backbone_out)

        # Extract sam3_fpn_features from backbone output (for upsampler).
        sam3_fpn_features = [
            feat.detach().contiguous()
            for feat in image_backbone_out["backbone_fpn"]
        ]

        # Get CLIP grid size from the image encoder.
        clip_image_cache = self._build_clip_image_cache(
            input=input,
            device=device,
        )
        if clip_image_cache is None:
            raise ValueError("CLIP image cache is required.")
        clip_grid_hw = clip_image_cache["clip_image_grid_hw"]

        semantic_logits_chunks: list[torch.Tensor] = []
        class_feature_low_chunks: list[torch.Tensor] = []
        merged_class_ids: list[int] = []

        for start in range(0, num_classes, chunk_size):
            end = min(start + chunk_size, num_classes)
            chunk_texts = class_texts[start:end]
            num_chunk_classes = len(chunk_texts)
            chunk_class_ids = list(range(start, end))
            chunk_text_cache = self._slice_text_cache(start=start, end=end)

            chunk_backbone_out = dict(image_backbone_out)
            chunk_backbone_out["language_features"] = chunk_text_cache[
                "language_features"
            ]
            chunk_backbone_out["language_mask"] = chunk_text_cache[
                "language_mask"
            ]
            if "language_embeds" in chunk_text_cache:
                chunk_backbone_out["language_embeds"] = chunk_text_cache[
                    "language_embeds"
                ]

            chunk_find_input = self._build_prompt_expanded_find_stage(
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
                device=device,
                base_find_input=base_find_input,
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

            if OUTPUT_KEYS.semantic_logits not in chunk_outputs:
                raise ValueError(
                    "Chunk outputs must contain semantic_logits for final mixer."
                )

            semantic_logits = self._ensure_4d_logits(
                chunk_outputs[OUTPUT_KEYS.semantic_logits],
                OUTPUT_KEYS.semantic_logits,
            )

            # Build class_feature_low from encoder output.
            encoder_out = raw_outputs.get("_encoder_out", None)
            if encoder_out is None:
                raise ValueError(
                    "Chunk raw_outputs must contain '_encoder_out' for "
                    "class_feature_low extraction."
                )
            class_feature_low_chunk = self._build_class_feature_low_from_encoder(
                encoder_out=encoder_out,
                clip_grid_hw=clip_grid_hw,
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
            )
            # class_feature_low_chunk: [B, C_chunk, D_sam, Hc, Wc]

            semantic_logits_chunks.append(semantic_logits.detach())
            class_feature_low_chunks.append(class_feature_low_chunk)
            merged_class_ids.extend(chunk_class_ids)

        if len(semantic_logits_chunks) == 0:
            raise ValueError("No chunk outputs were produced.")

        expected_class_ids = list(range(num_classes))
        if merged_class_ids != expected_class_ids:
            raise ValueError(
                "Chunk class ids must cover all classes in order without gaps. "
                f"Got {merged_class_ids}, expected {expected_class_ids}."
            )

        semantic_logits = torch.cat(semantic_logits_chunks, dim=1)
        class_feature_low = torch.cat(class_feature_low_chunks, dim=1)

        if tuple(semantic_logits.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "Merged semantic_logits shape mismatch: expected "
                f"{(batch_size, num_classes)}, got {tuple(semantic_logits.shape[:2])}."
            )

        if tuple(class_feature_low.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "Merged class_feature_low shape mismatch: expected "
                f"{(batch_size, num_classes)}, got {tuple(class_feature_low.shape[:2])}."
            )

        # Extract SAM3 text features from cache for final class token building.
        sam3_text_features = self._text_cache["language_features"].contiguous()
        sam3_text_mask = self._text_cache["language_mask"].contiguous()

        return {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.class_feature_low: class_feature_low,
            OUTPUT_KEYS.sam3_fpn_features: sam3_fpn_features,
            OUTPUT_KEYS.clip_mid_features: clip_image_cache[OUTPUT_KEYS.clip_mid_features],
            "clip_mid_layer_indices": clip_image_cache["clip_mid_layer_indices"],
            "sam3_text_features": sam3_text_features,
            "sam3_text_mask": sam3_text_mask,
            "clip_image_feat_map": clip_image_cache["clip_image_feat_map_native"],
            "class_names": class_texts,
            "class_ids": merged_class_ids,
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
        class_feature_low: torch.Tensor,
        sam3_fpn_features: List[torch.Tensor],
        sam3_text_features: torch.Tensor,
        sam3_text_mask: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        class_names: List[str],
        clip_mid_features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        semantic_logits = self._ensure_4d_logits(
            semantic_logits,
            OUTPUT_KEYS.semantic_logits,
        )

        if class_feature_low.dim() != 5:
            raise ValueError(
                f"class_feature_low must be [B, C, D, Hc, Wc], "
                f"got {tuple(class_feature_low.shape)}."
            )

        batch_size, num_classes, _, _, _ = class_feature_low.shape

        if tuple(semantic_logits.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "semantic_logits batch/class mismatch with class_feature_low."
            )

        mixer_outputs = self.final_mixer(
            semantic_logits=semantic_logits,
            class_feature_low=class_feature_low,
            sam3_fpn_features=sam3_fpn_features,
            sam3_text_features=sam3_text_features,
            sam3_text_mask=sam3_text_mask,
            clip_image_feat_map=clip_image_feat_map,
            class_names=class_names,
            clip_mid_features=clip_mid_features,
        )

        required_keys = (OUTPUT_KEYS.final_logits,)
        for key in required_keys:
            if key not in mixer_outputs:
                raise ValueError(f"final_mixer output is missing key={key!r}.")

        out = {
            OUTPUT_KEYS.semantic_logits: semantic_logits,
            OUTPUT_KEYS.final_logits: mixer_outputs[OUTPUT_KEYS.final_logits],
        }

        for key in (
            OUTPUT_KEYS.class_tokens,
            OUTPUT_KEYS.class_feature_low,
            OUTPUT_KEYS.clip_score_maps,
            OUTPUT_KEYS.sam3_score_low,
            OUTPUT_KEYS.clip_mid_features,
        ):
            if key in mixer_outputs:
                out[key] = mixer_outputs[key]

        return out

    def run_final_mixer_from_cache(
        self,
        final_mixer_cache: Dict[str, Any],
        batch: BatchedDatapoint,
    ) -> Dict[str, torch.Tensor]:
        if batch is None:
            raise ValueError("batch must be provided for final mixer inputs.")

        required_keys = (
            OUTPUT_KEYS.semantic_logits,
            OUTPUT_KEYS.class_feature_low,
        )
        for key in required_keys:
            if key not in final_mixer_cache:
                raise ValueError(
                    f"final_mixer_cache must contain {key!r}."
                )

        semantic_logits = final_mixer_cache[OUTPUT_KEYS.semantic_logits]
        class_feature_low = final_mixer_cache[OUTPUT_KEYS.class_feature_low]
        sam3_fpn_features = final_mixer_cache[OUTPUT_KEYS.sam3_fpn_features]
        sam3_text_features = final_mixer_cache["sam3_text_features"]
        sam3_text_mask = final_mixer_cache["sam3_text_mask"]

        clip_mid_features = final_mixer_cache[OUTPUT_KEYS.clip_mid_features]

        class_names = list(batch.find_text_batch)
        if len(class_names) == 0:
            raise ValueError("batch.find_text_batch is empty.")

        cached_class_names = final_mixer_cache.get("class_names", None)
        if cached_class_names is not None:
            cached_class_names = list(cached_class_names)
            if cached_class_names != class_names:
                raise ValueError(
                    "Cached class_names do not match batch.find_text_batch."
                )

        if semantic_logits.dim() != 4:
            raise ValueError(
                "semantic_logits must be [B, C, H, W], "
                f"got {tuple(semantic_logits.shape)}."
            )

        if int(semantic_logits.shape[1]) != len(class_names):
            raise ValueError(
                "semantic_logits class count must match batch.find_text_batch: "
                f"{semantic_logits.shape[1]} vs {len(class_names)}."
            )

        # Use cached CLIP image features (already computed in build_final_mixer_cache).
        clip_image_feat_map = final_mixer_cache.get("clip_image_feat_map", None)
        if clip_image_feat_map is None:
            clip_image_cache = self._build_clip_image_cache(
                input=batch,
                device=self.device,
            )
            if clip_image_cache is None:
                raise ValueError("CLIP image cache is required for final mixer.")
            clip_image_feat_map = clip_image_cache["clip_image_feat_map_native"]

        return self.run_final_mixer(
            semantic_logits=semantic_logits,
            class_feature_low=class_feature_low,
            sam3_fpn_features=sam3_fpn_features,
            sam3_text_features=sam3_text_features,
            sam3_text_mask=sam3_text_mask,
            clip_image_feat_map=clip_image_feat_map,
            class_names=class_names,
            clip_mid_features=clip_mid_features,
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

    def _build_class_feature_low_from_encoder(
        self,
        encoder_out: Dict[str, torch.Tensor],
        clip_grid_hw: Tuple[int, int],
        batch_size: int,
        num_chunk_classes: int,
    ) -> torch.Tensor:
        """
        Extract class-conditioned low-res features from encoder output.

        Takes the highest-res image tokens from encoder_hidden_states,
        reshapes to spatial, projects to D_sam, and pools to CLIP grid.

        Returns:
            class_feature_low: [B, C_chunk, D_sam, Hc, Wc]
        """
        encoder_hidden_states = encoder_out["encoder_hidden_states"]
        spatial_shapes = encoder_out["spatial_shapes"]
        level_start_index = encoder_out["level_start_index"]

        num_pairs = int(batch_size) * int(num_chunk_classes)

        # Determine image token range for the highest-res level
        if len(spatial_shapes) > 0:
            h_feat, w_feat = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            num_img_tokens = h_feat * w_feat
        else:
            raise ValueError("spatial_shapes is empty")

        # encoder_hidden_states: [total_tokens, num_pairs, D]
        mem = encoder_hidden_states[:num_img_tokens]  # [num_img_tokens, num_pairs, D]
        mem = mem.transpose(0, 1)  # [num_pairs, num_img_tokens, D]
        mem = mem.transpose(1, 2)  # [num_pairs, D, num_img_tokens]
        mem = mem.reshape(num_pairs, self.hidden_dim, h_feat, w_feat)

        Hc, Wc = int(clip_grid_hw[0]), int(clip_grid_hw[1])

        return self.final_mixer.project_class_feature_low(
            encoder_feature=mem,
            target_hw=(Hc, Wc),
            batch_size=batch_size,
            num_classes=num_chunk_classes,
        )

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

        # Save encoder_out for class_feature_low extraction (no class tokens here -
        # class tokens are built later in the final mixer).
        out["_encoder_out"] = encoder_out
        return out

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        final_mixer_cache = self.build_final_mixer_cache(input)
        return self.run_final_mixer_from_cache(
            final_mixer_cache=final_mixer_cache,
            batch=input,
        )