from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_misc import BatchedDatapoint, FindStage
from .encoder_refiner import ClassConditionedEncoderRefiner
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
        openclip_prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        encoder_refiner_fusion_layers: int = 4,
        encoder_refiner_num_heads: int = 8,
        encoder_refiner_dropout: float = 0.1,
        encoder_refiner_hidden_dim: int = 256,
        encoder_refiner_score_embed_dim: int = 256,
        encoder_refiner_conv_kernel: int = 7,
        encoder_refiner_window_size: int = 12,
        encoder_refiner_shift_size: int = 6,
        encoder_refiner_use_checkpoint: bool = True,
        encoder_refiner_early_prompt_attention: bool = False,
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

        self.task_mode = normalize_task_mode(task_mode)
        if self.task_mode != TASK_MODE_SEMANTIC:
            raise NotImplementedError("Sam3Image currently only supports semantic task mode.")

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
                    "Projected OpenCLIP text/image dimensions must match. "
                    f"Got text_dim={self.clip_text_dim}, image_dim={self.clip_image_dim}."
                )
            self.clip_align_dim = self.clip_text_dim

        if self.clip_align_dim is None:
            raise RuntimeError(
                "OpenCLIP image/text encoders are required by the encoder refiner."
            )

        self.encoder_refiner = ClassConditionedEncoderRefiner(
            clip_text_encoder=self.clip_text_encoder,
            hidden_dim=int(encoder_refiner_hidden_dim),
            clip_dim=self.clip_align_dim,
            score_embed_dim=int(encoder_refiner_score_embed_dim),
            num_heads=int(encoder_refiner_num_heads),
            window_size=int(encoder_refiner_window_size),
            shift_size=int(encoder_refiner_shift_size),
            fusion_layers=int(encoder_refiner_fusion_layers),
            dropout=float(encoder_refiner_dropout),
            prompt_templates=list(openclip_prompt_templates or []),
            normalize_label_for_clip=bool(normalize_label_for_clip),
            clip_score_conv_kernel=int(encoder_refiner_conv_kernel),
            use_checkpoint=bool(encoder_refiner_use_checkpoint),
        )

        self.encoder_refiner_early_prompt_attention = bool(
            encoder_refiner_early_prompt_attention
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

    def _build_clip_image_cache(
        self,
        input: BatchedDatapoint,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.clip_image_encoder is None:
            return None
        if input.raw_images is None:
            raise ValueError("clip_image_encoder is enabled, but BatchedDatapoint.raw_images is None.")

        clip_out = self.clip_image_encoder.encode_raw_images(
            raw_images=input.raw_images,
            device=device,
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

        image_encoder_trainable = (
            hasattr(self.clip_image_encoder, "has_trainable_params")
            and self.clip_image_encoder.has_trainable_params()
        )

        if image_encoder_trainable:
            clip_feat_map = clip_feat_map.contiguous()
        else:
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

    def _maybe_apply_early_prompt_attention(
        self,
        encoder_out: Dict[str, torch.Tensor],
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.encoder_refiner_early_prompt_attention:
            return encoder_out

        if self.segmentation_head is None:
            raise RuntimeError("segmentation_head is required for early prompt attention.")

        if not hasattr(self.segmentation_head, "apply_prompt_attention"):
            raise RuntimeError(
                "segmentation_head must expose apply_prompt_attention() "
                "when encoder_refiner_early_prompt_attention=True."
            )

        out = dict(encoder_out)
        out["encoder_hidden_states"] = self.segmentation_head.apply_prompt_attention(
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        return out

    def build_encoder_refiner_cache(
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

        # SAM3 image-backbone forward for this batch.
        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)
        image_backbone_out = self._detach_tree(image_backbone_out)

        # Save backbone_fpn for later use in segmentation_head.
        backbone_fpn = [
            feat.detach().contiguous()
            for feat in image_backbone_out["backbone_fpn"]
        ]

        # Get CLIP image cache.
        clip_image_cache = self._build_clip_image_cache(
            input=input,
            device=device,
        )
        if clip_image_cache is None:
            raise ValueError("CLIP image cache is required.")

        encoder_feature_chunks: list[torch.Tensor] = []
        encoder_out_chunks: list[Dict] = []
        chunk_prompts: list[torch.Tensor] = []
        chunk_prompt_masks: list[torch.Tensor] = []
        chunk_class_counts: list[int] = []
        merged_class_ids: list[int] = []
        sam_text_token_chunks: list[torch.Tensor] = []
        sam_text_token_mask_chunks: list[torch.Tensor] = []

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

            raw_outputs = self.forward_grounding_encoder_only(
                backbone_out=chunk_backbone_out,
                find_input=chunk_find_input,
                geometric_prompt=geometric_prompt,
            )

            encoder_out = raw_outputs["encoder_out"]
            prompt = raw_outputs["prompt"]
            prompt_mask = raw_outputs["prompt_mask"]

            encoder_out_for_refiner = self._maybe_apply_early_prompt_attention(
                encoder_out=encoder_out,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )

            encoder_feature_chunk = self._extract_encoder_last_feature(
                encoder_out=encoder_out_for_refiner,
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
            )

            sam_text_tokens_chunk, sam_text_token_mask_chunk = self._extract_sam_text_tokens(
                encoder_out=encoder_out_for_refiner,
                batch_size=batch_size,
                num_chunk_classes=num_chunk_classes,
            )
            sam_text_token_chunks.append(sam_text_tokens_chunk)
            sam_text_token_mask_chunks.append(sam_text_token_mask_chunk)

            encoder_feature_chunks.append(encoder_feature_chunk)
            encoder_out_chunks.append(encoder_out_for_refiner)
            chunk_prompts.append(prompt)
            chunk_prompt_masks.append(prompt_mask)
            merged_class_ids.extend(chunk_class_ids)
            chunk_class_counts.append(num_chunk_classes)

        if len(encoder_feature_chunks) == 0:
            raise ValueError("No chunk outputs were produced.")

        expected_class_ids = list(range(num_classes))
        if merged_class_ids != expected_class_ids:
            raise ValueError(
                "Chunk class ids must cover all classes in order without gaps. "
                f"Got {merged_class_ids}, expected {expected_class_ids}."
            )

        encoder_features_72 = torch.cat(encoder_feature_chunks, dim=1)

        if tuple(encoder_features_72.shape[:2]) != (batch_size, num_classes):
            raise ValueError(
                "Merged encoder features shape mismatch: expected "
                f"{(batch_size, num_classes)}, "
                f"got {tuple(encoder_features_72.shape[:2])}."
            )

        return {
            "encoder_features_72": encoder_features_72,
            "encoder_out_chunks": encoder_out_chunks,
            "chunk_prompts": chunk_prompts,
            "chunk_prompt_masks": chunk_prompt_masks,
            "backbone_fpn": backbone_fpn,
            "clip_image_feat_map": clip_image_cache["clip_image_feat_map_native"],
            "sam_text_tokens": torch.cat(sam_text_token_chunks, dim=1),
            "sam_text_token_mask": torch.cat(sam_text_token_mask_chunks, dim=1),
            OUTPUT_KEYS.clip_mid_features: clip_image_cache[OUTPUT_KEYS.clip_mid_features],
            "clip_mid_layer_indices": clip_image_cache["clip_mid_layer_indices"],
            "class_names": class_texts,
            "class_ids": merged_class_ids,
            "chunk_class_counts": chunk_class_counts,
        }

    @staticmethod
    def _extract_sam_text_tokens(
        encoder_out: Dict[str, torch.Tensor],
        batch_size: int,
        num_chunk_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = encoder_out.get(
            "prompt_after_enc",
            encoder_out.get("prompt_before_enc"),
        )
        prompt_mask = encoder_out.get("prompt_mask")

        if prompt_tokens is None:
            raise ValueError("encoder_out must contain prompt_after_enc or prompt_before_enc.")
        if prompt_mask is None:
            raise ValueError("encoder_out must contain prompt_mask.")

        if prompt_tokens.ndim != 3:
            raise ValueError(
                f"prompt_tokens must be [T, B*C, D], got {tuple(prompt_tokens.shape)}."
            )
        if prompt_mask.ndim != 2:
            raise ValueError(
                f"prompt_mask must be [B*C, T], got {tuple(prompt_mask.shape)}."
            )

        token_len, pair_count, hidden_dim = prompt_tokens.shape
        expected_pairs = batch_size * num_chunk_classes

        if pair_count != expected_pairs:
            raise ValueError(
                f"prompt_tokens pair count mismatch: expected {expected_pairs}, "
                f"got {pair_count}."
            )

        if tuple(prompt_mask.shape) != (expected_pairs, token_len):
            raise ValueError(
                f"prompt_mask shape mismatch: expected {(expected_pairs, token_len)}, "
                f"got {tuple(prompt_mask.shape)}."
            )

        tokens = prompt_tokens.transpose(0, 1)  # [B*C_chunk, T, D]
        tokens = tokens.reshape(
            batch_size,
            num_chunk_classes,
            token_len,
            hidden_dim,
        ).contiguous()

        mask = prompt_mask.reshape(
            batch_size,
            num_chunk_classes,
            token_len,
        ).bool().contiguous()

        return tokens, mask

    def _extract_encoder_last_feature(
        self,
        encoder_out: Dict[str, torch.Tensor],
        batch_size: int,
        num_chunk_classes: int,
    ) -> torch.Tensor:
        """
        Extract last-layer visual tokens from encoder_hidden_states
        and reshape to [B, C_chunk, D, H, W].

        Takes the highest-res image tokens only (first spatial_shapes level).
        """
        encoder_hidden_states = encoder_out["encoder_hidden_states"]
        spatial_shapes = encoder_out["spatial_shapes"]

        if len(spatial_shapes) > 0:
            h_feat, w_feat = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            num_img_tokens = h_feat * w_feat
        else:
            raise ValueError("spatial_shapes is empty")

        num_pairs = batch_size * num_chunk_classes

        mem = encoder_hidden_states[:num_img_tokens]  # [N_img, num_pairs, D]
        mem = mem.transpose(0, 1)  # [num_pairs, N_img, D]
        mem = mem.transpose(1, 2)  # [num_pairs, D, N_img]
        mem = mem.reshape(num_pairs, self.hidden_dim, h_feat, w_feat)

        return mem.reshape(
            batch_size, num_chunk_classes, self.hidden_dim, h_feat, w_feat
        ).contiguous()

    @staticmethod
    def _write_refined_encoder_features_to_encoder_hidden_states(
        encoder_out: Dict[str, torch.Tensor],
        refined_encoder_features_chunk: torch.Tensor,
        batch_size: int,
        num_chunk_classes: int,
    ) -> torch.Tensor:
        """
        Write refined encoder features back into encoder_hidden_states visual token region.

        Args:
            encoder_out: original encoder output dict
            refined_encoder_features_chunk: [B, C_chunk, D, H, W]
            batch_size: B
            num_chunk_classes: C_chunk

        Returns:
            new encoder_hidden_states with visual tokens replaced
        """
        encoder_hidden_states = encoder_out["encoder_hidden_states"].clone()
        spatial_shapes = encoder_out["spatial_shapes"]

        h_feat, w_feat = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
        num_img_tokens = h_feat * w_feat

        B, C_chunk, D, H, W = refined_encoder_features_chunk.shape

        refined_flat = refined_encoder_features_chunk.reshape(B * C_chunk, D, H * W)
        refined_flat = refined_flat.permute(2, 0, 1)  # [H*W, B*C_chunk, D]

        encoder_hidden_states[:num_img_tokens] = refined_flat.to(
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )

        return encoder_hidden_states

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

    def run_encoder_refiner(
        self,
        encoder_features_72: torch.Tensor,
        encoder_out_chunks: List[Dict],
        chunk_prompts: List[torch.Tensor],
        chunk_prompt_masks: List[torch.Tensor],
        chunk_class_counts: List[int],
        backbone_fpn: List[torch.Tensor],
        clip_image_feat_map: torch.Tensor,
        sam_text_tokens: torch.Tensor,
        sam_text_token_mask: torch.Tensor,
        class_names: List[str],
        clip_mid_features: List[torch.Tensor],
        clip_mid_layer_indices: tuple[int, ...],
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, C, D, H, W = encoder_features_72.shape

        sam_fpn_72 = backbone_fpn[-1].detach()

        # 1. Run encoder refiner:
        #    CLIP score + SAM FPN injection → refiner layers → output fusion with original encoder feature.
        refiner_out = self.encoder_refiner(
            encoder_features_72=encoder_features_72,
            clip_image_feat_map=clip_image_feat_map,
            sam_text_tokens=sam_text_tokens,
            sam_text_token_mask=sam_text_token_mask,
            class_names=class_names,
            sam_fpn_72=sam_fpn_72,
        )

        refined_encoder_features_72 = refiner_out["refined_encoder_features_72"]

        # 2. Write refined features back and get final logits from segmentation_head.
        final_logits_chunks: list[torch.Tensor] = []
        chunk_start = 0

        for chunk_idx, encoder_out in enumerate(encoder_out_chunks):
            C_chunk = chunk_class_counts[chunk_idx]

            refined_chunk = refined_encoder_features_72[
                :, chunk_start:chunk_start + C_chunk
            ]

            refined_hidden_states = self._write_refined_encoder_features_to_encoder_hidden_states(
                encoder_out=encoder_out,
                refined_encoder_features_chunk=refined_chunk,
                batch_size=B,
                num_chunk_classes=C_chunk,
            )

            chunk_find_input = self._build_prompt_expanded_find_stage(
                batch_size=B,
                num_chunk_classes=C_chunk,
                device=refined_encoder_features_72.device,
            )

            seg_outputs = self.segmentation_head(
                backbone_feats=backbone_fpn,
                obj_queries=torch.empty(0, device=refined_encoder_features_72.device),
                image_ids=chunk_find_input.img_ids,
                encoder_hidden_states=refined_hidden_states,
                prompt=chunk_prompts[chunk_idx],
                prompt_mask=chunk_prompt_masks[chunk_idx],
                apply_prompt_attention=not self.encoder_refiner_early_prompt_attention,
            )

            chunk_logits = seg_outputs["semantic_seg"]
            if chunk_logits.dim() == 4 and chunk_logits.shape[0] == B * C_chunk:
                chunk_logits = chunk_logits.reshape(
                    B, C_chunk, *chunk_logits.shape[-2:]
                )
            elif chunk_logits.dim() == 4 and chunk_logits.shape[0] == B and chunk_logits.shape[1] == C_chunk:
                pass
            elif chunk_logits.dim() == 4 and chunk_logits.shape[1] == 1:
                chunk_logits = chunk_logits.squeeze(1)
            else:
                raise ValueError(
                    f"Unexpected semantic_seg shape: {tuple(chunk_logits.shape)}, "
                    f"expected [B*C_chunk, 1, H, W] or [B, C_chunk, H, W]."
                )

            final_logits_chunks.append(chunk_logits)
            chunk_start += C_chunk

        final_logits = torch.cat(final_logits_chunks, dim=1)

        if tuple(final_logits.shape[:2]) != (B, C):
            raise ValueError(
                f"final_logits batch/class mismatch: expected {(B, C)}, "
                f"got {tuple(final_logits.shape[:2])}."
            )

        result = {
            OUTPUT_KEYS.final_logits: final_logits.contiguous(),
        }

        if return_debug:
            result.update({
                OUTPUT_KEYS.encoder_features: encoder_features_72.detach().contiguous(),
                OUTPUT_KEYS.refined_encoder_features: refined_encoder_features_72.detach().contiguous(),
                OUTPUT_KEYS.refiner_features_36: refiner_out["refiner_features_36"].detach().contiguous(),
                OUTPUT_KEYS.score_embed_36: refiner_out["score_embed_36"].detach().contiguous(),
                OUTPUT_KEYS.clip_score_embed_36: refiner_out["clip_score_embed_36"].detach().contiguous(),
                OUTPUT_KEYS.clip_score_maps: refiner_out["clip_score_maps_36"].detach().contiguous(),
                OUTPUT_KEYS.template_clip_text_features: refiner_out["template_clip_text"].detach().contiguous(),
                OUTPUT_KEYS.clip_mid_features: [
                    feat.detach().contiguous() for feat in clip_mid_features
                ],
            })

        return result

    def run_encoder_refiner_from_cache(
        self,
        encoder_refiner_cache: Dict[str, Any],
        batch: BatchedDatapoint,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if batch is None:
            raise ValueError("batch must be provided.")

        encoder_features_72 = encoder_refiner_cache["encoder_features_72"]
        encoder_out_chunks = encoder_refiner_cache["encoder_out_chunks"]
        chunk_prompts = encoder_refiner_cache["chunk_prompts"]
        chunk_prompt_masks = encoder_refiner_cache["chunk_prompt_masks"]
        chunk_class_counts = encoder_refiner_cache["chunk_class_counts"]
        backbone_fpn = encoder_refiner_cache["backbone_fpn"]
        clip_image_feat_map = encoder_refiner_cache["clip_image_feat_map"]
        sam_text_tokens = encoder_refiner_cache["sam_text_tokens"]
        sam_text_token_mask = encoder_refiner_cache["sam_text_token_mask"]
        clip_mid_features = encoder_refiner_cache[OUTPUT_KEYS.clip_mid_features]
        clip_mid_layer_indices = encoder_refiner_cache["clip_mid_layer_indices"]

        class_names = list(batch.find_text_batch)
        if len(class_names) == 0:
            raise ValueError("batch.find_text_batch is empty.")

        cached_class_names = list(encoder_refiner_cache["class_names"])
        if cached_class_names != class_names:
            raise ValueError(
                "Cached class_names do not match batch.find_text_batch."
            )

        return self.run_encoder_refiner(
            encoder_features_72=encoder_features_72,
            encoder_out_chunks=encoder_out_chunks,
            chunk_prompts=chunk_prompts,
            chunk_prompt_masks=chunk_prompt_masks,
            chunk_class_counts=chunk_class_counts,
            backbone_fpn=backbone_fpn,
            clip_image_feat_map=clip_image_feat_map,
            sam_text_tokens=sam_text_tokens,
            sam_text_token_mask=sam_text_token_mask,
            class_names=class_names,
            clip_mid_features=clip_mid_features,
            clip_mid_layer_indices=clip_mid_layer_indices,
            return_debug=return_debug,
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

    def forward_grounding_encoder_only(
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

        return {
            "encoder_out": encoder_out,
            "prompt": prompt,
            "prompt_mask": prompt_mask,
        }

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        encoder_refiner_cache = self.build_encoder_refiner_cache(input)
        return self.run_encoder_refiner_from_cache(
            encoder_refiner_cache=encoder_refiner_cache,
            batch=input,
        )