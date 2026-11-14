from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .data_misc import BatchedDatapoint, FindStage
from .feature_aggregation import RemoteCLIPGuidedAggregation
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
        remoteclip_prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        text_prompt_batch_size: int = 64,
        text_prompt_use_checkpoint: bool = True,
        aggregation_num_blocks: int = 4,
        aggregation_num_heads: int = 8,
        aggregation_dropout: float = 0.1,
        aggregation_feature_dim: int = 256,
        aggregation_semantic_dim: int = 256,
        aggregation_window_size: int = 12,
        aggregation_shift_size: int = 6,
        aggregation_use_checkpoint: bool = True,
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
            raise NotImplementedError(
                "Sam3Image currently only supports semantic task mode."
            )

        if (self.clip_text_encoder is None) != (
            self.clip_image_encoder is None
        ):
            raise RuntimeError(
                "RemoteCLIP is partially initialized: clip_text_encoder and "
                "clip_image_encoder must either both exist or both be None."
            )

        self.clip_text_dim = (
            self._infer_clip_text_dim()
            if self.clip_text_encoder is not None
            else None
        )
        self.clip_image_dim = (
            self._infer_clip_image_dim()
            if self.clip_image_encoder is not None
            else None
        )
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
                    "Projected RemoteCLIP text/image dimensions must match. "
                    f"Got text_dim={self.clip_text_dim}, "
                    f"image_dim={self.clip_image_dim}."
                )
            self.clip_align_dim = self.clip_text_dim

        if self.clip_align_dim is None:
            raise RuntimeError(
                "RemoteCLIP image/text encoders are required by the "
                "feature aggregation module."
            )

        self.feature_aggregation = RemoteCLIPGuidedAggregation(
            clip_text_encoder=self.clip_text_encoder,
            feature_dim=int(aggregation_feature_dim),
            clip_dim=self.clip_align_dim,
            semantic_dim=int(aggregation_semantic_dim),
            remoteclip_native_dim=self.clip_image_native_dim,
            num_heads=int(aggregation_num_heads),
            window_size=int(aggregation_window_size),
            shift_size=int(aggregation_shift_size),
            num_blocks=int(aggregation_num_blocks),
            dropout=float(aggregation_dropout),
            prompt_templates=list(remoteclip_prompt_templates or []),
            normalize_label_for_clip=bool(normalize_label_for_clip),
            use_checkpoint=bool(aggregation_use_checkpoint),
            text_prompt_batch_size=int(text_prompt_batch_size),
            text_prompt_use_checkpoint=bool(text_prompt_use_checkpoint),
        )

        self.prompt_chunk_size = None
        self._text_cache: Optional[Dict[str, torch.Tensor]] = None
        self._text_cache_key: Optional[Tuple[str, ...]] = None
        self._text_cache_device: Optional[str] = None
        self._last_clip_grid_hw: Optional[Tuple[int, int]] = None

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(
            self.parameters()
        ).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        self.clear_text_cache()
        return super().to(*args, **kwargs)

    @staticmethod
    def _normalize_text_cache_key(prompt_names: List[str]) -> Tuple[str, ...]:
        return tuple(str(x) for x in prompt_names)

    def clear_text_cache(self) -> None:
        self._text_cache = None
        self._text_cache_key = None
        self._text_cache_device = None
        self.clear_remoteclip_text_cache()

    def clear_remoteclip_text_cache(self) -> None:
        self.feature_aggregation.remoteclip_guidance.clear_text_cache()

    def prepare_text_cache(
        self,
        prompt_names: List[str],
        device: Optional[torch.device] = None,
        force: bool = False,
    ) -> None:
        if len(prompt_names) == 0:
            raise ValueError(
                "prompt_names is empty, cannot build text cache."
            )

        device = torch.device(device) if device is not None else self.device
        cache_key = self._normalize_text_cache_key(prompt_names)
        cache_device = str(device)

        if (
            not force
            and self._text_cache is not None
            and self._text_cache_key == cache_key
            and self._text_cache_device == cache_device
        ):
            return

        with torch.no_grad():
            text_out = self.backbone.forward_text(prompt_names, device=device)
        text_out = self._detach_tree(text_out)

        cache: Dict[str, torch.Tensor] = {
            "language_features": text_out["language_features"].contiguous(),
            "language_mask": text_out["language_mask"].contiguous(),
        }
        if text_out.get("language_embeds") is not None:
            cache["language_embeds"] = text_out[
                "language_embeds"
            ].contiguous()

        self._text_cache = cache
        self._text_cache_key = cache_key
        self._text_cache_device = cache_device

    def ensure_text_cache(
        self,
        prompt_names: List[str],
        device: Optional[torch.device] = None,
    ) -> None:
        self.prepare_text_cache(
            prompt_names=prompt_names, device=device, force=False
        )

    def _slice_text_cache(self, start: int, end: int) -> Dict[str, torch.Tensor]:
        if self._text_cache is None:
            raise RuntimeError("Text cache is not prepared.")

        out = {
            "language_features": self._text_cache["language_features"][
                :, start:end
            ].contiguous(),
            "language_mask": self._text_cache["language_mask"][
                start:end
            ].contiguous(),
        }

        if "language_embeds" in self._text_cache:
            out["language_embeds"] = self._text_cache["language_embeds"][
                :, start:end
            ].contiguous()

        return out

    def _get_prompt_chunk_size(self, num_prompts: int) -> int:
        chunk_size = getattr(self, "prompt_chunk_size", None)
        if chunk_size is None or int(chunk_size) <= 0:
            return num_prompts
        return min(int(chunk_size), num_prompts)

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
        raise AttributeError(
            "clip_text_encoder must expose a positive integer `output_dim`."
        )

    def _infer_clip_image_dim(self) -> int:
        output_dim = getattr(self.clip_image_encoder, "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim
        raise AttributeError(
            "clip_image_encoder must expose a positive integer `output_dim`."
        )

    def _build_remoteclip_image_cache(
        self,
        input: BatchedDatapoint,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.clip_image_encoder is None:
            return None
        if input.raw_images is None:
            raise ValueError(
                "clip_image_encoder is enabled, but "
                "BatchedDatapoint.raw_images is None."
            )

        clip_out = self.clip_image_encoder.encode_raw_images(
            raw_images=input.raw_images,
            device=device,
            keep_intermediate_grad=True,
        )

        if not isinstance(clip_out, dict):
            raise TypeError(
                "clip_image_encoder must return a dict with keys "
                "'feat_map', 'mid_features', and 'mid_layer_indices'."
            )

        remoteclip_feat_map = clip_out["feat_map"]
        clip_spatial_guidance_features = clip_out["mid_features"]
        clip_spatial_guidance_layer_indices = clip_out["mid_layer_indices"]

        if (
            not isinstance(remoteclip_feat_map, torch.Tensor)
            or remoteclip_feat_map.ndim != 4
        ):
            raise ValueError(
                "clip_out['feat_map'] must be [B, D_clip, Hc, Wc]."
            )

        if not isinstance(clip_spatial_guidance_features, list):
            raise TypeError(
                "clip_out['mid_features'] must be a list of tensors."
            )

        image_encoder_trainable = (
            hasattr(self.clip_image_encoder, "has_trainable_params")
            and self.clip_image_encoder.has_trainable_params()
        )

        keep_image_graph = bool(
            torch.is_grad_enabled() and image_encoder_trainable
        )

        if keep_image_graph:
            remoteclip_feat_map = remoteclip_feat_map.contiguous()
        else:
            remoteclip_feat_map = remoteclip_feat_map.detach().contiguous()
        clip_grid_hw = (
            int(remoteclip_feat_map.shape[-2]),
            int(remoteclip_feat_map.shape[-1]),
        )

        clean_mid_features = []
        for i, feat in enumerate(clip_spatial_guidance_features):
            if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                raise ValueError(
                    f"clip_spatial_guidance_features[{i}] must be "
                    f"[B, D, Hc, Wc], got "
                    f"{None if not isinstance(feat, torch.Tensor) else tuple(feat.shape)}."
                )
            if int(feat.shape[0]) != int(remoteclip_feat_map.shape[0]):
                raise ValueError(
                    f"clip_spatial_guidance_features[{i}] batch mismatch: "
                    f"{feat.shape[0]} vs {remoteclip_feat_map.shape[0]}."
                )
            if tuple(feat.shape[-2:]) != clip_grid_hw:
                raise ValueError(
                    f"clip_spatial_guidance_features[{i}] spatial size "
                    f"mismatch: {tuple(feat.shape[-2:])} vs {clip_grid_hw}."
                )
            clean_mid_features.append(
                feat.contiguous() if keep_image_graph
                else feat.detach().contiguous()
            )

        return {
            "remoteclip_feat_map_native": remoteclip_feat_map,
            "remoteclip_grid_hw": clip_grid_hw,
            OUTPUT_KEYS.clip_spatial_guidance_features: clean_mid_features,
            "clip_spatial_guidance_layer_indices": tuple(
                int(x) for x in clip_spatial_guidance_layer_indices
            ),
        }

    def build_aggregation_cache(
        self,
        input: BatchedDatapoint,
    ) -> Dict[str, Any]:
        device = self.device

        if len(input.find_inputs) != 1:
            raise ValueError(
                "Current semantic-only pipeline assumes exactly one find "
                "stage per batch."
            )

        base_find_input = input.find_inputs[0]
        prompt_names = list(input.find_text_batch)
        if len(prompt_names) == 0:
            raise ValueError("find_text_batch is empty.")

        self.ensure_text_cache(prompt_names=prompt_names, device=device)

        batch_size = int(input.img_batch.shape[0])
        num_prompts = len(prompt_names)
        chunk_size = self._get_prompt_chunk_size(num_prompts)

        # SAM3 image-backbone forward for this batch.
        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)
        image_backbone_out = self._detach_tree(image_backbone_out)

        # Save backbone_fpn for later use in segmentation_head.
        backbone_fpn = [
            feat.detach().contiguous()
            for feat in image_backbone_out["backbone_fpn"]
        ]

        remoteclip_image_cache = self._build_remoteclip_image_cache(
            input=input,
            device=device,
        )
        if remoteclip_image_cache is None:
            raise ValueError("RemoteCLIP image cache is required.")

        encoder_feature_chunks: list[torch.Tensor] = []
        chunk_prompt_counts: list[int] = []
        merged_prompt_ids: list[int] = []

        for start in range(0, num_prompts, chunk_size):
            end = min(start + chunk_size, num_prompts)
            chunk_prompts = prompt_names[start:end]
            num_chunk_prompts = len(chunk_prompts)
            chunk_prompt_ids = list(range(start, end))
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
                num_chunk_prompts=num_chunk_prompts,
                device=device,
                base_find_input=base_find_input,
            )

            geometric_prompt = Prompt(
                box_embeddings=chunk_find_input.input_boxes,
                box_mask=chunk_find_input.input_boxes_mask,
                box_labels=chunk_find_input.input_boxes_label,
            )

            chunk_out = self._encode_sam3_chunk(
                backbone_out=chunk_backbone_out,
                find_input=chunk_find_input,
                geometric_prompt=geometric_prompt,
                batch_size=batch_size,
                num_chunk_prompts=num_chunk_prompts,
            )

            encoder_feature_chunks.append(
                chunk_out[OUTPUT_KEYS.prompt_conditioned_features_72]
            )
            merged_prompt_ids.extend(chunk_prompt_ids)
            chunk_prompt_counts.append(num_chunk_prompts)

        if len(encoder_feature_chunks) == 0:
            raise ValueError("No chunk outputs were produced.")

        expected_prompt_ids = list(range(num_prompts))
        if merged_prompt_ids != expected_prompt_ids:
            raise ValueError(
                "Chunk prompt ids must cover all prompts in order without "
                f"gaps. Got {merged_prompt_ids}, "
                f"expected {expected_prompt_ids}."
            )

        prompt_conditioned_features_72 = torch.cat(
            encoder_feature_chunks, dim=1
        )

        if tuple(prompt_conditioned_features_72.shape[:2]) != (
            batch_size,
            num_prompts,
        ):
            raise ValueError(
                "Merged encoder features shape mismatch: expected "
                f"{(batch_size, num_prompts)}, "
                f"got {tuple(prompt_conditioned_features_72.shape[:2])}."
            )

        cache = {
            OUTPUT_KEYS.prompt_conditioned_features_72:
                prompt_conditioned_features_72,
            "backbone_fpn": backbone_fpn,
            "remoteclip_feat_map": (
                remoteclip_image_cache["remoteclip_feat_map_native"]
            ),
            OUTPUT_KEYS.clip_spatial_guidance_features:
                remoteclip_image_cache[
                    OUTPUT_KEYS.clip_spatial_guidance_features
                ],
            "clip_spatial_guidance_layer_indices": (
                remoteclip_image_cache[
                    "clip_spatial_guidance_layer_indices"
                ]
            ),
            "prompt_names": prompt_names,
            "prompt_ids": merged_prompt_ids,
            "chunk_prompt_counts": chunk_prompt_counts,
        }
        return cache

    def _hidden_states_to_feature_72(
        self,
        encoder_hidden_states: torch.Tensor,
        batch_size: int,
        num_chunk_prompts: int,
    ) -> torch.Tensor:
        expected_pairs = batch_size * num_chunk_prompts
        expected_tokens = 72 * 72

        if tuple(encoder_hidden_states.shape) != (
            expected_tokens,
            expected_pairs,
            self.hidden_dim,
        ):
            raise ValueError(
                f"encoder_hidden_states shape mismatch: expected "
                f"({expected_tokens}, {expected_pairs}, {self.hidden_dim}), "
                f"got {tuple(encoder_hidden_states.shape)}."
            )

        return (
            encoder_hidden_states
            .permute(1, 2, 0)
            .reshape(
                batch_size,
                num_chunk_prompts,
                self.hidden_dim,
                72,
                72,
            )
            .contiguous()
        )

    @staticmethod
    def _feature_72_to_hidden_states(
        feature_72: torch.Tensor,
    ) -> torch.Tensor:
        # [B, P_chunk, D, 72, 72] → [72*72, B*P_chunk, D]
        B, P_chunk, D, H, W = feature_72.shape
        return (
            feature_72
            .reshape(B * P_chunk, D, H * W)
            .permute(2, 0, 1)
            .contiguous()
        )

    @staticmethod
    def _has_nonempty_geometric_prompt(
        find_input: Optional[FindStage],
    ) -> bool:
        if find_input is None:
            return False
        for x in (
            getattr(find_input, "input_boxes", None),
            getattr(find_input, "input_points", None),
        ):
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                return True
        return False

    def _build_prompt_expanded_find_stage(
        self,
        batch_size: int,
        num_chunk_prompts: int,
        device: torch.device,
        base_find_input: Optional[FindStage] = None,
    ) -> FindStage:
        if self._has_nonempty_geometric_prompt(base_find_input):
            raise NotImplementedError(
                "Current stage-1 internal chunking only supports "
                "semantic-only batches without non-empty geometric prompts."
            )

        num_pairs = batch_size * num_chunk_prompts
        img_ids = torch.arange(
            batch_size, device=device, dtype=torch.long
        ).repeat_interleave(num_chunk_prompts)
        text_ids = torch.arange(
            num_chunk_prompts, device=device, dtype=torch.long
        ).repeat(batch_size)

        return FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=torch.zeros(
                (0, num_pairs, 4), dtype=torch.float32, device=device
            ),
            input_boxes_mask=torch.zeros(
                (num_pairs, 0), dtype=torch.bool, device=device
            ),
            input_boxes_label=torch.zeros(
                (0, num_pairs), dtype=torch.long, device=device
            ),
            input_points=torch.zeros(
                (0, num_pairs, 2), dtype=torch.float32, device=device
            ),
            input_points_mask=torch.zeros(
                (num_pairs, 0), dtype=torch.bool, device=device
            ),
        )

    # ------------------------------------------------------------------
    # Low-resolution: all-prompt aggregation at 36×36
    # ------------------------------------------------------------------

    def _run_aggregation_lowres(
        self,
        prompt_conditioned_features_72: torch.Tensor,
        remoteclip_feat_map: torch.Tensor,
        prompt_names: List[str],
        clip_spatial_guidance_features: list[torch.Tensor] | None = None,
        clip_spatial_guidance_layer_indices: tuple[int, ...] | None = None,
        sam3_fpn_72: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the 36×36 aggregation on all prompts simultaneously."""
        return self.feature_aggregation(
            prompt_conditioned_features_72=(
                prompt_conditioned_features_72
            ),
            remoteclip_feat_map=remoteclip_feat_map,
            class_names=prompt_names,
            clip_spatial_guidance_features=clip_spatial_guidance_features,
            clip_spatial_guidance_layer_indices=(
                clip_spatial_guidance_layer_indices
            ),
            sam3_fpn_72=sam3_fpn_72,
        )

    def run_aggregation_lowres_from_cache(
        self,
        aggregation_cache: Dict[str, Any],
        batch: BatchedDatapoint,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run all-prompt aggregation at 36×36 and return its output.

        Does NOT perform any 72/144/288 decoding. Returns aggregated features
        and semantic state for downstream per-chunk high-res decoding.
        """
        prompt_conditioned_features_72 = aggregation_cache[
            OUTPUT_KEYS.prompt_conditioned_features_72
        ]
        remoteclip_feat_map = aggregation_cache["remoteclip_feat_map"]

        cached_prompt_names = list(aggregation_cache["prompt_names"])
        batch_prompt_names = list(batch.find_text_batch)
        if cached_prompt_names != batch_prompt_names:
            raise ValueError(
                "Cached prompt_names do not match batch.find_text_batch."
            )

        aggregation_out = self._run_aggregation_lowres(
            prompt_conditioned_features_72=(
                prompt_conditioned_features_72
            ),
            remoteclip_feat_map=remoteclip_feat_map,
            prompt_names=batch_prompt_names,
            clip_spatial_guidance_features=aggregation_cache[
                OUTPUT_KEYS.clip_spatial_guidance_features
            ],
            clip_spatial_guidance_layer_indices=aggregation_cache[
                "clip_spatial_guidance_layer_indices"
            ],
            sam3_fpn_72=aggregation_cache["backbone_fpn"][2],
        )

        result: Dict[str, torch.Tensor] = {
            OUTPUT_KEYS.aggregated_features_36:
                aggregation_out[OUTPUT_KEYS.aggregated_features_36],
        }

        if return_debug:
            clip_spatial_guidance_features = aggregation_cache[
                OUTPUT_KEYS.clip_spatial_guidance_features
            ]
            result.update({
                OUTPUT_KEYS.prompt_conditioned_features_72:
                    prompt_conditioned_features_72.detach().contiguous(),
                OUTPUT_KEYS.semantic_state_36:
                    aggregation_out[
                        OUTPUT_KEYS.semantic_state_36
                    ].detach().contiguous(),
                OUTPUT_KEYS.clip_guidance_36:
                    aggregation_out[
                        OUTPUT_KEYS.clip_guidance_36
                    ].detach().contiguous(),
                OUTPUT_KEYS.clip_template_scores_36:
                    aggregation_out[
                        OUTPUT_KEYS.clip_template_scores_36
                    ].detach().contiguous(),
                OUTPUT_KEYS.remoteclip_template_text_features:
                    aggregation_out[
                        OUTPUT_KEYS.remoteclip_template_text_features
                    ].detach().contiguous(),
                OUTPUT_KEYS.clip_spatial_guidance_features: [
                    feat.detach().contiguous()
                    for feat in clip_spatial_guidance_features
                ],
            })

        return result

    # ------------------------------------------------------------------
    # Per-chunk high-resolution decoding
    # ------------------------------------------------------------------

    def decode_aggregation_chunk_from_cache(
        self,
        aggregation_cache: Dict[str, Any],
        aggregated_feature_36_chunk: torch.Tensor,
        class_start: int,
        class_end: int,
        return_teacher_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Single-scale fusion followed by shared frozen SAM3 decoding.

        Steps:
        1. Optionally decode the original SAM3 path (no_grad): original
           encoder72 → frozen Pixel Decoder → frozen semantic head →
           detached logits for training distillation only.
        2. Fuse aggregated36, prompt-conditioned encoder72 and FPN72 into 72×72.
        3. Student path (gradients enabled): fused 72×72 replaces the last
           FPN level of the same frozen Pixel Decoder and semantic head.
        """
        prompt_conditioned_features_72 = aggregation_cache[
            OUTPUT_KEYS.prompt_conditioned_features_72
        ]
        total_prompts = prompt_conditioned_features_72.shape[1]
        if not 0 <= class_start < class_end <= total_prompts:
            raise ValueError("Chunk indices are out of range.")

        backbone_fpn = aggregation_cache["backbone_fpn"]
        batch_size = prompt_conditioned_features_72.shape[0]
        num_prompts = class_end - class_start
        expected_shape = (batch_size, num_prompts, 256, 36, 36)
        if tuple(aggregated_feature_36_chunk.shape) != expected_shape:
            raise ValueError(
                f"Expected aggregated chunk {expected_shape}, "
                f"got {tuple(aggregated_feature_36_chunk.shape)}."
            )

        prompt_conditioned_chunk = prompt_conditioned_features_72[
            :, class_start:class_end
        ]
        image_ids = torch.arange(
            batch_size,
            device=prompt_conditioned_chunk.device,
            dtype=torch.long,
        ).repeat_interleave(num_prompts)

        result: Dict[str, torch.Tensor] = {}
        if self.training and return_teacher_logits:
            # Teacher runs first so its high-resolution intermediates do not
            # stay resident alongside the student graph.
            with torch.no_grad():
                teacher_outputs = self.segmentation_head(
                    backbone_feats=backbone_fpn,
                    image_ids=image_ids,
                    encoder_hidden_states=self._feature_72_to_hidden_states(
                        prompt_conditioned_chunk
                    ),
                )
                result[OUTPUT_KEYS.sam3_teacher_logits] = (
                    teacher_outputs["semantic_seg"]
                    .reshape(batch_size, num_prompts, 288, 288)
                    .detach()
                )
            del teacher_outputs

        num_pairs = batch_size * num_prompts
        decoder_features_72_flat = (
            self.feature_aggregation.fuse_semantic_detail_chunk(
                aggregated_feature_36=(
                    aggregated_feature_36_chunk.reshape(
                        num_pairs, 256, 36, 36
                    )
                ),
                prompt_conditioned_feature_72=(
                    prompt_conditioned_chunk.reshape(
                        num_pairs, 256, 72, 72
                    )
                ),
                sam3_fpn_72=backbone_fpn[2],
            )
        )
        student_hidden_states = self._feature_72_to_hidden_states(
            decoder_features_72_flat.reshape(
                batch_size, num_prompts, 256, 72, 72
            )
        )
        student_outputs = self.segmentation_head(
            backbone_feats=backbone_fpn,
            image_ids=image_ids,
            encoder_hidden_states=student_hidden_states,
        )
        result[OUTPUT_KEYS.final_logits] = student_outputs[
            "semantic_seg"
        ].reshape(
            batch_size, num_prompts, 288, 288
        )
        return result

    # ------------------------------------------------------------------
    # Inference / validation: full forward (not chunked backward)
    # ------------------------------------------------------------------

    def run_aggregation_from_cache(
        self,
        aggregation_cache: Dict[str, Any],
        batch: BatchedDatapoint,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full inference forward pass.

        For training this MUST NOT be called directly — use the Trainer's
        streaming chunk path instead (run_aggregation_lowres_from_cache +
        decode_aggregation_chunk_from_cache per chunk).

        Inference decodes the student once per prompt chunk. Teacher logits
        are produced only by the training streaming path when needed.
        """
        if self.training:
            raise RuntimeError(
                "run_aggregation_from_cache() must not be called during "
                "training. The Trainer must use the streaming chunk path: "
                "run_aggregation_lowres_from_cache() + per-chunk "
                "decode_aggregation_chunk_from_cache()."
            )

        if batch is None:
            raise ValueError("batch must be provided.")

        aggregation_out = self.run_aggregation_lowres_from_cache(
            aggregation_cache=aggregation_cache,
            batch=batch,
            return_debug=return_debug,
        )

        aggregated_features_36 = aggregation_out[
            OUTPUT_KEYS.aggregated_features_36
        ]
        B, P = aggregated_features_36.shape[:2]
        chunk_prompt_counts = aggregation_cache["chunk_prompt_counts"]

        final_logits = torch.empty(
            B, P, 288, 288,
            device=aggregated_features_36.device,
            dtype=aggregated_features_36.dtype,
        )

        chunk_start = 0
        for num_chunk_prompts in chunk_prompt_counts:
            chunk_end = chunk_start + num_chunk_prompts

            aggregated_feature_36_chunk = aggregated_features_36[
                :, chunk_start:chunk_end
            ]

            chunk_outputs = self.decode_aggregation_chunk_from_cache(
                aggregation_cache=aggregation_cache,
                aggregated_feature_36_chunk=aggregated_feature_36_chunk,
                class_start=chunk_start,
                class_end=chunk_end,
            )

            final_logits[:, chunk_start:chunk_end].copy_(
                chunk_outputs[OUTPUT_KEYS.final_logits]
            )

            chunk_start = chunk_end

        if chunk_start != P:
            raise ValueError(
                f"Chunk index mismatch: final chunk_start={chunk_start}, "
                f"expected P={P}."
            )

        result: Dict[str, torch.Tensor] = {
            OUTPUT_KEYS.final_logits: final_logits.contiguous(),
        }

        if return_debug:
            prompt_conditioned_features_72 = aggregation_cache[
                OUTPUT_KEYS.prompt_conditioned_features_72
            ]
            clip_spatial_guidance_features = aggregation_cache[
                OUTPUT_KEYS.clip_spatial_guidance_features
            ]
            result.update({
                OUTPUT_KEYS.prompt_conditioned_features_72:
                    prompt_conditioned_features_72.detach().contiguous(),
                OUTPUT_KEYS.aggregated_features_36:
                    aggregated_features_36.detach().contiguous(),
                OUTPUT_KEYS.semantic_state_36:
                    aggregation_out[OUTPUT_KEYS.semantic_state_36]
                    .detach().contiguous(),
                OUTPUT_KEYS.clip_guidance_36:
                    aggregation_out[OUTPUT_KEYS.clip_guidance_36]
                    .detach().contiguous(),
                OUTPUT_KEYS.clip_template_scores_36:
                    aggregation_out[OUTPUT_KEYS.clip_template_scores_36]
                    .detach().contiguous(),
                OUTPUT_KEYS.remoteclip_template_text_features:
                    aggregation_out[
                        OUTPUT_KEYS.remoteclip_template_text_features
                    ].detach().contiguous(),
                OUTPUT_KEYS.clip_spatial_guidance_features: [
                    feat.detach().contiguous()
                    for feat in clip_spatial_guidance_features
                ],
            })

        return result

    def _get_img_feats(self, backbone_out, img_ids):
        vis_feats = backbone_out["backbone_fpn"][
            -self.num_feature_levels:
        ]
        vis_pos_enc = backbone_out["vision_pos_enc"][
            -self.num_feature_levels:
        ]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        img_feats = [
            x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats
        ]
        img_pos_embeds = [
            x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc
        ]
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

        feat_tuple = self._get_img_feats(
            backbone_out, find_input.img_ids
        )
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

        if not encode_text:
            return (
                torch.cat([geo_feats, visual_prompt_embed], dim=0),
                torch.cat([geo_masks, visual_prompt_mask], dim=1),
                backbone_out,
            )

        prompt_list = [txt_feats, geo_feats, visual_prompt_embed]
        prompt_mask_list = [txt_masks, geo_masks, visual_prompt_mask]

        return (
            torch.cat(prompt_list, dim=0),
            torch.cat(prompt_mask_list, dim=1),
            backbone_out,
        )

    def _run_full_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        encoder_extra_kwargs=None,
    ):
        _, img_feats, img_pos_embeds, vis_feat_sizes = self._get_img_feats(
            backbone_out,
            find_input.img_ids,
        )

        return self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

    def _encode_sam3_chunk(
        self,
        backbone_out: Dict[str, torch.Tensor],
        find_input: FindStage,
        geometric_prompt: Prompt,
        batch_size: int,
        num_chunk_prompts: int,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out,
                find_input,
                geometric_prompt,
            )

            encoder_out = self._run_full_encoder(
                backbone_out=backbone_out,
                find_input=find_input,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )

            prompt_conditioned_hidden_states = (
                self.segmentation_head.apply_prompt_cross_attention(
                    encoder_hidden_states=encoder_out["memory"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
            )

            prompt_conditioned_features_72 = (
                self._hidden_states_to_feature_72(
                    prompt_conditioned_hidden_states,
                    batch_size=batch_size,
                    num_chunk_prompts=num_chunk_prompts,
                )
            )

        return {
            OUTPUT_KEYS.prompt_conditioned_features_72:
                prompt_conditioned_features_72,
        }

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError(
                "Sam3Image.forward() must not be called during training. "
                "The Trainer must use the streaming chunk path: "
                "build_aggregation_cache() + "
                "run_aggregation_lowres_from_cache() + per-chunk "
                "decode_aggregation_chunk_from_cache()."
            )
        aggregation_cache = self.build_aggregation_cache(input)
        return self.run_aggregation_from_cache(
            aggregation_cache=aggregation_cache,
            batch=input,
        )
