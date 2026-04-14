# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from copy import deepcopy
from typing import Dict, Optional, Iterator, Any

import torch
from .vl_combiner import SAM3VLBackbone
from .data_misc import BatchedDatapoint, FindStage

from .act_ckpt_utils import activation_ckpt_wrapper
from .box_ops import box_cxcywh_to_xyxy
from .geometry_encoders import Prompt
from .model_misc import inverse_sigmoid


def _update_out(out, out_name, out_value, auxiliary=True, update_aux=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


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

        self.clip_extra_token_templates = []
        self.num_clip_extra_tokens = 0
        self.normalize_label_for_clip = True

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

        self.clip_text_token_norm = torch.nn.LayerNorm(self.hidden_dim)
        self.clip_text_token_gate = torch.nn.Parameter(
            torch.tensor(float(getattr(openclip_cfg, "text_token_gate_init", 0.0)))
        )

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                self.instance_dot_prod_scoring = deepcopy(dot_prod_scoring)
        else:
            self.class_embed = torch.nn.Linear(self.hidden_dim, 1)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = deepcopy(self.class_embed)

        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score

        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output
        self.prompt_chunk_size = None

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        return super().to(*args, **kwargs)

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

        batch_size = int(input.img_batch.shape[0])
        num_classes = len(class_texts)
        chunk_size = self._get_prompt_chunk_size(num_classes)

        with torch.no_grad():
            image_backbone_out = self.backbone.forward_image(input.img_batch)
        image_backbone_out = self._detach_tree(image_backbone_out)

        for start in range(0, num_classes, chunk_size):
            end = min(start + chunk_size, num_classes)
            chunk_texts = class_texts[start:end]
            num_chunk_classes = len(chunk_texts)
            chunk_class_ids = list(range(start, end))

            with torch.no_grad():
                text_backbone_out = self.backbone.forward_text(
                    chunk_texts,
                    device=device,
                )
            text_backbone_out = self._detach_tree(text_backbone_out)

            chunk_backbone_out = dict(image_backbone_out)
            chunk_backbone_out.update(text_backbone_out)

            clip_extra_tokens = None
            if self.clip_text_encoder is not None and len(self.clip_extra_token_templates) > 0:
                # 这里不能再额外包 no_grad。
                # OpenCLIPTextEncoder 内部已经保证：
                # - 文本塔冻结
                # - resizer 保留梯度
                clip_extra_tokens = self.clip_text_encoder.encode_prompt_templates(
                    class_names=chunk_texts,
                    templates=self.clip_extra_token_templates,
                    device=device,
                    normalize_label=self.normalize_label_for_clip,
                )  # [num_chunk_classes, K, 256]

                clip_extra_tokens = self.clip_text_token_norm(clip_extra_tokens)
                gate = torch.tanh(self.clip_text_token_gate)
                clip_extra_tokens = gate * clip_extra_tokens
                clip_extra_tokens = clip_extra_tokens.permute(1, 0, 2).contiguous()  # [K, num_chunk_classes, 256]

                chunk_backbone_out["clip_language_features"] = clip_extra_tokens
                chunk_backbone_out["clip_language_mask"] = torch.zeros(
                    (num_chunk_classes, clip_extra_tokens.shape[0]),
                    dtype=torch.bool,
                    device=device,
                )

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

    def _build_clip_extra_text_tokens(self, class_names, device):
        if self.clip_text_encoder is None:
            return None
        if len(self.clip_extra_token_templates) == 0:
            return None

        pooled_tokens = self.clip_text_encoder.encode_prompt_templates(
            class_names=class_names,
            templates=self.clip_extra_token_templates,
            device=device,
            normalize_label=self.normalize_label_for_clip,
        )  # [B, K, 256]

        pooled_tokens = self.clip_text_token_norm(pooled_tokens)

        gate = torch.tanh(self.clip_text_token_gate)
        pooled_tokens = gate * pooled_tokens

        pooled_tokens = pooled_tokens.permute(1, 0, 2).contiguous()
        return pooled_tokens

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
            "pred_masks",
            "pred_logits",
            "semantic_seg",
            "presence_logit",
            "presence_logit_dec",
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
        prev_mask_pred=None,
    ):
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        clip_txt_feats = None
        clip_txt_masks = None
        if "clip_language_features" in backbone_out:
            clip_txt_feats = backbone_out["clip_language_features"][:, txt_ids]
            clip_txt_masks = backbone_out["clip_language_mask"][txt_ids]

        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]

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

    def _run_decoder(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
    ):
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        apply_dac = self.transformer.decoder.dac and self.transformer.decoder.training
        hs, reference_boxes, dec_presence_out, dec_presence_feats = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=apply_dac,
        )

        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)

        out["presence_feats"] = dec_presence_feats
        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )
        return out, hs

    def _update_scores_and_boxes(
        self,
        out,
        hs,
        reference_boxes,
        prompt,
        prompt_mask,
        dec_presence_out=None,
        is_instance_prompt=False,
    ):
        apply_dac = self.transformer.decoder.dac and self.transformer.decoder.training
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        assert num_o2m == (num_o2o if apply_dac else 0)

        out["queries"] = hs[-1][:, :num_o2o]

        if self.use_dot_prod_scoring:
            dot_prod_scoring_head = self.dot_prod_scoring
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                dot_prod_scoring_head = self.instance_dot_prod_scoring
            outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
        else:
            class_embed_head = self.class_embed
            if is_instance_prompt and self.instance_class_embed is not None:
                class_embed_head = self.instance_class_embed
            outputs_class = class_embed_head(hs)

        box_head = self.transformer.decoder.bbox_embed
        if is_instance_prompt and self.transformer.decoder.instance_bbox_embed is not None:
            box_head = self.transformer.decoder.instance_bbox_embed

        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        if dec_presence_out is not None:
            _update_out(
                out,
                "presence_logit_dec",
                dec_presence_out,
                update_aux=self.transformer.decoder.training,
            )

        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = dec_presence_out.clone().sigmoid()
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = prob_dec_presence_out.detach()

            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)

        _update_out(
            out,
            "pred_logits",
            outputs_class[:, :, :num_o2o],
            update_aux=self.transformer.decoder.training,
        )
        _update_out(
            out,
            "pred_boxes",
            outputs_coord[:, :, :num_o2o],
            update_aux=self.transformer.decoder.training,
        )
        _update_out(
            out,
            "pred_boxes_xyxy",
            outputs_boxes_xyxy[:, :, :num_o2o],
            update_aux=self.transformer.decoder.training,
        )

        if num_o2m > 0 and self.transformer.decoder.training:
            _update_out(
                out,
                "pred_logits_o2m",
                outputs_class[:, :, num_o2o:],
                update_aux=self.transformer.decoder.training,
            )
            _update_out(
                out,
                "pred_boxes_o2m",
                outputs_coord[:, :, num_o2o:],
                update_aux=self.transformer.decoder.training,
            )
            _update_out(
                out,
                "pred_boxes_xyxy_o2m",
                outputs_boxes_xyxy[:, :, num_o2o:],
                update_aux=self.transformer.decoder.training,
            )

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
    ):
        apply_dac = self.transformer.decoder.dac and self.transformer.decoder.training
        if self.segmentation_head is not None:
            num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
            num_o2m = hs.size(2) - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = activation_ckpt_wrapper(self.segmentation_head)(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                act_ckpt_enable=self.segmentation_head.training and self.use_act_checkpoint_seg_head,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            aux_masks = False
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if self.o2m_mask_predict and num_o2m > 0:
                        _update_out(
                            out,
                            f"{k}_o2m",
                            v[:, num_o2o:],
                            auxiliary=aux_masks,
                        )
                else:
                    out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

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

        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
            "prompt": prompt,
            "prompt_mask": prompt_mask,
        }

        with torch.profiler.record_function("Sam3Image._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        with torch.profiler.record_function("Sam3Image._run_segmentation_heads"):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        return out

    def forward(self, input: BatchedDatapoint) -> Dict[str, torch.Tensor]:
        chunk_outputs = []
        for chunk in self.iter_chunk_raw_outputs(input):
            chunk_outputs.append(chunk["raw_outputs"])

        merged_outputs = self._merge_chunk_outputs(chunk_outputs)
        return merged_outputs