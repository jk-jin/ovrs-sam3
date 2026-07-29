# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .model_misc import MLP


class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model):
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        return super().forward(hs)


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim, mask_dim):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries, pixel_embed):
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum(
                    "bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
        else:
            # Assumed to have aux masks
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum(
                    "lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )

        return mask_preds


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
    ):
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(
                hidden_dim, 1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt

        # used to update the output dictionary
        self.instance_keys = ["pred_masks"]

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _prepare_pixel_decoder_input(
        self,
        backbone_feats: List[torch.Tensor],
        image_ids,
        encoder_hidden_states,
    ) -> List[torch.Tensor]:
        """Prepare per-query backbone visual feats for the pixel decoder.

        Handles image_ids broadcasting (bs=1) or copying (bs>1), and replaces
        the last FPN level with the class-conditional encoder feature.
        """
        feature_device = backbone_feats[0].device
        model_device = self.device
        image_ids_ = image_ids.to(feature_device)

        if backbone_feats[0].shape[0] > 1:
            backbone_visual_feats = []
            for feat in backbone_feats:
                backbone_visual_feats.append(feat[image_ids_, ...].to(model_device))
        else:
            backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]

        encoder_hidden_states_ = encoder_hidden_states.permute(1, 2, 0)
        spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
        encoder_visual_embed = encoder_hidden_states_[..., :spatial_dim].reshape(
            -1, *backbone_feats[-1].shape[1:]
        )

        backbone_visual_feats[-1] = encoder_visual_embed
        return backbone_visual_feats

    def _embed_pixels(
        self,
        backbone_feats: List[torch.Tensor],
        image_ids,
        encoder_hidden_states,
    ) -> torch.Tensor:
        if not self.use_encoder_inputs:
            model_device = self.device
            backbone_feats_dev = [x.to(model_device) for x in backbone_feats]
            pixel_embed = self.pixel_decoder(backbone_feats_dev)
            if pixel_embed.shape[0] == 1:
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
            return pixel_embed

        assert encoder_hidden_states is not None

        backbone_visual_feats = self._prepare_pixel_decoder_input(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.act_ckpt and torch.is_grad_enabled():
            pixel_embed = checkpoint.checkpoint(
                self.pixel_decoder, backbone_visual_feats, use_reentrant=False
            )
        else:
            pixel_embed = self.pixel_decoder(backbone_visual_feats)

        return pixel_embed

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        conv_layers = []
        norms = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels
        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=compile_mode, dynamic=True, fullgraph=True
            )
            # Needed to make checkpointing happy. But we don't know if the module is checkpointed, so we disable it by default.
            torch._dynamo.config.optimize_ddp = False

    def forward_multiscale(
        self,
        backbone_feats: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(backbone_feats) != 3:
            raise ValueError(
                "Semantic Pixel Decoder pyramid requires exactly three "
                f"feature levels, got {len(backbone_feats)}."
            )

        # Verify expected spatial ordering: 288, 144, 72.
        expected_sizes = [(288, 288), (144, 144), (72, 72)]
        for i, (feat, expected) in enumerate(zip(backbone_feats, expected_sizes)):
            if tuple(feat.shape[-2:]) != expected:
                raise ValueError(
                    f"backbone_feats[{i}] must be {expected}, "
                    f"got {tuple(feat.shape[-2:])}."
                )

        multiscale_features = [backbone_feats[-1]]  # 72×72

        prev_fpn = backbone_feats[-1]

        for stage_idx, curr_fpn in enumerate(backbone_feats[:-1][::-1]):
            prev_fpn = curr_fpn + F.interpolate(
                prev_fpn,
                size=curr_fpn.shape[-2:],
                mode=self.interpolation_mode,
            )

            conv_idx = 0 if self.shared_conv else stage_idx
            prev_fpn = self.conv_layers[conv_idx](prev_fpn)
            prev_fpn = F.relu(
                self.norms[conv_idx](prev_fpn)
            )

            multiscale_features.append(prev_fpn)

        if len(multiscale_features) != 3:
            raise RuntimeError(
                f"forward_multiscale produced {len(multiscale_features)} "
                "features, expected 3."
            )

        pixel_feature_72, pixel_feature_144, pixel_feature_288 = (
            multiscale_features
        )

        if tuple(pixel_feature_72.shape[-2:]) != (72, 72):
            raise RuntimeError(
                f"pixel_feature_72 must be 72×72, "
                f"got {tuple(pixel_feature_72.shape[-2:])}."
            )
        if tuple(pixel_feature_144.shape[-2:]) != (144, 144):
            raise RuntimeError(
                f"pixel_feature_144 must be 144×144, "
                f"got {tuple(pixel_feature_144.shape[-2:])}."
            )
        if tuple(pixel_feature_288.shape[-2:]) != (288, 288):
            raise RuntimeError(
                f"pixel_feature_288 must be 288×288, "
                f"got {tuple(pixel_feature_288.shape[-2:])}."
            )

        return (
            pixel_feature_72,
            pixel_feature_144,
            pixel_feature_288,
        )

    def forward(self, backbone_feats: List[torch.Tensor]) -> torch.Tensor:
        return self.forward_multiscale(backbone_feats)[-1]


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation"""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, (
                "Specifying a dot product scorer without a presence head is likely a mistake"
            )

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1
        )

    def apply_prompt_cross_attention(
        self,
        encoder_hidden_states: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.cross_attend_prompt is None:
            raise RuntimeError(
                "Prompt cross-attention is required by the semantic pipeline."
            )

        update = self.cross_attn_norm(encoder_hidden_states)
        update = self.cross_attend_prompt(
            query=update,
            key=prompt,
            value=prompt,
            key_padding_mask=prompt_mask,
        )[0]
        return encoder_hidden_states + update

    def forward_semantic_pixel_pyramid(
        self,
        backbone_feats: List[torch.Tensor],
        image_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_logits: bool,
    ) -> Dict[str, torch.Tensor]:
        """Run the frozen Pixel Decoder and return all three scales.

        Returns a dict with "pixel_feature_72", "pixel_feature_144",
        "pixel_feature_288" and, when return_logits=True, "semantic_seg".
        This is the only interface for the original frozen branch; callers
        must wrap it in torch.no_grad().
        """
        backbone_visual_feats = self._prepare_pixel_decoder_input(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        (
            pixel_feature_72,
            pixel_feature_144,
            pixel_feature_288,
        ) = self.pixel_decoder.forward_multiscale(backbone_visual_feats)

        outputs = {
            "pixel_feature_72": pixel_feature_72,
            "pixel_feature_144": pixel_feature_144,
            "pixel_feature_288": pixel_feature_288,
        }

        if return_logits:
            outputs["semantic_seg"] = self.semantic_seg_head(
                pixel_feature_288
            )

        return outputs

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        image_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "semantic_seg": self.semantic_seg_head(pixel_embed),
        }
