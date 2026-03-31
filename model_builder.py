# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr

from .models.decoder import TransformerDecoder, TransformerDecoderLayer
from .models.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .models.geometry_encoders import SequenceGeometryEncoder
from .models.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .models.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .models.necks import Sam3DualViTDetNeck
from .models.position_encoding import PositionEmbeddingSine
from .models.sam3_image import Sam3Image
from .models.adapters.semantic_adapter import QueryMaskSemanticAdapter
from .models.segmentor import SAM3Segmentor
from .models.text_encoder_ve import VETextEncoder
from .models.tokenizer_ve import SimpleTokenizer
from .models.vitdet import ViT
from .models.vl_combiner import SAM3VLBackbone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def resolve_bpe_path(explicit_bpe_path=None):
    if explicit_bpe_path is not None:
        p = Path(explicit_bpe_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f'BPE vocab file not found: {p}')
        return str(p)

    candidate_paths = [
        PROJECT_ROOT / 'assets' / 'bpe_simple_vocab_16e6.txt.gz',
        PROJECT_ROOT / 'assets' / 'clip' / 'bpe_simple_vocab_16e6.txt.gz',
        PROJECT_ROOT / 'configs' / 'bpe_simple_vocab_16e6.txt.gz',
        PROJECT_ROOT / 'configs' / 'clip' / 'bpe_simple_vocab_16e6.txt.gz',
    ]

    for p in candidate_paths:
        if p.exists():
            return str(p)

    tried = '\n'.join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        'Cannot find bpe_simple_vocab_16e6.txt.gz. Tried:\n'
        f'{tried}\n'
        'Please pass `bpe_path` explicitly in config.'
    )


def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


@dataclass
class FreezeConfig:
    train_adapters_only: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)


@dataclass
class SegmentorBuildConfig:
    bpe_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_from_hf: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_mode: bool = True
    compile: bool = False
    semantic_topk: Optional[int] = 20
    semantic_aggregation: str = "weighted_sum"
    freeze_cfg: FreezeConfig = field(default_factory=FreezeConfig)


class FrozenModuleMixin:
    @staticmethod
    def set_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def set_model_requires_grad(model: nn.Module, requires_grad: bool) -> None:
        for p in model.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def get_named_modules(model: nn.Module) -> dict[str, nn.Module]:
        return dict(model.named_modules())

    @classmethod
    def set_modules_requires_grad(
        cls,
        model: nn.Module,
        module_names: list[str],
        requires_grad: bool,
        strict: bool = True,
    ) -> None:
        if not module_names:
            return

        named_modules = cls.get_named_modules(model)

        for name in module_names:
            if name not in named_modules:
                if strict:
                    available = '\n'.join(sorted(named_modules.keys()))
                    raise KeyError(
                        f'Unknown module name: {name}\n'
                        f'Available module names are:\n{available}'
                    )
                continue
            cls.set_requires_grad(named_modules[name], requires_grad)


class SAM3ModelBuilder(FrozenModuleMixin):
    @staticmethod
    def _create_position_encoding(precompute_resolution=None):
        return PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=precompute_resolution,
        )

    @staticmethod
    def _create_vit_backbone(compile_mode=None):
        return ViT(
            img_size=1008,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=32,
            num_heads=16,
            mlp_ratio=4.625,
            norm_layer="LayerNorm",
            drop_path_rate=0.1,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )

    @classmethod
    def _create_vit_neck(cls, position_encoding, vit_backbone, enable_inst_interactivity=False):
        return Sam3DualViTDetNeck(
            position_encoding=position_encoding,
            d_model=256,
            scale_factors=[4.0, 2.0, 1.0, 0.5],
            trunk=vit_backbone,
            add_sam2_neck=enable_inst_interactivity,
        )

    @staticmethod
    def _create_text_encoder(bpe_path: str) -> VETextEncoder:
        tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        return VETextEncoder(
            tokenizer=tokenizer,
            d_model=256,
            width=1024,
            heads=16,
            layers=24,
        )

    @staticmethod
    def _create_vl_backbone(vit_neck, text_encoder):
        return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)

    @staticmethod
    def _create_transformer_encoder() -> TransformerEncoderFusion:
        encoder_layer = TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False,
            pre_norm=True,
            self_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=True,
            ),
            cross_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=True,
            ),
        )
        return TransformerEncoderFusion(
            layer=encoder_layer,
            num_layers=6,
            d_model=256,
            num_feature_levels=1,
            frozen=False,
            use_act_checkpoint=True,
            add_pooled_text_to_img_feat=False,
            pool_text_with_mask=True,
        )

    @staticmethod
    def _create_transformer_decoder() -> TransformerDecoder:
        decoder_layer = TransformerDecoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            cross_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
            ),
            n_heads=8,
            use_text_cross_attention=True,
        )
        return TransformerDecoder(
            layer=decoder_layer,
            num_layers=6,
            num_queries=200,
            return_intermediate=True,
            box_refine=True,
            num_o2m_queries=0,
            dac=True,
            boxRPB="log",
            d_model=256,
            frozen=False,
            interaction_layer=None,
            dac_use_selfatt_ln=True,
            resolution=1008,
            stride=14,
            use_act_checkpoint=True,
            presence_token=True,
        )

    @staticmethod
    def _create_sam3_transformer() -> TransformerWrapper:
        encoder = SAM3ModelBuilder._create_transformer_encoder()
        decoder = SAM3ModelBuilder._create_transformer_decoder()
        return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)

    @staticmethod
    def _create_dot_product_scoring():
        prompt_mlp = MLP(
            input_dim=256,
            hidden_dim=2048,
            output_dim=256,
            num_layers=2,
            dropout=0.1,
            residual=True,
            out_norm=nn.LayerNorm(256),
        )
        return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)

    @staticmethod
    def _create_segmentation_head(compile_mode=None):
        pixel_decoder = PixelDecoder(
            num_upsampling_stages=3,
            interpolation_mode="nearest",
            hidden_dim=256,
            compile_mode=compile_mode,
        )
        cross_attend_prompt = MultiheadAttention(
            num_heads=8,
            dropout=0,
            embed_dim=256,
        )
        return UniversalSegmentationHead(
            hidden_dim=256,
            upsampling_stages=3,
            aux_masks=False,
            presence_head=False,
            dot_product_scorer=None,
            act_ckpt=True,
            cross_attend_prompt=cross_attend_prompt,
            pixel_decoder=pixel_decoder,
        )

    @classmethod
    def _create_geometry_encoder(cls):
        geo_pos_enc = cls._create_position_encoding()
        geo_layer = TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            pre_norm=True,
            self_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=False,
            ),
            pos_enc_at_cross_attn_queries=False,
            pos_enc_at_cross_attn_keys=True,
            cross_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=False,
            ),
        )
        return SequenceGeometryEncoder(
            pos_enc=geo_pos_enc,
            encode_boxes_as_points=False,
            points_direct_project=True,
            points_pool=True,
            points_pos_enc=True,
            boxes_direct_project=True,
            boxes_pool=True,
            boxes_pos_enc=True,
            d_model=256,
            num_layers=3,
            layer=geo_layer,
            use_act_ckpt=True,
            add_cls=True,
            add_post_encode_proj=True,
        )

    @staticmethod
    def _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
    ):
        common_params = {
            "backbone": backbone,
            "transformer": transformer,
            "input_geometry_encoder": input_geometry_encoder,
            "segmentation_head": segmentation_head,
            "num_feature_levels": 1,
            "o2m_mask_predict": True,
            "dot_prod_scoring": dot_prod_scoring,
            "use_instance_query": False,
            "multimask_output": True,
            "matcher": None,
        }
        return Sam3Image(**common_params)

    @staticmethod
    def _load_checkpoint(model, checkpoint_path: str):
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        sam3_image_ckpt = {
            k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
        }
        missing_keys, unexpected_keys = model.load_state_dict(sam3_image_ckpt, strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(
                f"Loaded {checkpoint_path} with missing keys={missing_keys} and unexpected keys={unexpected_keys}"
            )

    @staticmethod
    def download_ckpt_from_hf():
        model_id = "facebook/sam3"
        _ = hf_hub_download(repo_id=model_id, filename="config.json")
        return hf_hub_download(repo_id=model_id, filename="sam3.pt")

    @classmethod
    def apply_freeze_cfg(cls, model: nn.Module, freeze_cfg: FreezeConfig) -> None:
        if freeze_cfg.train_adapters_only:
            cls.set_model_requires_grad(model, False)
            cls.set_modules_requires_grad(
                model,
                freeze_cfg.trainable_modules,
                True,
                strict=True,
            )
        else:
            cls.set_model_requires_grad(model, True)
            cls.set_modules_requires_grad(
                model,
                freeze_cfg.frozen_modules,
                False,
                strict=True,
            )

    @classmethod
    def build_sam3_image_model(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        bpe_path = cfg.bpe_path
        if bpe_path is None:
            bpe_path = resolve_bpe_path(getattr(cfg, 'bpe_path', None))

        compile_mode = "default" if cfg.compile else None
        position_encoding = cls._create_position_encoding(precompute_resolution=1008)
        vit_backbone = cls._create_vit_backbone(compile_mode=compile_mode)
        vit_neck = cls._create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False)
        text_encoder = cls._create_text_encoder(bpe_path)
        backbone = cls._create_vl_backbone(vit_neck, text_encoder)
        transformer = cls._create_sam3_transformer()
        dot_prod_scoring = cls._create_dot_product_scoring()
        segmentation_head = cls._create_segmentation_head(compile_mode=compile_mode)
        input_geometry_encoder = cls._create_geometry_encoder()

        model = cls._create_sam3_model(
            backbone=backbone,
            transformer=transformer,
            input_geometry_encoder=input_geometry_encoder,
            segmentation_head=segmentation_head,
            dot_prod_scoring=dot_prod_scoring,
        )

        checkpoint_path = cfg.checkpoint_path
        if cfg.load_from_hf and checkpoint_path is None:
            checkpoint_path = cls.download_ckpt_from_hf()
        if checkpoint_path is not None:
            cls._load_checkpoint(model, checkpoint_path)

        return model

    @classmethod
    def build_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        sam3_image_model = cls.build_sam3_image_model(cfg)

        # semantic-only: make intent explicit
        sam3_image_model.matcher = None

        model = SAM3Segmentor(
            core=sam3_image_model,
            semantic_adapter=QueryMaskSemanticAdapter(
                topk=cfg.semantic_topk,
                aggregation=cfg.semantic_aggregation,
            ),
        )

        model = model.to(cfg.device)
        cls.apply_freeze_cfg(model, cfg.freeze_cfg)

        if hasattr(cfg, "prompt_chunk_size"):
            model.core.prompt_chunk_size = cfg.prompt_chunk_size

        if cfg.eval_mode:
            model.eval()
        else:
            model.train()

        return model


def build_segmentor_model(**kwargs) -> nn.Module:
    cfg = SegmentorBuildConfig(**kwargs)
    return SAM3ModelBuilder.build_segmentor(cfg)
