# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr

from .losses.semantic_criterion import (
    HybridCriterion,
    SemanticCriterion,
    SemanticCriterionConfig,
)
from .models.adapters.semantic_adapter import (
    HybridSegAdapter,
    SemanticSegAdapter,
)
from .models.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .models.geometry_encoders import SequenceGeometryEncoder
from .models.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .models.model_misc import (
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .models.necks import Sam3DualViTDetNeck
from .models.openclip_image_encoder import OpenCLIPImageEncoder
from .models.openclip_text_encoder import OpenCLIPTextEncoder
from .models.position_encoding import PositionEmbeddingSine
from .models.sam3_image import Sam3Image
from .models.segmentor import SAM3Segmentor
from .models.task_modes import (
    TASK_MODE_HYBRID,
    TASK_MODE_SEMANTIC,
    normalize_task_mode,
)
from .models.text_encoder_ve import VETextEncoder
from .models.tokenizer_ve import SimpleTokenizer
from .models.vitdet import ViT
from .models.vl_combiner import SAM3VLBackbone

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_bpe_path(explicit_bpe_path=None):
    if explicit_bpe_path is not None:
        p = Path(explicit_bpe_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"BPE vocab file not found: {p}")
        return str(p)

    candidate_paths = [
        PROJECT_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        PROJECT_ROOT / "assets" / "clip" / "bpe_simple_vocab_16e6.txt.gz",
        PROJECT_ROOT / "configs" / "bpe_simple_vocab_16e6.txt.gz",
        PROJECT_ROOT / "configs" / "clip" / "bpe_simple_vocab_16e6.txt.gz",
    ]

    for p in candidate_paths:
        if p.exists():
            return str(p)

    tried = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "Cannot find bpe_simple_vocab_16e6.txt.gz. Tried:\n"
        f"{tried}\n"
        "Please pass `bpe_path` explicitly in config."
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
class OpenCLIPConfig:
    enabled: bool = False
    model_name: str = "ViT-L-14"
    pretrained: Optional[str] = None
    default_output: str = "feat_map"

    extra_token_templates: list[str] = field(default_factory=lambda: [
        "a remote sensing image of {}.",
        "an aerial image of {}.",
    ])
    num_extra_tokens: int = 2
    normalize_label_for_clip: bool = True
    clip_token_global_scale: float = 0.3


@dataclass
class CriterionConfig:
    ignore_index: int = 255

    bce_weight: float = 1.0
    dice_weight: float = 1.0
    presence_bce_weight: float = 1.0
    final_bce_weight: float = 0.4
    final_dice_weight: float = 0.5
    final_ce_weight: float = 1.0
	
    bce_class_balance_clamp_min: float = 0.2
    bce_class_balance_clamp_max: float = 5.0
    eps: float = 1e-6
	
    presence_pos_weight: float = 1.0

@dataclass
class AdapterConfig:
    presence_base: float = 0.5
    init_presence_modulation_alpha: float = 1.0

@dataclass
class SegmentorBuildConfig:
    task_mode: str = TASK_MODE_SEMANTIC

    bpe_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_from_hf: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_mode: bool = True
    compile: bool = False

    prompt_chunk_size: Optional[int] = None

    freeze_cfg: FreezeConfig = field(default_factory=FreezeConfig)
    openclip_cfg: OpenCLIPConfig = field(default_factory=OpenCLIPConfig)
    criterion_cfg: CriterionConfig = field(default_factory=CriterionConfig)
    adapter_cfg: AdapterConfig = field(default_factory=AdapterConfig)

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

    @staticmethod
    def get_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
        return dict(model.named_parameters())

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
        named_parameters = cls.get_named_parameters(model)

        for name in module_names:
            if name in named_modules:
                cls.set_requires_grad(named_modules[name], requires_grad)
                continue

            if name in named_parameters:
                named_parameters[name].requires_grad = requires_grad
                continue

            if strict:
                available_modules = "\n".join(sorted(named_modules.keys()))
                available_parameters = "\n".join(sorted(named_parameters.keys()))
                raise KeyError(
                    f"Unknown module/parameter name: {name}\n"
                    f"Available module names are:\n{available_modules}\n\n"
                    f"Available parameter names are:\n{available_parameters}"
                )


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
    def _create_vit_neck(cls, position_encoding, vit_backbone):
        return Sam3DualViTDetNeck(
            position_encoding=position_encoding,
            d_model=256,
            scale_factors=[4.0, 2.0, 1.0, 0.5],
            trunk=vit_backbone,
            add_sam2_neck=False,
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
    def _create_encoder_only_transformer() -> TransformerWrapper:
        encoder = SAM3ModelBuilder._create_transformer_encoder()
        return TransformerWrapper(
            encoder=encoder,
            decoder=None,
            d_model=256,
        )

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
            no_dec=True,
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
    def _coerce_openclip_cfg(obj) -> OpenCLIPConfig:
        if isinstance(obj, OpenCLIPConfig):
            return obj
        if obj is None:
            return OpenCLIPConfig()
        if isinstance(obj, dict):
            return OpenCLIPConfig(**dict(obj))
        raise TypeError(f"Unsupported openclip_cfg type: {type(obj)}")

    @staticmethod
    def _coerce_criterion_cfg(obj) -> CriterionConfig:
        if isinstance(obj, CriterionConfig):
            return obj
        if obj is None:
            return CriterionConfig()
        if isinstance(obj, dict):
            return CriterionConfig(**dict(obj))
        raise TypeError(f"Unsupported criterion_cfg type: {type(obj)}")

    @staticmethod
    def _coerce_adapter_cfg(obj) -> AdapterConfig:
        if isinstance(obj, AdapterConfig):
            return obj
        if obj is None:
            return AdapterConfig()
        if isinstance(obj, dict):
            return AdapterConfig(**dict(obj))
        raise TypeError(f"Unsupported adapter_cfg type: {type(obj)}")

    @classmethod
    def _normalize_build_cfg(cls, cfg: SegmentorBuildConfig) -> SegmentorBuildConfig:
        cfg.task_mode = normalize_task_mode(cfg.task_mode)
        cfg.openclip_cfg = cls._coerce_openclip_cfg(cfg.openclip_cfg)
        cfg.criterion_cfg = cls._coerce_criterion_cfg(cfg.criterion_cfg)
        cfg.adapter_cfg = cls._coerce_adapter_cfg(cfg.adapter_cfg)
        return cfg

    @staticmethod
    def _resolve_openclip_pretrained(pretrained: Optional[str]) -> Optional[str]:
        if pretrained is None:
            return None

        p = Path(str(pretrained)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"openclip_cfg.pretrained: expected a local checkpoint file, but got {pretrained!r}."
            )

        return str(p.resolve())

    @classmethod
    def validate_openclip_cfg(cls, openclip_cfg: OpenCLIPConfig) -> OpenCLIPConfig:
        openclip_cfg = cls._coerce_openclip_cfg(openclip_cfg)

        if openclip_cfg.enabled:
            _ = cls._resolve_openclip_pretrained(openclip_cfg.pretrained)

        return openclip_cfg

    @classmethod
    def _create_openclip_encoders(
        cls,
        openclip_cfg: OpenCLIPConfig,
    ) -> tuple[OpenCLIPTextEncoder, OpenCLIPImageEncoder]:
        import open_clip

        pretrained = cls._resolve_openclip_pretrained(openclip_cfg.pretrained)
        if pretrained is None:
            raise ValueError(
                "openclip_cfg.enabled=True, but openclip_cfg.pretrained is None."
            )

        clip_model = open_clip.create_model(
            model_name=openclip_cfg.model_name,
            pretrained=pretrained,
            precision="fp32",
            device="cpu",
        )
        clip_model.eval()

        tokenizer = open_clip.get_tokenizer(openclip_cfg.model_name)

        text_width = getattr(getattr(clip_model, "transformer", None), "width", None)
        if text_width is None:
            raise AttributeError(
                "Cannot infer OpenCLIP text width from clip_model.transformer.width."
            )

        text_encoder = OpenCLIPTextEncoder(
            tokenizer=tokenizer,
            token_embedding=clip_model.token_embedding,
            positional_embedding=clip_model.positional_embedding,
            transformer=clip_model.transformer,
            ln_final=clip_model.ln_final,
            text_projection=clip_model.text_projection,
            attn_mask=getattr(clip_model, "attn_mask", None),
            context_length=getattr(clip_model, "context_length", 77),
            width=text_width,
        )

        image_encoder = OpenCLIPImageEncoder(
            visual=clip_model.visual,
            default_output=openclip_cfg.default_output,
        )

        return text_encoder, image_encoder

    @staticmethod
    def _load_checkpoint(model, checkpoint_path: str):
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        if any(k.startswith("detector.") for k in ckpt.keys()):
            ckpt = {
                k.replace("detector.", ""): v
                for k, v in ckpt.items()
                if k.startswith("detector.")
            }

        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(
                f"Loaded {checkpoint_path} with missing keys={missing_keys} "
                f"and unexpected keys={unexpected_keys}"
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
    def build_semantic_core_model(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        cfg = cls._normalize_build_cfg(cfg)

        bpe_path = cfg.bpe_path
        if bpe_path is None:
            bpe_path = resolve_bpe_path(getattr(cfg, "bpe_path", None))

        compile_mode = "default" if cfg.compile else None

        position_encoding = cls._create_position_encoding(precompute_resolution=1008)
        vit_backbone = cls._create_vit_backbone(compile_mode=compile_mode)
        vit_neck = cls._create_vit_neck(position_encoding, vit_backbone)
        text_encoder = cls._create_text_encoder(bpe_path)
        backbone = cls._create_vl_backbone(vit_neck, text_encoder)

        clip_text_encoder = None
        clip_image_encoder = None
        openclip_cfg_for_model = None

        if cfg.openclip_cfg.enabled:
            clip_text_encoder, clip_image_encoder = cls._create_openclip_encoders(
                cfg.openclip_cfg
            )
            openclip_cfg_for_model = cfg.openclip_cfg

        transformer = cls._create_encoder_only_transformer()
        segmentation_head = cls._create_segmentation_head(compile_mode=compile_mode)
        input_geometry_encoder = cls._create_geometry_encoder()

        model = Sam3Image(
            backbone=backbone,
            transformer=transformer,
            input_geometry_encoder=input_geometry_encoder,
            segmentation_head=segmentation_head,
            num_feature_levels=1,
            o2m_mask_predict=True,
            dot_prod_scoring=None,
            use_instance_query=True,
            multimask_output=True,
            matcher=None,
            clip_image_encoder=clip_image_encoder,
            clip_text_encoder=clip_text_encoder,
            openclip_cfg=openclip_cfg_for_model,
            task_mode=TASK_MODE_SEMANTIC,
        )

        checkpoint_path = cfg.checkpoint_path
        if cfg.load_from_hf and checkpoint_path is None:
            checkpoint_path = cls.download_ckpt_from_hf()

        if checkpoint_path is not None:
            cls._load_checkpoint(model, checkpoint_path)

        return model

    @classmethod
    def build_adapter(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        cfg = cls._normalize_build_cfg(cfg)

        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return SemanticSegAdapter(
                presence_base=float(cfg.adapter_cfg.presence_base),
                init_presence_modulation_alpha=float(
                    cfg.adapter_cfg.init_presence_modulation_alpha
                ),
            )

        if cfg.task_mode == TASK_MODE_HYBRID:
            return HybridSegAdapter()

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_criterion(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        cfg = cls._normalize_build_cfg(cfg)

        if cfg.task_mode == TASK_MODE_SEMANTIC:
            criterion_cfg = SemanticCriterionConfig(
                ignore_index=int(cfg.criterion_cfg.ignore_index),
                bce_weight=float(cfg.criterion_cfg.bce_weight),
                dice_weight=float(cfg.criterion_cfg.dice_weight),
                presence_bce_weight=float(cfg.criterion_cfg.presence_bce_weight),
                final_bce_weight=float(cfg.criterion_cfg.final_bce_weight),
                final_dice_weight=float(cfg.criterion_cfg.final_dice_weight),
                final_ce_weight=float(cfg.criterion_cfg.final_ce_weight),
                bce_class_balance_clamp_min=float(cfg.criterion_cfg.bce_class_balance_clamp_min),
                bce_class_balance_clamp_max=float(cfg.criterion_cfg.bce_class_balance_clamp_max),
                eps=float(cfg.criterion_cfg.eps),
                presence_pos_weight=float(cfg.criterion_cfg.presence_pos_weight),
            )
            return SemanticCriterion(cfg=criterion_cfg)

        if cfg.task_mode == TASK_MODE_HYBRID:
            return HybridCriterion()

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_semantic_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        cfg = cls._normalize_build_cfg(cfg)
        cfg.openclip_cfg = cls.validate_openclip_cfg(cfg.openclip_cfg)

        core_model = cls.build_semantic_core_model(cfg)
        adapter = cls.build_adapter(cfg)

        model = SAM3Segmentor(
            core=core_model,
            adapter=adapter,
            task_mode=TASK_MODE_SEMANTIC,
        )

        model = model.to(cfg.device)
        cls.apply_freeze_cfg(model, cfg.freeze_cfg)

        if cfg.prompt_chunk_size is not None:
            model.core.prompt_chunk_size = int(cfg.prompt_chunk_size)
        else:
            model.core.prompt_chunk_size = None

        if cfg.eval_mode:
            model.eval()
        else:
            model.train()

        return model

    @classmethod
    def build_hybrid_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        cfg = cls._normalize_build_cfg(cfg)
        raise NotImplementedError(
            "Hybrid task mode is not implemented yet. "
            "The current codebase only supports semantic mode."
        )

    @classmethod
    def build_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        cfg = cls._normalize_build_cfg(cfg)

        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return cls.build_semantic_segmentor(cfg)

        if cfg.task_mode == TASK_MODE_HYBRID:
            return cls.build_hybrid_segmentor(cfg)

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_training_components(cls, cfg: SegmentorBuildConfig) -> tuple[nn.Module, nn.Module]:
        model = cls.build_segmentor(cfg)
        criterion = cls.build_criterion(cfg)
        return model, criterion


def build_segmentor_model(**kwargs) -> nn.Module:
    cfg = SegmentorBuildConfig(**kwargs)
    return SAM3ModelBuilder.build_segmentor(cfg)


def build_training_components(**kwargs) -> tuple[nn.Module, nn.Module]:
    cfg = SegmentorBuildConfig(**kwargs)
    return SAM3ModelBuilder.build_training_components(cfg)