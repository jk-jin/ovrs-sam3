# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from __future__ import annotations

from pathlib import Path
from typing import Optional, TypeVar

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr

from .config_dataclasses import (
    AdapterConfig,
    CheckpointManagerConfig,
    DynamicPromptConfig,
    FinalMixerConfig,
    FreezeConfig,
    LoggerHookConfig,
    LowResMixerConfig,
    OpenCLIPConfig,
    SegmentorBuildConfig,
    SemanticCriterionConfig,
    TrainerConfig,
    UpsamplerConfig,
    VisualizerConfig,
)
from .losses.semantic_criterion import (
    HybridCriterion,
    SemanticCriterion,
)
from .engine.checkpoint import CheckpointManager
from .engine.hooks import LoggerHook
from .engine.visualization import VisualizationManager
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

ConfigT = TypeVar("ConfigT")

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
    def _coerce_config(obj, config_cls: type[ConfigT], name: str) -> ConfigT:
        if isinstance(obj, config_cls):
            return obj
        if obj is None:
            return config_cls()
        if isinstance(obj, dict):
            return config_cls(**dict(obj))
        raise TypeError(f"Unsupported {name} type: {type(obj)}")

    @classmethod
    def _coerce_freeze_cfg(cls, obj) -> FreezeConfig:
        return cls._coerce_config(obj, FreezeConfig, "freeze_cfg")

    @classmethod
    def _coerce_openclip_cfg(cls, obj) -> OpenCLIPConfig:
        return cls._coerce_config(obj, OpenCLIPConfig, "openclip_cfg")

    @classmethod
    def _coerce_dynamic_prompt_cfg(cls, obj) -> DynamicPromptConfig:
        return cls._coerce_config(
            obj,
            DynamicPromptConfig,
            "dynamic_prompt_cfg",
        )

    @classmethod
    def _coerce_lowres_cfg(cls, obj) -> LowResMixerConfig:
        return cls._coerce_config(obj, LowResMixerConfig, "lowres_cfg")

    @classmethod
    def _coerce_upsampler_cfg(cls, obj) -> UpsamplerConfig:
        return cls._coerce_config(obj, UpsamplerConfig, "upsampler_cfg")

    @classmethod
    def _coerce_final_mixer_cfg(cls, obj) -> FinalMixerConfig:
        if obj is None:
            return FinalMixerConfig()

        if isinstance(obj, FinalMixerConfig):
            cfg = obj
        elif isinstance(obj, dict):
            raw = dict(obj)
            nested_coercers = {
                "dynamic_prompt_cfg": cls._coerce_dynamic_prompt_cfg,
                "lowres_cfg": cls._coerce_lowres_cfg,
                "upsampler_cfg": cls._coerce_upsampler_cfg,
            }
            for key, coerce_fn in nested_coercers.items():
                if key in raw:
                    raw[key] = coerce_fn(raw[key])
            cfg = FinalMixerConfig(**raw)
        else:
            raise TypeError(f"Unsupported final_mixer_cfg type: {type(obj)}")

        cfg.dynamic_prompt_cfg = cls._coerce_dynamic_prompt_cfg(
            cfg.dynamic_prompt_cfg
        )
        cfg.lowres_cfg = cls._coerce_lowres_cfg(cfg.lowres_cfg)
        cfg.upsampler_cfg = cls._coerce_upsampler_cfg(cfg.upsampler_cfg)
        return cfg

    @classmethod
    def _coerce_criterion_cfg(cls, obj) -> SemanticCriterionConfig:
        return cls._coerce_config(obj, SemanticCriterionConfig, "criterion_cfg")

    @classmethod
    def _coerce_adapter_cfg(cls, obj) -> AdapterConfig:
        return cls._coerce_config(obj, AdapterConfig, "adapter_cfg")

    @classmethod
    def _normalize_build_cfg(cls, cfg: SegmentorBuildConfig) -> SegmentorBuildConfig:
        cfg.task_mode = normalize_task_mode(cfg.task_mode)
        cfg.freeze_cfg = cls._coerce_freeze_cfg(cfg.freeze_cfg)
        cfg.openclip_cfg = cls._coerce_openclip_cfg(cfg.openclip_cfg)
        cfg.final_mixer_cfg = cls._coerce_final_mixer_cfg(cfg.final_mixer_cfg)
        cfg.criterion_cfg = cls._coerce_criterion_cfg(cfg.criterion_cfg)
        cfg.adapter_cfg = cls._coerce_adapter_cfg(cfg.adapter_cfg)
        return cfg

    @classmethod
    def build_config(cls, **kwargs) -> SegmentorBuildConfig:
        cfg = SegmentorBuildConfig(**kwargs)
        cfg = cls._normalize_build_cfg(cfg)
        cfg.openclip_cfg = cls.validate_openclip_cfg(cfg.openclip_cfg)
        cfg.final_mixer_cfg = cls.validate_final_mixer_cfg(cfg.final_mixer_cfg)
        return cfg

    @staticmethod
    def _require_dict(obj, name: str) -> dict:
        if not isinstance(obj, dict):
            raise TypeError(f"{name} must be a dict, got {type(obj)}.")
        return dict(obj)

    @staticmethod
    def resolve_work_dir(cfg, work_dir_override: Optional[str] = None) -> str:
        if work_dir_override is not None:
            return str(work_dir_override)
        return str(cfg.get("work_dir", "./work_dirs/default"))

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

        if not openclip_cfg.enabled:
            return openclip_cfg

        _ = cls._resolve_openclip_pretrained(openclip_cfg.pretrained)

        if openclip_cfg.num_prompt_templates < 0:
            raise ValueError(
                "openclip_cfg.num_prompt_templates must be non-negative, "
                f"got {openclip_cfg.num_prompt_templates}."
            )

        if openclip_cfg.num_prompt_templates > len(openclip_cfg.prompt_templates):
            raise ValueError(
                "openclip_cfg.num_prompt_templates cannot exceed "
                "len(openclip_cfg.prompt_templates)."
            )
        return openclip_cfg

    @classmethod
    def validate_final_mixer_cfg(cls, cfg: FinalMixerConfig) -> FinalMixerConfig:
        cfg = cls._coerce_final_mixer_cfg(cfg)

        if not cfg.enabled:
            raise ValueError(
                "final_mixer_cfg.enabled=False is not supported by the current "
                "semantic training path."
            )

        if cfg.fusion_layers <= 0:
            raise ValueError(
                "final_mixer_cfg.fusion_layers must be positive, "
                f"got {cfg.fusion_layers}."
            )

        if cfg.num_heads <= 0:
            raise ValueError(
                "final_mixer_cfg.num_heads must be positive, "
                f"got {cfg.num_heads}."
            )

        dynamic_cfg = cfg.dynamic_prompt_cfg
        if dynamic_cfg.tokens_per_template <= 0:
            raise ValueError(
                "dynamic_prompt_cfg.tokens_per_template must be positive, "
                f"got {dynamic_cfg.tokens_per_template}."
            )

        lowres_cfg = cfg.lowres_cfg
        if lowres_cfg.window_size <= 0:
            raise ValueError(
                "final_mixer_cfg.lowres_cfg.window_size must be positive."
            )
        if not 0 <= lowres_cfg.shift_size < lowres_cfg.window_size:
            raise ValueError(
                "final_mixer_cfg.lowres_cfg.shift_size must satisfy "
                "0 <= shift_size < window_size."
            )

        upsampler_cfg = cfg.upsampler_cfg

        if len(upsampler_cfg.decoder_channels) < 2:
            raise ValueError(
                "upsampler_cfg.decoder_channels must contain at least "
                "one input channel and one decoder stage channel."
            )

        expected_stages = len(upsampler_cfg.decoder_channels) - 1

        if len(upsampler_cfg.sam_guidance_channels) != expected_stages:
            raise ValueError(
                "upsampler_cfg.sam_guidance_channels length must match "
                f"the number of upsample stages. Expected {expected_stages}, "
                f"got {len(upsampler_cfg.sam_guidance_channels)}."
            )

        if len(upsampler_cfg.score_channels) != expected_stages:
            raise ValueError(
                "upsampler_cfg.score_channels length must match "
                f"the number of upsample stages. Expected {expected_stages}, "
                f"got {len(upsampler_cfg.score_channels)}."
            )

        return cfg

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
            image_encoder_mode=openclip_cfg.image_encoder_mode,
            maskclip_skip_last_layers=openclip_cfg.maskclip_skip_last_layers,
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
        bpe_path = resolve_bpe_path(cfg.bpe_path)
        compile_mode = "default" if cfg.compile else None

        position_encoding = cls._create_position_encoding(precompute_resolution=1008)
        vit_backbone = cls._create_vit_backbone(compile_mode=compile_mode)
        vit_neck = cls._create_vit_neck(position_encoding, vit_backbone)
        text_encoder = cls._create_text_encoder(bpe_path)
        backbone = cls._create_vl_backbone(vit_neck, text_encoder)

        clip_text_encoder = None
        clip_image_encoder = None
        clip_prompt_templates: list[str] = []
        num_clip_prompt_templates = 0

        if cfg.openclip_cfg.enabled:
            clip_text_encoder, clip_image_encoder = cls._create_openclip_encoders(
                cfg.openclip_cfg
            )
            num_clip_prompt_templates = int(cfg.openclip_cfg.num_prompt_templates)
            clip_prompt_templates = list(
                cfg.openclip_cfg.prompt_templates[:num_clip_prompt_templates]
            )

        transformer = cls._create_encoder_only_transformer()
        segmentation_head = cls._create_segmentation_head(compile_mode=compile_mode)
        input_geometry_encoder = cls._create_geometry_encoder()

        final_cfg = cfg.final_mixer_cfg
        lowres_cfg = final_cfg.lowres_cfg
        upsampler_cfg = final_cfg.upsampler_cfg
        dynamic_cfg = final_cfg.dynamic_prompt_cfg

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
            clip_prompt_templates=clip_prompt_templates,
            num_clip_prompt_templates=num_clip_prompt_templates,
            normalize_label_for_clip=bool(cfg.openclip_cfg.normalize_label_for_clip),
            final_mixer_fusion_layers=int(final_cfg.fusion_layers),
            final_mixer_num_heads=int(final_cfg.num_heads),
            final_mixer_dropout=float(final_cfg.dropout),
            lowres_hidden_dim=int(lowres_cfg.hidden_dim),
            lowres_score_embed_dim=int(lowres_cfg.score_embed_dim),
            lowres_window_size=int(lowres_cfg.window_size),
            lowres_shift_size=int(lowres_cfg.shift_size),
            lowres_score_floor=float(lowres_cfg.score_floor),
            lowres_lambda_score=float(lowres_cfg.lambda_score),
            dynamic_tokens_per_template=int(dynamic_cfg.tokens_per_template),
            upsampler_class_chunk_size=int(upsampler_cfg.class_chunk_size),
            upsampler_decoder_channels=list(upsampler_cfg.decoder_channels),
            upsampler_sam_guidance_channels=list(upsampler_cfg.sam_guidance_channels),
            upsampler_score_channels=list(upsampler_cfg.score_channels),
            upsampler_score_input=str(upsampler_cfg.score_input),
            upsampler_upsample_mode=str(upsampler_cfg.upsample_mode),
            upsampler_norm=str(upsampler_cfg.norm),
            upsampler_act=str(upsampler_cfg.act),
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
        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return SemanticSegAdapter()

        if cfg.task_mode == TASK_MODE_HYBRID:
            return HybridSegAdapter()

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_criterion(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return SemanticCriterion(cfg=cfg.criterion_cfg)

        if cfg.task_mode == TASK_MODE_HYBRID:
            return HybridCriterion()

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_semantic_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        core_model = cls.build_semantic_core_model(cfg)
        adapter = cls.build_adapter(cfg)

        model = SAM3Segmentor(
            core=core_model,
            adapter=adapter,
            task_mode=TASK_MODE_SEMANTIC,
        )

        model = model.to(cfg.device)
        cls.apply_freeze_cfg(model, cfg.freeze_cfg)

        model.core.prompt_chunk_size = (
            None if cfg.prompt_chunk_size is None else int(cfg.prompt_chunk_size)
        )

        if cfg.eval_mode:
            model.eval()
        else:
            model.train()

        return model

    @classmethod
    def build_hybrid_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        raise NotImplementedError(
            "Hybrid task mode is not implemented yet. "
            "The current codebase only supports semantic mode."
        )

    @classmethod
    def build_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return cls.build_semantic_segmentor(cfg)

        if cfg.task_mode == TASK_MODE_HYBRID:
            return cls.build_hybrid_segmentor(cfg)

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_training_components(
        cls,
        cfg: SegmentorBuildConfig,
    ) -> tuple[nn.Module, nn.Module]:
        model = cls.build_segmentor(cfg)
        criterion = cls.build_criterion(cfg)
        return model, criterion

    @classmethod
    def build_trainer_config_from_cfg(
        cls,
        cfg,
        work_dir: str,
        auto_resume: bool = False,
    ) -> TrainerConfig:
        train_cfg = cls._require_dict(cfg.train_cfg, "train_cfg")
        train_cfg["save_dir"] = str(work_dir)
        train_cfg["auto_resume"] = bool(
            auto_resume or train_cfg.get("auto_resume", False)
        )
        train_cfg["tta_cfg"] = cfg.get("tta_cfg", None)
        train_cfg["eval_cfg"] = cfg.get("eval_cfg", None)
        return TrainerConfig(**train_cfg)

    @classmethod
    def build_checkpoint_manager(
        cls,
        trainer_cfg: TrainerConfig,
    ) -> CheckpointManager:
        checkpoint_cfg = CheckpointManagerConfig(
            save_dir=str(trainer_cfg.save_dir),
            monitor=str(trainer_cfg.monitor),
            mode=str(trainer_cfg.monitor_mode),
            max_keep=int(trainer_cfg.max_keep_ckpts),
            save_latest=True,
            save_best=True,
        )
        return CheckpointManager(checkpoint_cfg)

    @classmethod
    def build_hooks_from_cfg(cls, cfg) -> list:
        default_hooks = cls._require_dict(cfg.default_hooks, "default_hooks")
        logger_cfg = LoggerHookConfig(
            **cls._require_dict(default_hooks["logger"], "default_hooks.logger")
        )
        return [LoggerHook(logger_cfg)]

    @classmethod
    def build_visualizer_from_cfg(
        cls,
        cfg,
        work_dir: str,
    ) -> Optional[VisualizationManager]:
        visualization_cfg = cfg.get("visualization", None)
        if visualization_cfg is None:
            return None

        visualizer_cfg = VisualizerConfig(
            **cls._require_dict(visualization_cfg, "visualization")
        )
        if not visualizer_cfg.enabled:
            return None

        save_dir = Path(visualizer_cfg.save_dir)
        if not save_dir.is_absolute():
            visualizer_cfg.save_dir = str(Path(work_dir) / save_dir)

        return VisualizationManager(
            visualizer_cfg,
            eval_cfg=cfg.get("eval_cfg", None),
        )

    @classmethod
    def build_train_runtime_components(
        cls,
        cfg,
        work_dir_override: Optional[str] = None,
        auto_resume: bool = False,
    ) -> tuple[
        str,
        TrainerConfig,
        list,
        Optional[VisualizationManager],
        CheckpointManager,
    ]:
        work_dir = cls.resolve_work_dir(
            cfg,
            work_dir_override=work_dir_override,
        )
        trainer_cfg = cls.build_trainer_config_from_cfg(
            cfg,
            work_dir=work_dir,
            auto_resume=auto_resume,
        )
        return (
            work_dir,
            trainer_cfg,
            cls.build_hooks_from_cfg(cfg),
            cls.build_visualizer_from_cfg(cfg, work_dir=work_dir),
            cls.build_checkpoint_manager(trainer_cfg),
        )


def build_segmentor_model(**kwargs) -> nn.Module:
    cfg = SAM3ModelBuilder.build_config(**kwargs)
    return SAM3ModelBuilder.build_segmentor(cfg)


def build_training_components(**kwargs) -> tuple[nn.Module, nn.Module]:
    cfg = SAM3ModelBuilder.build_config(**kwargs)
    return SAM3ModelBuilder.build_training_components(cfg)


def build_train_runtime_components(
    cfg,
    work_dir_override: Optional[str] = None,
    auto_resume: bool = False,
):
    return SAM3ModelBuilder.build_train_runtime_components(
        cfg,
        work_dir_override=work_dir_override,
        auto_resume=auto_resume,
    )