"""CPU tests for fixed image Q/K guidance; no pretrained weights required."""
import ast
from contextlib import ExitStack
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from config_dataclasses import EncoderRefinerConfig, SegmentorBuildConfig
from models.encoder_refiner import ClassConditionedEncoderRefiner
from models.encoder_refiner_attention import (
    ClassScoreAttention, EncoderRefinerLayer, LocalScoreAttention,
    WindowScoreAttention, apply_layer_norm_bcdhw,
)
from models.openclip_image_encoder import OpenCLIPImageEncoder
from test_encoder_refiner_attention import FakeTextEncoder, intra_steps


ROOT = Path(__file__).resolve().parents[1]


def load_methods(path, name, methods, env, bases=None):
    """Execute actual method bodies without importing heavyweight SAM3 modules."""
    parsed = ast.parse((ROOT / path).read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if bases is not None:
        node.bases = bases
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    ), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), env)
    return env[name]


def lightweight_sam_class():
    env = dict(torch=torch, nn=nn, ClassConditionedEncoderRefiner=ClassConditionedEncoderRefiner,
               TASK_MODE_SEMANTIC="semantic", normalize_task_mode=lambda x: x,
               OUTPUT_KEYS=SimpleNamespace(clip_mid_features="clip_mid_features", encoder_features="encoder_features"))
    return load_methods("models/sam3_image.py", "Sam3Image", {
        "__init__", "_infer_clip_text_dim", "_infer_clip_image_dim",
        "_build_clip_image_cache", "_run_encoder_refiner_lowres", "run_encoder_refiner_lowres_from_cache",
    }, env), env


class GuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(31)

    def make_refiner(self, score_mode="window", feature_mode="window", dropout=0):
        return ClassConditionedEncoderRefiner(
            FakeTextEncoder(), clip_dim=8, clip_native_dim=5, fusion_layers=4,
            prompt_templates=["{}"] * 64, use_checkpoint=False, dropout=dropout,
            score_intra_attn_type=score_mode, feature_intra_attn_type=feature_mode,
        )

    def inputs(self):
        return dict(
            encoder_features_72=torch.randn(1, 2, 256, 72, 72),
            clip_image_feat_map=torch.randn(1, 8, 36, 36), class_names=["building", "road"],
            clip_mid_features=[torch.randn(1, 5, 36, 36, requires_grad=True) for _ in range(2)],
            clip_mid_layer_indices=(15, 7), sam_fpn_72=torch.randn(1, 256, 72, 72),
        )

    def test_fixed_image_guidance_is_prepared_once_and_routes_to_the_correct_stream(self):
        model = self.make_refiner()
        data = self.inputs()
        seen, mid_inputs = [], []
        with ExitStack() as stack:
            stack.enter_context(model.clip_mid_proj.register_forward_pre_hook(lambda m, args: mid_inputs.append(args[0].clone())))
            for layer in model.layers:
                stack.enter_context(layer.register_forward_pre_hook(lambda m, args, kwargs: seen.append(kwargs), with_kwargs=True))
            model(**data)
        self.assertEqual(len(mid_inputs), 1)
        torch.testing.assert_close(mid_inputs[0], torch.cat(data["clip_mid_features"][::-1], dim=1))
        projected = model.clip_mid_proj(mid_inputs[0])
        fpn = F.interpolate(data["sam_fpn_72"], size=(36, 36), mode="bilinear", align_corners=False)
        for layer, supplied in zip(model.layers, seen):
            self.assertEqual(layer.class_attn.q_proj.in_features, 512)
            for attention in intra_steps(layer):
                self.assertEqual(attention.q_proj.in_features, 768)
                self.assertEqual(attention.v_proj.in_features, 256)
            for cls in range(2):
                torch.testing.assert_close(supplied["clip_mid_36"][:, cls], projected)
                torch.testing.assert_close(supplied["sam_fpn_36"][:, cls], fpn)

    def test_image_guidance_and_streams_match_checkpoint_with_dropout(self):
        plain = self.make_refiner("local_3x3", "window", dropout=0.1).train()
        checked = deepcopy(plain)
        checked.use_checkpoint = True
        data, results, middle_grads = self.inputs(), [], []
        target = torch.randn(1, 2, 256, 36, 36)
        for model in (plain, checked):
            inputs = dict(data)
            inputs["clip_mid_features"] = [x.detach().clone().requires_grad_() for x in data["clip_mid_features"]]
            torch.manual_seed(72)
            out = model(**inputs)
            (out["refiner_features_36"] * target).mean().backward()
            results.append(out)
            middle_grads.append([x.grad for x in inputs["clip_mid_features"]])
            for value in (
                model.clip_mid_proj.weight.grad, model.layers[0].ffn_fc1_feature.weight.grad,
                model.layers[0].ffn_fc1_score.weight.grad,
                model.clip_score_embed.clip_text_encoder.templates.grad,
            ):
                self.assertTrue(torch.isfinite(value).all())
                self.assertGreater(value.abs().sum().item(), 0)
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key])
        for first, second in zip(middle_grads[0], middle_grads[1]):
            torch.testing.assert_close(first, second)
            self.assertGreater(first.abs().sum().item(), 0)
        for (name, first), (_, second) in zip(plain.named_parameters(), checked.named_parameters()):
            if first.grad is None:
                self.assertIsNone(second.grad, name)
            else:
                torch.testing.assert_close(first.grad, second.grad)

    def test_required_guidance_and_middle_indices_are_validated(self):
        model = self.make_refiner()
        for missing, message in (("clip_mid_features", "CLIP middle"), ("sam_fpn_72", "SAM FPN")):
            data = self.inputs()
            data.pop(missing)
            with self.assertRaisesRegex(ValueError, message):
                model(**data)
        data = self.inputs()
        data["clip_mid_layer_indices"] = (3, 7)
        with self.assertRaisesRegex(ValueError, "indices 7 and 15"):
            model(**data)

    def test_clip_middle_graph_survives_encoder_and_cache_but_validation_is_detached(self):
        Sam, _ = lightweight_sam_class()
        visual = nn.Module()
        visual.width, visual.output_dim = 5, 8
        visual.transformer = nn.Module()
        visual.transformer.resblocks = nn.ModuleList([nn.Identity() for _ in range(17)])
        visual.native_map = nn.Parameter(torch.randn(1, 5, 36, 36))
        encoder = OpenCLIPImageEncoder(visual, intermediate_layers=(15, 7))
        encoder.enable_grad = True
        visual.native_map.requires_grad_(True)
        sam = Sam.__new__(Sam)
        nn.Module.__init__(sam)
        sam.clip_image_encoder = encoder

        def dense_forward(images, return_intermediate):
            native = visual.native_map * 2
            dense = native.mean(1, keepdim=True).expand(-1, 8, -1, -1).flatten(2).transpose(1, 2)
            return dense, (36, 36), [native, native * 3]

        with (
            patch.object(encoder, "_forward_full_vit_dense_tokens", side_effect=dense_forward),
            patch.object(encoder, "preprocess_raw_images", return_value=torch.zeros(1, 3, 36, 36)),
        ):
            cache = sam._build_clip_image_cache(SimpleNamespace(raw_images=[torch.zeros(3, 36, 36)]), torch.device("cpu"))
            self.assertEqual(cache["clip_mid_layer_indices"], (7, 15))
            self.assertTrue(all(x.requires_grad for x in cache["clip_mid_features"]))
            sum(x.square().mean() for x in cache["clip_mid_features"]).backward()
            self.assertGreater(visual.native_map.grad.abs().sum().item(), 0)
            with torch.no_grad():
                cache = sam._build_clip_image_cache(SimpleNamespace(raw_images=[torch.zeros(3, 36, 36)]), torch.device("cpu"))
                self.assertFalse(cache["clip_image_feat_map_native"].requires_grad)
                self.assertTrue(all(not x.requires_grad for x in cache["clip_mid_features"]))
            encoder.enable_grad = False
            cache = sam._build_clip_image_cache(SimpleNamespace(raw_images=[torch.zeros(3, 36, 36)]), torch.device("cpu"))
            self.assertTrue(all(not x.requires_grad for x in cache["clip_mid_features"]))

    def test_lowres_cache_uses_fixed_inputs_and_detaches_debug_outputs(self):
        Sam, _ = lightweight_sam_class()
        sam = Sam.__new__(Sam)
        nn.Module.__init__(sam)
        sam.encoder_refiner = self.make_refiner()
        data = self.inputs()
        cache = dict(
            cross_attended_encoder_features_72=data["encoder_features_72"], clip_image_feat_map=data["clip_image_feat_map"],
            clip_mid_features=data["clip_mid_features"], clip_mid_layer_indices=data["clip_mid_layer_indices"],
            class_names=data["class_names"], backbone_fpn=[torch.empty(0), torch.empty(0), data["sam_fpn_72"]],
        )
        result = sam.run_encoder_refiner_lowres_from_cache(cache, SimpleNamespace(find_text_batch=data["class_names"]), return_debug=True)
        self.assertTrue(result["refiner_features_36"].requires_grad)
        for key in ("score_embed_36", "clip_score_embed_36", "clip_score_maps_36", "template_clip_text"):
            self.assertFalse(result[key].requires_grad)
        self.assertTrue(all(not x.requires_grad for x in result["clip_mid_features"]))
        (result["refiner_features_36"] * torch.randn_like(result["refiner_features_36"])).mean().backward()
        for middle in data["clip_mid_features"]:
            self.assertGreater(middle.grad.abs().sum().item(), 0)

    def test_config_builder_preserves_spatial_modes_and_has_no_guidance_switches(self):
        from dataclasses import fields
        self.assertFalse(any("_use_" in item.name for item in fields(EncoderRefinerConfig)))
        Sam, env = lightweight_sam_class()
        env.update(EncoderRefinerConfig=EncoderRefinerConfig, resolve_bpe_path=lambda x: x)
        Builder = load_methods("model_builder.py", "SAM3ModelBuilder", {
            "_coerce_config", "_coerce_encoder_refiner_cfg", "validate_encoder_refiner_cfg", "build_semantic_core_model",
        }, env, bases=[])
        for name in ("_create_position_encoding", "_create_vit_backbone", "_create_vit_neck", "_create_text_encoder",
                     "_create_vl_backbone", "_create_segmentation_head", "_create_geometry_encoder"):
            setattr(Builder, name, staticmethod(lambda *args, **kwargs: nn.Identity()))
        transformer = nn.Identity()
        transformer.d_model = 256
        Builder._create_encoder_only_transformer = staticmethod(lambda: transformer)
        text, image = FakeTextEncoder(), nn.Identity()
        text.output_dim, image.output_dim, image.native_dim = 8, 8, 5
        Builder._create_openclip_encoders = staticmethod(lambda cfg: (text, image))
        for score_mode, feature_mode in product(("window", "local_3x3"), repeat=2):
            cfg_data = dict(score_intra_attn_type=score_mode, feature_intra_attn_type=feature_mode)
            refiner_cfg = Builder.validate_encoder_refiner_cfg(cfg_data)
            cfg = SegmentorBuildConfig(encoder_refiner_cfg=refiner_cfg, load_from_hf=False)
            cfg.openclip_cfg.enabled = True
            model = Builder.build_semantic_core_model(cfg)
            refiner = model.encoder_refiner
            self.assertEqual([layer.intra_attn_type for layer in refiner.layers], [score_mode, feature_mode, score_mode, feature_mode])
            self.assertFalse(hasattr(refiner, "sam_text_mean_proj"))
            self.assertFalse(hasattr(refiner, "clip_text_mean_proj"))
            self.assertFalse(hasattr(refiner, "template_text_proj"))
            self.assertTrue(all(not hasattr(layer, "template_attention") for layer in refiner.layers))
            self.assertFalse(hasattr(refiner.decoder_input_fusion, "layers"))
            self.assertFalse(hasattr(refiner.decoder_input_fusion, "input_fusion"))
            self.assertFalse(hasattr(refiner_cfg, "decoder_input_fusion_layers"))
            for layer in refiner.layers:
                self.assertEqual(layer.ffn_norm_feature.normalized_shape, (256,))
                self.assertEqual(layer.ffn_norm_score.normalized_shape, (256,))
                self.assertEqual(layer.ffn_fc1_feature.in_features, 256)
                self.assertEqual(layer.ffn_fc1_score.in_features, 256)
        cfg.openclip_cfg.image_intermediate_layers = [3, 7]
        with self.assertRaisesRegex(ValueError, "include"):
            Builder.build_semantic_core_model(cfg)


if __name__ == "__main__":
    unittest.main()
