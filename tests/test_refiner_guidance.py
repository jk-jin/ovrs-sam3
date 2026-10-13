"""CPU tests for optional Q/K guidance; no pretrained weights required."""
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


FLAGS = (
    "score_intra_use_clip_mid", "score_inter_use_clip_text_mean",
    "feature_intra_use_sam_fpn", "feature_inter_use_sam_text_mean",
)
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
        "__init__", "_infer_clip_text_dim", "_infer_clip_image_dim", "_masked_sam_text_mean",
        "_build_clip_image_cache", "_run_encoder_refiner_lowres", "run_encoder_refiner_lowres_from_cache",
    }, env), env


class GuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(31)

    def test_all_switches_route_only_their_own_attention(self):
        for enabled, kind, mode in product(product((False, True), repeat=4), ("intra", "inter"), ("window", "local_3x3")):
            flags = dict(zip(FLAGS, enabled))
            with self.subTest(flags=enabled, kind=kind, mode=mode):
                layer = EncoderRefinerLayer(8, 8, 2, 3, 1, 0, kind, mode, mode, clip_text_dim=12, **flags)
                feature = torch.randn(1, 2, 8, 6, 6)
                score = torch.randn_like(feature)
                guides = [torch.randn(1, 2, channels, 6, 6) for channels in (256, 12, 256, 256)]
                intra_index, class_index = (0, 3) if kind == "intra" else (2, 1)
                seen = []

                def record(attention, args, kwargs):
                    guide = kwargs["guidance"]
                    index = class_index if attention is layer.class_attn else intra_index
                    guide_dim = guides[index].shape[2]
                    self.assertEqual(attention.guidance_dim, guide_dim if enabled[index] else 0)
                    self.assertEqual(attention.q_proj.in_features, 16 + attention.guidance_dim)
                    self.assertEqual(attention.v_proj.in_features, 8)
                    if enabled[index]:
                        expected = F.layer_norm(guides[index].permute(0, 1, 3, 4, 2), (guide_dim,)).permute(0, 1, 4, 2, 3)
                        torch.testing.assert_close(guide, expected)
                    else:
                        self.assertIsNone(guide)
                    seen.append(attention)

                with ExitStack() as stack:
                    for attention in [layer.class_attn] + intra_steps(layer):
                        stack.enter_context(attention.register_forward_pre_hook(record, with_kwargs=True))
                    layer(feature, score, *guides)
                self.assertEqual(len(seen), 3 if mode == "window" else 5)
                self.assertIs(seen[-1] if kind == "intra" else seen[0], layer.class_attn)

    def test_guidance_changes_weights_but_never_supplies_values(self):
        feature, score = torch.randn(1, 2, 8, 6, 6), torch.randn(1, 2, 8, 6, 6)
        for factory, stream in product((ClassScoreAttention, WindowScoreAttention, LocalScoreAttention), ("score", "feature")):
            kwargs = dict(hidden_dim=8, score_embed_dim=8, num_heads=2, dropout=0, value_stream=stream, guidance_dim=256)
            if factory is WindowScoreAttention:
                kwargs.update(window_size=3, shift_size=1)
            attention = factory(**kwargs)
            guide = torch.randn(1, 2, 256, 6, 6, requires_grad=True)
            first = attention(feature, score, guide)
            second = attention(feature, score, torch.zeros_like(guide))
            self.assertGreater((first - second).abs().sum().item(), 0)
            (first * torch.randn_like(first)).sum().backward()
            self.assertTrue(torch.isfinite(guide.grad).all())
            self.assertGreater(guide.grad.abs().sum().item(), 0)
            with torch.no_grad():
                attention.v_proj.weight.zero_()
                attention.v_proj.bias.zero_()
                attention.out_proj.bias.zero_()
            torch.testing.assert_close(attention(feature, score, guide), torch.zeros_like(first))

    def make_refiner(self, score_mode="window", feature_mode="window", **flags):
        return ClassConditionedEncoderRefiner(
            FakeTextEncoder(), clip_dim=8, clip_native_dim=5, fusion_layers=4,
            prompt_templates=["{}"] * 64, use_checkpoint=False, dropout=0,
            score_intra_attn_type=score_mode, feature_intra_attn_type=feature_mode, **flags,
        )

    def test_raw_clip_text_channels_reach_qk_and_receive_gradients(self):
        # Real RemoteCLIP width: only the usual Q/K projections reduce channels.
        layer = EncoderRefinerLayer(
            256, 256, 8, 3, 1, 0, "inter", score_inter_use_clip_text_mean=True,
            clip_text_dim=768,
        )
        self.assertEqual(layer.class_attn.q_proj.in_features, 1280)
        self.assertEqual(layer.class_attn.k_proj.in_features, 1280)
        self.assertEqual(layer.class_norm_guidance.normalized_shape, (768,))
        self.assertEqual(layer.class_attn.v_proj.in_features, 256)
        self.assertEqual(layer.class_attn.out_proj.out_features, 256)
        templates = torch.randn(3, 64, 768, requires_grad=True)
        mean = templates.mean(dim=1)
        guide = mean[None, :, :, None, None].expand(1, -1, -1, 6, 6)
        feature = torch.randn(1, 3, 256, 6, 6)
        score = torch.randn_like(feature)
        captured = []
        with layer.class_attn.q_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0])):
            out, _ = layer(feature, score, clip_text_mean=guide)
        expected = layer.class_norm_guidance(mean)[None].expand(36, -1, -1)
        torch.testing.assert_close(captured[0][..., 512:], expected)
        (out * torch.randn_like(out)).mean().backward()
        self.assertTrue(torch.isfinite(templates.grad).all())
        self.assertGreater(templates.grad.abs().sum().item(), 0)

    def inputs(self):
        return dict(
            encoder_features_72=torch.randn(1, 2, 256, 72, 72),
            clip_image_feat_map=torch.randn(1, 8, 36, 36), class_names=["building", "road"],
            # Deliberately reverse metadata order to check layer-index selection.
            clip_mid_features=[torch.randn(1, 5, 36, 36, requires_grad=True) for _ in range(2)],
            clip_mid_layer_indices=(15, 7), sam_fpn_72=torch.randn(1, 256, 72, 72),
            sam_text_mean=torch.randn(2, 256),
        )

    def test_guidance_is_prepared_once_with_correct_means_and_broadcasts(self):
        model = self.make_refiner(**dict.fromkeys(FLAGS, True))
        inputs = self.inputs()
        seen, mid_inputs = [], []

        def record_layer(layer, args, kwargs):
            seen.append(kwargs)

        with ExitStack() as stack:
            stack.enter_context(model.clip_mid_proj.register_forward_pre_hook(lambda m, args: mid_inputs.append(args[0].clone())))
            for layer in model.layers:
                stack.enter_context(layer.register_forward_pre_hook(record_layer, with_kwargs=True))
            out = model(**inputs)
        self.assertEqual(len(mid_inputs), 1)
        self.assertFalse(hasattr(model, "clip_text_mean_proj"))
        torch.testing.assert_close(mid_inputs[0], torch.cat(inputs["clip_mid_features"][::-1], dim=1))
        expected_mid = model.clip_mid_proj(mid_inputs[0])
        expected_text = out["template_clip_text"].mean(dim=1)
        expected_fpn = F.interpolate(inputs["sam_fpn_72"], size=(36, 36), mode="bilinear", align_corners=False)
        for supplied in seen:
            for cls in range(2):
                torch.testing.assert_close(supplied["clip_mid_36"][:, cls], expected_mid)
                torch.testing.assert_close(supplied["sam_fpn_36"][:, cls], expected_fpn)
                torch.testing.assert_close(supplied["clip_text_mean"][0, cls], expected_text[cls, :, None, None].expand(-1, 36, 36))
                torch.testing.assert_close(supplied["sam_text_mean"][0, cls], inputs["sam_text_mean"][cls, :, None, None].expand(-1, 36, 36))

    def test_all_guidance_checkpoint_matches_outputs_and_gradients(self):
        # Full native grid and channels, covering all four spatial-mode combinations.
        for score_mode, feature_mode in product(("window", "local_3x3"), repeat=2):
            with self.subTest(score_mode=score_mode, feature_mode=feature_mode):
                plain = self.make_refiner(score_mode, feature_mode, **dict.fromkeys(FLAGS, True)).train()
                checked = deepcopy(plain)
                checked.use_checkpoint = True
                inputs = self.inputs()
                target = torch.randn(1, 2, 256, 36, 36)
                results, mid_grads = [], []
                for model in (plain, checked):
                    data = dict(inputs)
                    data["clip_mid_features"] = [x.detach().clone().requires_grad_() for x in inputs["clip_mid_features"]]
                    out = model(**data)
                    (out["refiner_features_36"] * target).mean().backward()
                    results.append(out)
                    mid_grads.append([x.grad for x in data["clip_mid_features"]])
                    self.assertFalse(hasattr(model, "clip_text_mean_proj"))
                    for projection in (model.clip_mid_proj,):
                        self.assertTrue(torch.isfinite(projection.weight.grad).all())
                        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
                for key in results[0]:
                    torch.testing.assert_close(results[0][key], results[1][key])
                for first, second in zip(mid_grads[0], mid_grads[1]):
                    torch.testing.assert_close(first, second)
                    self.assertGreater(first.abs().sum().item(), 0)
                for (name, first), (_, second) in zip(plain.named_parameters(), checked.named_parameters()):
                    if first.grad is None:
                        self.assertIsNone(second.grad, name)
                    else:
                        self.assertTrue(torch.isfinite(first.grad).all(), name)
                        torch.testing.assert_close(first.grad, second.grad)

    def test_missing_enabled_sources_fail_and_disabled_sources_are_optional(self):
        for flag, missing, message in (
            (FLAGS[0], "clip_mid_features", "CLIP middle"),
            (FLAGS[2], "sam_fpn_72", "SAM FPN"),
            (FLAGS[3], "sam_text_mean", "SAM text"),
        ):
            model = self.make_refiner(**{flag: True})
            data = self.inputs()
            data.pop(missing)
            with self.assertRaisesRegex(ValueError, message):
                model(**data)
        model = self.make_refiner(**{FLAGS[0]: True})
        data = self.inputs()
        data["clip_mid_layer_indices"] = (3, 7)
        with self.assertRaisesRegex(ValueError, "indices 7 and 15"):
            model(**data)
        baseline = self.make_refiner()
        self.assertFalse(hasattr(baseline, "clip_mid_proj"))
        self.assertFalse(hasattr(baseline, "clip_text_mean_proj"))
        data = self.inputs()
        with torch.no_grad():
            first = baseline(**data)
            second = baseline(**{key: data[key] for key in ("encoder_features_72", "clip_image_feat_map", "class_names")})
        torch.testing.assert_close(first["refiner_features_36"], second["refiner_features_36"], rtol=0, atol=0)

    def test_sam_token_mean_excludes_padding_and_checks_empty_prompts(self):
        Sam, _ = lightweight_sam_class()
        tokens = torch.randn(4, 2, 256)
        mask = torch.tensor([[False, False, True, True], [False, False, False, True]])
        tokens.masked_fill_(mask.T.unsqueeze(-1), float("nan"))
        actual = Sam._masked_sam_text_mean(tokens, mask)
        torch.testing.assert_close(actual[0], tokens[:2, 0].mean(0))
        torch.testing.assert_close(actual[1], tokens[:3, 1].mean(0))
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            Sam._masked_sam_text_mean(tokens, torch.ones_like(mask))

    def test_clip_middle_gradient_survives_encoder_and_cache_only_when_requested(self):
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
            for enabled in (False, True):
                sam.encoder_refiner = SimpleNamespace(score_intra_use_clip_mid=enabled)
                cache = sam._build_clip_image_cache(SimpleNamespace(raw_images=[torch.zeros(3, 36, 36)]), torch.device("cpu"))
                self.assertEqual(cache["clip_mid_layer_indices"], (7, 15))
                for middle in cache["clip_mid_features"]:
                    self.assertEqual(middle.requires_grad, enabled)
                if enabled:
                    sum(x.square().mean() for x in cache["clip_mid_features"]).backward()
                    self.assertGreater(visual.native_map.grad.abs().sum().item(), 0)
            with torch.no_grad():
                cache = sam._build_clip_image_cache(SimpleNamespace(raw_images=[torch.zeros(3, 36, 36)]), torch.device("cpu"))
                self.assertFalse(cache["clip_image_feat_map_native"].requires_grad)
                self.assertTrue(all(not x.requires_grad for x in cache["clip_mid_features"]))

    def test_lowres_cache_routes_guidance_and_detaches_only_debug_outputs(self):
        Sam, _ = lightweight_sam_class()
        sam = Sam.__new__(Sam)
        nn.Module.__init__(sam)
        sam.encoder_refiner = self.make_refiner(**dict.fromkeys(FLAGS, True))
        data = self.inputs()
        cache = dict(
            cross_attended_encoder_features_72=data["encoder_features_72"],
            clip_image_feat_map=data["clip_image_feat_map"],
            clip_mid_features=data["clip_mid_features"], clip_mid_layer_indices=data["clip_mid_layer_indices"],
            sam_text_mean=data["sam_text_mean"], class_names=data["class_names"],
            backbone_fpn=[torch.empty(0), torch.empty(0), data["sam_fpn_72"]],
        )
        result = sam.run_encoder_refiner_lowres_from_cache(cache, SimpleNamespace(find_text_batch=data["class_names"]), return_debug=True)
        self.assertTrue(result["refiner_features_36"].requires_grad)
        for key in ("score_embed_36", "clip_score_embed_36", "clip_score_maps_36", "template_clip_text"):
            self.assertFalse(result[key].requires_grad)
        self.assertTrue(all(not x.requires_grad for x in result["clip_mid_features"]))
        self.assertTrue(all(x.requires_grad for x in data["clip_mid_features"]))
        (result["refiner_features_36"] * torch.randn_like(result["refiner_features_36"])).mean().backward()
        for middle in data["clip_mid_features"]:
            self.assertGreater(middle.grad.abs().sum().item(), 0)

    def test_config_validation_and_actual_builder_constructor_route_all_switches(self):
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
        for enabled in product((False, True), repeat=4):
            flags = dict(zip(FLAGS, enabled))
            refiner_cfg = Builder.validate_encoder_refiner_cfg(flags)
            cfg = SegmentorBuildConfig(encoder_refiner_cfg=refiner_cfg, load_from_hf=False)
            cfg.openclip_cfg.enabled = True
            model = Builder.build_semantic_core_model(cfg)
            refiner = model.encoder_refiner
            for flag in FLAGS:
                self.assertEqual(getattr(refiner, flag), flags[flag])
            self.assertEqual([layer.use_class_guidance for layer in refiner.layers], [enabled[3], enabled[1], enabled[3], enabled[1]])
            self.assertEqual([layer.use_intra_guidance for layer in refiner.layers], [enabled[0], enabled[2], enabled[0], enabled[2]])
            self.assertEqual(hasattr(refiner, "clip_mid_proj"), enabled[0])
            self.assertFalse(hasattr(refiner, "clip_text_mean_proj"))
            for layer in refiner.layers:
                dim = (256 if layer.score_attention_type == "intra" else 8) if layer.use_class_guidance else 0
                self.assertEqual(layer.class_attn.guidance_dim, dim)
                self.assertEqual(layer.class_attn.q_proj.in_features, 512 + dim)
        for flag in FLAGS:
            with self.assertRaisesRegex(ValueError, flag):
                Builder.validate_encoder_refiner_cfg({flag: "False"})
        cfg.encoder_refiner_cfg = EncoderRefinerConfig(score_intra_use_clip_mid=True)
        cfg.openclip_cfg.image_intermediate_layers = [3, 7]
        with self.assertRaisesRegex(ValueError, "include"):
            Builder.build_semantic_core_model(cfg)


if __name__ == "__main__":
    unittest.main()
