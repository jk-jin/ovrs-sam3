"""CPU regressions; no pretrained weights or datasets required."""
from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_ovrs_sam3_test"
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, pkg)
attention = import_module(f"{PACKAGE}.models.encoder_refiner_attention")
Refiner = import_module(f"{PACKAGE}.models.encoder_refiner").ClassConditionedEncoderRefiner
Config = import_module(f"{PACKAGE}.engine.config").Config
configs = import_module(f"{PACKAGE}.config_dataclasses")


class TinyTextEncoder(nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.templates = nn.Parameter(torch.randn(3, 64, self.output_dim))

    def encode_prompt_templates(self, class_names, device, **kwargs):
        return self.templates[:len(class_names)].to(device)


class TinyImageEncoder(nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.dense = nn.Parameter(torch.randn(1, 8, 36, 36))

    def has_trainable_params(self):
        return self.dense.requires_grad

    def encode_raw_images(self, raw_images, device, keep_intermediate_grad):
        self.kept_intermediate_grad = keep_intermediate_grad
        return {"feat_map": self.dense * 2, "mid_features": [self.dense.square()]}


def make_refiner(**kwargs):
    return Refiner(
        TinyTextEncoder(), clip_dim=8, prompt_templates=["aerial {}"] * 64,
        **kwargs,
    )


def make_layer(stream):
    return attention.EncoderRefinerLayer(
        hidden_dim=8, score_embed_dim=8, num_heads=2,
        window_size=2, shift_size=1, dropout=0, value_stream=stream,
    )


def uniform_attention(module):
    with torch.no_grad():
        for projection in (module.q_proj, module.k_proj):
            projection.weight.zero_()
            projection.bias.zero_()
        for projection in (module.v_proj, module.out_proj):
            projection.weight.copy_(torch.eye(projection.weight.shape[0]))
            projection.bias.zero_()
        if hasattr(module, "relative_position_bias_table"):
            module.relative_position_bias_table.zero_()


class RefinerAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_update_order_uses_latest_target_and_only_one_ffn(self):
        for stream in ("score", "feature"):
            with self.subTest(stream=stream):
                layer = make_layer(stream)
                for module in (layer.class_attn, layer.window_attn_regular, layer.window_attn_shifted):
                    self.assertEqual(module.value_stream, stream)
                events = []

                class RecordAttention(nn.Module):
                    def __init__(self, name):
                        super().__init__()
                        self.name = name

                    def forward(self, feature, score_embed, guidance=None):
                        target = score_embed if stream == "score" else feature
                        events.append((self.name, target.mean().item()))
                        return target

                class RecordFFN(nn.Module):
                    def forward(self, target):
                        events.append(("ffn", target.mean().item()))
                        return target

                layer.class_attn = RecordAttention("inter")
                layer.window_attn_regular = RecordAttention("regular")
                layer.window_attn_shifted = RecordAttention("shifted")
                for name in ("class_norm_feature", "class_norm_score", "regular_norm_feature",
                             "regular_norm_score", "shifted_norm_feature", "shifted_norm_score"):
                    setattr(layer, name, nn.Identity())
                layer.ffn = RecordFFN()
                feature = torch.full((1, 2, 8, 4, 4), 2.)
                score = torch.full_like(feature, 3.)
                out_f, out_s = layer(feature, score, torch.randn(1, 2, 256, 4, 4))
                initial = 3. if stream == "score" else 2.
                self.assertEqual(events, [(name, initial * 2**i) for i, name in enumerate(
                    ("inter", "regular", "shifted", "ffn"))])
                torch.testing.assert_close(out_s if stream == "score" else out_f,
                                           (score if stream == "score" else feature) * 16)
                self.assertIs(out_f if stream == "score" else out_s,
                              feature if stream == "score" else score)

    def test_unchanged_stream_and_fpn_still_receive_guidance_gradients(self):
        for stream in ("score", "feature"):
            with self.subTest(stream=stream):
                layer = make_layer(stream)
                feature = torch.randn(2, 3, 8, 4, 4, requires_grad=True)
                score = torch.randn_like(feature, requires_grad=True)
                fpn = torch.randn(2, 3, 256, 4, 4, requires_grad=True)
                out_f, out_s = layer(feature, score, fpn)
                (out_s if stream == "score" else out_f).square().mean().backward()
                self.assertIs(out_f if stream == "score" else out_s,
                              feature if stream == "score" else score)
                for tensor in (feature, score, fpn):
                    self.assertIsNotNone(tensor.grad)
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)
                for name, parameter in layer.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertEqual(sum(isinstance(m, nn.Linear) for m in layer.ffn), 2)

    def test_inter_class_aggregates_classes_at_same_pixel_including_self(self):
        for stream in ("feature", "score"):
            for classes in (1, 3):
                with self.subTest(stream=stream, classes=classes):
                    module = attention.ClassScoreAttention(
                        hidden_dim=8, score_embed_dim=8, num_heads=2,
                        dropout=0, value_stream=stream,
                    )
                    uniform_attention(module)
                    feature = torch.randn(2, classes, 8, 4, 4)
                    score = torch.randn_like(feature)
                    target = feature if stream == "feature" else score
                    torch.testing.assert_close(module(feature, score),
                                               target.mean(1, keepdim=True).expand_as(target))

    def test_intra_qk_order_is_feature_score_fpn_for_both_targets(self):
        for stream in ("feature", "score"):
            for shift in (0, 1):
                with self.subTest(stream=stream, shift=shift):
                    module = attention.WindowScoreAttention(
                        hidden_dim=8, score_embed_dim=8, num_heads=2,
                        window_size=2, shift_size=shift, guidance_dim=256,
                        value_stream=stream, dropout=0,
                    )
                    captured = []
                    handle = module.q_proj.register_forward_pre_hook(
                        lambda _, args: captured.append(args[0].detach()))
                    feature = torch.ones(1, 2, 8, 4, 4)
                    module(feature, feature * 2, torch.full((1, 2, 256, 4, 4), 3.))
                    handle.remove()
                    qk = captured[0]
                    self.assertTrue((qk[..., :8] == 1).all())
                    self.assertTrue((qk[..., 8:16] == 2).all())
                    self.assertTrue((qk[..., 16:] == 3).all())

    def test_windows_cross_internal_boundaries_without_wrapping_image_edges(self):
        modules = [attention.WindowScoreAttention(
            hidden_dim=2, score_embed_dim=2, num_heads=1, window_size=2,
            shift_size=shift, value_stream="feature", guidance_dim=256, dropout=0,
        ) for shift in (0, 1)]
        for module in modules:
            uniform_attention(module)
        feature = torch.zeros(1, 1, 2, 4, 4)
        score = torch.zeros_like(feature)
        guidance = torch.zeros(1, 1, 256, 4, 4)
        feature[..., 1, 1] = 1
        regular, shifted = [m(feature, score, guidance) for m in modules]
        self.assertEqual(regular[..., 2, 2].abs().sum().item(), 0)
        self.assertGreater(shifted[..., 2, 2].sum().item(), 0)
        feature.zero_()
        feature[..., 0, 0] = 1
        shifted = modules[1](feature, score, guidance)
        self.assertLess(shifted[..., 3, 3].abs().max().item(), 1e-20)

    def test_full_grid_initialization_fpn_broadcast_and_layer_targets(self):
        model = make_refiner(use_checkpoint=False, dropout=0)
        self.assertEqual([layer.value_stream for layer in model.layers],
                         ["score", "feature", "score", "feature"])
        encoder = torch.randn(2, 2, 256, 72, 72)
        clip = torch.randn(2, 8, 36, 36)
        fpn = torch.randn(2, 256, 72, 72)
        inputs = []
        unchanged = []
        handles = []
        for layer in model.layers:
            handles.append(layer.register_forward_pre_hook(
                lambda _, args, kwargs: inputs.append(kwargs), with_kwargs=True))
            def check_unchanged(module, args, kwargs, output):
                index = 0 if module.value_stream == "score" else 1
                key = "feature_36" if index == 0 else "score_embed_36"
                unchanged.append(output[index] is kwargs[key])
            handles.append(layer.register_forward_hook(check_unchanged, with_kwargs=True))
        with torch.no_grad():
            out = model(encoder, clip, ["ship", "bridge"], sam_fpn_72=fpn)
        for handle in handles:
            handle.remove()
        expected = F.interpolate(fpn, size=(36, 36), mode="bilinear", align_corners=False)
        for item in inputs:
            torch.testing.assert_close(item["sam_fpn_36"], expected[:, None].expand(-1, 2, -1, -1, -1))
        torch.testing.assert_close(inputs[0]["feature_36"], F.interpolate(
            encoder.flatten(0, 1), size=(36, 36), mode="bilinear", align_corners=False,
        ).reshape(2, 2, 256, 36, 36))
        torch.testing.assert_close(inputs[0]["score_embed_36"], out["clip_score_embed_36"])
        self.assertTrue(all(unchanged))
        self.assertEqual(out["refiner_features_36"].shape, (2, 2, 256, 36, 36))
        self.assertFalse(hasattr(model, "clip_mid_proj"))

    def test_four_layer_checkpoint_matches_output_and_gradients_with_dropout(self):
        plain = make_refiner(use_checkpoint=False, dropout=0.1).train()
        checked = copy.deepcopy(plain)
        checked.use_checkpoint = True
        encoder = torch.randn(1, 2, 256, 72, 72)
        fpn = torch.randn(1, 256, 72, 72)
        clip = torch.randn(1, 8, 36, 36)
        probe = torch.randn(1, 2, 256, 36, 36)
        results = []
        for model in (plain, checked):
            dense = clip.clone().requires_grad_()
            model.clip_score_embed.clip_text_encoder.zero_grad(set_to_none=True)
            torch.manual_seed(33)
            out = model(encoder, dense, ["ship", "bridge"], sam_fpn_72=fpn)
            (out["refiner_features_36"] * probe).mean().backward()
            gradients = {name: p.grad.clone() for name, p in model.named_parameters()
                         if p.grad is not None}
            for name, parameter in model.named_parameters():
                if not name.startswith("decoder_input_fusion."):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            text_grad = model.clip_score_embed.clip_text_encoder.templates.grad
            self.assertGreater(text_grad.abs().sum().item(), 0)
            self.assertGreater(dense.grad.abs().sum().item(), 0)
            results.append((out["refiner_features_36"].detach(), out["score_embed_36"].detach(),
                            gradients, dense.grad.clone(), text_grad.clone()))
        for index in (0, 1, 3, 4):
            torch.testing.assert_close(results[0][index], results[1][index], rtol=1e-5, atol=1e-7)
        self.assertEqual(results[0][2].keys(), results[1][2].keys())
        for name in results[0][2]:
            torch.testing.assert_close(results[0][2][name], results[1][2][name],
                                       rtol=1e-5, atol=1e-7, msg=name)

    def test_fusion_backpropagates_through_frozen_head_to_both_stream_layers(self):
        model = make_refiner(use_checkpoint=True, dropout=0).train()
        dense = torch.randn(1, 8, 36, 36, requires_grad=True)
        encoder = torch.randn(1, 2, 256, 72, 72)
        fpn = torch.randn(1, 256, 72, 72)
        out = model(encoder, dense, ["ship", "bridge"], sam_fpn_72=fpn)
        # Follow trainer's proxy-leaf strategy across two prompt chunks.
        proxy = out["refiner_features_36"].detach().requires_grad_()
        head = nn.Conv2d(256, 1, 1).requires_grad_(False)
        for prompt in range(2):
            fused = model.fuse_decoder_input_chunk(proxy[:, prompt], encoder[:, prompt], fpn)
            F.binary_cross_entropy_with_logits(head(fused), torch.zeros(1, 1, 72, 72)).backward()
        out["refiner_features_36"].backward(proxy.grad)
        self.assertIsNone(head.weight.grad)
        for layer in model.layers:
            self.assertGreater(layer.ffn[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(dense.grad.abs().sum().item(), 0)
        for parameter in model.decoder_input_fusion.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_missing_or_wrong_fpn_fails(self):
        layer = make_layer("score")
        feature = torch.randn(1, 2, 8, 4, 4)
        for bad in (None, torch.randn(1, 1, 256, 4, 4), torch.randn(1, 2, 8, 4, 4)):
            with self.subTest(shape=None if bad is None else bad.shape):
                with self.assertRaises(ValueError):
                    layer(feature, feature, bad)

    def test_config_and_builder_to_sam3_chain(self):
        Builder = import_module(f"{PACKAGE}.model_builder").SAM3ModelBuilder
        cfg = Config.fromfile(ROOT / "configs/train/isaid_loveda_full.py")
        refiner_cfg = cfg["model"]["encoder_refiner_cfg"]
        Builder.validate_encoder_refiner_cfg(refiner_cfg)
        for field in ("score_intra_attn_type", "feature_intra_attn_type"):
            self.assertNotIn(field, refiner_cfg)
            with self.assertRaises(TypeError):
                configs.EncoderRefinerConfig(**{field: "window"})
        for overrides in ({"window_size": 5}, {"shift_size": 12}):
            with self.assertRaises(ValueError):
                Builder.validate_encoder_refiner_cfg(dict(refiner_cfg, **overrides))
        model_cfg = Builder._normalize_build_cfg(configs.SegmentorBuildConfig(**cfg["model"]))
        model_cfg.load_from_hf = False
        model_cfg.checkpoint_path = None
        model_cfg.openclip_cfg.image_intermediate_layers = []
        image_encoder = TinyImageEncoder()
        transformer = nn.Identity()
        transformer.d_model = 256
        methods = {name: Mock(return_value=nn.Identity()) for name in (
            "_create_position_encoding", "_create_vit_backbone", "_create_vit_neck",
            "_create_text_encoder", "_create_vl_backbone", "_create_segmentation_head",
            "_create_geometry_encoder",
        )}
        methods["_create_encoder_only_transformer"] = Mock(return_value=transformer)
        methods["_create_openclip_encoders"] = Mock(return_value=(TinyTextEncoder(), image_encoder))
        with patch.multiple(Builder, **methods):
            core = Builder.build_semantic_core_model(model_cfg)
        self.assertEqual([layer.value_stream for layer in core.encoder_refiner.layers],
                         ["score", "feature", "score", "feature"])
        image_batch = types.SimpleNamespace(raw_images=[torch.zeros(3, 4, 4)])
        cache = core._build_clip_image_cache(image_batch, torch.device("cpu"))
        self.assertFalse(image_encoder.kept_intermediate_grad)
        self.assertTrue(cache["clip_image_feat_map_native"].requires_grad)
        self.assertFalse(cache["clip_mid_features"][0].requires_grad)
        cache["clip_image_feat_map_native"].sum().backward()
        self.assertGreater(image_encoder.dense.grad.abs().sum().item(), 0)
        with torch.no_grad():
            cache = core._build_clip_image_cache(image_batch, torch.device("cpu"))
            self.assertFalse(cache["clip_image_feat_map_native"].requires_grad)
            result = core.run_encoder_refiner_lowres_from_cache(
                {"cross_attended_encoder_features_72": torch.randn(1, 2, 256, 72, 72),
                 "clip_image_feat_map": torch.randn(1, 8, 36, 36),
                 "class_names": ["ship", "bridge"],
                 "backbone_fpn": [None, None, torch.randn(1, 256, 72, 72)]},
                types.SimpleNamespace(find_text_batch=["ship", "bridge"]),
            )
        self.assertEqual(result["refiner_features_36"].shape, (1, 2, 256, 36, 36))


if __name__ == "__main__":
    unittest.main()
