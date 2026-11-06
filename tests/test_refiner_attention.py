"""CPU regressions for pure inter/intra blocks; no pretrained weights required."""

import copy
import itertools
import runpy
import unittest
from pathlib import Path

import torch
from torch import nn

from config_dataclasses import EncoderRefinerConfig
from models.encoder_refiner import ClassConditionedEncoderRefiner
from models.encoder_refiner_attention import (
    ClassScoreAttention,
    EncoderRefinerLayer,
    LocalScoreAttention,
    WindowScoreAttention,
)


ROOT = Path(__file__).resolve().parents[1]
Config = runpy.run_path(str(ROOT / "engine/config.py"))["Config"]


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("templates", torch.randn(2, 64, 8))

    def encode_prompt_templates(self, class_names, device, **kwargs):
        return self.templates[:len(class_names)].to(device)


class RefinerAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def make_layer(self, kind="inter", clip=True, fpn=True, modes=("window", "window")):
        return EncoderRefinerLayer(
            hidden_dim=16, score_embed_dim=16, num_heads=4, dropout=0,
            window_size=2, shift_size=1, attention_type=kind,
            intra_score_use_clip_mid=clip, intra_feature_use_sam_fpn=fpn,
            score_intra_attn_type=modes[0], feature_intra_attn_type=modes[1],
        )

    def test_blocks_have_only_their_declared_attention_type(self):
        for kind in ("inter", "intra"):
            for modes in itertools.product(("window", "local_3x3"), repeat=2):
                with self.subTest(kind=kind, modes=modes):
                    layer = self.make_layer(kind, modes=modes)
                    for stream, mode in zip((layer.score_attention, layer.feature_attention), modes):
                        expected = (ClassScoreAttention if kind == "inter" else
                                    WindowScoreAttention if mode == "window" else LocalScoreAttention)
                        expected_steps = 1 if kind == "inter" else 2 if mode == "window" else 4
                        self.assertEqual(len(stream.attentions), expected_steps)
                        self.assertTrue(all(isinstance(a, expected) for a in stream.attentions))
                    self.assertIsNot(layer.score_attention.attentions[0].q_proj.weight,
                                     layer.feature_attention.attentions[0].q_proj.weight)

    def test_feature_attention_reads_updated_score(self):
        for kind in ("inter", "intra"):
            with self.subTest(kind=kind):
                layer = self.make_layer(kind, clip=False, fpn=False)
                feature, score = torch.randn(2, 3, 16, 4, 4), torch.randn(2, 3, 16, 4, 4)
                captured = {}
                def after_score(module, args, output):
                    captured["score"] = output[1].detach().clone()
                def before_feature(module, args):
                    captured["feature_input"] = args[1].detach().clone()
                handles = [layer.score_attention.register_forward_hook(after_score),
                           layer.feature_attention.register_forward_pre_hook(before_feature)]
                layer(feature, score)
                for handle in handles:
                    handle.remove()
                torch.testing.assert_close(captured["feature_input"], captured["score"])
                self.assertFalse(torch.allclose(captured["score"], score))
                # Each attention group changes only the selected residual stream.
                f, s = layer.score_attention(feature, score)
                torch.testing.assert_close(f, feature)
                f2, s2 = layer.feature_attention(f, s)
                torch.testing.assert_close(s2, s)
                self.assertFalse(torch.allclose(f2, feature))

    def test_independent_guidance_switches_and_gradients(self):
        for modes in itertools.product(("window", "local_3x3"), repeat=2):
            for clip, fpn in itertools.product((False, True), repeat=2):
                with self.subTest(modes=modes, clip=clip, fpn=fpn):
                    layer = self.make_layer("intra", clip, fpn, modes)
                    feature = torch.randn(1, 2, 16, 4, 4, requires_grad=True)
                    score = torch.randn_like(feature, requires_grad=True)
                    clip_input = torch.randn(1, 2, 256, 4, 4, requires_grad=True)
                    fpn_input = torch.randn_like(clip_input, requires_grad=True)
                    outputs = layer(feature, score, clip_input, fpn_input)
                    for stream, enabled in ((layer.score_attention, clip), (layer.feature_attention, fpn)):
                        self.assertEqual(len(stream.guidance_norms), len(stream.attentions) if enabled else 0)
                        self.assertEqual(stream.attentions[0].q_proj.in_features, 32 + 256 * enabled)
                    sum(x.square().mean() for x in outputs).backward()
                    for tensor, enabled in ((clip_input, clip), (fpn_input, fpn)):
                        self.assertEqual(tensor.grad is not None, enabled)
                        if enabled:
                            self.assertTrue(torch.isfinite(tensor.grad).all())
                            self.assertGreater(tensor.grad.abs().sum().item(), 0)
                    for parameter in layer.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    # Disabled guidance may be omitted entirely.
                    omitted = layer(feature, score, clip_input if clip else None, fpn_input if fpn else None)
                    for actual, expected in zip(omitted, outputs):
                        torch.testing.assert_close(actual, expected)

    def test_enabled_guidance_is_required(self):
        feature = score = torch.randn(1, 2, 16, 4, 4)
        guide = torch.randn(1, 2, 256, 4, 4)
        with self.assertRaises(ValueError):
            self.make_layer("intra")(feature, score, None, guide)
        with self.assertRaises(ValueError):
            self.make_layer("intra")(feature, score, guide, None)
        # Class blocks never require either image guidance.
        self.make_layer("inter")(feature, score)

    def make_refiner(self, clip, fpn, checkpoint=False, depth=4, modes=("window", "local_3x3")):
        return ClassConditionedEncoderRefiner(
            clip_text_encoder=FakeTextEncoder(), clip_dim=8, clip_native_dim=8 if clip else None,
            prompt_templates=["a satellite image of {}."] * 64,
            fusion_layers=depth, dropout=0, use_checkpoint=checkpoint,
            intra_score_use_clip_mid=clip, intra_feature_use_sam_fpn=fpn,
            score_intra_attn_type=modes[0], feature_intra_attn_type=modes[1],
        )

    def test_full_refiner_checkpoint_outputs_and_gradients(self):
        for clip, fpn in itertools.product((False, True), repeat=2):
            with self.subTest(clip=clip, fpn=fpn):
                normal = self.make_refiner(clip, fpn)
                checked = copy.deepcopy(normal)
                checked.use_checkpoint = True
                self.assertEqual([layer.attention_type for layer in normal.layers],
                                 ["inter", "intra", "inter", "intra"])
                self.assertEqual(normal.clip_mid_proj is not None, clip)
                source = [torch.randn(1, 2, 256, 72, 72), torch.randn(1, 8, 36, 36),
                          torch.randn(1, 8, 36, 36), torch.randn(1, 8, 36, 36),
                          torch.randn(1, 256, 72, 72)]
                results = []
                for model in (normal, checked):
                    inputs = [x.clone().requires_grad_() for x in source]
                    result = model(
                        inputs[0], inputs[1], ["ship", "bridge"],
                        clip_mid_features=inputs[2:4] if clip else None,
                        clip_mid_layer_indices=(7, 15) if clip else None,
                        sam_fpn_72=inputs[4] if fpn else None,
                    )
                    feature, score = result["refiner_features_36"], result["score_embed_36"]
                    self.assertEqual(feature.shape, (1, 2, 256, 36, 36))
                    # A nonuniform objective also checks gradients through final LayerNorm.
                    weights = torch.linspace(-1, 1, 256)[None, None, :, None, None]
                    ((feature * weights).mean() + score.square().mean()).backward()
                    for index, enabled in ((0, True), (1, True), (2, clip), (3, clip), (4, fpn)):
                        self.assertEqual(inputs[index].grad is not None, enabled)
                        if enabled:
                            self.assertTrue(torch.isfinite(inputs[index].grad).all())
                            self.assertGreater(inputs[index].grad.abs().sum().item(), 0)
                    results.append((feature.detach(), score.detach(), inputs))
                for left, right in zip(results[0][:2], results[1][:2]):
                    torch.testing.assert_close(left, right)
                for left, right in zip(results[0][2], results[1][2]):
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad)
                for (name, left), (_, right) in zip(normal.named_parameters(), checked.named_parameters()):
                    self.assertEqual(left.grad is None, right.grad is None, name)
                    if left.grad is not None:
                        self.assertTrue(torch.isfinite(left.grad).all(), name)
                        torch.testing.assert_close(left.grad, right.grad, msg=name)

    def test_default_window_refiner_backward(self):
        refiner = self.make_refiner(True, True, checkpoint=True, modes=("window", "window"))
        result = refiner(
            torch.randn(1, 2, 256, 72, 72), torch.randn(1, 8, 36, 36), ["ship", "bridge"],
            clip_mid_features=[torch.randn(1, 8, 36, 36), torch.randn(1, 8, 36, 36)],
            clip_mid_layer_indices=(7, 15), sam_fpn_72=torch.randn(1, 256, 72, 72),
        )
        (result["refiner_features_36"].square().mean() + result["score_embed_36"].square().mean()).backward()
        for name, parameter in refiner.named_parameters():
            if name.startswith("decoder_input_fusion."):
                continue  # This test ends at the Refiner's 36x36 output.
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_one_inter_block_needs_no_guidance(self):
        refiner = self.make_refiner(True, True, depth=1)
        self.assertIsNone(refiner.clip_mid_proj)
        refiner(torch.randn(1, 1, 256, 72, 72), torch.randn(1, 8, 36, 36), ["ship"])

    def test_config_defaults_and_overrides(self):
        root = Path(__file__).resolve().parents[1]
        cfg = Config.fromfile(root / "configs/train/isaid_loveda_full.py")
        defaults = EncoderRefinerConfig(**cfg["model"]["encoder_refiner_cfg"])
        self.assertTrue(defaults.intra_score_use_clip_mid)
        self.assertTrue(defaults.intra_feature_use_sam_fpn)
        for clip, fpn in itertools.product((False, True), repeat=2):
            options = dict(cfg["model"]["encoder_refiner_cfg"],
                           intra_score_use_clip_mid=clip, intra_feature_use_sam_fpn=fpn)
            parsed = EncoderRefinerConfig(**options)
            self.assertEqual(parsed.intra_score_use_clip_mid, clip)
            self.assertEqual(parsed.intra_feature_use_sam_fpn, fpn)


if __name__ == "__main__":
    unittest.main()
