"""CPU regression tests; no model weights or datasets required."""
import copy
import importlib
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


# The checkout directory may contain a hyphen; use a stable package name.
ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("ovrs_test")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("ovrs_test", package)
Config = importlib.import_module("ovrs_test.config_dataclasses").SemanticCriterionConfig
Criterion = importlib.import_module("ovrs_test.losses.semantic_criterion").SemanticCriterion
Attention = importlib.import_module("ovrs_test.models.post_fusion_attention").PostFusionAttention
intra = importlib.import_module("ovrs_test.models.encoder_refiner_attention")
Refiner = importlib.import_module("ovrs_test.models.encoder_refiner").ClassConditionedEncoderRefiner
torch.set_num_threads(1)


def reference_bce(logits, labels, mapping, a, b):
    """Independent per-image/original-class/prompt loop reference."""
    present, absent = [], []
    for image in range(labels.shape[0]):
        valid = labels[image] != 255
        for cls in sorted(set(mapping.tolist())):
            exists = bool((labels[image] == cls).any())
            aliases = []
            for prompt in (mapping == cls).nonzero().flatten():
                loss = F.binary_cross_entropy_with_logits(
                    logits[image, prompt].float(), (labels[image] == cls).float(), reduction="none"
                )
                valid_loss = loss[valid].mean() if valid.any() else loss.sum() * 0
                ignore_loss = loss[~valid].mean() if (~valid).any() else loss.sum() * 0
                aliases.append(a * valid_loss + (1 - a) * ignore_loss if exists else valid_loss)
            (present if exists else absent).append(torch.stack(aliases).mean())
    zero = logits.sum() * 0
    return b * (torch.stack(present).mean() if present else zero) + (1 - b) * (
        torch.stack(absent).mean() if absent else zero
    )


class GroupedBCETest(unittest.TestCase):
    def check_case(self, labels, mapping, a=0.7, b=0.6):
        criterion = Criterion(Config(final_bce_valid_weight=a, final_bce_present_weight=b,
                                     final_balanced_bce_weight=1.7, sam3_mask_distill_weight=0))
        logits = torch.randn(labels.shape[0], len(mapping), *labels.shape[-2:], requires_grad=True)
        expected = reference_bce(logits, labels, mapping, a, b)
        expected_grad, = torch.autograd.grad(expected * 1.7, logits)
        context = criterion.prepare_streaming_context(
            {"label_map": labels}, len(mapping), mapping, int(mapping.max()) + 1, labels.shape[-2:]
        )
        for chunk_size in (1, 2, len(mapping)):
            actual = logits.detach().clone().requires_grad_(True)
            raw = 0
            for start in range(0, len(mapping), chunk_size):
                end = min(start + chunk_size, len(mapping))
                losses = criterion.forward_chunk({"final_logits": actual[:, start:end]}, context, start, end)
                raw = raw + losses["loss_final_bce"].detach()
                losses["total_loss"].backward()
            torch.testing.assert_close(raw, expected.detach())
            torch.testing.assert_close(actual.grad, expected_grad)

    def test_mixed_regions_and_aliases(self):
        self.check_case(torch.tensor([[[0, 0, 255], [1, 255, 0]],
                                      [[2, 255, 2], [255, 255, 255]]]), torch.tensor([0, 1, 0, 2]))

    def test_empty_regions_and_groups(self):
        for labels in (torch.full((2, 2, 3), 255), torch.zeros(1, 2, 3, dtype=torch.long),
                       torch.tensor([[[0, 1, 2], [0, 1, 2]]])):
            for a, b in ((0.0, 0.0), (1.0, 1.0), (0.7, 0.6)):
                self.check_case(labels, torch.tensor([0, 1, 2]), a, b)

    def test_invalid_weights(self):
        for field in ("final_bce_valid_weight", "final_bce_present_weight"):
            for value in (-0.1, 1.1, float("nan")):
                with self.assertRaises(ValueError):
                    Criterion(Config(**{field: value}))

    def test_large_half_precision_regions_are_finite(self):
        criterion = Criterion(Config(sam3_mask_distill_weight=0))
        logits = torch.ones(1, 2, 288, 288, dtype=torch.float16, requires_grad=True)
        labels = torch.zeros(1, 288, 288, dtype=torch.long)
        labels[:, 144:] = 255
        losses = criterion({"final_logits": logits}, {"label_map": labels}, torch.tensor([0, 1]), 2)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class AttentionTest(unittest.TestCase):
    def test_query_blocks_match_dense_output_and_gradients(self):
        for qk_dim, value_dim in ((64, 256), (32, 256), (32, 32)):
            dense = Attention(qk_dim, value_dim, dropout=0, query_chunk_size=1000,
                              use_checkpoint=False)
            blocked = copy.deepcopy(dense)
            blocked.query_chunk_size = 7
            x = torch.randn(4, 256, 4, 4, requires_grad=True)
            e, g = torch.randn_like(x), torch.randn(2, 256, 4, 4)
            y = x.detach().clone().requires_grad_(True)
            expected, actual = dense(x, e, g), blocked(y, e, g)
            torch.testing.assert_close(actual, expected)
            expected.square().mean().backward()
            actual.square().mean().backward()
            torch.testing.assert_close(y.grad, x.grad)
            for p1, p2 in zip(dense.parameters(), blocked.parameters()):
                torch.testing.assert_close(p1.grad, p2.grad)

    def test_checkpoint_preserves_dropout_and_gradients(self):
        a = Attention(dropout=0.1, query_chunk_size=5, use_checkpoint=False)
        b = copy.deepcopy(a)
        b.use_checkpoint = True
        x = torch.randn(2, 256, 3, 3, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        e, g = torch.randn_like(x), torch.randn(1, 256, 3, 3)
        torch.manual_seed(25)
        expected = a(x, e, g)
        torch.manual_seed(25)
        actual = b(y, e, g)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(y.grad, x.grad)
        for p1, p2 in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p1.grad, p2.grad)

    def test_prompt_chunks_match_full_and_keep_global_keys(self):
        attention = Attention(dropout=0, query_chunk_size=5).eval()
        x, e = torch.randn(2, 3, 256, 3, 3), torch.randn(2, 3, 256, 3, 3)
        fpn = torch.randn(2, 256, 3, 3)
        calls = []
        sdpa = F.scaled_dot_product_attention
        def record(q, k, v, **kwargs):
            calls.append((q.shape, k.shape, v.shape))
            return sdpa(q, k, v, **kwargs)
        with patch.object(F, "scaled_dot_product_attention", side_effect=record):
            full = attention(x.flatten(0, 1), e.flatten(0, 1), fpn).reshape_as(x)
        chunks = [attention(x[:, i:i+1].flatten(0, 1), e[:, i:i+1].flatten(0, 1), fpn)
                  .reshape(2, 1, 256, 3, 3) for i in range(3)]
        torch.testing.assert_close(torch.cat(chunks, dim=1), full)
        self.assertTrue(all(q[-2] <= 5 and k[-2] == v[-2] == 9 for q, k, v in calls))
        self.assertTrue(all(q[-1] == 16 and v[-1] == 64 for q, _, v in calls))

    def test_actual_72_grid_backward(self):
        attention = Attention(dropout=0, query_chunk_size=256)
        x = torch.randn(1, 256, 72, 72, requires_grad=True)
        out = attention(x, torch.randn_like(x), torch.randn_like(x))
        out.square().mean().backward()
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_invalid_parameters(self):
        for cfg in ({"qk_dim": 63}, {"value_dim": 255}, {"query_chunk_size": 0},
                    {"num_heads": 0}, {"dropout": 1}):
            with self.assertRaises(ValueError):
                Attention(**cfg)


class PositionTest(unittest.TestCase):
    def test_global_sine_coordinates_and_value_is_unmodified(self):
        for kind in ("window", "local"):
            args = dict(hidden_dim=8, score_embed_dim=8, num_heads=2,
                        guidance_dim=8, position_encoding="sam3_sine", dropout=0)
            module = (intra.WindowScoreAttention(window_size=2, shift_size=1, **args)
                      if kind == "window" else intra.LocalScoreAttention(**args))
            x = torch.zeros(1, 2, 8, 4, 4)
            captured = {}
            def capture(name):
                def hook(_, inputs):
                    captured[name] = inputs[0].detach().clone()
                return hook
            module.q_proj.register_forward_pre_hook(capture("q"))
            module.k_proj.register_forward_pre_hook(capture("k"))
            module.v_proj.register_forward_pre_hook(capture("v"))
            out = module(x, x, x)
            pos = intra.PositionEmbeddingSine(24)(x[0, :1])
            if kind == "window":
                pos = module._window_partition(torch.roll(pos, (-1, -1), (-2, -1)))
                pos = pos.repeat(2, 1, 1)
            else:
                pos = pos.flatten(2).transpose(1, 2).expand(2, -1, -1)
            torch.testing.assert_close(captured["q"], pos)
            torch.testing.assert_close(captured["k"], pos)
            self.assertEqual(captured["v"].count_nonzero(), 0)
            self.assertIsNone(module.relative_position_bias_table)
            self.assertTrue(torch.isfinite(out).all())

    def test_both_intra_modes_and_position_choices_backward(self):
        for kind in ("window", "local_3x3"):
            for pos in ("learned_relative", "sam3_sine"):
                for stream in ("intra", "inter"):
                    layer = intra.EncoderRefinerLayer(
                        hidden_dim=8, score_embed_dim=8, num_heads=2, window_size=2,
                        shift_size=1, dropout=0, score_attention_type=stream,
                        score_intra_attn_type=kind, feature_intra_attn_type=kind,
                        intra_position_encoding=pos,
                    )
                    f = torch.randn(1, 3, 8, 4, 4, requires_grad=True)
                    s = torch.randn_like(f, requires_grad=True)
                    g = torch.randn(1, 3, 256, 4, 4)
                    a, b = layer(f, s, g, g)
                    (a.square().mean() + b.square().mean()).backward()
                    self.assertTrue(torch.isfinite(f.grad).all())
                    self.assertTrue(torch.isfinite(s.grad).all())


class FusionIntegrationTest(unittest.TestCase):
    def test_config_defaults_and_fusion_chunk_backward(self):
        cfg = runpy.run_path(str(ROOT / "configs/_base_/model/ovrs_sam3.py"))["model"]
        c = cfg["encoder_refiner_cfg"]
        post_cfg = {name.removeprefix("post_fusion_"): value for name, value in c.items()
                    if name.startswith("post_fusion_")}
        post_cfg["layers"] = post_cfg.pop("attn_layers")
        model = Refiner(
            clip_text_encoder=nn.Identity(), prompt_templates=cfg["openclip_cfg"]["prompt_templates"],
            clip_native_dim=8, fusion_layers=1, dropout=0,
            post_fusion_attn_cfg=post_cfg, intra_position_encoding=c["intra_position_encoding"],
        )
        self.assertEqual(len(model.post_fusion_attention), 2)
        self.assertEqual(model.post_fusion_attention[0].v_proj.out_features, 256)
        x = torch.randn(1, 256, 36, 36, requires_grad=True)
        e, g = torch.randn(1, 256, 72, 72), torch.randn(1, 256, 72, 72)
        out = model.fuse_decoder_input_chunk(x, e, g)
        out.square().mean().backward()
        self.assertEqual(out.shape, e.shape)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(model.post_fusion_attention[0].v_proj.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
