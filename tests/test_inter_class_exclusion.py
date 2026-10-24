"""CPU regressions; run: python -m unittest discover -s tests -v."""
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Load these independent modules without importing the heavyweight models package.
_PACKAGE = "_refiner_exclusion_tests"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "models")]
sys.modules[_PACKAGE] = package
attention = importlib.import_module(f"{_PACKAGE}.encoder_refiner_attention")
refiner_module = importlib.import_module(f"{_PACKAGE}.encoder_refiner")
torch.set_num_threads(2)


class InterClassExclusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_matches_explicit_other_class_selection(self):
        # Independent oracle physically removes forbidden keys/values.
        ids = torch.tensor([4, 4, 9, 12])
        blocked = ids[:, None] == ids[None, :]
        feature = torch.randn(2, 4, 8, 2, 3)
        score = torch.randn_like(feature)
        for stream in ("feature", "score"):
            with self.subTest(stream=stream):
                module = attention.ClassScoreAttention(8, 8, 2, 0, stream)
                actual = module(feature, score, blocked)
                expected = torch.empty_like(actual)
                for b in range(2):
                    for y in range(2):
                        for x in range(3):
                            f, s = feature[b, :, :, y, x], score[b, :, :, y, x]
                            inputs = torch.cat((s, f), dim=-1)
                            values = f if stream == "feature" else s
                            for i in range(4):
                                keep = ids != ids[i]
                                q = module.q_proj(inputs[i]).reshape(2, 1, 4)
                                k = module.k_proj(inputs[keep]).reshape(-1, 2, 4).transpose(0, 1)
                                v = module.v_proj(values[keep]).reshape(-1, 2, 4).transpose(0, 1)
                                weights = (q @ k.transpose(-2, -1) / 2).softmax(-1)
                                expected[b, i, :, y, x] = module.out_proj((weights @ v).reshape(8))
                torch.testing.assert_close(actual, expected)

    def test_same_class_values_have_no_effect_or_gradient(self):
        ids = torch.tensor([0, 0, 1, 2])
        blocked = ids[:, None] == ids[None, :]
        for stream in ("feature", "score"):
            module = attention.ClassScoreAttention(8, 8, 2, 0, stream)
            # Uniform weights let us isolate the value path from query changes.
            with torch.no_grad():
                module.q_proj.weight.zero_()
                module.q_proj.bias.zero_()
            f = torch.randn(1, 4, 8, 1, 1, requires_grad=True)
            s = torch.randn_like(f, requires_grad=True)
            output = module(f, s, blocked)[:, 0]
            target = f if stream == "feature" else s
            gradient = torch.autograd.grad(output.sum(), target)[0]
            self.assertEqual(torch.count_nonzero(gradient[:, :2]).item(), 0)
            self.assertGreater(torch.count_nonzero(gradient[:, 2:]).item(), 0)
            changed = target.detach().clone()
            changed[:, :2] += 100
            new = module(changed, s, blocked) if stream == "feature" else module(f, changed, blocked)
            torch.testing.assert_close(output, new[:, 0])

    def test_one_original_class_is_exact_zero_with_nonzero_bias(self):
        for prompts in (1, 3):
            for stream in ("feature", "score"):
                module = attention.ClassScoreAttention(8, 8, 2, 0, stream)
                with torch.no_grad():
                    module.out_proj.bias.fill_(7)
                f = torch.randn(1, prompts, 8, 2, 2, requires_grad=True)
                s = torch.randn_like(f, requires_grad=True)
                update = module(f, s, torch.ones(prompts, prompts, dtype=torch.bool))
                target = f if stream == "feature" else s
                self.assertTrue(torch.isfinite(update).all())
                torch.testing.assert_close(update, torch.zeros_like(target), rtol=0, atol=0)
                (target + update).sum().backward()
                torch.testing.assert_close(target.grad, torch.ones_like(target))
                self.assertIsNone(module.out_proj.bias.grad)

    def test_two_classes_exchange_only_other_values(self):
        for stream in ("feature", "score"):
            module = attention.ClassScoreAttention(8, 8, 2, 0, stream)
            f, s = torch.randn(1, 2, 8, 1, 1), torch.randn(1, 2, 8, 1, 1)
            target = f if stream == "feature" else s
            expected = module.out_proj(module.v_proj(target[0, [1, 0], :, 0, 0]))
            actual = module(f, s, torch.eye(2, dtype=torch.bool))
            torch.testing.assert_close(actual[0, :, :, 0, 0], expected)

    def test_alternating_layers_checkpoint_outputs_and_gradients(self):
        for mode in ("window", "local_3x3"):
            for ids in (torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0])):
                with self.subTest(mode=mode, ids=ids.tolist()):
                    layers = nn.ModuleList([
                        attention.EncoderRefinerLayer(
                            hidden_dim=8, score_embed_dim=8, num_heads=2,
                            window_size=2, shift_size=1, dropout=0,
                            score_attention_type=kind,
                            score_intra_attn_type=mode, feature_intra_attn_type=mode,
                        ) for kind in ("intra", "inter", "intra", "inter")
                    ])
                    checked = copy.deepcopy(layers)
                    inputs = [torch.randn(1, 3, d, 4, 4, requires_grad=True) for d in (8, 8, 256, 256)]
                    cloned = [x.detach().clone().requires_grad_() for x in inputs]
                    blocked = ids[:, None] == ids[None, :]

                    def run(stack, tensors, use_checkpoint):
                        f, s, clip, fpn = tensors
                        for layer in stack:
                            args = (f, s, blocked, clip, fpn)
                            f, s = checkpoint(layer, *args, use_reentrant=False) if use_checkpoint else layer(*args)
                        (f.square().mean() + s.square().mean()).backward()
                        return f, s

                    expected = run(layers, inputs, False)
                    actual = run(checked, cloned, True)
                    for a, b in zip(expected, actual):
                        torch.testing.assert_close(a, b)
                    for a, b in zip(inputs, cloned):
                        self.assertTrue(torch.isfinite(b.grad).all())
                        torch.testing.assert_close(a.grad, b.grad)
                    for a, b in zip(layers.parameters(), checked.parameters()):
                        if a.grad is None:
                            self.assertIsNone(b.grad)
                        else:
                            torch.testing.assert_close(a.grad, b.grad)

    def test_full_refiner_passes_same_class_mask_to_every_layer(self):
        class ScoreStub(nn.Module):
            def forward(self, class_names, remoteclip_feat_map):
                score = remoteclip_feat_map[:, None].expand(-1, len(class_names), -1, -1, -1)
                return score, score[:, :, :64], score.new_zeros(len(class_names), 64, 256)

        for use_checkpoint in (False, True):
            model = refiner_module.ClassConditionedEncoderRefiner(
                None, prompt_templates=["{}"] * 64, clip_native_dim=8,
                use_checkpoint=use_checkpoint, dropout=0,
                score_intra_attn_type="local_3x3", feature_intra_attn_type="local_3x3",
            )
            model.clip_score_embed = ScoreStub()
            ids = torch.tensor([2, 2, 7])
            expected = ids[:, None] == ids[None, :]
            seen = []

            def check_mask(module, args, kwargs):
                torch.testing.assert_close(kwargs["inter_class_mask"], expected)
                seen.append(module.value_stream)

            for layer in model.layers:
                layer.class_attn.register_forward_pre_hook(check_mask, with_kwargs=True)
            image = torch.randn(1, 256, 36, 36, requires_grad=True)
            output = model(
                torch.randn(1, 3, 256, 72, 72), image,
                ["ship", "vessel", "road"], ids,
                [torch.randn(1, 8, 36, 36) for _ in range(2)], (7, 15),
                torch.randn(1, 256, 72, 72),
            )
            self.assertEqual(seen, ["feature", "score", "feature", "score"])
            output["refiner_features_36"].square().mean().backward()
            self.assertTrue(torch.isfinite(image.grad).all())
            self.assertGreater(image.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
