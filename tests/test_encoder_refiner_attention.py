"""CPU regression tests for alternating single-value Refiner attention.

Run with: python -m unittest discover -s tests -v
"""
from copy import deepcopy
from contextlib import ExitStack
from itertools import product
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.encoder_refiner import ClassConditionedEncoderRefiner
from models.encoder_refiner_attention import (
    ClassScoreAttention,
    EncoderRefinerLayer,
    LocalScoreAttention,
    WindowScoreAttention,
)


def uniform_attention(module):
    """Make attention a uniform average of the selected stream."""
    with torch.no_grad():
        for projection in (module.q_proj, module.k_proj):
            projection.weight.zero_()
            projection.bias.zero_()
        for projection in (module.v_proj, module.out_proj):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
        if hasattr(module, "relative_position_bias_table"):
            module.relative_position_bias_table.zero_()


def intra_steps(layer):
    if layer.intra_attn_type == "window":
        return [layer.window_attn_regular, layer.window_attn_shifted]
    return list(layer.local_attn)


class FakeTextEncoder(nn.Module):
    """Trainable deterministic template features, without SAM3/RemoteCLIP weights."""

    def __init__(self):
        super().__init__()
        self.templates = nn.Parameter(torch.randn(2, 64, 8))

    def encode_prompt_templates(self, **kwargs):
        return self.templates


class AlternatingAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_class_attention_averages_only_selected_values_across_classes(self):
        feature = torch.randn(2, 3, 4, 6, 6)
        score = torch.randn_like(feature) + 10
        for stream, values in (("feature", feature), ("score", score)):
            with self.subTest(stream=stream):
                attention = ClassScoreAttention(4, 4, 1, 0, value_stream=stream)
                uniform_attention(attention)
                actual = attention(feature, score)
                expected = values.mean(dim=1, keepdim=True).expand_as(values)
                torch.testing.assert_close(actual, expected)

    def test_windows_do_not_mix_classes_or_wrap_across_image_edges(self):
        for stream in ("feature", "score"):
            for shift in (0, 3):
                with self.subTest(stream=stream, shift=shift):
                    attention = WindowScoreAttention(
                        4, 4, 1, window_size=6, shift_size=shift,
                        dropout=0, value_stream=stream,
                    )
                    uniform_attention(attention)
                    values = torch.zeros(1, 2, 4, 12, 12)
                    values[:, 0, :, -1, -1] = 1
                    guidance = torch.full_like(values, 100)
                    feature, score = (
                        (values, guidance) if stream == "feature" else (guidance, values)
                    )
                    out = attention(feature, score)
                    torch.testing.assert_close(out[:, 1], torch.zeros_like(out[:, 1]))
                    torch.testing.assert_close(out[..., 0, 0], torch.zeros_like(out[..., 0, 0]))
                    self.assertGreater(out[0, 0, 0, -1, -1].item(), 0)

    def test_local_attention_matches_explicit_neighbors_with_position_bias(self):
        # A direct per-pixel reference checks neighbor layout, mask, and single V.
        for stream in ("feature", "score"):
            attention = LocalScoreAttention(4, 6, 2, dropout=0, value_stream=stream)
            feature = torch.randn(2, 2, 4, 3, 4)
            score = torch.randn(2, 2, 6, 3, 4)
            actual = attention(feature, score)
            expected = torch.empty_like(actual)
            with torch.no_grad():
                for batch, cls, y, x in product(range(2), range(2), range(3), range(4)):
                    query_input = torch.cat([score[batch, cls, :, y, x], feature[batch, cls, :, y, x]])
                    query = attention.q_proj(query_input).reshape(2, 2)
                    keys, values, biases = [], [], []
                    for index, (dy, dx) in enumerate(product((-1, 0, 1), repeat=2)):
                        ny, nx = y + dy, x + dx
                        if not (0 <= ny < 3 and 0 <= nx < 4):
                            continue
                        key_input = torch.cat([score[batch, cls, :, ny, nx], feature[batch, cls, :, ny, nx]])
                        selected = feature if stream == "feature" else score
                        keys.append(attention.k_proj(key_input).reshape(2, 2))
                        values.append(attention.v_proj(selected[batch, cls, :, ny, nx]).reshape(2, 2))
                        biases.append(attention.relative_position_bias_table[index])
                    logits = (torch.stack(keys) * query).sum(-1) / (2 ** 0.5) + torch.stack(biases)
                    weights = logits.softmax(dim=0)
                    out = (weights.unsqueeze(-1) * torch.stack(values)).sum(0).flatten()
                    expected[batch, cls, :, y, x] = attention.out_proj(out)
            torch.testing.assert_close(actual, expected)

    def test_local_attention_masks_edges_and_does_not_mix_classes(self):
        for stream, grid in product(("feature", "score"), ((1, 1), (2, 3), (6, 6))):
            with self.subTest(stream=stream, grid=grid):
                attention = LocalScoreAttention(4, 4, 1, dropout=0, value_stream=stream)
                uniform_attention(attention)
                values = torch.ones(1, 2, 4, *grid)
                values[:, 1] = 7
                guidance = torch.randn_like(values) * 100
                feature, score = (values, guidance) if stream == "feature" else (guidance, values)
                # Constant class-specific maps stay constant, including all edges.
                torch.testing.assert_close(attention(feature, score), values)
                values.zero_()
                values[:, 0, :, -1, -1] = 1
                out = attention(feature, score)
                self.assertTrue(torch.isfinite(out).all())
                torch.testing.assert_close(out[:, 1], torch.zeros_like(out[:, 1]))
                if grid == (6, 6):
                    torch.testing.assert_close(out[..., 0, 0], torch.zeros_like(out[..., 0, 0]))
                    torch.testing.assert_close(out[0, 0, :, -1, -1], torch.full((4,), 0.25))

    def test_invalid_intra_attention_types_are_rejected(self):
        for name in ("score_intra_attn_type", "feature_intra_attn_type"):
            for kind in ("intra", "local", "", None):
                with self.subTest(name=name, kind=kind):
                    with self.assertRaisesRegex(ValueError, name):
                        EncoderRefinerLayer(**{name: kind})

    def test_layer_order_and_residual_targets_use_updated_score(self):
        feature = torch.randn(1, 2, 4, 6, 6)
        score = torch.randn_like(feature)
        for kind, score_type, feature_type in product(
            ("intra", "inter"), ("window", "local_3x3"), ("window", "local_3x3")
        ):
            with self.subTest(kind=kind, score_type=score_type, feature_type=feature_type):
                layer = EncoderRefinerLayer(
                    4, 4, 1, window_size=3, shift_size=1,
                    dropout=0, score_attention_type=kind,
                    score_intra_attn_type=score_type, feature_intra_attn_type=feature_type,
                )
                steps = intra_steps(layer)
                events = []

                def update(name, amount):
                    def forward(feature, score_embed):
                        events.append((name, feature.clone(), score_embed.clone()))
                        return torch.full_like(feature, amount)
                    return forward

                with ExitStack() as stack:
                    stack.enter_context(patch(
                        "models.encoder_refiner_attention.apply_layer_norm_bcdhw",
                        side_effect=lambda x, norm: x,
                    ))
                    stack.enter_context(patch.object(layer.class_attn, "forward", side_effect=update("class", 3)))
                    for index, attention in enumerate(steps, 1):
                        stack.enter_context(patch.object(attention, "forward", side_effect=update(str(index), index)))
                    stack.enter_context(patch.object(layer, "_ffn_feature_update", side_effect=lambda x: torch.full_like(x, 4)))
                    stack.enter_context(patch.object(layer, "_ffn_score_update", side_effect=lambda x: torch.full_like(x, 5)))
                    out_feature, out_score = layer(feature, score)
                total = sum(range(1, len(steps) + 1))
                torch.testing.assert_close(out_feature, feature + (3 if kind == "intra" else total) + 4)
                torch.testing.assert_close(out_score, score + (total if kind == "intra" else 3) + 5)
                names = [str(index) for index in range(1, len(steps) + 1)]
                expected_order = (
                    names + ["class"] if kind == "intra" else ["class"] + names
                )
                self.assertEqual([event[0] for event in events], expected_order)
                expected_feature, expected_score = feature.clone(), score.clone()
                for name, seen_feature, seen_score in events:
                    torch.testing.assert_close(seen_feature, expected_feature)
                    torch.testing.assert_close(seen_score, expected_score)
                    amount = 3 if name == "class" else int(name)
                    if (name == "class") == (kind == "inter"):
                        expected_score = expected_score + amount
                    else:
                        expected_feature = expected_feature + amount

    def test_guidance_and_selected_values_both_receive_gradients(self):
        for kind, intra_type in product(("intra", "inter"), ("window", "local_3x3")):
            with self.subTest(kind=kind, intra_type=intra_type):
                layer = EncoderRefinerLayer(
                    8, 8, 2, window_size=3, shift_size=1,
                    dropout=0, score_attention_type=kind,
                    score_intra_attn_type=intra_type, feature_intra_attn_type=intra_type,
                )
                feature = torch.randn(1, 3, 8, 6, 6, requires_grad=True)
                score = torch.randn_like(feature, requires_grad=True)
                # Feature output alone must supervise the earlier score update.
                out_feature, _ = layer(feature, score)
                (out_feature * torch.randn_like(out_feature)).sum().backward()
                for tensor in (feature, score):
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)
                for attention in [layer.class_attn] + intra_steps(layer):
                    for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.out_proj):
                        self.assertTrue(torch.isfinite(projection.weight.grad).all())
                        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
                for attention in intra_steps(layer):
                    self.assertGreater(attention.relative_position_bias_table.grad.abs().sum().item(), 0)

    def test_full_refiner_checkpoint_matches_outputs_and_gradients(self):
        for score_type, feature_type in product(("window", "local_3x3"), repeat=2):
            with self.subTest(score_type=score_type, feature_type=feature_type):
                self.check_full_refiner_checkpoint(score_type, feature_type)

    def check_full_refiner_checkpoint(self, score_type, feature_type):
        plain = ClassConditionedEncoderRefiner(
            FakeTextEncoder(), clip_dim=8, fusion_layers=4,
            prompt_templates=["{}"] * 64, use_checkpoint=False,
            score_intra_attn_type=score_type, feature_intra_attn_type=feature_type,
        ).train()
        checked = deepcopy(plain)
        checked.use_checkpoint = True
        self.assertEqual(
            [layer.score_attention_type for layer in plain.layers],
            ["intra", "inter", "intra", "inter"],
        )
        self.assertEqual(
            [layer.class_attn.value_stream for layer in plain.layers],
            ["feature", "score", "feature", "score"],
        )
        self.assertEqual(
            [intra_steps(layer)[0].value_stream for layer in plain.layers],
            ["score", "feature", "score", "feature"],
        )
        self.assertEqual(
            [layer.intra_attn_type for layer in plain.layers],
            [score_type, feature_type, score_type, feature_type],
        )
        encoder = torch.randn(1, 2, 256, 72, 72)
        clip = torch.randn(1, 8, 36, 36)
        target = torch.randn(1, 2, 256, 36, 36)
        outputs, input_grads = [], []
        for model in (plain, checked):
            image = clip.clone().requires_grad_()
            torch.manual_seed(91)  # Match dropout, including checkpoint recomputation.
            out = model(encoder, image, ["building", "road"])
            self.assertEqual(tuple(out["refiner_features_36"].shape), (1, 2, 256, 36, 36))
            loss = (out["refiner_features_36"] * target).mean()
            loss.backward()
            outputs.append(out)
            input_grads.append(image.grad)
        for name in outputs[0]:
            torch.testing.assert_close(outputs[0][name], outputs[1][name])
        torch.testing.assert_close(input_grads[0], input_grads[1])
        self.assertGreater(input_grads[0].abs().sum().item(), 0)
        for (name, first), (other_name, second) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertEqual(name, other_name)
            if first.grad is None:
                self.assertIsNone(second.grad)
            else:
                self.assertTrue(torch.isfinite(first.grad).all(), name)
                torch.testing.assert_close(first.grad, second.grad)
        for layer in plain.layers:
            self.assertGreater(layer.class_attn.v_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(plain.clip_score_embed.score_stem[0].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
