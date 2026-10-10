"""CPU regression tests for alternating single-value Refiner attention.

Run with: python -m unittest discover -s tests -v
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.encoder_refiner import ClassConditionedEncoderRefiner
from models.encoder_refiner_attention import (
    ClassScoreAttention,
    EncoderRefinerLayer,
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
        if isinstance(module, WindowScoreAttention):
            module.relative_position_bias_table.zero_()


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

    def test_layer_order_and_residual_targets_use_updated_score(self):
        feature = torch.randn(1, 2, 4, 6, 6)
        score = torch.randn_like(feature)
        for kind in ("intra", "inter"):
            with self.subTest(kind=kind):
                layer = EncoderRefinerLayer(
                    4, 4, 1, window_size=3, shift_size=1,
                    dropout=0, score_attention_type=kind,
                )
                # Bypass normalization to observe exact residual inputs.
                for name, module in list(layer.named_children()):
                    if isinstance(module, nn.LayerNorm):
                        setattr(layer, name, nn.Identity())
                events = []

                def update(name, amount):
                    def forward(feature, score_embed):
                        events.append((name, feature.clone(), score_embed.clone()))
                        return torch.full_like(feature, amount)
                    return forward

                with (
                    patch.object(layer.class_attn, "forward", side_effect=update("class", 3)),
                    patch.object(layer.window_attn_regular, "forward", side_effect=update("regular", 1)),
                    patch.object(layer.window_attn_shifted, "forward", side_effect=update("shifted", 2)),
                    patch.object(layer, "_ffn_feature_update", side_effect=lambda x: torch.full_like(x, 4)),
                    patch.object(layer, "_ffn_score_update", side_effect=lambda x: torch.full_like(x, 5)),
                ):
                    out_feature, out_score = layer(feature, score)
                torch.testing.assert_close(out_feature, feature + 7)
                torch.testing.assert_close(out_score, score + 8)
                expected_order = (
                    ["regular", "shifted", "class"] if kind == "intra"
                    else ["class", "regular", "shifted"]
                )
                self.assertEqual([event[0] for event in events], expected_order)
                torch.testing.assert_close(events[0][1], feature)
                torch.testing.assert_close(events[0][2], score)
                torch.testing.assert_close(events[1][1], feature)
                torch.testing.assert_close(events[1][2], score + (1 if kind == "intra" else 3))
                torch.testing.assert_close(events[2][1], feature + (0 if kind == "intra" else 1))
                torch.testing.assert_close(events[2][2], score + 3)

    def test_guidance_and_selected_values_both_receive_gradients(self):
        for kind in ("intra", "inter"):
            with self.subTest(kind=kind):
                layer = EncoderRefinerLayer(
                    8, 8, 2, window_size=3, shift_size=1,
                    dropout=0, score_attention_type=kind,
                )
                feature = torch.randn(1, 3, 8, 6, 6, requires_grad=True)
                score = torch.randn_like(feature, requires_grad=True)
                # Feature output alone must supervise the earlier score update.
                out_feature, _ = layer(feature, score)
                (out_feature * torch.randn_like(out_feature)).sum().backward()
                for tensor in (feature, score):
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)
                for attention in (layer.class_attn, layer.window_attn_regular, layer.window_attn_shifted):
                    for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.out_proj):
                        self.assertTrue(torch.isfinite(projection.weight.grad).all())
                        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
                for attention in (layer.window_attn_regular, layer.window_attn_shifted):
                    self.assertGreater(attention.relative_position_bias_table.grad.abs().sum().item(), 0)

    def test_full_refiner_checkpoint_matches_outputs_and_gradients(self):
        plain = ClassConditionedEncoderRefiner(
            FakeTextEncoder(), clip_dim=8, fusion_layers=4,
            prompt_templates=["{}"] * 64, use_checkpoint=False,
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
            [layer.window_attn_regular.value_stream for layer in plain.layers],
            ["score", "feature", "score", "feature"],
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
