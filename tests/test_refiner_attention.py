"""Content-only intra-class attention, masks and alternating-layer gradients."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovrs_sam3', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)],
)
sys.modules.setdefault('ovrs_sam3', importlib.util.module_from_spec(spec))
from ovrs_sam3.models.encoder_refiner_attention import (
    EncoderRefinerLayer, LocalScoreAttention, WindowScoreAttention,
)


def content_reference(module, feature, score, guidance, local=False):
    """Explicit valid key sets; no window partition, unfold or position terms."""
    batch, classes, channels, height, width = feature.shape
    f = feature.permute(0, 1, 3, 4, 2)
    s = score.permute(0, 1, 3, 4, 2)
    g = guidance.permute(0, 1, 3, 4, 2)
    qk = torch.cat([s, f, g], dim=-1)
    heads, head_dim = module.num_heads, channels // module.num_heads
    shape = (batch, classes, height, width, heads, head_dim)
    queries = module.q_proj(qk).reshape(shape)
    keys = module.k_proj(qk).reshape(shape)
    values = module.v_proj(f if module.value_stream == 'feature' else s).reshape(shape)
    outputs = []
    for b in range(batch):
        for c in range(classes):
            for y in range(height):
                for x in range(width):
                    if local:
                        positions = [
                            (yy, xx)
                            for yy in range(max(y-1, 0), min(y+2, height))
                            for xx in range(max(x-1, 0), min(x+2, width))
                        ]
                    else:
                        window, shift = module.window_size, module.shift_size
                        positions = [
                            (yy, xx) for yy in range(height) for xx in range(width)
                            if (yy-shift)//window == (y-shift)//window
                            and (xx-shift)//window == (x-shift)//window
                        ]
                    k = torch.stack([keys[b, c, yy, xx] for yy, xx in positions], dim=1)
                    v = torch.stack([values[b, c, yy, xx] for yy, xx in positions], dim=1)
                    logits = (queries[b, c, y, x][:, None] * k).sum(-1) / head_dim**0.5
                    outputs.append((logits.softmax(-1)[..., None] * v).sum(1).reshape(channels))
    output = torch.stack(outputs).reshape(batch, classes, height, width, channels)
    return module.out_proj(output).permute(0, 1, 4, 2, 3).contiguous()


class RefinerAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def check_content_reference(self, module, local=False):
        inputs = (
            torch.randn(1, 2, 4, 6, 6, requires_grad=True),
            torch.randn(1, 2, 6, 6, 6, requires_grad=True),
            torch.randn(1, 2, 3, 6, 6, requires_grad=True),
        )
        actual = module(*inputs)
        expected = content_reference(module, *inputs, local=local)
        torch.testing.assert_close(actual, expected)
        variables = inputs + tuple(module.parameters())
        actual_grads = torch.autograd.grad(actual.square().mean(), variables)
        expected_grads = torch.autograd.grad(expected.square().mean(), variables)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_regular_and_shifted_windows_match_content_only_reference(self):
        for stream in ('feature', 'score'):
            for shift in (0, 1):
                with self.subTest(stream=stream, shift=shift):
                    module = WindowScoreAttention(
                        hidden_dim=4, score_embed_dim=6, num_heads=2,
                        window_size=3, shift_size=shift, dropout=0,
                        value_stream=stream, guidance_dim=3,
                    )
                    self.check_content_reference(module)

    def test_local_attention_matches_reference_with_clipped_image_edges(self):
        for stream in ('feature', 'score'):
            with self.subTest(stream=stream):
                module = LocalScoreAttention(
                    hidden_dim=4, score_embed_dim=6, num_heads=2,
                    dropout=0, value_stream=stream, guidance_dim=3,
                )
                self.check_content_reference(module, local=True)

    def test_all_intra_modes_remove_position_parameters_and_buffers(self):
        for score_kind in ('window', 'local_3x3'):
            for feature_kind in ('window', 'local_3x3'):
                for order in ('intra', 'inter'):
                    with self.subTest(score=score_kind, feature=feature_kind, order=order):
                        layer = EncoderRefinerLayer(
                            hidden_dim=8, score_embed_dim=8, num_heads=2,
                            window_size=3, shift_size=1, dropout=0,
                            score_attention_type=order,
                            score_intra_attn_type=score_kind,
                            feature_intra_attn_type=feature_kind,
                        )
                        self.assertFalse(any('relative_position' in key for key in layer.state_dict()))
                        for module in layer.modules():
                            if isinstance(module, (WindowScoreAttention, LocalScoreAttention)):
                                self.assertFalse(hasattr(module, 'relative_position_bias_table'))
                                self.assertFalse(hasattr(module, 'relative_position_index'))
                                self.assertFalse(hasattr(module, '_get_relative_position_bias'))

    def test_alternating_mode_combinations_match_checkpoint_outputs_and_gradients(self):
        for score_kind in ('window', 'local_3x3'):
            for feature_kind in ('window', 'local_3x3'):
                with self.subTest(score=score_kind, feature=feature_kind):
                    layers = torch.nn.ModuleList([
                        EncoderRefinerLayer(
                            hidden_dim=8, score_embed_dim=8, num_heads=2,
                            window_size=3, shift_size=1, dropout=0,
                            score_attention_type=order,
                            score_intra_attn_type=score_kind,
                            feature_intra_attn_type=feature_kind,
                        ) for order in ('intra', 'inter', 'intra', 'inter')
                    ])
                    checked_layers = copy.deepcopy(layers)
                    inputs = tuple(torch.randn(1, 3, d, 6, 6, requires_grad=True)
                                   for d in (8, 8, 256, 256))
                    checked_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
                    def run(modules, values, use_checkpoint):
                        feature, score, clip, fpn = values
                        for layer in modules:
                            if use_checkpoint:
                                feature, score = checkpoint(
                                    layer, feature, score, clip, fpn, use_reentrant=False,
                                )
                            else:
                                feature, score = layer(feature, score, clip, fpn)
                        loss = feature.square().mean() + score.square().mean()
                        grads = torch.autograd.grad(loss, values + tuple(modules.parameters()))
                        return (feature, score), grads
                    expected, expected_grads = run(layers, inputs, False)
                    actual, actual_grads = run(checked_layers, checked_inputs, True)
                    for a, e in zip(actual + actual_grads, expected + expected_grads):
                        self.assertTrue(torch.isfinite(a).all())
                        torch.testing.assert_close(a, e)


if __name__ == '__main__':
    unittest.main()
