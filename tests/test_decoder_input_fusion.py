from __future__ import annotations

import __future__
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.decoder_input_fusion import DecoderInputFusion
from models.encoder_refiner import ClassConditionedEncoderRefiner
from models.encoder_refiner_attention import EncoderRefinerLayer
from models.maskformer_segmentation import UniversalSegmentationHead

ROOT = Path(__file__).resolve().parents[1]
KEYS = SimpleNamespace(final_logits='final_logits', sam3_teacher_logits='teacher_logits')


def sam_method(name):
    """Use production routing without importing pretrained encoder dependencies."""
    tree = ast.parse((ROOT / 'models/sam3_image.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Sam3Image')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    scope = {'torch': torch, 'F': F, 'OUTPUT_KEYS': KEYS}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 'models/sam3_image.py', 'exec', flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


class PromptHead(nn.Module):
    """Production prompt methods with a small frozen decoder stand-in."""
    apply_prompt_cross_attention = UniversalSegmentationHead.apply_prompt_cross_attention
    apply_prompt_cross_attention_to_features = UniversalSegmentationHead.apply_prompt_cross_attention_to_features

    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.cross_attn_norm = nn.LayerNorm(256)
        self.cross_attend_prompt = nn.MultiheadAttention(256, 8, dropout=0)
        self.proj = nn.Conv2d(256, 1, 1)
        self.requires_grad_(False).eval()
        self.calls = []

    def forward(self, *, backbone_feats, image_ids, encoder_hidden_states):
        self.calls.append((encoder_hidden_states.detach().clone(), torch.is_grad_enabled()))
        x = encoder_hidden_states.permute(1, 2, 0).reshape(-1, 256, 72, 72)
        return {'semantic_seg': F.interpolate(self.proj(x), size=(288, 288), mode='nearest')}


class ToyScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(8, 256, 1)

    def forward(self, *, class_names, remoteclip_feat_map):
        p = len(class_names)
        score = self.proj(remoteclip_feat_map)[:, None].expand(-1, p, -1, -1, -1)
        return score, score[:, :, :64], score.new_zeros(p, 64, 8)


def make_refiner(use_checkpoint=False, score_type='window', feature_type='window'):
    model = ClassConditionedEncoderRefiner(
        clip_text_encoder=None, clip_dim=8, clip_native_dim=8,
        prompt_templates=['{}'] * 64, dropout=0, use_checkpoint=use_checkpoint,
        score_intra_attn_type=score_type, feature_intra_attn_type=feature_type,
    )
    model.clip_score_embed = ToyScores()
    return model


def inputs(num_prompts=2):
    return dict(
        encoder_features_72=torch.randn(1, num_prompts, 256, 72, 72),
        clip_image_feat_map=torch.randn(1, 8, 36, 36),
        class_names=[str(i) for i in range(num_prompts)],
        prompt_features=torch.randn(1, num_prompts, 5, 256),
        prompt_mask=torch.tensor([[[False, False, False, True, True]] * num_prompts]),
        clip_mid_features=[torch.randn(1, 8, 36, 36), torch.randn(1, 8, 36, 36)],
        clip_mid_layer_indices=(7, 15), sam_fpn_72=torch.randn(1, 256, 72, 72),
    )


class ConstantAttention(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, *, feature, **kwargs):
        return torch.full_like(feature, self.value)


class PromptRefinerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)

    def test_feature_replacement_has_no_outer_residual(self):
        for kind in ('intra', 'inter'):
            for mode in ('window', 'local_3x3'):
                with self.subTest(kind=kind, mode=mode):
                    layer = EncoderRefinerLayer(window_size=2, shift_size=1, dropout=0,
                        score_attention_type=kind, score_intra_attn_type=mode,
                        feature_intra_attn_type=mode)
                    layer.class_attn = ConstantAttention(2)
                    constants = [3, 5] if mode == 'window' else [3, 4, 5, 6]
                    if mode == 'window':
                        layer.window_attn_regular, layer.window_attn_shifted = map(ConstantAttention, constants)
                    else:
                        layer.local_attn = nn.ModuleList(map(ConstantAttention, constants))
                    calls = []
                    def prompt_op(x, tokens, mask):
                        calls.append(x.detach().clone())
                        return x + 10  # the prompt module's internal residual is already included
                    x = torch.full((1, 2, 256, 4, 4), 100.)
                    score = torch.zeros_like(x)
                    out, updated_score = layer(x, score, prompt_op,
                        torch.zeros(1, 2, 1, 256), torch.zeros(1, 2, 1, dtype=torch.bool), x, x)
                    torch.testing.assert_close(out, torch.full_like(x, 12 if kind == 'intra' else constants[-1] + 10))
                    self.assertEqual(len(calls), 1 if kind == 'intra' else len(constants))
                    self.assertTrue(torch.isfinite(updated_score).all())
                    self.assertFalse(any('ffn' in n and 'feature' in n for n, _ in layer.named_parameters()))
                    self.assertTrue(any('ffn_fc1_score' in n for n, _ in layer.named_parameters()))

    def test_prompt_adapter_mask_order_and_single_internal_residual(self):
        head = PromptHead()
        x = torch.randn(2, 2, 256, 4, 4, requires_grad=True)
        tokens = torch.randn(2, 2, 3, 256)
        mask = torch.tensor([[[False, False, True]] * 2] * 2)
        actual = head.apply_prompt_cross_attention_to_features(x, tokens, mask)
        for b in range(2):
            for p in range(2):
                expected = head.apply_prompt_cross_attention(x[b, p].flatten(1).T[:, None],
                    tokens[b, p, :, :][:, None], mask[b, p][None])
                torch.testing.assert_close(actual[b, p], expected[:, 0].T.reshape(256, 4, 4))
        poisoned = tokens.clone()
        poisoned[:, :, 2] = 1e5
        torch.testing.assert_close(actual, head.apply_prompt_cross_attention_to_features(x, poisoned, mask))
        actual.square().mean().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        with torch.no_grad():
            self.assertFalse(head.apply_prompt_cross_attention_to_features(x, tokens, mask).requires_grad)

    def test_prompt_chunk_padding_preserves_image_prompt_order(self):
        merge = sam_method('_merge_prompt_chunks')
        a = torch.randn(2, 2, 3, 256, requires_grad=True)
        b = torch.randn(2, 1, 5, 256, requires_grad=True)
        x, mask = merge([a, b], [torch.zeros(2, 2, 3, dtype=torch.bool), torch.zeros(2, 1, 5, dtype=torch.bool)])
        torch.testing.assert_close(x[:, :2, :3], a)
        torch.testing.assert_close(x[:, 2:], b)
        self.assertTrue(mask[:, :2, 3:].all())
        self.assertFalse(mask[:, :2, :3].any())
        self.assertFalse(mask[:, 2:].any())
        self.assertFalse(x.requires_grad)

    def test_raw_encoder_cache_skips_prompt_attention(self):
        encode = sam_method('_encode_sam3_chunk')
        convert = sam_method('_hidden_states_to_feature_72')
        class Encoder:
            hidden_dim = 256
            _hidden_states_to_feature_72 = convert
            _encode_sam3_chunk = encode
            def _encode_prompt(self, *args):
                return self.tokens, self.mask, {}
            def _run_full_encoder(self, **kwargs):
                return {'memory': self.memory}
        model = Encoder()
        model.tokens = torch.randn(3, 4, 256, requires_grad=True)
        model.mask = torch.zeros(4, 3, dtype=torch.bool)
        model.memory = torch.randn(72 * 72, 4, 256, requires_grad=True)
        result = model._encode_sam3_chunk({}, None, None, 2, 2)
        torch.testing.assert_close(result['encoder_features_72'], convert(model, model.memory, 2, 2))
        torch.testing.assert_close(result['prompt_features'].reshape(4, 3, 256), model.tokens.permute(1, 0, 2))
        self.assertTrue(all(not x.requires_grad for x in result.values()))

    def test_all_spatial_modes_keep_checkpoint_outputs_and_gradients(self):
        for mode in ('window', 'local_3x3'):
            with self.subTest(mode=mode):
                head = PromptHead()
                plain = make_refiner(False, mode, mode).train()
                checked = copy.deepcopy(plain)
                checked.use_checkpoint = True
                data = inputs(1)
                results = []
                for model in (plain, checked):
                    x = data['encoder_features_72'].clone().requires_grad_()
                    args = dict(data, encoder_features_72=x,
                        prompt_cross_attention=head.apply_prompt_cross_attention_to_features)
                    out = model(**args)['refiner_features_36']
                    (out * torch.linspace(-1, 1, 256)[None, None, :, None, None]).mean().backward()
                    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                    results.append((out.detach(), x.grad, grads))
                torch.testing.assert_close(results[0][0], results[1][0])
                torch.testing.assert_close(results[0][1], results[1][1])
                self.assertGreater(results[0][1].abs().sum().item(), 0)
                self.assertEqual(results[0][2].keys(), results[1][2].keys())
                for name, grad in results[0][2].items():
                    torch.testing.assert_close(grad, results[1][2][name], atol=1e-7, rtol=1e-4)
                self.assertTrue(all(p.grad is None for p in head.parameters()))
                self.assertFalse(any('cross_attend_prompt' in n for n, _ in plain.named_parameters()))

    def test_default_four_layers_have_six_prompt_calls(self):
        model, head, data = make_refiner(), PromptHead(), inputs(1)
        calls = []
        def prompt_op(x, tokens, mask):
            calls.append(x.shape[-2:])
            return head.apply_prompt_cross_attention_to_features(x, tokens, mask)
        with torch.no_grad():
            model(**data, prompt_cross_attention=prompt_op)
        self.assertEqual(calls, [(36, 36)] * 6)

    def test_dual_branch_and_chunk_teacher_student_routing(self):
        convert = sam_method('_feature_72_to_hidden_states')
        decode = sam_method('decode_encoder_refiner_chunk_from_cache')
        class Model(nn.Module):
            _feature_72_to_hidden_states = staticmethod(convert)
            decode_encoder_refiner_chunk_from_cache = decode
            def __init__(self):
                super().__init__()
                self.encoder_refiner = make_refiner(True)
                self.segmentation_head = PromptHead()
        model, data = Model(), inputs(2)
        original = data['encoder_features_72'].requires_grad_()
        cache = {'encoder_features_72': original, 'backbone_fpn': [None, None, data['sam_fpn_72']],
                 'prompt_features': data['prompt_features'], 'prompt_mask': data['prompt_mask']}
        x = torch.randn(1, 1, 256, 36, 36, requires_grad=True)
        tokens, mask = data['prompt_features'][:, 1:2], data['prompt_mask'][:, 1:2]
        with torch.no_grad():
            teacher = model.segmentation_head.apply_prompt_cross_attention_to_features(original[:, 1:2], tokens, mask)
            fused = model.encoder_refiner.fuse_decoder_input_chunk(x.flatten(0, 1), original[:, 1:2].flatten(0, 1), data['sam_fpn_72'])
            student = model.segmentation_head.apply_prompt_cross_attention_to_features(fused[:, None], tokens, mask)
        result = model.decode_encoder_refiner_chunk_from_cache(cache, x, 1, 2, True)
        calls = model.segmentation_head.calls
        torch.testing.assert_close(calls[0][0], convert(teacher))
        torch.testing.assert_close(calls[1][0], convert(student))
        self.assertEqual([x[1] for x in calls], [False, True])
        self.assertFalse(result['teacher_logits'].requires_grad)
        result['final_logits'].square().mean().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder_refiner.decoder_input_fusion.encoder_proj[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.segmentation_head.parameters()))

    def test_streaming_proxy_matches_joint_backward(self):
        data, head = inputs(2), PromptHead()
        plain = make_refiner(False).train()
        streaming = copy.deepcopy(plain)
        results = []
        for model, use_proxy in ((plain, False), (streaming, True)):
            low = model(**data, prompt_cross_attention=head.apply_prompt_cross_attention_to_features)['refiner_features_36']
            features = low.detach().requires_grad_() if use_proxy else low
            losses = []
            for i in range(2):
                fused = model.fuse_decoder_input_chunk(features[:, i:i+1].flatten(0, 1),
                    data['encoder_features_72'][:, i:i+1].flatten(0, 1), data['sam_fpn_72'])
                updated = head.apply_prompt_cross_attention_to_features(fused[:, None],
                    data['prompt_features'][:, i:i+1], data['prompt_mask'][:, i:i+1])
                loss = head.proj(updated[:, 0]).square().mean() / 2
                if use_proxy:
                    loss.backward()
                else:
                    losses.append(loss)
            if use_proxy:
                low.backward(features.grad)
            else:
                sum(losses).backward()
            results.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        self.assertEqual(results[0].keys(), results[1].keys())
        for name, grad in results[0].items():
            torch.testing.assert_close(grad, results[1][name], atol=2e-7, rtol=2e-3)


if __name__ == '__main__':
    unittest.main()
