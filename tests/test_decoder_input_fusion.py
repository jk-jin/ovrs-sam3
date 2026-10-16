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

from models.decoder_input_fusion import DecoderInputFusion


ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name):
    """Load the actual wrapper without heavyweight SAM3/CLIP dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch, "OUTPUT_KEYS": SimpleNamespace(
        final_logits="final_logits", sam3_teacher_logits="teacher_logits",
    )}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[method_name]


class FrozenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(256, 1, 1).requires_grad_(False)
        self.calls = []

    def forward(self, *, backbone_feats, image_ids, encoder_hidden_states):
        self.calls.append((encoder_hidden_states.detach().clone(), image_ids.clone(), torch.is_grad_enabled()))
        feature = encoder_hidden_states.permute(1, 2, 0).reshape(-1, 256, 72, 72)
        return {"semantic_seg": F.interpolate(self.proj(feature), size=(288, 288), mode="nearest")}


class DecoderInputFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_image_prompt_order_matches_individual_decoding(self):
        model = DecoderInputFusion(use_checkpoint=False).eval()
        refiner = torch.randn(4, 256, 36, 36)
        fpn = torch.randn(2, 256, 72, 72)
        with torch.no_grad():
            actual = model(refiner, fpn)
            expected = torch.cat([
                model(refiner[i:i + 1], fpn[i // 2:i // 2 + 1])
                for i in range(4)
            ])
        self.assertEqual(actual.shape, (4, 256, 72, 72))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_checkpoint_preserves_outputs_and_gradients(self):
        plain = DecoderInputFusion(use_checkpoint=False).train()
        checked = copy.deepcopy(plain)
        checked.use_checkpoint = True
        refiner = torch.randn(1, 256, 36, 36)
        fpn = torch.randn(1, 256, 72, 72)
        results = []
        for model in (plain, checked):
            x, y = refiner.clone().requires_grad_(), fpn.clone().requires_grad_()
            out = model(x, y)
            out.square().mean().backward()
            results.append((out.detach(), x.grad, y.grad, [p.grad for p in model.parameters()]))
        for a, b in zip(results[0][:3], results[1][:3]):
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.isfinite(a).all())
        for a, b in zip(results[0][3], results[1][3]):
            self.assertIsNotNone(a)
            self.assertGreater(a.abs().sum().item(), 0)
            torch.testing.assert_close(a, b)

    def test_frozen_head_keeps_input_gradients_and_no_grad_is_respected(self):
        model = DecoderInputFusion().train()
        head = nn.Conv2d(256, 1, 1).requires_grad_(False).eval()
        x = torch.randn(1, 256, 36, 36, requires_grad=True)
        fpn = torch.randn(1, 256, 72, 72)
        head(model(x, fpn)).square().mean().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        with torch.no_grad():
            self.assertFalse(model(x, fpn).requires_grad)

    def test_invalid_shapes_and_pair_counts(self):
        model = DecoderInputFusion()
        cases = [
            (torch.empty(1, 256, 35, 36), torch.empty(1, 256, 72, 72)),
            (torch.empty(1, 256, 36, 36), torch.empty(1, 128, 72, 72)),
            (torch.empty(3, 256, 36, 36), torch.empty(2, 256, 72, 72)),
            (torch.empty(0, 256, 36, 36), torch.empty(1, 256, 72, 72)),
            (torch.empty(1, 256, 36, 36), torch.empty(0, 256, 72, 72)),
            (torch.empty(1, 256, 36, 36, device="meta"), torch.empty(1, 256, 72, 72)),
        ]
        for x, fpn in cases:
            with self.subTest(x=x.shape, fpn=fpn.shape), self.assertRaises(ValueError):
                model(x, fpn)

    def test_actual_chunk_wrapper_uses_fpn_only_and_keeps_teacher_separate(self):
        wrapper = load_method("models/encoder_refiner.py", "ClassConditionedEncoderRefiner", "fuse_decoder_input_chunk")
        convert = load_method("models/sam3_image.py", "Sam3Image", "_feature_72_to_hidden_states")
        decode = load_method("models/sam3_image.py", "Sam3Image", "decode_encoder_refiner_chunk_from_cache")

        class Refiner(nn.Module):
            fuse_decoder_input_chunk = wrapper

            def __init__(self):
                super().__init__()
                self.decoder_input_fusion = DecoderInputFusion(False)

        class Model(nn.Module):
            _feature_72_to_hidden_states = staticmethod(convert)
            decode_encoder_refiner_chunk_from_cache = decode

            def __init__(self):
                super().__init__()
                self.encoder_refiner = Refiner()
                self.segmentation_head = FrozenHead()

        model = Model()
        original = torch.randn(1, 3, 256, 72, 72, requires_grad=True)
        cache = {"cross_attended_encoder_features_72": original,
                 "backbone_fpn": [None, None, torch.randn(1, 256, 72, 72)]}
        x = torch.randn(1, 1, 256, 36, 36, requires_grad=True)
        result = model.decode_encoder_refiner_chunk_from_cache(cache, x, 1, 2, True)
        head = model.segmentation_head
        torch.testing.assert_close(head.calls[0][0], convert(original[:, 1:2]).detach())
        self.assertFalse(head.calls[0][2])
        self.assertTrue(head.calls[1][2])
        self.assertFalse(result["teacher_logits"].requires_grad)
        self.assertEqual(result["final_logits"].shape, (1, 1, 288, 288))
        result["final_logits"].square().mean().backward()
        self.assertIsNone(original.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in model.encoder_refiner.parameters()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        with torch.no_grad():
            changed = dict(cache, cross_attended_encoder_features_72=original + 100)
            other = model.decode_encoder_refiner_chunk_from_cache(changed, x, 1, 2, True)
            torch.testing.assert_close(result["final_logits"].detach(), other["final_logits"])
            self.assertFalse(torch.allclose(result["teacher_logits"], other["teacher_logits"]))


if __name__ == "__main__":
    unittest.main()
