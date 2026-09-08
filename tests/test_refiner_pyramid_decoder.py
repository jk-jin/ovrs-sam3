"""CPU regression checks; run with python -m unittest discover -s tests."""

import copy
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


# Load this standalone module without importing the SAM3 model dependencies.
_path = Path(__file__).resolve().parents[1] / "models/refiner_pyramid_decoder.py"
_spec = importlib.util.spec_from_file_location("pyramid_decoder", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
SemanticDetailFusionStage = _module.SemanticDetailFusionStage
RefinerPyramidDecoder = _module.RefinerPyramidDecoder


class ZeroFeatures(nn.Module):
    def forward(self, features):
        return torch.zeros_like(features)


class PyramidDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_default_channels_and_class_chunk_equivalence(self):
        stage = SemanticDetailFusionStage()
        refiner = torch.randn(6, 256, 6, 6, requires_grad=True)
        pixel = torch.randn(6, 256, 12, 12)
        fpn = torch.randn(2, 256, 12, 12)
        result = stage(refiner, pixel, fpn)
        self.assertEqual(result.shape, (6, 256, 12, 12))
        with torch.no_grad():
            chunks = [
                stage(
                    refiner.reshape(2, 3, 256, 6, 6)[:, c],
                    pixel.reshape(2, 3, 256, 12, 12)[:, c],
                    fpn,
                )
                for c in range(3)
            ]
        chunked = torch.stack(chunks, dim=1).reshape_as(result)
        torch.testing.assert_close(result, chunked, atol=2e-6, rtol=2e-5)
        result.square().mean().backward()
        self.assertGreater(refiner.grad.abs().sum().item(), 0)
        for name, parameter in stage.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_pixel_path_survives_without_branch_outputs(self):
        stage = SemanticDetailFusionStage()
        stage.semantic_block = ZeroFeatures()
        stage.detail_block = ZeroFeatures()
        refiner = torch.randn(2, 256, 4, 4)
        pixel = torch.randn(2, 256, 8, 8, requires_grad=True)
        fpn = torch.randn(1, 256, 8, 8)
        result = stage(refiner, pixel, fpn)
        # With both block outputs zeroed, old input residuals must not leak in.
        torch.testing.assert_close(result, stage(refiner * 2, pixel, fpn * 3))
        self.assertFalse(torch.allclose(result, stage(refiner, pixel.flip(-1), fpn)))
        result.square().mean().backward()
        self.assertGreater(pixel.grad.abs().sum().item(), 0)

    def test_full_resolution_checkpoint_output_and_gradients(self):
        # Smaller channels keep the three real spatial scales cheap on CPU.
        plain = RefinerPyramidDecoder(hidden_dim=16, branch_dim=8, use_checkpoint=False)
        checked = copy.deepcopy(plain)
        checked.use_checkpoint = True
        refiner = torch.randn(2, 16, 36, 36, requires_grad=True)
        checked_refiner = refiner.detach().clone().requires_grad_()
        pixels = [torch.randn(2, 16, hw, hw) for hw in (72, 144, 288)]
        fpns = [torch.randn(1, 16, hw, hw) for hw in (72, 144, 288)]
        expected = plain(refiner, *pixels, *fpns)
        actual = checked(checked_refiner, *pixels, *fpns)
        self.assertEqual(actual.shape, (2, 16, 288, 288))
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(checked_refiner.grad, refiner.grad)
        for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            torch.testing.assert_close(q.grad, p.grad)


if __name__ == "__main__":
    unittest.main()
