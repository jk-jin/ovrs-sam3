"""CPU regression tests; run: python -m unittest discover -s tests -v."""

import importlib
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


# Load the real loss and config without importing SAM3/CLIP model builders.
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_ovrs_confidence_tests"
for suffix in ("", ".losses", ".models"):
    module = types.ModuleType(PACKAGE + suffix)
    module.__path__ = [str(ROOT.joinpath(*suffix.strip(".").split(".")))]
    sys.modules[module.__name__] = module

SemanticCriterionConfig = importlib.import_module(
    f"{PACKAGE}.config_dataclasses"
).SemanticCriterionConfig
SemanticCriterion = importlib.import_module(
    f"{PACKAGE}.losses.semantic_criterion"
).SemanticCriterion


class ConfidenceBCETest(unittest.TestCase):
    def criterion(self, **kwargs):
        config = dict(
            final_balanced_bce_weight=0.0,
            final_dice_weight=0.0,
            sam3_mask_distill_weight=0.0,
            confidence_bce_weight=0.2,
        )
        config.update(kwargs)
        return SemanticCriterion(SemanticCriterionConfig(**config))

    def loss(self, criterion, logits, labels, mapping):
        return criterion(
            {"final_logits": logits},
            {"label_map": labels},
            torch.tensor(mapping),
            max(mapping) + 1,
        )

    def test_threshold_targets_and_gradient_direction(self):
        # p=0.5 equals the threshold: target must be 1, gradient negative.
        logits = torch.tensor([[[[-2.0, 0.0, 2.0]]]], requires_grad=True)
        labels = torch.full((1, 1, 3), 255)
        result = self.loss(
            self.criterion(confidence_bce_present_only=False), logits, labels, [0]
        )
        expected_target = torch.tensor([[[[0.0, 1.0, 1.0]]]])
        expected = F.binary_cross_entropy_with_logits(logits, expected_target)
        torch.testing.assert_close(result["loss_confidence_bce"], expected)
        result["total_loss"].backward()
        torch.testing.assert_close(
            logits.grad, 0.2 * (logits.detach().sigmoid() - expected_target) / 3
        )

    def test_configured_threshold_is_applied_to_probability(self):
        logits = torch.tensor([[[[0.0, 0.5, 2.0]]]], requires_grad=True)
        result = self.loss(
            self.criterion(
                confidence_bce_threshold=0.7, confidence_bce_present_only=False
            ),
            logits, torch.zeros((1, 1, 3), dtype=torch.long), [0],
        )
        expected = F.binary_cross_entropy_with_logits(
            logits, torch.tensor([[[[0.0, 0.0, 1.0]]]])
        )
        torch.testing.assert_close(result["loss_confidence_bce"], expected)

    def test_presence_is_per_image_and_aliases_include_ignore_pixels(self):
        labels = torch.tensor([[[0, 255]], [[1, 255]]])
        logits = torch.zeros((2, 4, 1, 2), requires_grad=True)
        mapping = [0, 1, 0, 2]
        criterion = self.criterion()
        context = criterion.prepare_streaming_context(
            {"label_map": labels}, 4, torch.tensor(mapping), 3, (1, 2)
        )
        self.assertEqual(context.confidence_pixel_denom, 6)
        result = self.loss(criterion, logits, labels, mapping)
        result["total_loss"].backward()
        selected = torch.tensor([[True, False, True, False],
                                 [False, True, False, False]])
        self.assertTrue((logits.grad[selected] < 0).all())
        self.assertTrue((logits.grad[~selected] == 0).all())
        # Both labeled and ignored positions receive equal confidence gradients.
        torch.testing.assert_close(logits.grad[..., 0], logits.grad[..., 1])
        torch.testing.assert_close(
            result["loss_confidence_bce"], torch.tensor(2.0).log()
        )

    def test_chunk_values_and_gradients_match_full_with_existing_losses(self):
        torch.manual_seed(7)
        labels = torch.tensor([[[0, 255, 0], [255, 0, 0]],
                               [[1, 1, 255], [255, 1, 1]]])
        mapping = torch.tensor([0, 1, 0, 2, 1])
        source = torch.randn(2, 5, 2, 3)
        teacher = torch.randn_like(source)
        for present_only in (True, False):
            for chunk_size in (1, 2, 3):
                with self.subTest(present_only=present_only, chunk_size=chunk_size):
                    criterion = self.criterion(
                        confidence_bce_present_only=present_only,
                        confidence_bce_threshold=0.65,
                        final_balanced_bce_weight=1.0,
                        final_dice_weight=0.3,
                        sam3_mask_distill_weight=0.5,
                    )
                    full = source.clone().requires_grad_()
                    full_result = criterion(
                        {"final_logits": full, "sam3_teacher_logits": teacher},
                        {"label_map": labels}, mapping, 3,
                    )
                    full_result["total_loss"].backward()
                    chunked = source.clone().requires_grad_()
                    context = criterion.prepare_streaming_context(
                        {"label_map": labels}, 5, mapping, 3, (2, 3)
                    )
                    accumulated = {key: torch.tensor(0.0) for key in full_result}
                    for start in range(0, 5, chunk_size):
                        end = min(start + chunk_size, 5)
                        result = criterion.forward_chunk(
                            {"final_logits": chunked[:, start:end],
                             "sam3_teacher_logits": teacher[:, start:end]},
                            context, start, end,
                        )
                        result["total_loss"].backward()
                        for key, value in result.items():
                            accumulated[key] += value.detach()
                    for key, expected in full_result.items():
                        torch.testing.assert_close(accumulated[key], expected)
                    torch.testing.assert_close(chunked.grad, full.grad)

    def test_all_ignore_presence_mode_returns_differentiable_zero(self):
        logits = torch.randn(2, 2, 2, 3, requires_grad=True)
        result = self.loss(
            self.criterion(), logits, torch.full((2, 2, 3), 255), [0, 1]
        )
        self.assertEqual(result["loss_confidence_bce"].item(), 0.0)
        result["total_loss"].backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_all_ignore_all_class_mode_still_computes_loss(self):
        logits = torch.zeros((1, 2, 1, 2), requires_grad=True)
        result = self.loss(
            self.criterion(confidence_bce_present_only=False),
            logits, torch.full((1, 1, 2), 255), [0, 1],
        )
        torch.testing.assert_close(
            result["loss_confidence_bce"], torch.tensor(2.0).log()
        )
        result["total_loss"].backward()
        self.assertTrue((logits.grad < 0).all())

    def test_zero_weight_preserves_original_bce_and_gradients(self):
        logits = torch.tensor([[[[-0.4, 1.3]]]], requires_grad=True)
        result = self.loss(
            self.criterion(confidence_bce_weight=0.0, final_balanced_bce_weight=1.0),
            logits, torch.tensor([[[0, 255]]]), [0],
        )
        expected = F.binary_cross_entropy_with_logits(
            logits[..., :1], torch.ones_like(logits[..., :1])
        )
        torch.testing.assert_close(result["total_loss"], expected)
        self.assertEqual(result["loss_confidence_bce"].item(), 0.0)
        self.assertEqual(result["loss_confidence_weighted"].item(), 0.0)
        result["total_loss"].backward()
        self.assertEqual(logits.grad[..., 1].item(), 0.0)
        torch.testing.assert_close(
            logits.grad[..., :1], logits.detach()[..., :1].sigmoid() - 1
        )

    def test_invalid_configuration(self):
        for weight in (-0.1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                self.criterion(confidence_bce_weight=weight)
        for threshold in (0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.criterion(confidence_bce_threshold=threshold)
        with self.assertRaises(TypeError):
            self.criterion(confidence_bce_present_only="false")

    def test_autocast_and_extreme_logits_have_finite_loss_and_gradient(self):
        logits = torch.tensor(
            [[[[-100.0, -0.2, 0.0, 0.2, 100.0]]]],
            dtype=torch.bfloat16, requires_grad=True,
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = self.loss(
                self.criterion(confidence_bce_present_only=False),
                logits, torch.zeros((1, 1, 5), dtype=torch.long), [0],
            )
        self.assertTrue(torch.isfinite(result["total_loss"]))
        result["total_loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
