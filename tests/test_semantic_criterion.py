"""CPU regressions for class-wise region BCE; no model weights needed."""

import importlib
import importlib.util
from pathlib import Path
import runpy
import sys
import unittest

import torch
import torch.nn.functional as F


# Load the repository as a package even when its checkout name contains '-'.
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ovrs_sam3_test"
if PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)

SemanticCriterion = importlib.import_module(
    f"{PACKAGE}.losses.semantic_criterion"
).SemanticCriterion
SemanticCriterionConfig = importlib.import_module(
    f"{PACKAGE}.config_dataclasses"
).SemanticCriterionConfig


class ClasswiseRegionBceTest(unittest.TestCase):
    def criterion(self, **kwargs):
        return SemanticCriterion(SemanticCriterionConfig(
            sam3_mask_distill_weight=kwargs.get("distill_weight", 0.0),
            sam3_mask_distill_boundary_width=1,
            final_bce_positive_weight=kwargs.get("positive_weight", 0.5),
        ))

    @staticmethod
    def reference(logits, labels, mapping, positive_weight=0.5):
        """Independent region selection, with one loss per image and prompt."""
        losses = []
        for image in range(logits.shape[0]):
            for prompt, class_id in enumerate(mapping.tolist()):
                prediction = logits[image, prompt]
                positive = labels[image] == class_id
                zero = prediction.sum() * 0.0
                if positive.any():
                    positive_loss = F.binary_cross_entropy_with_logits(
                        prediction[positive], torch.ones_like(prediction[positive])
                    )
                    negative = ~positive
                    negative_loss = F.binary_cross_entropy_with_logits(
                        prediction[negative], torch.zeros_like(prediction[negative])
                    ) if negative.any() else zero
                    loss = positive_weight * positive_loss + (1 - positive_weight) * negative_loss
                else:
                    valid = labels[image] != 255
                    loss = F.binary_cross_entropy_with_logits(
                        prediction[valid], torch.zeros_like(prediction[valid])
                    ) if valid.any() else zero
                losses.append(loss)
        return torch.stack(losses).mean()

    def test_per_image_presence_and_ignore_gradients(self):
        labels = torch.tensor([[[0, 255], [0, 255]], [[1, 1], [255, 255]]])
        logits = torch.tensor([
            [[[0.2, 1.1], [-0.4, 0.7]], [[3., 3.], [3., 3.]], [[4., 4.], [4., 4.]]],
            [[[5., 5.], [5., 5.]], [[-0.1, 0.8], [1.2, -0.6]], [[6., 6.], [6., 6.]]],
        ], requires_grad=True)
        result = self.criterion()(
            {"final_logits": logits}, {"label_map": labels}, torch.arange(3), 3
        )
        expected = self.reference(logits, labels, torch.arange(3))
        torch.testing.assert_close(result["loss_final_bce"], expected)
        result["total_loss"].backward()
        self.assertTrue(torch.all(logits.grad[0, 0, :, 1] > 0))
        self.assertTrue(torch.all(logits.grad[1, 1, 1, :] > 0))
        for image, channel in [(0, 1), (0, 2), (1, 0), (1, 2)]:
            valid = labels[image] != 255
            self.assertTrue(torch.all(logits.grad[image, channel][valid] > 0))
            self.assertEqual(torch.count_nonzero(logits.grad[image, channel][~valid]).item(), 0)

    def test_chunk_loss_and_gradients_match_full_reference_with_aliases(self):
        labels = torch.tensor([[[0, 0, 255], [2, 255, 2]], [[1, 255, 1], [1, 1, 255]]])
        mapping = torch.tensor([0, 0, 1, 2, 3])
        logits = torch.linspace(-2, 2, 60, dtype=torch.float64).reshape(2, 5, 2, 3)
        reference_logits = logits.clone().requires_grad_()
        # Class 0 aliases count independently; class 3 is always absent.
        reference = self.reference(reference_logits, labels, mapping, positive_weight=0.7)
        reference.backward()
        criterion = self.criterion(positive_weight=0.7)
        full = criterion(
            {"final_logits": logits}, {"label_map": labels}, mapping, 4
        )
        torch.testing.assert_close(full["loss_final_bce"], reference)
        for ranges in [[(0, 1), (1, 3), (3, 4), (4, 5)], [(0, 2), (2, 5)]]:
            chunk_logits = logits.clone().requires_grad_()
            context = criterion.prepare_streaming_context(
                {"label_map": labels}, 5, mapping, 4, target_hw=(2, 3)
            )
            contributions = []
            for start, end in ranges:
                loss = criterion.forward_chunk(
                    {"final_logits": chunk_logits[:, start:end]}, context, start, end
                )["total_loss"]
                contributions.append(loss.detach())
                loss.backward()
            torch.testing.assert_close(sum(contributions), reference)
            torch.testing.assert_close(chunk_logits.grad, reference_logits.grad)

    def test_empty_image_contributes_zero_but_keeps_batch_denominator(self):
        labels = torch.tensor([[[0, 255]], [[255, 255]]])
        logits = torch.tensor([[[[0.5, -0.5]]], [[[9., 9.]]]], requires_grad=True)
        result = self.criterion()(
            {"final_logits": logits}, {"label_map": labels}, torch.tensor([0]), 1
        )
        expected = self.reference(logits, labels, torch.tensor([0]))
        torch.testing.assert_close(result["total_loss"], expected)
        result["total_loss"].backward()
        self.assertEqual(torch.count_nonzero(logits.grad[1]).item(), 0)

    def test_all_ignore_returns_differentiable_zero(self):
        logits = torch.ones(2, 3, 2, 2, requires_grad=True)
        result = self.criterion()(
            {"final_logits": logits},
            {"label_map": torch.full((2, 2, 2), 255)}, torch.arange(3), 3,
        )
        self.assertEqual(result["total_loss"].item(), 0.0)
        result["total_loss"].backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_empty_negative_region_keeps_original_region_weights(self):
        logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
        result = self.criterion(positive_weight=0.3)(
            {"final_logits": logits}, {"label_map": torch.zeros(1, 2, 3, dtype=torch.long)},
            torch.arange(2), 2,
        )
        # Class 0: 0.3 * log(2); absent class 1: log(2); average both.
        expected = 0.65 * torch.tensor(2.0).log()
        torch.testing.assert_close(result["loss_final_bce"], expected)
        result["total_loss"].backward()
        self.assertTrue(torch.all(logits.grad[0, 0] < 0))
        self.assertTrue(torch.all(logits.grad[0, 1] > 0))

    def test_region_weights_are_independent_of_region_area(self):
        for negative_count in [1, 9]:
            labels = torch.tensor([[[0] + [255] * negative_count]])
            logits = torch.tensor([[[[0.4] + [1.2] * negative_count]]], requires_grad=True)
            for alpha in [0.0, 0.25, 1.0]:
                with self.subTest(negative_count=negative_count, alpha=alpha):
                    result = self.criterion(positive_weight=alpha)(
                        {"final_logits": logits}, {"label_map": labels}, torch.tensor([0]), 1
                    )
                    expected = alpha * F.softplus(torch.tensor(-0.4)) + (1 - alpha) * F.softplus(torch.tensor(1.2))
                    torch.testing.assert_close(result["loss_final_bce"], expected)

    def test_positive_weight_validation(self):
        for weight in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "final_bce_positive_weight"):
                self.criterion(positive_weight=weight)

    def test_base_config_constructs_criterion(self):
        settings = runpy.run_path(str(ROOT / "configs/_base_/model/ovrs_sam3.py"))
        cfg = SemanticCriterionConfig(**settings["model"]["criterion_cfg"])
        self.assertEqual(cfg.final_bce_positive_weight, 0.5)
        self.assertEqual(SemanticCriterionConfig().final_bce_positive_weight, 0.5)
        SemanticCriterion(cfg)

    def test_distillation_keeps_valid_pixels_and_outer_ignore_ring(self):
        labels = torch.full((1, 4, 4), 255)
        labels[0, 0, 0] = 0
        logits = torch.linspace(-1, 1, 32).reshape(1, 2, 4, 4).requires_grad_()
        teacher = torch.linspace(1, -1, 32).reshape_as(logits).requires_grad_()
        criterion = self.criterion(distill_weight=0.5)
        result = criterion(
            {"final_logits": logits, "sam3_teacher_logits": teacher},
            {"label_map": labels}, torch.arange(2), 2,
        )
        expected = F.binary_cross_entropy_with_logits(
            logits[0, 0, :2, :2], teacher[0, 0, :2, :2].detach().sigmoid()
        )
        torch.testing.assert_close(result["loss_sam3_mask_distill_bce"], expected)
        torch.testing.assert_close(
            result["total_loss"], result["loss_final_bce"] + 0.5 * expected
        )
        result["total_loss"].backward()
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
