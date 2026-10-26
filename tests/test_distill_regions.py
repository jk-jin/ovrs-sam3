import importlib
from pathlib import Path
import runpy
import sys
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
SemanticCriterion = importlib.import_module(
    f"{ROOT.name}.losses.semantic_criterion"
).SemanticCriterion
SemanticCriterionConfig = importlib.import_module(
    f"{ROOT.name}.config_dataclasses"
).SemanticCriterionConfig


class DistillRegionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.labels = torch.full((2, 7, 8), 255, dtype=torch.long)
        self.labels[0, 1, 1] = 0
        self.labels[0, 5, 6] = 1
        self.labels[1, 3, 4] = 1
        # Two prompts for class 0; class 2 is absent everywhere.
        self.mapping = torch.tensor([0, 0, 1, 2])
        self.student = torch.randn(2, 4, 7, 8)
        self.teacher = torch.randn_like(self.student, requires_grad=True)

    def criterion(self, alpha=0.7, width=1, weight=0.6):
        return SemanticCriterion(SemanticCriterionConfig(
            final_balanced_bce_weight=0.0,
            sam3_mask_distill_weight=weight,
            sam3_mask_distill_boundary_width=width,
            sam3_mask_distill_region_alpha=alpha,
        ))

    def reference(self, student, teacher, labels, mapping, width, alpha, weight):
        # Independent pixel-loop region construction: valid pixels of ANY
        # class plus ignore pixels within the current class's square band.
        selected = torch.zeros_like(student, dtype=torch.bool)
        other = torch.zeros_like(selected)
        for b in range(labels.shape[0]):
            for p, cls in enumerate(mapping.tolist()):
                positions = (labels[b] == cls).nonzero().tolist()
                if not positions:
                    continue
                for y in range(labels.shape[1]):
                    for x in range(labels.shape[2]):
                        near = any(max(abs(y - cy), abs(x - cx)) <= width
                                   for cy, cx in positions)
                        inside = labels[b, y, x] != 255 or near
                        selected[b, p, y, x] = inside
                        other[b, p, y, x] = not inside
        ce = F.binary_cross_entropy_with_logits(
            student.float(), teacher.detach().float().sigmoid(), reduction="none"
        )
        first = ce[selected].sum() / max(int(selected.sum()), 1)
        second = ce[other].sum() / max(int(other.sum()), 1)
        return weight * (alpha * first + (1 - alpha) * second), first, second

    def forward_loss(self, criterion, student=None, teacher=None, labels=None):
        return criterion(
            {"final_logits": self.student if student is None else student,
             "sam3_teacher_logits": self.teacher if teacher is None else teacher},
            {"label_map": self.labels if labels is None else labels},
            self.mapping, 3,
        )

    def test_matches_independent_full_image_reference(self):
        student = self.student.clone().requires_grad_()
        result = self.forward_loss(self.criterion(), student=student)
        expected, first, second = self.reference(
            student, self.teacher, self.labels, self.mapping, 1, 0.7, 0.6
        )
        torch.testing.assert_close(result["total_loss"], expected)
        torch.testing.assert_close(result["loss_sam3_mask_distill_valid_ring_bce"], first)
        torch.testing.assert_close(result["loss_sam3_mask_distill_other_bce"], second)
        result["total_loss"].backward()
        self.assertIsNone(self.teacher.grad)
        self.assertGreater(float(student.grad[0, 0, 6, 0].abs()), 0)
        self.assertEqual(float(student.grad[:, 3].abs().sum()), 0)
        self.assertEqual(float(student.grad[1, :2].abs().sum()), 0)

    def test_chunk_values_and_gradients_match_full_forward(self):
        criterion = self.criterion()
        full_student = self.student.clone().requires_grad_()
        full = self.forward_loss(criterion, student=full_student)
        full["total_loss"].backward()
        context = criterion.prepare_streaming_context(
            {"label_map": self.labels}, 4, self.mapping, 3, (7, 8)
        )
        self.assertEqual(
            context.distill_pixel_denom + context.distill_other_pixel_denom,
            context.num_present_pairs * 7 * 8,
        )
        for bounds in ((0, 1, 3, 4), (0, 2, 4), (0, 4)):
            with self.subTest(bounds=bounds):
                student = self.student.clone().requires_grad_()
                sums = {}
                for start, end in zip(bounds, bounds[1:]):
                    chunk = criterion.forward_chunk(
                        {"final_logits": student[:, start:end],
                         "sam3_teacher_logits": self.teacher[:, start:end]},
                        context, start, end,
                    )
                    chunk["total_loss"].backward()
                    for key, value in chunk.items():
                        sums[key] = sums.get(key, 0) + value.detach()
                for key in full:
                    torch.testing.assert_close(sums[key], full[key])
                torch.testing.assert_close(student.grad, full_student.grad)

    def test_alpha_endpoints_and_zero_boundary(self):
        for alpha in (0.0, 1.0):
            for width in (0, 1):
                with self.subTest(alpha=alpha, width=width):
                    actual = self.forward_loss(self.criterion(alpha, width))
                    expected, _, _ = self.reference(
                        self.student, self.teacher, self.labels, self.mapping,
                        width, alpha, 0.6,
                    )
                    torch.testing.assert_close(actual["total_loss"], expected)

    def test_empty_other_region_keeps_original_coefficient(self):
        labels = self.labels.clone().clamp_max(1)
        actual = self.forward_loss(self.criterion(), labels=labels)
        self.assertEqual(float(actual["loss_sam3_mask_distill_other_bce"]), 0)
        torch.testing.assert_close(
            actual["total_loss"],
            0.6 * 0.7 * actual["loss_sam3_mask_distill_valid_ring_bce"],
        )

    def test_all_ignore_and_disabled_distillation_need_no_teacher(self):
        for criterion, labels in (
            (self.criterion(), torch.full_like(self.labels, 255)),
            (self.criterion(weight=0), self.labels),
        ):
            student = self.student.clone().requires_grad_()
            result = criterion(
                {"final_logits": student}, {"label_map": labels}, self.mapping, 3
            )
            self.assertEqual(float(result["total_loss"].detach()), 0)
            result["total_loss"].backward()
            self.assertEqual(float(student.grad.abs().sum()), 0)

    def test_enabled_distillation_requires_teacher(self):
        with self.assertRaisesRegex(ValueError, "sam3_teacher_logits"):
            self.criterion()(
                {"final_logits": self.student}, {"label_map": self.labels},
                self.mapping, 3,
            )

    def test_large_half_precision_distillation_is_finite(self):
        criterion = self.criterion()
        student = torch.zeros(1, 1, 288, 288, dtype=torch.float16, requires_grad=True)
        result = criterion(
            {"final_logits": student, "sam3_teacher_logits": torch.zeros_like(student)},
            {"label_map": torch.zeros(1, 288, 288, dtype=torch.long)},
            torch.tensor([0]), 1,
        )
        value = result["loss_sam3_mask_distill_weighted"]
        self.assertTrue(bool(torch.isfinite(value)))
        torch.testing.assert_close(value, torch.tensor(0.6 * 0.7 * 0.69314718))
        value.backward()
        self.assertTrue(bool(torch.isfinite(student.grad).all()))

    def test_alpha_validation_and_base_config(self):
        for alpha in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "region_alpha"):
                self.criterion(alpha)
        cfg = runpy.run_path(str(ROOT / "configs/_base_/model/ovrs_sam3.py"))
        criterion = SemanticCriterion(SemanticCriterionConfig(**cfg["model"]["criterion_cfg"]))
        self.assertEqual(criterion.cfg.sam3_mask_distill_region_alpha, 0.5)


if __name__ == "__main__":
    unittest.main()
