"""CPU loss/streaming regressions; no pretrained model weights required."""
from importlib import import_module
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


# Match tools/train.py so the checkout directory can have any name.
PACKAGE = "_ovrs_sam3_distill_tests"
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = pkg
SemanticCriterionConfig = import_module(f"{PACKAGE}.config_dataclasses").SemanticCriterionConfig
SemanticCriterion = import_module(f"{PACKAGE}.losses.semantic_criterion").SemanticCriterion
Config = import_module(f"{PACKAGE}.engine.config").Config
Trainer = import_module(f"{PACKAGE}.engine.trainer").Trainer


class DistillationTest(unittest.TestCase):
    def criterion(self, **kwargs):
        cfg = dict(final_balanced_bce_weight=0.0, sam3_mask_distill_weight=1.0)
        cfg.update(kwargs)
        return SemanticCriterion(SemanticCriterionConfig(**cfg))

    def run_loss(self, criterion, student, teacher=None, labels=None, mapping=None):
        if labels is None:
            labels = torch.zeros(student.shape[0], *student.shape[-2:], dtype=torch.long)
        if mapping is None:
            mapping = torch.arange(student.shape[1])
        outputs = {"final_logits": student}
        if teacher is not None:
            outputs["sam3_teacher_logits"] = teacher
        return criterion(outputs, {"label_map": labels}, mapping, int(mapping.max()) + 1)

    def test_separate_means_and_soft_targets(self):
        student = torch.tensor([[[[2.0, -1.0, 0.3, -0.4]]]], requires_grad=True)
        probabilities = torch.tensor([[[[0.9, 0.1, 0.2, 0.3]]]])
        teacher = torch.logit(probabilities).requires_grad_()
        result = self.run_loss(self.criterion(sam3_mask_distill_high_weight=0.7), student, teacher)
        ce = F.binary_cross_entropy_with_logits(student, probabilities, reduction="none").flatten()
        expected = 0.7 * ce[0] + 0.3 * ce[1:].mean()
        torch.testing.assert_close(result["total_loss"], expected)
        result["total_loss"].backward()
        expected_grad = (student.detach().sigmoid() - probabilities) * torch.tensor([[[[0.7, 0.1, 0.1, 0.1]]]])
        torch.testing.assert_close(student.grad, expected_grad)
        self.assertIsNone(teacher.grad)

    def test_equal_threshold_is_low_and_threshold_is_configurable(self):
        student = torch.tensor([[[[-2.0, 1.0, 3.0]]]])
        teacher = torch.logit(torch.tensor([[[[0.5, 0.7, 0.9]]]]))
        for threshold, high_indices, low_indices in ((0.5, [1, 2], [0]), (0.8, [2], [0, 1])):
            with self.subTest(threshold=threshold):
                cfg = self.criterion(sam3_mask_distill_teacher_threshold=threshold, sam3_mask_distill_high_weight=0.8)
                ce = F.binary_cross_entropy_with_logits(student, teacher.sigmoid(), reduction="none").flatten()
                expected = 0.8 * ce[high_indices].mean() + 0.2 * ce[low_indices].mean()
                torch.testing.assert_close(self.run_loss(cfg, student, teacher)["total_loss"], expected)

    def test_full_image_includes_far_ignore_pixels_in_present_mode(self):
        student = torch.zeros(1, 1, 1, 9, requires_grad=True)
        labels = torch.full((1, 1, 9), 255)
        labels[0, 0, 0] = 0
        result = self.run_loss(self.criterion(), student, torch.ones_like(student), labels)
        result["total_loss"].backward()
        self.assertTrue(bool((student.grad != 0).all()))
        # A single present GT pixel must not restrict any part of the image.
        torch.testing.assert_close(student.grad, student.grad[..., :1].expand_as(student))
        self.assertEqual(result["loss_final_bce"].item(), torch.log(torch.tensor(2.0)).item())

    def test_scope_and_alias_presence(self):
        labels = torch.tensor([[[0, 255]], [[1, 255]]])
        mapping = torch.tensor([0, 0, 1])
        for scope, expected_mask in (
            ("present", [[True, True, False], [False, False, True]]),
            ("all", [[True, True, True], [True, True, True]]),
        ):
            with self.subTest(scope=scope):
                cfg = self.criterion(sam3_mask_distill_class_scope=scope)
                student = torch.full((2, 3, 1, 2), 0.7, requires_grad=True)
                result = self.run_loss(cfg, student, torch.full_like(student, 1.0), labels, mapping)
                result["total_loss"].backward()
                self.assertEqual((student.grad.abs().sum((-2, -1)) > 0).tolist(), expected_mask)

    def test_empty_regions_and_endpoint_weights(self):
        student = torch.tensor([[[[-1.0, 2.0]]]], requires_grad=True)
        for probability in (0.2, 0.8):
            for weight in (0.0, 0.3, 1.0):
                with self.subTest(probability=probability, weight=weight):
                    teacher = torch.full_like(student, torch.logit(torch.tensor(probability)))
                    cfg = self.criterion(sam3_mask_distill_high_weight=weight)
                    coefficient = weight if probability > 0.5 else 1 - weight
                    expected = coefficient * F.binary_cross_entropy_with_logits(student, teacher.sigmoid())
                    torch.testing.assert_close(self.run_loss(cfg, student, teacher)["total_loss"], expected)

    def test_no_selected_pairs_and_disabled_distillation_need_no_teacher(self):
        for cfg, labels in (
            (self.criterion(), torch.full((1, 2, 3), 255)),
            (self.criterion(sam3_mask_distill_weight=0.0), torch.zeros(1, 2, 3, dtype=torch.long)),
        ):
            student = torch.zeros(1, 1, 2, 3, requires_grad=True)
            result = self.run_loss(cfg, student, labels=labels)
            result["total_loss"].backward()
            self.assertEqual(result["total_loss"].item(), 0.0)
            self.assertEqual(student.grad.abs().sum().item(), 0.0)

    def test_all_scope_distills_all_ignore_image(self):
        student = torch.zeros(1, 2, 2, 3, requires_grad=True)
        cfg = self.criterion(sam3_mask_distill_class_scope="all")
        labels = torch.full((1, 2, 3), 255)
        result = self.run_loss(cfg, student, torch.ones_like(student), labels)
        result["total_loss"].backward()
        self.assertGreater(result["total_loss"].item(), 0)
        self.assertTrue(bool((student.grad != 0).all()))

    def test_streaming_matches_full_loss_and_gradients(self):
        torch.manual_seed(9)
        mapping = torch.tensor([0, 1, 1, 2, 3])
        labels = torch.tensor([[[0, 255, 0], [2, 0, 255]], [[1, 255, 1], [1, 255, 255]]])
        initial = torch.randn(2, 5, 2, 3)
        teacher = torch.randn_like(initial)
        for scope in ("present", "all"):
            cfg = self.criterion(sam3_mask_distill_class_scope=scope, sam3_mask_distill_high_weight=0.7, sam3_mask_distill_weight=0.6)
            full = initial.clone().requires_grad_()
            expected = self.run_loss(cfg, full, teacher, labels, mapping)
            expected["total_loss"].backward()
            context = cfg.prepare_streaming_context({"label_map": labels}, 5, mapping, 4, (2, 3))
            for boundaries in ((0, 1, 2, 3, 4, 5), (0, 2, 5), (0, 5)):
                with self.subTest(scope=scope, boundaries=boundaries):
                    student = initial.clone().requires_grad_()
                    totals = {key: 0.0 for key in expected}
                    for start, end in zip(boundaries, boundaries[1:]):
                        output = {"final_logits": student[:, start:end], "sam3_teacher_logits": teacher[:, start:end]}
                        result = cfg.forward_chunk(output, context, start, end)
                        result["total_loss"].backward()
                        for key, value in result.items():
                            totals[key] += value.detach()
                    for key in expected:
                        torch.testing.assert_close(totals[key], expected[key])
                    torch.testing.assert_close(student.grad, full.grad)

    def test_large_half_precision_distillation_reduces_in_float32(self):
        student = torch.zeros(1, 1, 288, 288, dtype=torch.float16, requires_grad=True)
        cfg = self.criterion(sam3_mask_distill_class_scope="all")
        result = self.run_loss(cfg, student, torch.full_like(student, 2.0))
        self.assertEqual(result["loss_sam3_mask_distill_bce"].dtype, torch.float32)
        torch.testing.assert_close(result["total_loss"], torch.tensor(0.5 * torch.log(torch.tensor(2.0)).item()))
        result["total_loss"].backward()
        self.assertTrue(bool(torch.isfinite(student.grad).all()))

    def test_configuration_validation_and_base_config(self):
        root = Path(__file__).resolve().parents[1]
        cfg = Config.fromfile(root / "configs/train/isaid_loveda_full.py")
        SemanticCriterion(SemanticCriterionConfig(**cfg["model"]["criterion_cfg"]))
        for name, invalid_values in (
            ("sam3_mask_distill_teacher_threshold", (-0.1, 1.1, float("nan"))),
            ("sam3_mask_distill_high_weight", (-0.1, 1.1, float("inf"))),
            ("sam3_mask_distill_class_scope", ("invalid",)),
        ):
            for value in invalid_values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.criterion(**{name: value})
        with self.assertRaises(ValueError):
            self.run_loss(self.criterion(), torch.zeros(1, 1, 1, 1))

    def test_trainer_teacher_gate_and_proxy_backward(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature = torch.nn.Parameter(torch.zeros(1, 2, 1, 1, 1))
                self.teacher_calls = []

            def build_encoder_refiner_cache(self, batch):
                return {"chunk_class_counts": [1, 1]}

            def run_encoder_refiner_lowres_from_cache(self, *args, **kwargs):
                return {"refiner_features_36": self.feature * 1.0}

            def decode_encoder_refiner_chunk_from_cache(self, **kwargs):
                student = kwargs["refiner_feature_36_chunk"][:, :, 0].expand(-1, -1, 288, 288)
                self.teacher_calls.append(kwargs["return_teacher_logits"])
                output = {"final_logits": student}
                if kwargs["return_teacher_logits"]:
                    output["sam3_teacher_logits"] = torch.ones_like(student)
                return output

        for scope, weight, needs_teacher in (("all", 1.0, True), ("present", 1.0, False), ("all", 0.0, False)):
            with self.subTest(scope=scope, weight=weight):
                model = TinyModel()
                trainer = Trainer.__new__(Trainer)
                trainer.model = model
                trainer.criterion = self.criterion(sam3_mask_distill_class_scope=scope, sam3_mask_distill_weight=weight)
                trainer.cfg = types.SimpleNamespace(use_amp=False)
                trainer.device = torch.device("cpu")
                trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
                batch = types.SimpleNamespace(
                    find_targets=[types.SimpleNamespace(semantic_label_map=torch.full((1, 288, 288), 255))],
                    find_metadatas=[types.SimpleNamespace(num_prompts=2, num_classes=2, prompt_to_class_id=[0, 1])],
                )
                trainer._compute_train_loss_streaming(batch)
                self.assertEqual(model.teacher_calls, [needs_teacher, needs_teacher])
                self.assertEqual(bool((model.feature.grad != 0).all()), needs_teacher)


if __name__ == "__main__":
    unittest.main()
