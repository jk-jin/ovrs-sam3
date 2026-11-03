"""CPU loss/streaming regression tests; no SAM3 weights or image libraries needed."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovrs_sam3', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules.setdefault('ovrs_sam3', package)
from ovrs_sam3.config_dataclasses import SemanticCriterionConfig
from ovrs_sam3.losses.semantic_criterion import SemanticCriterion


def reference_loss(logits, teacher, labels, mapping, cfg):
    """Independent image/class loop, with explicit pixel and synonym means."""
    main_groups, distill_groups = [[], []], [[], []]
    for image in range(labels.shape[0]):
        valid = labels[image] != cfg.ignore_index
        for class_id in range(int(mapping.max()) + 1):
            target = labels[image] == class_id
            present = bool(target.any())
            ring = torch.zeros_like(valid)
            width = cfg.sam3_mask_distill_boundary_width
            if width:
                # Explicit neighbourhoods independently verify boundary morphology.
                for y, x in target.nonzero().tolist():
                    ring[max(y-width, 0):y+width+1, max(x-width, 0):x+width+1] = True
                ring &= ~valid
            primary = valid | ring
            main_prompts, distill_prompts = [], []
            for prompt in (mapping == class_id).nonzero().flatten().tolist():
                main = F.binary_cross_entropy_with_logits(
                    logits[image, prompt], target.float(), reduction='none',
                )
                distill = F.binary_cross_entropy_with_logits(
                    logits[image, prompt], teacher[image, prompt].detach().sigmoid(), reduction='none',
                )
                mean = lambda values, mask: values[mask].mean() if mask.any() else values.sum() * 0
                main_value = mean(main, valid)
                distill_value = mean(distill, valid)
                if present:
                    a = cfg.final_bce_present_valid_weight
                    main_value = a * main_value + (1-a) * mean(main, ~valid)
                    a = cfg.sam3_mask_distill_present_region_weight
                    distill_value = a * mean(distill, primary) + (1-a) * mean(distill, ~primary)
                main_prompts.append(main_value)
                distill_prompts.append(distill_value)
            main_groups[int(present)].append(torch.stack(main_prompts).mean())
            distill_groups[int(present)].append(torch.stack(distill_prompts).mean())
    def grouped(groups, weight):
        mean = lambda group: torch.stack(group).mean() if group else logits.sum() * 0
        return weight * mean(groups[1]) + (1-weight) * mean(groups[0])
    return (
        grouped(main_groups, cfg.final_bce_present_class_weight),
        grouped(distill_groups, cfg.sam3_mask_distill_present_class_weight),
    )


class SemanticCriterionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.labels = torch.full((2, 9, 9), 255, dtype=torch.long)
        self.labels[0, 2:4, 2:4] = 0
        self.labels[0, 6, 6] = 1
        self.labels[1, :2] = 1
        self.labels[1, 4:6, 4:6] = 2
        self.mapping = torch.tensor([0, 1, 1, 2, 3])
        self.logits = torch.randn(2, 5, 9, 9)
        self.teacher = torch.randn_like(self.logits)
        self.cfg = SemanticCriterionConfig(
            final_balanced_bce_weight=1.7,
            final_bce_present_valid_weight=0.83,
            final_bce_present_class_weight=0.37,
            sam3_mask_distill_weight=0.6,
            sam3_mask_distill_boundary_width=1,
            sam3_mask_distill_present_region_weight=0.71,
            sam3_mask_distill_present_class_weight=0.62,
        )

    def evaluate(self, logits=None, teacher=None, labels=None, mapping=None, cfg=None):
        logits = self.logits if logits is None else logits
        teacher = self.teacher if teacher is None else teacher
        labels = self.labels if labels is None else labels
        mapping = self.mapping if mapping is None else mapping
        outputs = {'final_logits': logits}
        if teacher is not False:
            outputs['sam3_teacher_logits'] = teacher
        return SemanticCriterion(cfg or self.cfg)(
            outputs, {'label_map': labels}, mapping, int(mapping.max()) + 1,
        )

    def context(self, criterion, labels=None, mapping=None):
        labels = self.labels if labels is None else labels
        mapping = self.mapping if mapping is None else mapping
        return criterion.prepare_streaming_context(
            {'label_map': labels}, len(mapping), mapping, int(mapping.max()) + 1,
            target_hw=labels.shape[-2:],
        )

    def test_grouped_losses_and_gradients_match_independent_reference(self):
        logits = self.logits.clone().requires_grad_()
        teacher = self.teacher.clone().requires_grad_()
        actual = self.evaluate(logits, teacher)
        main, distill = reference_loss(logits, teacher, self.labels, self.mapping, self.cfg)
        torch.testing.assert_close(actual['loss_final_bce'], main)
        torch.testing.assert_close(actual['loss_sam3_mask_distill_bce'], distill)
        total = 1.7 * main + 0.6 * distill
        torch.testing.assert_close(actual['total_loss'], total)
        expected_grad, = torch.autograd.grad(total, logits)
        actual['total_loss'].backward()
        torch.testing.assert_close(logits.grad, expected_grad)
        self.assertIsNone(teacher.grad)

    def test_chunk_values_and_immediate_backward_match_full_forward(self):
        full = self.logits.clone().requires_grad_()
        expected = self.evaluate(full)
        expected['total_loss'].backward()
        for chunk_size in (1, 2, 3, 5):
            with self.subTest(chunk_size=chunk_size):
                criterion = SemanticCriterion(self.cfg)
                context = self.context(criterion)
                proxy = self.logits.clone().requires_grad_()
                totals = {}
                for start in range(0, 5, chunk_size):
                    end = min(start + chunk_size, 5)
                    result = criterion.forward_chunk(
                        {'final_logits': proxy[:, start:end], 'sam3_teacher_logits': self.teacher[:, start:end]},
                        context, start, end,
                    )
                    result['total_loss'].backward()
                    for key, value in result.items():
                        totals[key] = totals.get(key, 0) + value.detach()
                for key in totals:
                    torch.testing.assert_close(totals[key], expected[key])
                torch.testing.assert_close(proxy.grad, full.grad)

    def test_synonym_duplication_does_not_change_class_weight_or_gradient(self):
        mapping = torch.arange(4)
        logits = torch.randn(2, 4, 9, 9, requires_grad=True)
        teacher = torch.randn_like(logits)
        expected = self.evaluate(logits, teacher, mapping=mapping)
        expected['total_loss'].backward()
        duplicated = logits.detach()[:, self.mapping].clone().requires_grad_()
        actual = self.evaluate(duplicated, teacher[:, self.mapping])
        actual['total_loss'].backward()
        torch.testing.assert_close(actual['total_loss'], expected['total_loss'])
        combined = torch.zeros_like(logits)
        combined.index_add_(1, self.mapping, duplicated.grad)
        torch.testing.assert_close(combined, logits.grad)

    def test_absent_class_distills_teacher_only_on_valid_pixels(self):
        self.cfg.final_balanced_bce_weight = 0
        self.cfg.sam3_mask_distill_present_class_weight = 0
        logits = self.logits.clone().requires_grad_()
        # Class 3 is absent in both images. A positive teacher gives negative
        # gradients at zero student logits, proving targets are not forced to 0.
        logits = torch.zeros_like(logits, requires_grad=True)
        teacher = torch.full_like(logits, 3)
        result = self.evaluate(logits, teacher)
        result['total_loss'].backward()
        valid = self.labels != 255
        self.assertTrue((logits.grad[:, 4][valid] < 0).all())
        self.assertEqual(logits.grad[:, 4][~valid].abs().sum().item(), 0)
        criterion = SemanticCriterion(self.cfg)
        self.assertTrue(self.context(criterion).needs_teacher_logits(4, 5))

    def test_present_region_switch_selects_complement_of_valid_plus_ring(self):
        self.cfg.final_balanced_bce_weight = 0
        self.cfg.sam3_mask_distill_present_class_weight = 1
        for region_weight in (0, 1):
            self.cfg.sam3_mask_distill_present_region_weight = region_weight
            logits = self.logits.clone().requires_grad_()
            result = self.evaluate(logits)
            result['total_loss'].backward()
            criterion = SemanticCriterion(self.cfg)
            context = self.context(criterion)
            primary = context.valid_mask[0] | context.distill_outer_boundary_by_class[0, 0]
            selected = primary if region_weight else ~primary
            grad = logits.grad[0, 0]
            self.assertGreater(grad[selected].abs().sum().item(), 0)
            self.assertEqual(grad[~selected].abs().sum().item(), 0)

    def test_zero_width_boundary_still_distills_other_region(self):
        self.cfg.sam3_mask_distill_boundary_width = 0
        actual = self.evaluate()
        _, expected = reference_loss(self.logits, self.teacher, self.labels, self.mapping, self.cfg)
        torch.testing.assert_close(actual['loss_sam3_mask_distill_bce'], expected)
        self.assertFalse(self.context(SemanticCriterion(self.cfg)).distill_outer_boundary_by_class.any())

    def test_empty_regions_and_groups_keep_weights_without_renormalizing(self):
        labels = torch.zeros((1, 2, 2), dtype=torch.long)
        logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
        result = self.evaluate(logits, torch.zeros_like(logits), labels, torch.tensor([0]))
        ce = torch.tensor(2.).log()
        torch.testing.assert_close(result['loss_final_bce'], ce * 0.83 * 0.37)
        torch.testing.assert_close(result['loss_sam3_mask_distill_bce'], ce * 0.71 * 0.62)
        self.cfg.sam3_mask_distill_present_region_weight = 0
        criterion = SemanticCriterion(self.cfg)
        self.assertFalse(self.context(criterion, labels, torch.tensor([0])).needs_teacher_logits(0, 1))
        self.evaluate(logits, False, labels, torch.tensor([0]))

    def test_all_ignore_batch_returns_differentiable_zero_without_teacher(self):
        logits = self.logits.clone().requires_grad_()
        labels = torch.full_like(self.labels, 255)
        criterion = SemanticCriterion(self.cfg)
        self.assertFalse(self.context(criterion, labels).needs_teacher_logits(0, 5))
        result = self.evaluate(logits, False, labels)
        self.assertEqual(result['total_loss'].item(), 0)
        result['total_loss'].backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_disabled_or_zero_weight_group_does_not_require_teacher(self):
        self.cfg.sam3_mask_distill_weight = 0
        self.evaluate(teacher=False)
        self.cfg.sam3_mask_distill_weight = 1
        self.cfg.sam3_mask_distill_present_class_weight = 1
        criterion = SemanticCriterion(self.cfg)
        context = self.context(criterion)
        self.assertFalse(context.needs_teacher_logits(4, 5))
        criterion.forward_chunk({'final_logits': self.logits[:, 4:5]}, context, 4, 5)
        with self.assertRaises(ValueError):
            self.evaluate(teacher=False)

    def test_fp16_reductions_are_finite_at_actual_output_resolution(self):
        self.cfg.sam3_mask_distill_weight = 0
        logits = torch.full((1, 2, 288, 288), 10., dtype=torch.float16, requires_grad=True)
        labels = torch.zeros((1, 288, 288), dtype=torch.long)
        labels[:, 144:] = 255
        result = self.evaluate(logits, False, labels, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(result['total_loss']))
        result['total_loss'].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_config_wiring_and_ratio_validation(self):
        config = runpy.run_path(str(ROOT / 'configs/_base_/model/ovrs_sam3.py'))
        SemanticCriterion(SemanticCriterionConfig(**config['model']['criterion_cfg']))
        for name in (
            'final_bce_present_valid_weight', 'final_bce_present_class_weight',
            'sam3_mask_distill_present_region_weight', 'sam3_mask_distill_present_class_weight',
        ):
            for value in (-0.1, 1.1, float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    SemanticCriterion(SemanticCriterionConfig(**{name: value}))

    def test_actual_trainer_loop_routes_teacher_and_propagates_proxy_gradient(self):
        # Execute the actual method in isolation from heavyweight image/model
        # imports. The fake decoder exposes only the streaming model interface.
        tree = ast.parse((ROOT / 'engine/trainer.py').read_text())
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_train_loss_streaming')
        namespace = {'torch': torch, 'autocast': torch.autocast, 'Dict': dict}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<trainer streaming method>', 'exec'), namespace)
        labels = torch.full((1, 288, 288), 255, dtype=torch.long)
        labels[:, :100] = 0
        cfg = SemanticCriterionConfig(sam3_mask_distill_present_class_weight=0)
        parameter = torch.nn.Parameter(torch.randn(1, 2, 1, 1))
        calls = []
        def decode(**kwargs):
            start, end = kwargs['class_start'], kwargs['class_end']
            calls.append(kwargs['return_teacher_logits'])
            out = {'final_logits': kwargs['refiner_feature_36_chunk'].expand(-1, -1, 288, 288)}
            if kwargs['return_teacher_logits']:
                out['sam3_teacher_logits'] = torch.ones_like(out['final_logits'])
            return out
        model = SimpleNamespace(
            build_encoder_refiner_cache=lambda batch: {'chunk_class_counts': [1, 1]},
            run_encoder_refiner_lowres_from_cache=lambda *args, **kwargs: {'refiner_features_36': parameter * 2},
            decode_encoder_refiner_chunk_from_cache=decode,
        )
        batch = SimpleNamespace(
            find_targets=[SimpleNamespace(semantic_label_map=labels)],
            find_metadatas=[SimpleNamespace(num_prompts=2, num_classes=2, prompt_to_class_id=[0, 1])],
        )
        runner = SimpleNamespace(
            model=model, cfg=SimpleNamespace(use_amp=False), device=torch.device('cpu'),
            criterion=SemanticCriterion(cfg), scaler=SimpleNamespace(scale=lambda value: value),
        )
        actual = namespace['_compute_train_loss_streaming'](runner, batch)
        self.assertEqual(calls, [False, True])
        expected_input = parameter.detach().clone().requires_grad_()
        logits = (expected_input * 2).expand(-1, -1, 288, 288)
        expected = SemanticCriterion(cfg)(
            {'final_logits': logits, 'sam3_teacher_logits': torch.ones_like(logits)},
            {'label_map': labels}, torch.arange(2), 2,
        )
        expected['total_loss'].backward()
        torch.testing.assert_close(actual['total_loss'], expected['total_loss'])
        torch.testing.assert_close(parameter.grad, expected_input.grad)


if __name__ == '__main__':
    unittest.main()
