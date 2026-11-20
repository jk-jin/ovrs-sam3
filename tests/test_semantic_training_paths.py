from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tests._bootstrap import load_module


core_mod = load_module("models.sam3_image")
model_mod = load_module("models.reca_sam3")
adapter_mod = load_module("models.adapters.semantic_adapter")
criterion_mod = load_module("losses.semantic_criterion")
config_mod = load_module("config_dataclasses")
trainer_mod = load_module("engine.trainer")
evaluator_mod = load_module("engine.evaluator")
OUTPUT_KEYS = load_module("models.task_modes").OUTPUT_KEYS


class CountingHead(torch.nn.Module):
    """Small frozen decoder substitute; records actual decode grad contexts."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.grad_modes = []

    def forward(self, backbone_feats, image_ids, encoder_hidden_states):
        self.grad_modes.append(torch.is_grad_enabled())
        features = encoder_hidden_states[:, :, 0].T.reshape(-1, 1, 72, 72)
        return {"semantic_seg": F.interpolate(features * self.gain, (288, 288))}


class ToyFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def fuse_semantic_detail_chunk(self, aggregated_feature, **kwargs):
        values = aggregated_feature.mean(dim=(1, 2, 3), keepdim=True)
        return (values * self.scale).expand(-1, 256, 72, 72)


class ToyCore(core_mod.Sam3Image):
    """Use production decode/chunk orchestration with synthetic cached features."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.pipeline = "reca"
        self.backbone = torch.nn.Identity()
        self.transformer = torch.nn.Identity()
        self.geometry_encoder = torch.nn.Identity()
        self.segmentation_head = CountingHead()
        self.feature_aggregation = ToyFusion()
        self.aggregation_scale = torch.nn.Parameter(torch.tensor(0.2))
        self.cache = {
            OUTPUT_KEYS.prompt_conditioned_features_72:
                torch.ones(1, 3, 1, 1, 1).expand(1, 3, 256, 72, 72),
            "backbone_fpn": [None, None, torch.zeros(1, 256, 72, 72)],
            "chunk_prompt_counts": [2, 1],
        }

    def clear_remoteclip_text_cache(self):
        pass

    def build_aggregation_cache(self, batch):
        return self.cache

    def run_aggregation_lowres_from_cache(self, aggregation_cache, batch,
                                         return_debug=False):
        return {OUTPUT_KEYS.aggregated_features:
                self.aggregation_scale.expand(1, 3, 256, 36, 36)}


def make_model_and_batch(ignore_all=False):
    core = ToyCore()
    model = model_mod.ReCASAM3(core, adapter_mod.SemanticSegAdapter(), "semantic")
    metadata = SimpleNamespace(
        num_prompts=3, num_classes=3, prompt_to_class_id=[0, 1, 2],
        class_names=["building", "water", "trees"], background_enabled=False,
    )
    labels = torch.full((1, 4, 4), 255 if ignore_all else 0, dtype=torch.long)
    batch = SimpleNamespace(
        img_batch=torch.zeros(1, 3, 4, 4), raw_images=None,
        find_metadatas=[metadata],
        find_targets=[SimpleNamespace(semantic_label_map=labels)],
    )
    return model, batch


class TestDecodePaths(unittest.TestCase):
    def test_validation_and_tta_decode_once_per_chunk_and_view(self):
        model, batch = make_model_and_batch()
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
        self.assertEqual(model.core.segmentation_head.grad_modes, [False, False])
        self.assertEqual(outputs[OUTPUT_KEYS.final_score_map].shape, (1, 3, 288, 288))
        self.assertNotIn(OUTPUT_KEYS.sam3_teacher_logits, outputs)

        model.core.segmentation_head.grad_modes.clear()
        evaluator_mod.inference_with_tta(
            model, batch,
            dict(enabled=True, scales=[1.0], flip_modes=["none", "h"], size_divisor=1),
        )
        self.assertEqual(model.core.segmentation_head.grad_modes, [False] * 4)

    def test_eval_cannot_decode_teacher_even_when_requested(self):
        model, batch = make_model_and_batch()
        model.eval()
        core = model.core
        features = core.run_aggregation_lowres_from_cache(core.cache, batch)
        outputs = core.decode_aggregation_chunk_from_cache(
            core.cache, features[OUTPUT_KEYS.aggregated_features][:, :2],
            0, 2, return_teacher_logits=True,
        )
        self.assertEqual(len(core.segmentation_head.grad_modes), 1)
        self.assertNotIn(OUTPUT_KEYS.sam3_teacher_logits, outputs)

    def test_streaming_training_teacher_only_when_needed(self):
        for weight, ignore_all in [(0.5, False), (0.0, False), (0.5, True)]:
            with self.subTest(distill_weight=weight, ignore_all=ignore_all):
                model, batch = make_model_and_batch(ignore_all)
                model.train()
                trainer = object.__new__(trainer_mod.Trainer)
                trainer.model = model
                trainer.global_iter = 0
                trainer.cfg = SimpleNamespace(use_amp=False)
                trainer.device = torch.device("cpu")
                trainer.scaler = trainer_mod._build_grad_scaler("cpu", False)
                trainer.criterion = criterion_mod.SemanticCriterion(
                    config_mod.SemanticCriterionConfig(reference_distill_weight=weight)
                )
                losses = trainer._compute_train_loss_streaming(batch)
                expected = [False, True, False, True] if weight and not ignore_all else [True, True]
                self.assertEqual(model.core.segmentation_head.grad_modes, expected)
                self.assertIsNone(model.core.segmentation_head.gain.grad)
                for parameter in [model.core.aggregation_scale,
                                  model.core.feature_aggregation.scale]:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad))
                    if not ignore_all:
                        self.assertGreater(parameter.grad.abs().item(), 0)
                if not weight or ignore_all:
                    self.assertEqual(losses["loss_reference_distill_bce"].item(), 0)

    def test_fixed_weight_shared_by_chunks_at_early_and_late_iterations(self):
        for iteration, expected_weight in ((1, 0.6), (1000, 0.6), (12000, 0.6), (20000, 0.6), (20000, 0.0)):
            with self.subTest(iteration=iteration):
                model, batch = make_model_and_batch()
                model.train()
                trainer = object.__new__(trainer_mod.Trainer)
                trainer.model = model
                trainer.global_iter = iteration - 1
                trainer.cfg = SimpleNamespace(use_amp=False)
                trainer.device = torch.device("cpu")
                trainer.scaler = trainer_mod._build_grad_scaler("cpu", False)
                trainer.criterion = criterion_mod.SemanticCriterion(
                    config_mod.SemanticCriterionConfig(
                        reference_distill_weight=expected_weight,
                    )
                )
                losses = trainer._compute_train_loss_streaming(batch)
                self.assertAlmostEqual(losses["reference_distill_weight"].item(), expected_weight)
                torch.testing.assert_close(
                    losses["loss_reference_distill_weighted"],
                    expected_weight * losses["loss_reference_distill_bce"],
                )
                expected_modes = [False, True, False, True] if expected_weight else [True, True]
                self.assertEqual(model.core.segmentation_head.grad_modes, expected_modes)
                if not expected_weight:
                    self.assertEqual(losses["loss_reference_distill_bce"].item(), 0.0)


class TestStreamingLoss(unittest.TestCase):
    def test_distillation_excludes_all_ignore_pixels_and_absent_prompts(self):
        criterion = criterion_mod.SemanticCriterion(
            config_mod.SemanticCriterionConfig(seg_bce_weight=0.0, reference_distill_weight=1.0)
        )
        labels = torch.tensor([
            [[0, 255, 1], [255, 255, 1], [0, 255, 255]],
            [[2, 2, 255], [255, 2, 255], [255, 255, 255]],
        ])
        mapping = torch.tensor([0, 0, 1, 2, 3])
        logits = torch.linspace(-2, 2, 90).reshape(2, 5, 3, 3).requires_grad_()
        teacher = torch.ones_like(logits, requires_grad=True)
        context = criterion.prepare_streaming_context(
            {"label_map": labels}, 5, mapping, 4, target_hw=(3, 3),
        )
        self.assertEqual(context.distill_pixel_denom, 15)
        losses = criterion(
            {OUTPUT_KEYS.final_logits: logits, OUTPUT_KEYS.sam3_teacher_logits: teacher},
            {"label_map": labels}, mapping, 4,
        )
        mask = (labels != 255)[:, None] & context.presence_target[:, :, None, None]
        expected = F.binary_cross_entropy_with_logits(
            logits[mask], teacher.detach().sigmoid()[mask]
        )
        torch.testing.assert_close(losses["loss_reference_distill_bce"], expected)
        losses["total_loss"].backward()
        torch.testing.assert_close(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
        self.assertGreater(logits.grad[mask].abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)

    def test_group_means_and_image_means_match_reference_loss_and_gradient(self):
        # Unequal valid areas, unequal group counts, and prompt aliases.
        labels = torch.tensor([[[0, 255], [1, 0]], [[2, 2], [2, 2]]])
        mapping = torch.tensor([0, 0, 1, 2, 3])
        for weight in (0.0, 0.8, 1.0):
            with self.subTest(weight=weight):
                criterion = criterion_mod.SemanticCriterion(
                    config_mod.SemanticCriterionConfig(
                        present_class_weight=weight, reference_distill_weight=0.0,
                    )
                )
                logits = torch.linspace(-2, 2, 40).reshape(2, 5, 2, 2).requires_grad_()
                loss = criterion(
                    {OUTPUT_KEYS.final_logits: logits},
                    {"label_map": labels}, mapping, 4,
                )["loss_seg_bce"]
                image_losses = []
                for image in range(2):
                    valid = labels[image] != 255
                    present, absent = [], []
                    for prompt, class_id in enumerate(mapping):
                        target = labels[image] == class_id
                        pair_loss = F.binary_cross_entropy_with_logits(
                            logits[image, prompt][valid], target[valid].float()
                        )
                        (present if target.any() else absent).append(pair_loss)
                    image_losses.append(
                        weight * torch.stack(present).mean()
                        + (1 - weight) * torch.stack(absent).mean()
                    )
                expected = torch.stack(image_losses).mean()
                torch.testing.assert_close(loss, expected)
                expected_grad, = torch.autograd.grad(expected, logits, retain_graph=True)
                loss.backward()
                torch.testing.assert_close(logits.grad, expected_grad)

    def test_absent_group_count_does_not_change_total_share(self):
        criterion = criterion_mod.SemanticCriterion(
            config_mod.SemanticCriterionConfig(reference_distill_weight=0.0)
        )
        for prompts in (2, 17):
            logits = torch.zeros(1, prompts, 1, 1, requires_grad=True)
            loss = criterion(
                {OUTPUT_KEYS.final_logits: logits},
                {"label_map": torch.zeros(1, 1, 1, dtype=torch.long)},
                torch.arange(prompts), prompts,
            )["loss_seg_bce"]
            torch.testing.assert_close(loss, torch.tensor(2.0).log())
            loss.backward()
            self.assertAlmostEqual(logits.grad[0, 0].abs().sum().item(), 0.4)
            self.assertAlmostEqual(logits.grad[0, 1:].abs().sum().item(), 0.1)

    def test_all_present_uses_full_weight_even_at_zero_present_share(self):
        for weight in (0.0, 0.8, 1.0):
            criterion = criterion_mod.SemanticCriterion(
                config_mod.SemanticCriterionConfig(
                    present_class_weight=weight, reference_distill_weight=0.0,
                )
            )
            labels = torch.tensor([[[0, 1]]])
            logits = torch.tensor([[[[-1.0, 2.0]], [[3.0, -4.0]]]], requires_grad=True)
            loss = criterion(
                {OUTPUT_KEYS.final_logits: logits}, {"label_map": labels},
                torch.arange(2), 2,
            )["loss_seg_bce"]
            target = (labels[:, None] == torch.arange(2)[None, :, None, None]).float()
            torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(logits, target))

    def test_ignore_only_image_contributes_zero_to_image_mean(self):
        criterion = criterion_mod.SemanticCriterion(
            config_mod.SemanticCriterionConfig(reference_distill_weight=0.0)
        )
        logits = torch.zeros(2, 2, 1, 1, requires_grad=True)
        loss = criterion(
            {OUTPUT_KEYS.final_logits: logits},
            {"label_map": torch.tensor([[[0]], [[255]]])}, torch.arange(2), 2,
        )["loss_seg_bce"]
        torch.testing.assert_close(loss, torch.tensor(2.0).log() / 2)
        loss.backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]))

    def test_ignore_only_batch_has_zero_loss_and_gradient(self):
        criterion = criterion_mod.SemanticCriterion()
        logits = torch.ones(2, 3, 2, 2, requires_grad=True)
        loss = criterion(
            {OUTPUT_KEYS.final_logits: logits},
            {"label_map": torch.full((2, 2, 2), 255)}, torch.arange(3), 3,
        )["total_loss"]
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_invalid_present_weights_are_rejected(self):
        for weight in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                criterion_mod.SemanticCriterion(
                    config_mod.SemanticCriterionConfig(present_class_weight=weight)
                )

    def test_chunked_loss_matches_full_loss_and_gradients_with_aliases(self):
        for weight in (0.0, 0.5):
            with self.subTest(weight=weight):
                self._check_chunk_equivalence(weight)

    def _check_chunk_equivalence(self, weight):
        criterion = criterion_mod.SemanticCriterion(
            config_mod.SemanticCriterionConfig(reference_distill_weight=weight)
        )
        labels = torch.tensor([[[0, 255], [1, 0]], [[2, 2], [255, 1]]])
        mapping = torch.tensor([0, 0, 1, 2])
        logits = torch.linspace(-1, 1, 32).reshape(2, 4, 2, 2).requires_grad_()
        teacher = torch.zeros_like(logits, requires_grad=True)
        full = criterion(
            {OUTPUT_KEYS.final_logits: logits, OUTPUT_KEYS.sam3_teacher_logits: teacher},
            {"label_map": labels}, mapping, 3,
        )
        full["total_loss"].backward()
        expected_grad = logits.grad.clone()
        self.assertIsNone(teacher.grad)

        chunk_logits = logits.detach().clone().requires_grad_()
        context = criterion.prepare_streaming_context(
            {"label_map": labels}, 4, mapping, 3, target_hw=(2, 2),
        )
        contributions = []
        for start, end in [(0, 1), (1, 3), (3, 4)]:
            losses = criterion.forward_chunk(
                {OUTPUT_KEYS.final_logits: chunk_logits[:, start:end],
                 OUTPUT_KEYS.sam3_teacher_logits: teacher[:, start:end]},
                context, start, end,
            )
            contributions.append(losses["total_loss"])
        total = sum(contributions)
        torch.testing.assert_close(total, full["total_loss"])
        total.backward()
        torch.testing.assert_close(chunk_logits.grad, expected_grad)


    def test_ignore_pixels_have_no_segmentation_gradient(self):
        criterion = criterion_mod.SemanticCriterion(
            config_mod.SemanticCriterionConfig(reference_distill_weight=0.0)
        )
        logits = torch.zeros(1, 2, 1, 2, requires_grad=True)
        losses = criterion(
            {OUTPUT_KEYS.final_logits: logits},
            {"label_map": torch.tensor([[[0, 255]]])}, torch.tensor([0, 1]), 2,
        )
        losses["total_loss"].backward()
        torch.testing.assert_close(logits.grad[..., 1], torch.zeros(1, 2, 1))
        self.assertGreater(logits.grad[..., 0].abs().sum().item(), 0)



if __name__ == "__main__":
    unittest.main()
