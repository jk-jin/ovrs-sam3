"""CPU regression tests; no SAM3/RemoteCLIP weights or datasets required."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

# The checkout name may contain a hyphen; load its empty root as a package.
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ovrs_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

from ovrs_test.config_dataclasses import SemanticCriterionConfig, TrainerConfig
from ovrs_test.engine.evaluator import inference_with_tta
from ovrs_test.engine.trainer import Trainer
from ovrs_test.losses.semantic_criterion import SemanticCriterion
from ovrs_test.models.adapters.semantic_adapter import SemanticSegAdapter
from ovrs_test.models.decoder_input_fusion import DecoderInputFusion
from ovrs_test.models.mask_feedback_refiner import MaskFeedbackRefiner
from ovrs_test.models.sam3_image import Sam3Image
from ovrs_test.models.task_modes import OUTPUT_KEYS as K

torch.set_num_threads(2)


def metadata(mapping=(0, 1, 2)):
    return SimpleNamespace(num_prompts=len(mapping), num_classes=max(mapping) + 1,
                           prompt_to_class_id=list(mapping))


class SecondStageLossTests(unittest.TestCase):
    def test_full_image_bce_matches_torch_and_uneven_chunks(self):
        criterion = SemanticCriterion(SemanticCriterionConfig(
            sam3_mask_distill_weight=0, second_stage_bce_weight=0.7,
        ))
        labels = torch.tensor([[[0, 1, 255], [1, 0, 255]]])
        mapping = torch.tensor([0, 1, 0, 2])  # Alias and an absent class.
        context = criterion.prepare_streaming_context(
            {"label_map": labels}, 4, mapping, 3, target_hw=(2, 3),
        )
        logits = torch.randn(1, 4, 2, 3, requires_grad=True)
        full = criterion.forward_second_stage_chunk(logits, context, 0, 4)
        chunks = [criterion.forward_second_stage_chunk(logits[:, :3], context, 0, 3),
                  criterion.forward_second_stage_chunk(logits[:, 3:], context, 3, 4)]
        targets = (labels[:, None] == mapping[None, :, None, None]).float()
        valid = (labels != 255)[:, None].expand_as(logits)
        expected = F.binary_cross_entropy_with_logits(logits[valid], targets[valid])
        torch.testing.assert_close(full["loss_second_stage_bce"], expected)
        torch.testing.assert_close(full["total_loss"], 0.7 * expected)
        torch.testing.assert_close(sum(c["total_loss"] for c in chunks), full["total_loss"])
        full["total_loss"].backward()
        self.assertTrue((logits.grad[:, :, :, 2] == 0).all())
        self.assertGreater(logits.grad[:, 3, :, :2].abs().sum().item(), 0)

    def test_all_ignored_and_invalid_weights(self):
        criterion = SemanticCriterion(SemanticCriterionConfig(sam3_mask_distill_weight=0))
        context = criterion.prepare_streaming_context(
            {"label_map": torch.full((1, 2, 2), 255)}, 2, torch.tensor([0, 1]), 2, (2, 2),
        )
        logits = torch.randn(1, 2, 2, 2, requires_grad=True)
        loss = criterion.forward_second_stage_chunk(logits, context, 0, 2)["total_loss"]
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue((logits.grad == 0).all())
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SemanticCriterion(SemanticCriterionConfig(second_stage_bce_weight=weight))


class InferenceTests(unittest.TestCase):
    def test_whole_pixel_selection_threshold_equality_aliases_and_high_ties(self):
        adapter = SemanticSegAdapter()
        # At pixel 0, exactly 0.5 selects stage one; pixel 1 selects stage
        # two. Pixel 2 keeps stage one despite the near tie 0.9 vs 0.89.
        first = torch.tensor([[[[0.1, 0.3, 0.9]], [[0.4, 0.49, 0.89]],
                               [[0.5, 0.2, 0.1]], [[0.1, 0.1, 0.1]]]])
        second = torch.tensor([[[[0.1, 0.1, 0.1]], [[0.99, 0.2, 0.1]],
                                [[0.1, 0.1, 0.1]], [[0.1, 0.8, 0.99]]]])
        batch = SimpleNamespace(find_metadatas=[metadata((0, 1, 0, 2))])
        result = adapter({K.final_logits: torch.logit(first),
                          K.second_stage_logits: torch.logit(second)}, batch, "infer")
        self.assertEqual(result[K.final_pred].tolist(), [[[0, 2, 0]]])
        self.assertEqual(result[K.first_stage_selected].tolist(), [[[True, False, True]]])
        expected = torch.tensor([[[[0.5, 0.1, 0.9]], [[0.4, 0.2, 0.89]], [[0.1, 0.8, 0.1]]]])
        torch.testing.assert_close(result[K.raw_final_score_map], expected)
        torch.testing.assert_close(result[K.final_logits].sigmoid(), expected)

    def test_threshold_configuration(self):
        adapter = SemanticSegAdapter(stage_switch_threshold=0.8)
        first = torch.tensor([[[[0.7]], [[0.1]]]])
        second = torch.tensor([[[[0.1]], [[0.9]]]])
        result = adapter.build_infer_score_outputs(first, metadata((0, 1)), second)
        self.assertEqual(result[K.final_pred].item(), 1)
        for threshold in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                SemanticSegAdapter(stage_switch_threshold=threshold)

    def test_tta_averages_stages_before_routing(self):
        class Views(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = SemanticSegAdapter()
                self.calls = 0

            def forward(self, batch):
                value = (0.6, 0.2)[self.calls]
                self.calls += 1
                first = torch.tensor([value, 0.1]).view(1, 2, 1, 1).expand(1, 2, 2, 2)
                second = torch.tensor([0.05, 0.95]).view(1, 2, 1, 1).expand_as(first)
                return self.adapter({K.final_logits: torch.logit(first),
                                     K.second_stage_logits: torch.logit(second)}, batch, "infer")

        batch = SimpleNamespace(img_batch=torch.zeros(1, 3, 28, 28), raw_images=None,
                                find_metadatas=[metadata((0, 1))])
        result = inference_with_tta(Views(), batch, {
            "enabled": True, "scales": [1.0], "flip_modes": ["none", "h"],
        })
        self.assertTrue((result[K.final_pred] == 1).all())
        self.assertFalse(result[K.first_stage_selected].any())
        torch.testing.assert_close(result[K.raw_final_score_map][:, 1], torch.full((1, 28, 28), 0.95))


class FeedbackTests(unittest.TestCase):
    def test_checkpoint_gradients_and_detached_mask_input(self):
        torch.manual_seed(9)
        plain = MaskFeedbackRefiner(dropout=0.1, use_checkpoint=False).train()
        checked = copy.deepcopy(plain)
        checked.use_checkpoint = True
        source = torch.randn(1, 2, 256, 72, 72)
        encoder = torch.randn_like(source)
        masks = torch.randn(1, 2, 288, 288, requires_grad=True)
        gradients = []
        outputs = []
        for module in (plain, checked):
            feature = source.clone().requires_grad_(True)
            torch.manual_seed(20)
            out = module(feature, encoder, masks, chunk_size=1)
            out.square().mean().backward()
            outputs.append(out.detach())
            gradients.append(feature.grad)
            self.assertEqual(len(module.layers), 2)
            self.assertGreater(module.mask_embedding[0].weight.grad.abs().sum().item(), 0)
            self.assertIsNone(masks.grad)
        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(gradients[0], gradients[1])
        for (_, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=1e-4)


class FrozenHead(nn.Module):
    """Cheap substitute for SAM3 head, with the same input/output protocol."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(256, 1, 1)
        self.requires_grad_(False)
        self.calls = []

    def forward(self, backbone_feats, image_ids, encoder_hidden_states):
        self.calls.append((torch.is_grad_enabled(), image_ids.tolist()))
        feature = encoder_hidden_states.permute(1, 2, 0).reshape(-1, 256, 72, 72)
        logits = F.interpolate(self.proj(feature), (288, 288), mode="bilinear", align_corners=False)
        return {"semantic_seg": logits}


class RefinerFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Conv2d(4, 256, 1)
        self.decoder_input_fusion = DecoderInputFusion(use_checkpoint=True)
        self.mask_feedback_refiner = MaskFeedbackRefiner(dropout=0, use_checkpoint=True)

    def fuse_decoder_input_chunk(self, **kwargs):
        return self.decoder_input_fusion(**kwargs)


class PipelineFixture(nn.Module):
    # Exercise the production cache/decode helpers rather than copying them.
    fuse_encoder_refiner_chunk_from_cache = Sam3Image.fuse_encoder_refiner_chunk_from_cache
    refine_second_stage_from_cache = Sam3Image.refine_second_stage_from_cache
    decode_encoder_refiner_chunk_from_cache = Sam3Image.decode_encoder_refiner_chunk_from_cache
    decode_fused_feature_chunk_from_cache = Sam3Image.decode_fused_feature_chunk_from_cache
    run_encoder_refiner_from_cache = Sam3Image.run_encoder_refiner_from_cache
    _feature_72_to_hidden_states = staticmethod(Sam3Image._feature_72_to_hidden_states)

    def __init__(self):
        super().__init__()
        self.encoder_refiner = RefinerFixture()
        self.segmentation_head = FrozenHead()
        self.register_buffer("source", torch.randn(1, 3, 4, 36, 36))
        self.register_buffer("original", torch.randn(1, 3, 256, 72, 72))
        self.register_buffer("fpn", torch.randn(1, 256, 72, 72))

    def build_encoder_refiner_cache(self, batch):
        return {"cross_attended_encoder_features_72": self.original,
                "backbone_fpn": [self.fpn, self.fpn, self.fpn], "chunk_class_counts": [2, 1]}

    def run_encoder_refiner_lowres_from_cache(self, encoder_refiner_cache, batch, return_debug=False):
        return {"refiner_features_36": self.encoder_refiner.low(
            self.source.flatten(0, 1)).reshape(1, 3, 256, 36, 36)}


class StreamingTests(unittest.TestCase):
    def test_streaming_matches_full_graph_losses_and_scaled_gradients(self):
        torch.manual_seed(8)
        streamed = PipelineFixture().train()
        reference = copy.deepcopy(streamed)
        labels = torch.zeros(1, 288, 288, dtype=torch.long)
        labels[:, :, 144:] = 1  # Class 2 absent.
        labels[:, :8] = 255
        batch = SimpleNamespace(find_targets=[SimpleNamespace(semantic_label_map=labels)],
                                find_metadatas=[metadata()])
        criterion = SemanticCriterion(SemanticCriterionConfig(
            sam3_mask_distill_weight=0.5, sam3_mask_distill_boundary_width=2,
            second_stage_bce_weight=0.7,
        ))
        context = criterion.prepare_streaming_context(
            {"label_map": labels}, 3, torch.arange(3), 3,
        )
        cache = reference.build_encoder_refiner_cache(batch)
        low = reference.run_encoder_refiner_lowres_from_cache(cache, batch)["refiner_features_36"]
        first, fused, teachers, losses = [], [], [], []
        for start, end in ((0, 2), (2, 3)):
            out = reference.decode_encoder_refiner_chunk_from_cache(cache, low[:, start:end], start, end, True)
            first.append(out[K.final_logits])
            fused.append(out["fused_feature_72"])
            teachers.append(out[K.sam3_teacher_logits])
            losses.append(criterion.forward_chunk(out, context, start, end)["total_loss"])
        refined = reference.refine_second_stage_from_cache(cache, torch.cat(fused, 1), torch.cat(first, 1))
        second = torch.cat([reference.decode_fused_feature_chunk_from_cache(
            cache, refined[:, start:end], start, end) for start, end in ((0, 2), (2, 3))], 1)
        losses.append(criterion.forward_second_stage_chunk(second, context, 0, 3)["total_loss"])
        total = sum(losses)
        combined = criterion(
            {K.final_logits: torch.cat(first, 1), K.second_stage_logits: second,
             K.sam3_teacher_logits: torch.cat(teachers, 1)},
            {"label_map": labels}, torch.arange(3), 3,
        )
        torch.testing.assert_close(combined["total_loss"], total)
        (total * 8).backward()  # Non-unit scale detects accidental double scaling.

        trainer = Trainer.__new__(Trainer)
        trainer.model, trainer.criterion = streamed, criterion
        trainer.device = torch.device("cpu")
        trainer.cfg = TrainerConfig(device="cpu", use_amp=False)
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=8)
        result = trainer._compute_train_loss_streaming(batch)
        torch.testing.assert_close(result["total_loss"], total.detach())
        for (name, expected), (_, actual) in zip(reference.named_parameters(), streamed.named_parameters()):
            if not expected.requires_grad:
                self.assertIsNone(actual.grad, name)
            else:
                self.assertIsNotNone(actual.grad, name)
                torch.testing.assert_close(actual.grad, expected.grad, atol=5e-6, rtol=3e-3, msg=name)
        self.assertGreater(streamed.encoder_refiner.low.weight.grad.abs().sum().item(), 0)
        self.assertEqual([grad for grad, _ in streamed.segmentation_head.calls],
                         [False, True, False, True, True, True])

        streamed.eval()
        with torch.no_grad():
            inferred = streamed.run_encoder_refiner_from_cache(streamed.build_encoder_refiner_cache(batch), batch)
        self.assertEqual(inferred[K.final_logits].shape, (1, 3, 288, 288))
        self.assertEqual(inferred[K.second_stage_logits].shape, (1, 3, 288, 288))
        torch.testing.assert_close(inferred[K.final_logits], torch.cat(first, 1).detach())
        torch.testing.assert_close(inferred[K.second_stage_logits], second.detach(), atol=2e-5, rtol=1e-4)

    def test_decode_preserves_batch_major_image_order(self):
        model = PipelineFixture().eval()
        cache = model.build_encoder_refiner_cache(None)
        cache["cross_attended_encoder_features_72"] = model.original.expand(2, -1, -1, -1, -1)
        feature = torch.arange(1, 5, dtype=torch.float).reshape(2, 2, 1, 1, 1).expand(2, 2, 256, 72, 72)
        with torch.no_grad():
            result = model.decode_fused_feature_chunk_from_cache(cache, feature, 1, 3)
        self.assertEqual(result.shape, (2, 2, 288, 288))
        self.assertEqual(model.segmentation_head.calls[-1][1], [0, 0, 1, 1])
        expected = torch.arange(1, 5).reshape(2, 2).float() * model.segmentation_head.proj.weight.sum()
        expected = expected + model.segmentation_head.proj.bias.item()
        torch.testing.assert_close(result[:, :, 0, 0], expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
