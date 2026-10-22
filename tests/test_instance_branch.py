"""CPU regressions; no SAM3/RemoteCLIP downloads or dataset required."""

import copy
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if "ovrs_sam3" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "ovrs_sam3", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)


def load(name):
    return importlib.import_module("ovrs_sam3." + name)


branch_module = load("models.learnable_instance_branch")
LearnableInstanceBranch = branch_module.LearnableInstanceBranch
aggregate_teacher_masks = branch_module.aggregate_teacher_masks
SemanticCriterion = load("losses.semantic_criterion").SemanticCriterion
SemanticCriterionConfig = load("config_dataclasses").SemanticCriterionConfig
Sam3Image = load("models.sam3_image").Sam3Image
SAM3ModelBuilder = load("model_builder").SAM3ModelBuilder
OUTPUT_KEYS = load("models.task_modes").OUTPUT_KEYS
torch.set_num_threads(1)


def criterion(**kwargs):
    values = dict(sam3_mask_distill_weight=0.0, final_dice_weight=0.0)
    values.update(kwargs)
    return SemanticCriterion(SemanticCriterionConfig(**values))


class FakeHead(nn.Module):
    """Small spatial output, with real frozen projections and input gradients."""
    def __init__(self):
        super().__init__()
        self.semantic = nn.Conv2d(256, 1, 1)
        self.instance = nn.Conv2d(256, 256, 1)
        self.calls = []
        self.requires_grad_(False)

    def forward(self, backbone_feats, image_ids, encoder_hidden_states, return_instance_features=False):
        self.calls.append((torch.is_grad_enabled(), return_instance_features))
        feature = encoder_hidden_states.permute(1, 2, 0).reshape(-1, 256, 72, 72)
        feature = F.adaptive_avg_pool2d(feature, 4)
        result = {"semantic_seg": self.semantic(feature)}
        if return_instance_features:
            result["instance_features"] = self.instance(feature)
        return result


class FakeRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Conv2d(256, 256, 1)
        self.instance_branch = LearnableInstanceBranch(dropout=0.0, use_checkpoint=False)

    def fuse_decoder_input_chunk(self, refiner_feature_36, original_encoder_feature_72, sam_fpn_72):
        return self.fusion(F.interpolate(refiner_feature_36, size=72))


def decoding_fixture(prompts=1):
    core = Sam3Image.__new__(Sam3Image)
    nn.Module.__init__(core)
    core.instance_fusion_enabled = True
    core.instance_teacher_score_threshold = 0.5
    core.segmentation_head = FakeHead()
    core.encoder_refiner = FakeRefiner()
    cache = {
        "cross_attended_encoder_features_72": torch.randn(1, prompts, 256, 72, 72),
        "backbone_fpn": [torch.zeros(1), torch.zeros(1), torch.zeros(1)],
        "mask_queries": torch.randn(1, prompts, 200, 256),
        "matching_probabilities": torch.rand(1, prompts, 200),
    }
    return core, cache


class InstanceBranchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_teacher_strict_threshold_soft_max_and_empty_pair(self):
        logits = torch.tensor([[[[10., -8.]], [[-2., 3.]], [[1., 2.]]],
                               [[[9., 9.]], [[8., 8.]], [[7., 7.]]]], requires_grad=True)
        scores = torch.tensor([[0.5, 0.6, 0.9], [0.1, 0.5, 0.0]])
        teacher, valid = aggregate_teacher_masks(logits, scores, 0.5)
        torch.testing.assert_close(teacher[0], logits[0, 1:].sigmoid().amax(0))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(teacher[1].sum().item(), 0)
        self.assertFalse(teacher.requires_grad)

    def test_learned_vectors_and_frozen_head_input_receive_gradients(self):
        branch = LearnableInstanceBranch(dropout=0.0, use_checkpoint=False)
        head = nn.Conv2d(256, 256, 1).requires_grad_(False)
        fused = torch.randn(2, 256, 2, 3, requires_grad=True)
        pixels = torch.randn(2, 256, 4, 5, requires_grad=True)
        queries = torch.randn(2, 200, 256)
        output = branch(fused, queries, head(pixels))
        self.assertEqual(output.shape, (2, 1, 4, 5))
        output.square().mean().backward()
        for tensor in (fused, pixels, branch.vectors.weight):
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_checkpoint_matches_outputs_and_gradients_with_dropout(self):
        plain = LearnableInstanceBranch(dropout=0.1, use_checkpoint=False).train()
        checked = copy.deepcopy(plain)
        checked.use_checkpoint = True
        inputs = [torch.randn(1, 256, 2, 2), torch.randn(1, 200, 256), torch.randn(1, 256, 3, 3)]
        outputs = []
        gradients = []
        for module in (plain, checked):
            args = [x.clone().requires_grad_(i != 1) for i, x in enumerate(inputs)]
            torch.manual_seed(99)
            out = module(*args)
            out.square().mean().backward()
            outputs.append(out.detach())
            gradients.append([p.grad for p in module.parameters()] + [args[0].grad, args[2].grad])
        torch.testing.assert_close(*outputs)
        for left, right in zip(*gradients):
            torch.testing.assert_close(left, right)

    def test_instance_gt_supervises_absent_classes_and_ignores_255(self):
        loss_fn = criterion(final_balanced_bce_weight=0.0, instance_distill_weight=0.0)
        labels = torch.tensor([[[0, 255], [1, 1]]])
        logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
        losses = loss_fn(
            {OUTPUT_KEYS.final_logits: torch.zeros_like(logits), OUTPUT_KEYS.instance_logits: logits},
            {"label_map": labels}, torch.arange(3), 3,
        )
        losses["total_loss"].backward()
        self.assertGreater(logits.grad[0, 2].sum().item(), 0)
        self.assertEqual(logits.grad[0, :, 0, 1].abs().sum().item(), 0)
        torch.testing.assert_close(losses["loss_instance_bce"], torch.tensor(2.).log())

    def test_instance_distill_present_valid_pairs_full_image_and_detached_teacher(self):
        loss_fn = criterion(final_balanced_bce_weight=0.0, instance_bce_weight=0.0, instance_distill_weight=1.0)
        labels = torch.tensor([[[0, 255], [1, 1]]])
        logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
        teacher = torch.full_like(logits, 0.9, requires_grad=True)
        losses = loss_fn({
            OUTPUT_KEYS.final_logits: torch.zeros_like(logits), OUTPUT_KEYS.instance_logits: logits,
            OUTPUT_KEYS.instance_teacher_probs: teacher,
            OUTPUT_KEYS.instance_teacher_valid: torch.tensor([[True, False, True]]),
        }, {"label_map": labels}, torch.arange(3), 3)
        losses["total_loss"].backward()
        self.assertGreater(logits.grad[0, 0, 0, 1].abs().item(), 0)
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)

    def test_all_queries_rejected_skips_distill_keeps_gt(self):
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        losses = criterion()({
            OUTPUT_KEYS.final_logits: logits, OUTPUT_KEYS.instance_logits: logits,
            OUTPUT_KEYS.instance_teacher_valid: torch.zeros(1, 2, dtype=torch.bool),
        }, {"label_map": torch.zeros(1, 2, 2, dtype=torch.long)}, torch.arange(2), 2)
        self.assertEqual(losses["loss_instance_distill_bce"].item(), 0)
        self.assertGreater(losses["loss_instance_bce"].item(), 0)
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_streamed_losses_and_gradients_match_full_with_aliases(self):
        loss_fn = criterion()
        labels = torch.tensor([[[0, 255], [1, 1]], [[0, 0], [255, 0]]])
        mapping = torch.tensor([0, 1, 0])
        valid = torch.tensor([[True, False, True], [False, True, True]])
        teacher = torch.rand(2, 3, 2, 2)
        semantic = torch.randn(2, 3, 2, 2, requires_grad=True)
        instance = torch.randn(2, 3, 2, 2, requires_grad=True)
        outputs = {OUTPUT_KEYS.final_logits: semantic, OUTPUT_KEYS.instance_logits: instance,
                   OUTPUT_KEYS.instance_teacher_probs: teacher, OUTPUT_KEYS.instance_teacher_valid: valid}
        full = loss_fn(outputs, {"label_map": labels}, mapping, 2)
        full["total_loss"].backward()
        reference_gradients = [semantic.grad.clone(), instance.grad.clone()]
        semantic.grad = instance.grad = None
        context = loss_fn.prepare_streaming_context(
            {"label_map": labels}, 3, mapping, 2, (2, 2), instance_teacher_valid=valid,
        )
        accum = {}
        for start, end in ((0, 1), (1, 3)):
            losses = loss_fn.forward_chunk(
                {key: value[:, start:end] for key, value in outputs.items()}, context, start, end,
            )
            losses["total_loss"].backward()
            for key, value in losses.items():
                accum[key] = accum.get(key, 0) + value.detach()
        for key in full:
            torch.testing.assert_close(full[key], accum[key])
        for actual, expected in zip((semantic.grad, instance.grad), reference_gradients):
            torch.testing.assert_close(actual, expected)

    def test_two_shared_head_passes_and_student_gradient_path(self):
        core, cache = decoding_fixture()
        core.train()
        refiner = torch.randn(1, 1, 256, 36, 36, requires_grad=True)
        out = core.decode_encoder_refiner_chunk_from_cache(cache, refiner, 0, 1, True, True)
        self.assertEqual(core.segmentation_head.calls, [(False, True), (True, True)])
        self.assertFalse(out[OUTPUT_KEYS.instance_teacher_probs].requires_grad)
        self.assertFalse(out[OUTPUT_KEYS.sam3_teacher_logits].requires_grad)
        (out[OUTPUT_KEYS.instance_logits].square().mean() + out[OUTPUT_KEYS.final_logits].square().mean()).backward()
        self.assertGreater(refiner.grad.abs().sum().item(), 0)
        self.assertGreater(core.encoder_refiner.instance_branch.vectors.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in core.segmentation_head.parameters()))

    def test_inference_max_and_disabled_branch_skip(self):
        core, cache = decoding_fixture()
        core.eval()
        refiner = torch.randn(1, 1, 256, 36, 36)
        with torch.no_grad():
            enabled = core.decode_encoder_refiner_chunk_from_cache(cache, refiner, 0, 1)
            core.instance_fusion_enabled = False
            disabled = core.decode_encoder_refiner_chunk_from_cache(cache, refiner, 0, 1)
        torch.testing.assert_close(
            enabled[OUTPUT_KEYS.final_logits].sigmoid(),
            torch.maximum(disabled[OUTPUT_KEYS.final_logits].sigmoid(), enabled[OUTPUT_KEYS.instance_logits].sigmoid()),
        )
        self.assertNotIn(OUTPUT_KEYS.instance_logits, disabled)
        self.assertEqual(core.segmentation_head.calls, [(False, True), (False, False)])

    def test_original_decoder_200_queries_presence_and_mask_mlp(self):
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.transformer = nn.Module()
        core.transformer.decoder = SAM3ModelBuilder._create_transformer_decoder().eval()
        core.dot_prod_scoring = SAM3ModelBuilder._create_dot_product_scoring().eval().requires_grad_(False)
        core.segmentation_head = SAM3ModelBuilder._create_segmentation_head().eval().requires_grad_(False)
        self.assertTrue(all(not p.requires_grad for p in core.transformer.decoder.parameters()))
        self.assertEqual(len(core.transformer.decoder.layers), 6)
        encoder = {
            "memory": torch.randn(6, 2, 256), "padding_mask": None,
            "pos_embed": torch.randn(6, 2, 256), "level_start_index": torch.tensor([0]),
            "spatial_shapes": torch.tensor([[2, 3]]), "valid_ratios": torch.ones(2, 1, 2),
        }
        outputs = core._run_instance_queries(encoder, torch.randn(4, 2, 256), torch.zeros(2, 4, dtype=torch.bool), 1, 2)
        self.assertEqual(outputs["mask_queries"].shape, (1, 2, 200, 256))
        self.assertEqual(outputs["presence_probabilities"].shape, (1, 2))
        self.assertTrue(all(not x.requires_grad and torch.isfinite(x).all() for x in outputs.values()))

    def test_query_decoder_reads_pre_prompt_attention_memory(self):
        core, _ = decoding_fixture()
        core.hidden_dim = 256
        raw_memory = torch.zeros(72 * 72, 1, 256)
        core._encode_prompt = lambda *args: (torch.zeros(2, 1, 256), torch.zeros(1, 2, dtype=torch.bool), {})
        core._run_full_encoder = lambda **kwargs: {"memory": raw_memory}
        seen = []
        core._run_instance_queries = lambda **kwargs: seen.append(kwargs["encoder_out"]["memory"].clone()) or {}
        core.segmentation_head.apply_prompt_cross_attention = lambda **kwargs: kwargs["encoder_hidden_states"] + 5
        out = core._encode_sam3_chunk({}, None, None, 1, 1)
        self.assertEqual(seen[0].sum().item(), 0)
        self.assertEqual(out["encoder_features_72"].mean().item(), 5)

    def test_config_defaults_and_new_parameters_in_refiner_group(self):
        cfg = SAM3ModelBuilder.build_config()
        self.assertTrue(cfg.encoder_refiner_cfg.instance_fusion_enabled)
        self.assertEqual(cfg.encoder_refiner_cfg.instance_teacher_score_threshold, 0.5)
        cfg = SAM3ModelBuilder.build_config(encoder_refiner_cfg={"instance_fusion_enabled": False})
        self.assertFalse(cfg.encoder_refiner_cfg.instance_fusion_enabled)
        with self.assertRaises(ValueError):
            SAM3ModelBuilder.build_config(encoder_refiner_cfg={"instance_teacher_score_threshold": float("nan")})
        core, _ = decoding_fixture()
        core.backbone = nn.Linear(1, 1)
        core.transformer = nn.Linear(1, 1)
        core.geometry_encoder = nn.Linear(1, 1)
        core.dot_prod_scoring = nn.Linear(1, 1)
        core.clip_image_encoder = core.clip_text_encoder = None
        core.clear_remoteclip_text_cache = lambda: None
        cfg = SAM3ModelBuilder.build_config(device="cpu", eval_mode=False)
        with patch.object(SAM3ModelBuilder, "build_semantic_core_model", return_value=core):
            model = SAM3ModelBuilder.build_semantic_segmentor(cfg)
        model.eval().train()
        self.assertTrue(all(p.requires_grad for p in core.encoder_refiner.instance_branch.parameters()))
        for module in (core.backbone, core.transformer, core.geometry_encoder, core.segmentation_head, core.dot_prod_scoring):
            self.assertFalse(module.training)
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_missing_original_instance_weights_are_rejected(self):
        core = nn.Module()
        core.transformer = nn.Module()
        core.transformer.decoder = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pt"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(RuntimeError, "original SAM3 pretrained weights"):
                SAM3ModelBuilder._load_checkpoint(core, str(path))

    def test_trainer_proxy_backward_matches_one_full_backward(self):
        Trainer = load("engine.trainer").Trainer

        class StreamingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lowres_weight = nn.Parameter(torch.tensor(0.4))
                self.instance_weight = nn.Parameter(torch.tensor(0.7))

            def build_encoder_refiner_cache(self, batch):
                return {"chunk_class_counts": [1, 2], OUTPUT_KEYS.instance_teacher_valid: torch.tensor([[True, False, True]])}

            def run_encoder_refiner_lowres_from_cache(self, *args, **kwargs):
                return {"refiner_features_36": self.lowres_weight * torch.tensor([1., 2., 3.]).view(1, 3, 1, 1, 1)}

            def decode_encoder_refiner_chunk_from_cache(self, encoder_refiner_cache, refiner_feature_36_chunk,
                                                       class_start, class_end, return_teacher_logits,
                                                       return_instance_teacher):
                logits = refiner_feature_36_chunk[..., 0].expand(1, class_end - class_start, 288, 288)
                outputs = {OUTPUT_KEYS.final_logits: logits, OUTPUT_KEYS.instance_logits: logits * self.instance_weight}
                if return_instance_teacher:
                    outputs[OUTPUT_KEYS.instance_teacher_probs] = torch.full_like(logits, 0.8)
                    outputs[OUTPUT_KEYS.instance_teacher_valid] = encoder_refiner_cache[OUTPUT_KEYS.instance_teacher_valid][:, class_start:class_end]
                return outputs

        model = StreamingModel()
        reference = copy.deepcopy(model)
        labels = torch.zeros(1, 288, 288, dtype=torch.long)
        labels[:, 144:] = 1
        labels[:, :, 20:30] = 255
        batch = types.SimpleNamespace(
            find_targets=[types.SimpleNamespace(semantic_label_map=labels)],
            find_metadatas=[types.SimpleNamespace(num_prompts=3, num_classes=2, prompt_to_class_id=[0, 1, 0])],
        )
        loss_fn = criterion()
        trainer = Trainer.__new__(Trainer)
        trainer.model, trainer.criterion = model, loss_fn
        trainer.cfg = types.SimpleNamespace(use_amp=False)
        trainer.device = torch.device("cpu")
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        actual = trainer._compute_train_loss_streaming(batch)
        cache = reference.build_encoder_refiner_cache(batch)
        features = reference.run_encoder_refiner_lowres_from_cache()["refiner_features_36"]
        outputs = reference.decode_encoder_refiner_chunk_from_cache(cache, features, 0, 3, False, True)
        expected = loss_fn(outputs, {"label_map": labels}, torch.tensor([0, 1, 0]), 2)
        expected["total_loss"].backward()
        for key, value in actual.items():
            torch.testing.assert_close(value, expected[key])
        for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_param.grad, expected_param.grad)


if __name__ == "__main__":
    unittest.main()
