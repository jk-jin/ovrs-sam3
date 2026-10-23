"""CPU regressions for presence-guided Refiner and three-branch fusion."""

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
from torch.utils.checkpoint import checkpoint

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


DecoderInputFusion = load("models.decoder_input_fusion").DecoderInputFusion
ClassScoreAttention = load("models.encoder_refiner_attention").ClassScoreAttention
EncoderRefinerLayer = load("models.encoder_refiner_attention").EncoderRefinerLayer
ClassConditionedEncoderRefiner = load("models.encoder_refiner").ClassConditionedEncoderRefiner
Sam3Image = load("models.sam3_image").Sam3Image
SAM3ModelBuilder = load("model_builder").SAM3ModelBuilder
SemanticCriterion = load("losses.semantic_criterion").SemanticCriterion
SemanticCriterionConfig = load("config_dataclasses").SemanticCriterionConfig
OUTPUT_KEYS = load("models.task_modes").OUTPUT_KEYS
torch.set_num_threads(1)


def empty_core():
    core = Sam3Image.__new__(Sam3Image)
    nn.Module.__init__(core)
    core.hidden_dim = 256
    return core


def fusion_inputs():
    return (
        torch.randn(1, 256, 36, 36, requires_grad=True),
        torch.randn(1, 256, 72, 72), torch.randn(1, 256, 72, 72),
        torch.randn(1, 256, 72, 72), torch.randn(1, 200, 256),
    )


class QueryGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_both_inter_class_kinds_use_presence_only_in_qk(self):
        for kind in ("intra", "inter"):
            for mode in ("window", "local_3x3"):
                with self.subTest(kind=kind, mode=mode):
                    layer = EncoderRefinerLayer(
                        score_attention_type=kind, score_intra_attn_type=mode,
                        feature_intra_attn_type=mode, window_size=2, shift_size=1, dropout=0.0,
                    )
                    self.assertEqual(layer.class_attn.q_proj.in_features, 768)
                    self.assertEqual(layer.class_attn.k_proj.in_features, 768)
                    self.assertEqual(layer.class_attn.v_proj.in_features, 256)
                    feature = torch.randn(1, 2, 256, 2, 2, requires_grad=True)
                    score = torch.randn_like(feature, requires_grad=True)
                    presence = torch.randn(1, 2, 256)
                    guidance = torch.randn_like(feature)
                    captured = []
                    hook = layer.class_attn.register_forward_pre_hook(
                        lambda module, args, kwargs: captured.append(kwargs["guidance"]), with_kwargs=True,
                    )
                    outputs = layer(feature, score, presence, guidance, guidance)
                    expected = layer.class_norm_presence(presence)[..., None, None].expand_as(feature)
                    torch.testing.assert_close(captured[0], expected)
                    sum(x.square().mean() for x in outputs).backward()
                    self.assertIsNotNone(layer.class_norm_presence.weight.grad)
                    self.assertGreater(feature.grad.abs().sum().item(), 0)
                    changed = presence.clone()
                    changed[0, 0] = changed[0, 0].roll(3)
                    with torch.no_grad():
                        other = layer(feature, score, changed, guidance, guidance)
                    self.assertGreater(sum((a - b).abs().sum().item() for a, b in zip(outputs, other)), 0)
                    hook.remove()

    def test_inter_class_single_value_remains_target_stream(self):
        feature = torch.randn(1, 3, 256, 2, 2)
        score = torch.randn_like(feature)
        guidance = torch.randn_like(feature)
        for stream, target in (("feature", feature), ("score", score)):
            attention = ClassScoreAttention(value_stream=stream, guidance_dim=256, dropout=0.0)
            with torch.no_grad():
                attention.q_proj.weight.zero_(); attention.q_proj.bias.zero_()
                attention.k_proj.weight.zero_(); attention.k_proj.bias.zero_()
                for projection in (attention.v_proj, attention.out_proj):
                    projection.weight.copy_(torch.eye(256)); projection.bias.zero_()
            actual = attention(feature, score, guidance)
            torch.testing.assert_close(actual, target.mean(1, keepdim=True).expand_as(target))

    def test_refiner_layer_checkpoint_carries_presence_and_matches_gradients(self):
        plain = EncoderRefinerLayer(score_attention_type="inter", window_size=2, shift_size=1, dropout=0.1)
        checked = copy.deepcopy(plain)
        inputs = [torch.randn(1, 2, 256, 2, 2), torch.randn(1, 2, 256, 2, 2),
                  torch.randn(1, 2, 256), torch.randn(1, 2, 256, 2, 2), torch.randn(1, 2, 256, 2, 2)]
        results, gradients = [], []
        for use_checkpoint, module in ((False, plain), (True, checked)):
            args = [value.clone().requires_grad_(i < 2) for i, value in enumerate(inputs)]
            torch.manual_seed(99)
            outputs = checkpoint(module, *args, use_reentrant=False) if use_checkpoint else module(*args)
            sum(value.square().mean() for value in outputs).backward()
            results.append(outputs)
            gradients.append([p.grad for p in module.parameters()] + [args[0].grad, args[1].grad])
        for actual, expected in zip(*results):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(*gradients):
            torch.testing.assert_close(actual, expected)

    def test_query_attention_direction_all_queries_and_residual(self):
        module = DecoderInputFusion(dropout=0.0, use_checkpoint=False)
        raw = torch.randn(2, 256, 2, 3)
        queries = torch.randn(2, 200, 256)
        seen = []
        hook = module.query_cross_attention.register_forward_pre_hook(
            lambda module, args: seen.append([x.shape for x in args[:3]])
        )
        with torch.no_grad():
            module.query_cross_attention.out_proj.weight.zero_()
            module.query_cross_attention.out_proj.bias.zero_()
        updated = module.attend_instance_queries(raw, queries)
        torch.testing.assert_close(updated, raw)
        self.assertEqual(seen, [[(2, 6, 256), (2, 200, 256), (2, 200, 256)]])
        hook.remove()

    def test_query_attention_trains_with_both_inputs_frozen(self):
        module = DecoderInputFusion(dropout=0.0, use_checkpoint=False)
        raw = torch.randn(1, 256, 2, 3)
        queries = torch.randn(1, 200, 256)
        module.attend_instance_queries(raw, queries).square().mean().backward()
        self.assertGreater(module.query_cross_attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(module.query_norm_memory.weight.grad)
        self.assertIsNone(raw.grad)
        self.assertIsNone(queries.grad)

    def test_three_branch_fusion_shape_and_all_branch_gradients(self):
        module = DecoderInputFusion(dropout=0.0, use_checkpoint=False)
        inputs = fusion_inputs()
        output = module(*inputs)
        self.assertEqual(output.shape, (1, 256, 72, 72))
        output.square().mean().backward()
        self.assertGreater(inputs[0].grad.abs().sum().item(), 0)
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertEqual(len({id(module.semantic_refiner_proj), id(module.detail_refiner_proj), id(module.query_refiner_proj)}), 3)

    def test_fusion_checkpoint_matches_outputs_and_gradients(self):
        plain = DecoderInputFusion(dropout=0.1, use_checkpoint=False)
        checked = copy.deepcopy(plain)
        checked.use_checkpoint = True
        base = fusion_inputs()
        results, gradients = [], []
        for module in (plain, checked):
            args = [value.detach().clone().requires_grad_(i == 0) for i, value in enumerate(base)]
            torch.manual_seed(55)
            output = module(*args)
            output.square().mean().backward()
            results.append(output.detach())
            gradients.append([p.grad for p in module.parameters()] + [args[0].grad])
        torch.testing.assert_close(*results)
        for actual, expected in zip(*gradients):
            torch.testing.assert_close(actual, expected)

    def test_raw_queries_and_presence_are_unmodified_decoder_hidden_outputs(self):
        core = empty_core()
        core.transformer = nn.Module()
        decoder = SAM3ModelBuilder._create_transformer_decoder().eval()
        core.transformer.decoder = decoder
        seen = []
        hook = decoder.register_forward_hook(lambda module, args, output: seen.append(output))
        encoder = {"memory": torch.randn(6, 2, 256), "padding_mask": None,
                   "pos_embed": torch.randn(6, 2, 256), "level_start_index": torch.tensor([0]),
                   "spatial_shapes": torch.tensor([[2, 3]]), "valid_ratios": torch.ones(2, 1, 2)}
        output = core._run_instance_queries(encoder, torch.randn(4, 2, 256), torch.zeros(2, 4, dtype=torch.bool), 1, 2)
        torch.testing.assert_close(output["instance_queries"], seen[0][0][-1].transpose(0, 1).reshape(1, 2, 200, 256))
        torch.testing.assert_close(output["presence_tokens"], seen[0][3][0].reshape(1, 2, 256))
        self.assertTrue(all(not value.requires_grad for value in output.values()))
        self.assertTrue(all(not parameter.requires_grad for parameter in decoder.parameters()))
        self.assertEqual(set(output), {"instance_queries", "presence_tokens"})
        hook.remove()

    def test_encoder_cache_keeps_raw_memory_before_segmentation_attention(self):
        core = empty_core()
        raw = torch.zeros(72 * 72, 1, 256)
        core._encode_prompt = lambda *args: (torch.zeros(2, 1, 256), torch.zeros(1, 2, dtype=torch.bool), {})
        core._run_full_encoder = lambda **kwargs: {"memory": raw}
        seen = []
        core._run_instance_queries = lambda encoder, *args: seen.append(encoder["memory"]) or {
            "instance_queries": torch.ones(1, 1, 200, 256), "presence_tokens": torch.ones(1, 1, 256),
        }
        core.segmentation_head = nn.Module()
        core.segmentation_head.apply_prompt_cross_attention = lambda **kwargs: kwargs["encoder_hidden_states"] + 5
        output = core._encode_sam3_chunk({}, None, None, 1, 1)
        self.assertIs(seen[0], raw)
        self.assertEqual(output["raw_encoder_features_72"].sum().item(), 0)
        self.assertEqual(output["encoder_features_72"].mean().item(), 5)

    def test_prompt_chunk_cache_preserves_image_and_prompt_order(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward_image(self, images):
                return {"backbone_fpn": [torch.zeros(2, 256, 1, 1)] * 3}

        core = empty_core()
        core.backbone = Backbone()
        core.prompt_chunk_size = 2
        core.ensure_text_cache = lambda **kwargs: None
        core._slice_text_cache = lambda start, end: {
            "language_features": torch.arange(start, end).view(1, -1, 1).float(),
            "language_mask": torch.zeros(end - start, 1, dtype=torch.bool),
        }
        core._build_clip_image_cache = lambda **kwargs: {
            "clip_image_feat_map_native": torch.zeros(2, 8, 36, 36),
            OUTPUT_KEYS.clip_mid_features: [], "clip_mid_layer_indices": (),
        }

        def encode(backbone_out, batch_size, num_chunk_classes, **kwargs):
            ids = torch.arange(batch_size)[:, None] * 100 + backbone_out["language_features"][0, :, 0]
            return {"encoder_features_72": ids[..., None, None, None].expand(-1, -1, 256, 72, 72),
                    "raw_encoder_features_72": (ids + 1)[..., None, None, None].expand(-1, -1, 256, 72, 72),
                    "instance_queries": (ids + 2)[..., None, None].expand(-1, -1, 200, 256),
                    "presence_tokens": (ids + 3)[..., None].expand(-1, -1, 256)}

        core._encode_sam3_chunk = encode
        batch = types.SimpleNamespace(img_batch=torch.zeros(2, 3, 1, 1), find_inputs=[None], find_text_batch=["a", "b", "c"])
        cache = core.build_encoder_refiner_cache(batch)
        expected = torch.tensor([[0., 1., 2.], [100., 101., 102.]])
        torch.testing.assert_close(cache["cross_attended_encoder_features_72"][..., 0, 0, 0], expected)
        torch.testing.assert_close(cache["raw_encoder_features_72"][..., 0, 0, 0], expected + 1)
        torch.testing.assert_close(cache["instance_queries"][..., 0, 0], expected + 2)
        torch.testing.assert_close(cache["presence_tokens"][..., 0], expected + 3)
        self.assertEqual(cache["chunk_class_counts"], [2, 1])

    def test_full_refiner_forwards_presence_through_checkpoint(self):
        class Scores(nn.Module):
            def forward(self, class_names, remoteclip_feat_map):
                return (torch.zeros(1, 2, 256, 36, 36), torch.zeros(1, 2, 64, 36, 36), torch.zeros(2, 64, 8))

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(256, 256)
                self.seen = []

            def forward(self, feature_36, score_embed_36, presence_tokens, clip_mid_36, sam_fpn_36):
                self.seen.append(presence_tokens.detach().clone())
                return feature_36 + self.projection(presence_tokens)[..., None, None], score_embed_36

        for checked in (False, True):
            refiner = ClassConditionedEncoderRefiner(
                clip_text_encoder=None, clip_dim=8, clip_native_dim=4,
                prompt_templates=["{}"] * 64, use_checkpoint=checked,
            )
            refiner.clip_score_embed = Scores()
            refiner.layers = nn.ModuleList([Layer() for _ in range(4)])
            presence = torch.randn(1, 2, 256)
            result = refiner(
                torch.randn(1, 2, 256, 72, 72), torch.zeros(1, 8, 36, 36), ["a", "b"], presence,
                [torch.randn(1, 4, 36, 36)] * 2, (7, 15), torch.randn(1, 256, 72, 72),
            )
            result["refiner_features_36"].sin().mean().backward()
            for layer in refiner.layers:
                torch.testing.assert_close(layer.seen[0], presence)
                self.assertGreater(layer.projection.weight.grad.abs().sum().item(), 0)
            self.assertFalse(hasattr(refiner, "instance_branch"))

    def test_removed_instance_config_is_rejected_and_semantic_loss_preserved(self):
        with self.assertRaises(TypeError):
            SAM3ModelBuilder.build_config(encoder_refiner_cfg={"instance_fusion_enabled": True})
        with self.assertRaises(TypeError):
            SemanticCriterionConfig(instance_distill_weight=0.5)
        loss_fn = SemanticCriterion(SemanticCriterionConfig(sam3_mask_distill_weight=0.5))
        labels = torch.tensor([[[0, 255], [1, 1]]])
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        teacher = torch.ones_like(logits)
        loss = loss_fn({OUTPUT_KEYS.final_logits: logits, OUTPUT_KEYS.sam3_teacher_logits: teacher},
                       {"label_map": labels}, torch.arange(2), 2)
        self.assertFalse(any("instance" in key for key in loss))
        self.assertGreater(loss["loss_sam3_mask_distill_bce"].item(), 0)
        loss["total_loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_missing_original_decoder_weights_fail(self):
        core = nn.Module()
        core.transformer = nn.Module()
        core.transformer.decoder = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pt"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(RuntimeError, "original SAM3 decoder pretrained weights"):
                SAM3ModelBuilder._load_checkpoint(core, str(path))

    def test_shared_semantic_head_never_runs_instance_prediction_and_backpropagates(self):
        class Refiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder_input_fusion = DecoderInputFusion(dropout=0.0, use_checkpoint=False)

            def fuse_decoder_input_chunk(self, **kwargs):
                return self.decoder_input_fusion(**kwargs)

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.semantic = nn.Conv2d(256, 1, 1).requires_grad_(False)
                self.calls = []

            def forward(self, backbone_feats, image_ids, encoder_hidden_states):
                self.calls.append(torch.is_grad_enabled())
                feature = encoder_hidden_states.permute(1, 2, 0).reshape(-1, 256, 72, 72)
                logits = self.semantic(F.adaptive_avg_pool2d(feature, 4))
                return {"semantic_seg": F.interpolate(logits, size=288)}

        core = empty_core()
        core.encoder_refiner, core.segmentation_head = Refiner(), Head()
        cache = {"cross_attended_encoder_features_72": torch.randn(1, 2, 256, 72, 72),
                 "raw_encoder_features_72": torch.randn(1, 2, 256, 72, 72),
                 "instance_queries": torch.randn(1, 2, 200, 256),
                 "backbone_fpn": [torch.empty(1), torch.empty(1), torch.randn(1, 256, 72, 72)]}
        feature = torch.randn(1, 1, 256, 36, 36, requires_grad=True)
        output = core.decode_encoder_refiner_chunk_from_cache(cache, feature, 1, 2, return_teacher_logits=True)
        self.assertEqual(set(output), {OUTPUT_KEYS.final_logits, OUTPUT_KEYS.sam3_teacher_logits})
        self.assertEqual(core.segmentation_head.calls, [False, True])
        self.assertFalse(output[OUTPUT_KEYS.sam3_teacher_logits].requires_grad)
        output[OUTPUT_KEYS.final_logits].square().mean().backward()
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertGreater(core.encoder_refiner.decoder_input_fusion.query_cross_attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in core.segmentation_head.parameters()))
        core.eval()
        with torch.no_grad():
            output = core.decode_encoder_refiner_chunk_from_cache(cache, feature, 1, 2)
        self.assertEqual(set(output), {OUTPUT_KEYS.final_logits})
        self.assertEqual(core.segmentation_head.calls[-1], False)


if __name__ == "__main__":
    unittest.main()
