"""CPU regression tests, no SAM3/RemoteCLIP checkpoint or dataset required."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

# The repository directory can contain a hyphen; use a stable package alias.
ROOT = Path(__file__).resolve().parents[1]
if "ovrs_test" not in sys.modules:
    package = types.ModuleType("ovrs_test")
    package.__path__ = [str(ROOT)]
    sys.modules["ovrs_test"] = package

from ovrs_test.config_dataclasses import SemanticCriterionConfig
from ovrs_test.losses.semantic_criterion import SemanticCriterion
from ovrs_test.models.instance_scoring import (
    InstanceScoreAdapter,
    aggregate_instance_probabilities,
)
from ovrs_test.models.adapters.semantic_adapter import SemanticSegAdapter
from ovrs_test.models.maskformer_segmentation import (
    UniversalSegmentationHead,
    PixelDecoder,
    MaskPredictor,
)
from ovrs_test.models.model_misc import DotProductScoring, MLP

torch.set_num_threads(2)


class InstanceScoringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_quality_ignores_other_objects_and_unknown_pixels(self):
        # Two objects of class 1; a mask of one object still deserves quality 1.
        label = torch.tensor([[[1, 0, 1, 255]]])
        target = (label == 1)[:, None]
        masks = torch.tensor(
            [
                [
                    [
                        [[8.0, -8.0, -8.0, -8.0]],
                        [[8.0, 8.0, -8.0, -8.0]],
                        [[-8.0, -8.0, -8.0, 8.0]],
                        [[-8.0, -8.0, -8.0, -8.0]],
                    ]
                ]
            ]
        )
        quality, supervised = SemanticCriterion.build_instance_quality_targets(
            masks,
            target,
            label != 255,
            0.5,
        )
        torch.testing.assert_close(quality, torch.tensor([[[1.0, 0.5, 0.0, 0.0]]]))
        self.assertEqual(supervised.tolist(), [[[True, True, False, True]]])

    def test_adapter_gradients_do_not_reach_frozen_inputs_or_scoring_heads(self):
        adapter = InstanceScoreAdapter()
        scorer = DotProductScoring(256, 256).requires_grad_(False).eval()
        presence_head = nn.Sequential(
            nn.LayerNorm(256), MLP(256, 256, 1, 3)
        ).requires_grad_(False)
        queries = torch.randn(2, 4, 256, requires_grad=True)
        original_queries = queries.detach().clone()
        presence = torch.randn(2, 1, 256, requires_grad=True)
        fused = torch.randn(2, 256, 4, 4, requires_grad=True)
        updated, updated_presence = adapter(queries, presence, fused)
        prompt = torch.randn(3, 2, 256)
        logits = scorer(updated[None], prompt, torch.zeros(2, 3, dtype=torch.bool))
        (
            logits.square().mean() + presence_head(updated_presence).square().mean()
        ).backward()
        self.assertIsNone(queries.grad)
        self.assertIsNone(presence.grad)
        self.assertIsNone(fused.grad)
        torch.testing.assert_close(queries, original_queries)
        for layer in adapter.layers:
            self.assertGreater(
                layer.attention.in_proj_weight.grad.abs().sum().item(), 0
            )
        self.assertTrue(all(p.grad is None for p in scorer.parameters()))
        self.assertTrue(all(p.grad is None for p in presence_head.parameters()))

    def test_joint_threshold_filters_without_weighting_mask(self):
        masks = torch.logit(torch.tensor([[[[[0.9, 0.2]], [[0.4, 0.99]]]]]))
        score = torch.logit(torch.tensor([[[0.8, 0.2]]]))
        presence = torch.logit(torch.tensor([[0.9]]))
        output = aggregate_instance_probabilities(masks, score, presence, 0.5)
        torch.testing.assert_close(output, torch.tensor([[[[0.9, 0.2]]]]))
        output = aggregate_instance_probabilities(masks, score, presence, 1.0)
        self.assertEqual(output.count_nonzero().item(), 0)
        # Strictly above the threshold, matching the requested filtering rule.
        self.assertEqual(
            aggregate_instance_probabilities(
                masks,
                torch.zeros_like(score),
                torch.zeros_like(presence),
                0.25,
            )
            .count_nonzero()
            .item(),
            0,
        )

    def test_prompt_alias_fusion_and_semantic_only_training_output(self):
        adapter = SemanticSegAdapter()
        metadata = SimpleNamespace(
            num_prompts=3, num_classes=2, prompt_to_class_id=[0, 0, 1]
        )
        batch = SimpleNamespace(find_metadatas=[metadata])
        semantic = torch.logit(torch.tensor([[[[0.2]], [[0.4]], [[0.6]]]]))
        raw = {
            "final_logits": semantic,
            "instance_score_map": torch.tensor([[[[0.8]], [[0.3]], [[0.1]]]]),
        }
        infer = adapter(raw, batch, "infer")
        torch.testing.assert_close(
            infer["raw_final_score_map"], torch.tensor([[[[0.8]], [[0.6]]]])
        )
        torch.testing.assert_close(
            infer["final_logits"].sigmoid(), infer["raw_final_score_map"]
        )
        torch.testing.assert_close(
            adapter(raw, batch, "final")["final_logits"], semantic
        )

    def make_loss_fixture(self):
        label = torch.tensor([[[0, 1], [255, 2]], [[1, 1], [0, 255]]])
        mapping = torch.tensor([0, 1, 1, 2])
        criterion = SemanticCriterion(
            SemanticCriterionConfig(sam3_mask_distill_weight=0)
        )
        outputs = {
            "final_logits": torch.randn(2, 4, 2, 2, requires_grad=True),
            "instance_mask_logits": torch.randn(2, 4, 5, 2, 2),
            "instance_score_logits": torch.randn(2, 4, 5, requires_grad=True),
            "instance_presence_logits": torch.randn(2, 4, requires_grad=True),
        }
        context = criterion.prepare_streaming_context(
            {"label_map": label}, 4, mapping, 3, (2, 2)
        )
        return criterion, outputs, context

    def test_streaming_loss_and_gradients_equal_full_prompt_forward(self):
        criterion, outputs, context = self.make_loss_fixture()
        full = criterion.forward_chunk(outputs, context, 0, 4)
        full["total_loss"].backward()
        grads = {k: v.grad.clone() for k, v in outputs.items() if v.requires_grad}
        for value in outputs.values():
            value.grad = None
        accumulated = {}
        for start, end in [(0, 1), (1, 3), (3, 4)]:
            loss = criterion.forward_chunk(
                {k: v[:, start:end] for k, v in outputs.items()}, context, start, end
            )
            loss["total_loss"].backward()
            for key, value in loss.items():
                accumulated[key] = accumulated.get(key, 0) + value.detach()
        for key in full:
            torch.testing.assert_close(full[key], accumulated[key])
        for key in grads:
            torch.testing.assert_close(outputs[key].grad, grads[key])

    def test_all_ignored_and_absent_classes(self):
        criterion, outputs, _ = self.make_loss_fixture()
        label = torch.full((2, 2, 2), 255)
        context = criterion.prepare_streaming_context(
            {"label_map": label}, 4, torch.tensor([0, 1, 1, 2]), 3, (2, 2)
        )
        losses = criterion.forward_chunk(outputs, context, 0, 4)
        self.assertEqual(losses["total_loss"].item(), 0)
        losses["total_loss"].backward()
        for key in ["instance_score_logits", "instance_presence_logits"]:
            self.assertEqual(outputs[key].grad.abs().sum().item(), 0)
        quality, supervised = criterion.build_instance_quality_targets(
            torch.ones(1, 1, 2, 2, 2),
            torch.zeros(1, 1, 2, 2, dtype=torch.bool),
            torch.ones(1, 2, 2, dtype=torch.bool),
            0.5,
        )
        self.assertEqual(quality.sum().item(), 0)
        self.assertTrue(supervised.all())

    def test_presence_detects_tiny_objects_before_downsampling(self):
        criterion = SemanticCriterion(
            SemanticCriterionConfig(sam3_mask_distill_weight=0)
        )
        label = torch.zeros(1, 4, 4, dtype=torch.long)
        label[0, 3, 3] = 1
        ctx = criterion.prepare_streaming_context(
            {"label_map": label}, 2, torch.tensor([0, 1]), 2, (1, 1)
        )
        self.assertFalse(ctx.presence_target[0, 1])  # Existing semantic loss unchanged.
        self.assertTrue(ctx.instance_presence_target[0, 1])

    def test_semantic_loss_matches_original_plain_bce(self):
        criterion, outputs, context = self.make_loss_fixture()
        loss = criterion.forward_chunk(outputs, context, 0, 4)
        target = (
            context.label_map[:, None]
            == context.prompt_to_class_id[None, :, None, None]
        ).float()
        per_pixel = nn.functional.binary_cross_entropy_with_logits(
            outputs["final_logits"], target, reduction="none"
        )
        expected = (
            per_pixel * context.valid_mask[:, None]
        ).sum() / context.total_valid_pixels
        torch.testing.assert_close(loss["loss_final_bce"], expected)

    def test_restored_mask_predictor_matches_official_dot_product(self):
        class SmallPixelDecoder(nn.Module):
            out_dim = 256

            def forward(self, features):
                return nn.functional.interpolate(
                    features[-1], size=(8, 8), mode="nearest"
                )

        pixel_decoder = SmallPixelDecoder()
        head = (
            UniversalSegmentationHead(256, 3, pixel_decoder, no_dec=False)
            .requires_grad_(False)
            .eval()
        )
        fpn = [
            torch.randn(1, 256, 8, 8),
            torch.randn(1, 256, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]
        memory = torch.randn(4, 1, 256, requires_grad=True)
        queries = torch.randn(1, 3, 256)
        out = head(fpn, torch.tensor([0]), memory, queries)
        pixels = head._embed_pixels(fpn, torch.tensor([0]), memory)
        expected = torch.einsum(
            "bqc,bchw->bqhw",
            head.mask_predictor.mask_embed(queries),
            head.instance_seg_head(pixels),
        )
        torch.testing.assert_close(out["pred_masks"], expected)
        out["semantic_seg"].mean().backward()
        self.assertGreater(memory.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_bad_config_values_rejected(self):
        for value in [-1.0, float("nan")]:
            with self.assertRaises(ValueError):
                SemanticCriterion(
                    SemanticCriterionConfig(instance_score_bce_weight=value)
                )
        with self.assertRaises(ValueError):
            InstanceScoreAdapter(num_heads=3)
        with self.assertRaises(ValueError):
            SemanticCriterion(SemanticCriterionConfig(instance_mask_threshold=0))


class IntegrationTests(unittest.TestCase):
    def test_base_config_and_restored_head_builders(self):
        import runpy
        from ovrs_test.model_builder import SAM3ModelBuilder
        from ovrs_test.config_dataclasses import SegmentorBuildConfig

        config = runpy.run_path(str(ROOT / "configs/_base_/model/ovrs_sam3.py"))[
            "model"
        ]
        # Coerce configuration without loading external checkpoint files.
        cfg = SAM3ModelBuilder._normalize_build_cfg(SegmentorBuildConfig(**config))
        self.assertIn("core.instance_score_adapter", cfg.freeze_cfg.trainable_modules)
        self.assertEqual(cfg.instance_scoring_cfg.query_keep_threshold, 0.5)
        head = SAM3ModelBuilder._create_segmentation_head()
        self.assertIsInstance(head.mask_predictor, MaskPredictor)
        scorer = SAM3ModelBuilder._create_dot_product_scoring()
        self.assertEqual(len(scorer.prompt_mlp.layers), 2)

    def test_builder_forces_sam3_frozen_across_mode_switches(self):
        from unittest.mock import patch
        from ovrs_test.model_builder import SAM3ModelBuilder
        from ovrs_test.config_dataclasses import SegmentorBuildConfig

        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                for name in [
                    "backbone",
                    "transformer",
                    "geometry_encoder",
                    "segmentation_head",
                    "dot_prod_scoring",
                    "encoder_refiner",
                    "instance_score_adapter",
                ]:
                    setattr(self, name, nn.Linear(2, 2))

            def clear_remoteclip_text_cache(self):
                pass

        core = Core()
        with patch.object(
            SAM3ModelBuilder, "build_semantic_core_model", return_value=core
        ):
            model = SAM3ModelBuilder.build_semantic_segmentor(
                SegmentorBuildConfig(device="cpu", eval_mode=False),
            )
        model.eval()
        model.train()
        for name in [
            "backbone",
            "transformer",
            "geometry_encoder",
            "segmentation_head",
            "dot_prod_scoring",
        ]:
            module = getattr(core, name)
            self.assertFalse(module.training)
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        self.assertTrue(core.instance_score_adapter.training)
        self.assertTrue(
            all(p.requires_grad for p in core.instance_score_adapter.parameters())
        )

    def test_original_six_layer_decoder_runs_with_200_queries_on_cpu(self):
        from ovrs_test.model_builder import SAM3ModelBuilder

        decoder = SAM3ModelBuilder._create_transformer_decoder().eval()
        memory = torch.randn(16, 1, 256)
        with torch.no_grad():
            queries, boxes, presence, feats = decoder(
                decoder.query_embed.weight[:, None],
                memory,
                pos=torch.zeros_like(memory),
                spatial_shapes=torch.tensor([[4, 4]]),
                valid_ratios=torch.ones(1, 1, 2),
                memory_text=torch.randn(2, 1, 256),
                text_attention_mask=torch.zeros(1, 2, dtype=torch.bool),
                apply_dac=False,
            )
        self.assertEqual(queries.shape, (6, 200, 1, 256))
        self.assertEqual(feats.shape, (1, 1, 256))
        self.assertEqual(presence.shape, (6, 1, 1))
        self.assertTrue(torch.isfinite(queries).all())

    def test_encoder_uses_raw_memory_for_queries_and_one_prompt_attention_for_pixels(
        self,
    ):
        from ovrs_test.models.sam3_image import Sam3Image
        from unittest.mock import Mock

        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.hidden_dim = 256
        memory = torch.ones(72 * 72, 2, 256)
        decoder = Mock()
        decoder.query_embed.weight = torch.randn(3, 256)
        decoder.num_queries = 3
        decoder.return_value = (
            torch.ones(1, 3, 2, 256),
            None,
            None,
            torch.ones(1, 2, 256),
        )
        core.transformer = SimpleNamespace(decoder=decoder)
        prompt = torch.randn(4, 2, 256)
        prompt_mask = torch.zeros(2, 4, dtype=torch.bool)
        core._encode_prompt = Mock(return_value=(prompt, prompt_mask, {}))
        core._run_full_encoder = Mock(
            return_value={
                "memory": memory,
                "padding_mask": None,
                "pos_embed": torch.zeros_like(memory),
                "level_start_index": torch.tensor([0]),
                "spatial_shapes": torch.tensor([[72, 72]]),
                "valid_ratios": torch.ones(2, 1, 2),
            }
        )
        core.segmentation_head = SimpleNamespace(
            apply_prompt_cross_attention=Mock(
                side_effect=lambda **kw: kw["encoder_hidden_states"] + 2
            ),
        )
        output = core._encode_sam3_chunk({}, None, None, 1, 2)
        self.assertIs(decoder.call_args.kwargs["memory"], memory)
        self.assertFalse(decoder.call_args.kwargs["apply_dac"])
        self.assertEqual(
            core.segmentation_head.apply_prompt_cross_attention.call_count, 1
        )
        self.assertEqual(output["encoder_features_72"].mean().item(), 3.0)
        self.assertEqual(output["instance_queries"].shape, (1, 2, 3, 256))

    def test_chunk_decode_reuses_teacher_and_separates_score_gradients(self):
        from ovrs_test.models.sam3_image import Sam3Image

        class Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Conv2d(256, 256, 1)

            def fuse_decoder_input_chunk(self, refiner_feature_36, **kw):
                return self.proj(
                    nn.functional.interpolate(refiner_feature_36, size=(72, 72))
                )

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(
                self, backbone_feats, image_ids, encoder_hidden_states, obj_queries=None
            ):
                self.calls.append((torch.is_grad_enabled(), obj_queries))
                semantic = nn.functional.interpolate(
                    encoder_hidden_states.mean(dim=-1).T.reshape(-1, 1, 72, 72),
                    size=(288, 288),
                )
                out = {"semantic_seg": semantic}
                if obj_queries is not None:
                    out["pred_masks"] = semantic + obj_queries.mean(-1)[..., None, None]
                return out

        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.segmentation_head = Head()
        core.encoder_refiner = Fusion()
        core.instance_score_adapter = InstanceScoreAdapter()
        core.dot_prod_scoring = DotProductScoring(256, 256).requires_grad_(False).eval()
        decoder = nn.Module()
        decoder.presence_token_head = MLP(256, 256, 1, 3)
        decoder.presence_token_out_norm = nn.LayerNorm(256)
        decoder.clamp_presence_logits = True
        decoder.clamp_presence_logit_max_val = 10.0
        core.transformer = nn.Module()
        core.transformer.decoder = decoder.requires_grad_(False).eval()
        cache = {
            "cross_attended_encoder_features_72": torch.randn(1, 1, 256, 72, 72),
            "instance_queries": torch.randn(1, 1, 3, 256),
            "presence_features": torch.randn(1, 1, 1, 256),
            "backbone_fpn": [None, None, torch.randn(1, 256, 72, 72)],
            "prompt_chunks": {
                (0, 1): (torch.randn(2, 1, 256), torch.zeros(1, 2, dtype=torch.bool))
            },
        }
        refiner = torch.randn(1, 1, 256, 36, 36, requires_grad=True)
        output = core.decode_encoder_refiner_chunk_from_cache(
            cache, refiner, 0, 1, True
        )
        self.assertEqual(len(core.segmentation_head.calls), 2)
        self.assertFalse(core.segmentation_head.calls[0][0])
        self.assertTrue(core.segmentation_head.calls[1][0])
        torch.testing.assert_close(
            core.segmentation_head.calls[0][1],
            cache["instance_queries"].reshape(1, 3, 256),
        )
        self.assertFalse(output["instance_mask_logits"].requires_grad)
        self.assertFalse(output["sam3_teacher_logits"].requires_grad)
        (
            output["instance_score_logits"].sum()
            + output["instance_presence_logits"].sum()
        ).backward()
        self.assertIsNone(refiner.grad)
        self.assertIsNone(core.encoder_refiner.proj.weight.grad)
        # Detached score memory allows a separate semantic backward without retain_graph.
        output["final_logits"].mean().backward()
        self.assertGreater(refiner.grad.abs().sum().item(), 0)
        self.assertGreater(core.encoder_refiner.proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in core.dot_prod_scoring.parameters()))


class NativeResolutionTests(unittest.TestCase):
    def test_native_200_mask_pipeline_and_backward(self):
        from ovrs_test.model_builder import SAM3ModelBuilder
        from ovrs_test.models.sam3_image import Sam3Image
        from ovrs_test.models.decoder_input_fusion import DecoderInputFusion

        torch.set_num_threads(2)
        torch.manual_seed(11)
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.hidden_dim = 256
        core.transformer = nn.Module()
        core.transformer.decoder = (
            SAM3ModelBuilder._create_transformer_decoder().requires_grad_(False).eval()
        )
        core.segmentation_head = (
            SAM3ModelBuilder._create_segmentation_head().requires_grad_(False).eval()
        )
        core.dot_prod_scoring = (
            SAM3ModelBuilder._create_dot_product_scoring().requires_grad_(False).eval()
        )
        core.instance_score_adapter = InstanceScoreAdapter()
        fusion = DecoderInputFusion()

        class Refiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder_input_fusion = fusion

            def fuse_decoder_input_chunk(self, **kw):
                return self.decoder_input_fusion(**kw)

        core.encoder_refiner = Refiner()
        memory = torch.randn(72 * 72, 1, 256)
        prompt = torch.randn(8, 1, 256)
        prompt_mask = torch.zeros(1, 8, dtype=torch.bool)
        with torch.no_grad():
            q, _, _, presence = core.transformer.decoder(
                core.transformer.decoder.query_embed.weight[:, None],
                memory,
                pos=torch.randn_like(memory),
                spatial_shapes=torch.tensor([[72, 72]]),
                valid_ratios=torch.ones(1, 1, 2),
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )
            pixel_memory = core.segmentation_head.apply_prompt_cross_attention(
                memory, prompt, prompt_mask
            )
        cache = {
            "cross_attended_encoder_features_72": pixel_memory.permute(1, 2, 0).reshape(
                1, 1, 256, 72, 72
            ),
            "instance_queries": q[-1].permute(1, 0, 2).reshape(1, 1, 200, 256),
            "presence_features": presence.permute(1, 0, 2).reshape(1, 1, 1, 256),
            "backbone_fpn": [
                torch.randn(1, 256, 288, 288),
                torch.randn(1, 256, 144, 144),
                torch.randn(1, 256, 72, 72),
            ],
            "prompt_chunks": {(0, 1): (prompt, prompt_mask)},
        }
        refiner = torch.randn(1, 1, 256, 36, 36, requires_grad=True)
        out = core.decode_encoder_refiner_chunk_from_cache(cache, refiner, 0, 1, True)
        criterion = SemanticCriterion(SemanticCriterionConfig())
        label = torch.zeros(1, 288, 288, dtype=torch.long)
        label[:, :, :20] = 255
        ctx = criterion.prepare_streaming_context(
            {"label_map": label}, 1, torch.tensor([0]), 1
        )
        losses = criterion.forward_chunk(out, ctx, 0, 1)
        losses["total_loss"].backward()
        assert torch.isfinite(losses["total_loss"])
        assert refiner.grad.abs().sum() > 0
        assert all(
            layer.attention.in_proj_weight.grad.abs().sum() > 0
            for layer in core.instance_score_adapter.layers
        )
        assert all(p.grad is None for p in core.segmentation_head.parameters())


if __name__ == "__main__":
    unittest.main()
