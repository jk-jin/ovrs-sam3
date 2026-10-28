"""CPU regression checks; run: python -m unittest discover -s tests -v."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Load this repository as a package independently of its checkout directory name.
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ovrs_sam3", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
if "ovrs_sam3" not in sys.modules:
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

from ovrs_sam3.config_dataclasses import SemanticCriterionConfig
from ovrs_sam3.engine.trainer import Trainer
from ovrs_sam3.losses.semantic_criterion import SemanticCriterion
from ovrs_sam3.models.decoder_input_fusion import DecoderInputFusion
from ovrs_sam3.models.encoder_refiner import ClassConditionedEncoderRefiner
from ovrs_sam3.models.encoder_refiner_attention import (
    EncoderRefinerLayer,
    InterClassAttention,
    WindowAttention,
)
from ovrs_sam3.models.sam3_image import Sam3Image
from ovrs_sam3.models.task_modes import OUTPUT_KEYS


class TinyTextEncoder(nn.Module):
    """Trainable template features stand in for pretrained RemoteCLIP text."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(2, 64, 8))

    def encode_prompt_templates(self, class_names, **kwargs):
        return self.embeddings[: len(class_names)]


def make_refiner(use_checkpoint=False, dropout=0):
    text = TinyTextEncoder()
    return ClassConditionedEncoderRefiner(
        text,
        clip_dim=8,
        prompt_templates=["a photo of {}"] * 64,
        use_checkpoint=use_checkpoint,
        dropout=dropout,
    ), text


def assert_gradients(test, tensors):
    for tensor in tensors:
        test.assertIsNotNone(tensor.grad)
        test.assertTrue(torch.isfinite(tensor.grad).all())
        test.assertGreater(tensor.grad.abs().sum().item(), 0)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_odd_even_order_uses_updated_guidance(self):
        class Update(nn.Module):
            def __init__(self, label, amount, log):
                super().__init__()
                self.label, self.amount, self.log = label, amount, log

            def forward(self, target, other, score):
                self.log.append((self.label, target.clone(), other.clone(), score))
                return torch.full_like(target, self.amount)

        for target_stream in ("fpn", "encoder"):
            log = []
            layer = EncoderRefinerLayer(
                hidden_dim=8, score_embed_dim=8, num_heads=2, intra_target=target_stream
            )
            layer.window_regular = Update("regular", 1, log)
            layer.window_shifted = Update("shifted", 2, log)
            layer.inter_attention = Update("inter", 3, log)
            layer.encoder_ffn = nn.Identity()
            layer.fpn_ffn = nn.Identity()
            encoder, fpn, score = [
                torch.full((1, 2, 8, 3, 3), v) for v in (10.0, 20.0, 30.0)
            ]
            initial = fpn if target_stream == "fpn" else encoder
            other = encoder if target_stream == "fpn" else fpn
            e, f = layer(encoder, fpn, score)
            self.assertEqual([x[0] for x in log], ["regular", "shifted", "inter"])
            torch.testing.assert_close(log[1][1], initial + 1)
            torch.testing.assert_close(log[1][2], other)
            torch.testing.assert_close(log[2][1], other)
            torch.testing.assert_close(log[2][2], initial + 3)
            self.assertTrue(all(x[3] is score for x in log))
            torch.testing.assert_close(e, encoder + 3)
            torch.testing.assert_close(f, fpn + 3)

    def test_window_cannot_mix_prompts_or_wrap_image_edges(self):
        for shift in (0, 1):
            attention = WindowAttention(8, 8, 2, 0, window_size=3, shift_size=shift)
            target, other, score = [torch.randn(2, 3, 8, 6, 6) for _ in range(3)]
            baseline = attention(target, other, score)
            changed = target.clone()
            changed[:, 1] += torch.randn_like(changed[:, 1]) * 20
            torch.testing.assert_close(
                attention(changed, other, score)[:, 0], baseline[:, 0]
            )
            # Bottom/right edge changes must not leak across a cyclic shift to top-left.
            changed = target.clone()
            changed[..., -1, :] += torch.randn_like(changed[..., -1, :]) * 20
            changed[..., :, -1] += torch.randn_like(changed[..., :, -1]) * 20
            torch.testing.assert_close(
                attention(changed, other, score)[..., 0, 0], baseline[..., 0, 0]
            )

    def test_inter_attention_mixes_prompts_only_and_has_target_only_values(self):
        attention = InterClassAttention(8, 8, 2, 0)
        target, other, score = [torch.randn(2, 3, 8, 6, 6) for _ in range(3)]
        baseline = attention(target, other, score)
        changed = target.clone()
        changed[..., 5, 5] += torch.randn_like(changed[..., 5, 5]) * 20
        torch.testing.assert_close(
            attention(changed, other, score)[..., 0, 0], baseline[..., 0, 0]
        )
        changed = target.clone()
        changed[:, 1] += torch.randn_like(changed[:, 1]) * 20
        self.assertFalse(
            torch.allclose(attention(changed, other, score)[:, 0], baseline[:, 0])
        )
        # With a single prompt, guidance cannot change the only available value.
        one = [x[:, :1] for x in (target, other, score)]
        torch.testing.assert_close(
            attention(*one), attention(one[0], one[1] * -7, one[2] * 9)
        )
        # Any prompt reordering must simply reorder the output.
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(
            attention(target[:, order], other[:, order], score[:, order]),
            baseline[:, order],
        )

    def test_layer_checkpoint_output_and_gradients_with_dropout(self):
        for target_stream in ("fpn", "encoder"):
            direct = EncoderRefinerLayer(8, 8, 2, 3, 1, 0.1, target_stream).train()
            checked = copy.deepcopy(direct)
            inputs = [torch.randn(2, 3, 8, 6, 6, requires_grad=True) for _ in range(3)]
            copies = [x.detach().clone().requires_grad_() for x in inputs]
            weights = [torch.randn_like(inputs[0]) for _ in range(2)]
            torch.manual_seed(19)
            output = direct(*inputs)
            sum((x * w).sum() for x, w in zip(output, weights)).backward()
            torch.manual_seed(19)
            other_output = checkpoint(checked, *copies, use_reentrant=False)
            sum((x * w).sum() for x, w in zip(other_output, weights)).backward()
            for x, y in zip(output, other_output):
                torch.testing.assert_close(x, y)
            for x, y in zip(inputs, copies):
                torch.testing.assert_close(x.grad, y.grad)
            for x, y in zip(direct.parameters(), checked.parameters()):
                torch.testing.assert_close(x.grad, y.grad)

    def test_cpu_autocast_and_invalid_windows(self):
        layer = EncoderRefinerLayer(8, 8, 2, 3, 1, 0)
        inputs = [torch.randn(1, 2, 8, 6, 6, requires_grad=True) for _ in range(3)]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(*inputs)
            loss = sum(x.square().mean() for x in output)
        loss.backward()
        assert_gradients(self, inputs)
        with self.assertRaises(ValueError):
            WindowAttention(window_size=12, shift_size=12)
        with self.assertRaises(ValueError):
            layer(*(x[..., :5, :] for x in inputs))


class RefinerFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)

    def test_full_36_grid_fixed_score_and_four_layer_checkpoint_parity(self):
        direct, text = make_refiner(False)
        checked, checked_text = make_refiner(True)
        checked.load_state_dict(direct.state_dict())
        checked_text.load_state_dict(text.state_dict())
        inputs = (
            torch.randn(1, 2, 256, 72, 72, requires_grad=True),
            torch.randn(1, 8, 36, 36, requires_grad=True),
            torch.randn(1, 256, 72, 72, requires_grad=True),
        )
        copies = tuple(x.detach().clone().requires_grad_() for x in inputs)
        weights = [torch.randn(1, 2, 256, 36, 36) for _ in range(2)]
        observed = []
        handle = direct.clip_score_embed.register_forward_hook(
            lambda m, a, out: observed.append(out[0])
        )
        output = direct(inputs[0], inputs[1], ["road", "water"], inputs[2])
        handle.remove()
        self.assertIs(output["score_embed_36"], observed[0])
        self.assertIs(output["score_embed_36"], output["clip_score_embed_36"])
        score_before = output["score_embed_36"].detach().clone()
        self.assertEqual(
            [x.intra_target for x in direct.layers],
            ["fpn", "encoder", "fpn", "encoder"],
        )
        keys = ("refined_encoder_36", "refined_fpn_36")
        sum((output[k] * w).mean() for k, w in zip(keys, weights)).backward()
        torch.testing.assert_close(output["score_embed_36"], score_before)
        checked_out = checked(copies[0], copies[1], ["road", "water"], copies[2])
        sum((checked_out[k] * w).mean() for k, w in zip(keys, weights)).backward()
        for key in keys:
            self.assertEqual(output[key].shape, (1, 2, 256, 36, 36))
            torch.testing.assert_close(output[key], checked_out[key])
        self.assertFalse(
            torch.allclose(
                output["refined_fpn_36"][:, 0], output["refined_fpn_36"][:, 1]
            )
        )
        assert_gradients(self, (*inputs, text.embeddings, *direct.layers.parameters()))
        for x, y in zip(inputs, copies):
            torch.testing.assert_close(x.grad, y.grad)
        for x, y in zip(direct.parameters(), checked.parameters()):
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad)
        torch.testing.assert_close(text.embeddings.grad, checked_text.embeddings.grad)

    def test_fusion_batch_prompt_order_and_checkpoint_gradients(self):
        fusion = DecoderInputFusion(use_checkpoint=False).train()
        inputs = (
            torch.randn(6, 256, 36, 36, requires_grad=True),
            torch.randn(6, 256, 36, 36, requires_grad=True),
            torch.randn(6, 256, 72, 72),
            torch.randn(2, 256, 72, 72),
        )
        # One batched call must equal separate calls for each image/prompt pair.
        with torch.no_grad():
            batched = fusion(*inputs)
            single = torch.cat(
                [
                    fusion(
                        *(x[i : i + 1] for x in inputs[:3]),
                        inputs[3][i // 3 : i // 3 + 1],
                    )
                    for i in range(6)
                ]
            )
        torch.testing.assert_close(batched, single, rtol=2e-5, atol=2e-6)
        copies = tuple(
            x.detach().clone().requires_grad_(x.requires_grad) for x in inputs
        )
        output = fusion(*inputs)
        weight = torch.randn_like(output)
        (output * weight).mean().backward()
        expected_params = [p.grad.clone() for p in fusion.parameters()]
        fusion.zero_grad(set_to_none=True)
        fusion.use_checkpoint = True
        other_output = fusion(*copies)
        (other_output * weight).mean().backward()
        torch.testing.assert_close(output, other_output)
        for x, y in zip(inputs[:2], copies[:2]):
            torch.testing.assert_close(x.grad, y.grad)
        for actual, expected in zip(fusion.parameters(), expected_params):
            torch.testing.assert_close(actual.grad, expected)
        assert_gradients(self, inputs[:2])

    def test_fusion_rejects_misaligned_streams(self):
        fusion = DecoderInputFusion()
        with self.assertRaises(ValueError):
            fusion(
                torch.empty(2, 256, 36, 36),
                torch.empty(1, 256, 36, 36),
                torch.empty(2, 256, 72, 72),
                torch.empty(1, 256, 72, 72),
            )


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_core_chunk_decode_teacher_and_inference_debug(self):
        class FrozenHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Conv2d(256, 1, 1)
                self.requires_grad_(False)
                self.calls = []

            def forward(self, backbone_feats, image_ids, encoder_hidden_states):
                self.calls.append(
                    (torch.is_grad_enabled(), encoder_hidden_states.detach().clone())
                )
                image = encoder_hidden_states.permute(1, 2, 0).reshape(-1, 256, 72, 72)
                return {
                    "semantic_seg": F.interpolate(
                        self.projection(image),
                        (288, 288),
                        mode="bilinear",
                        align_corners=False,
                    )
                }

        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.encoder_refiner, text = make_refiner(True)
        core.segmentation_head = FrozenHead().eval()
        cache = {
            "cross_attended_encoder_features_72": torch.randn(1, 2, 256, 72, 72),
            "clip_image_feat_map": torch.randn(1, 8, 36, 36, requires_grad=True),
            "backbone_fpn": [None, None, torch.randn(1, 256, 72, 72)],
            "class_names": ["road", "water"],
            "chunk_class_counts": [1, 1],
        }
        batch = SimpleNamespace(find_text_batch=["road", "water"])
        lowres = core.run_encoder_refiner_lowres_from_cache(cache, batch)
        output = core.decode_encoder_refiner_chunk_from_cache(
            cache,
            lowres["refined_encoder_36"][:, 1:],
            lowres["refined_fpn_36"][:, 1:],
            1,
            2,
            return_teacher_logits=True,
        )
        self.assertFalse(output[OUTPUT_KEYS.sam3_teacher_logits].requires_grad)
        self.assertTrue(output[OUTPUT_KEYS.final_logits].requires_grad)
        self.assertEqual([x[0] for x in core.segmentation_head.calls], [False, True])
        torch.testing.assert_close(
            core.segmentation_head.calls[0][1],
            core._feature_72_to_hidden_states(
                cache["cross_attended_encoder_features_72"][:, 1:]
            ),
        )
        output[OUTPUT_KEYS.final_logits].square().mean().backward()
        assert_gradients(
            self,
            (
                cache["clip_image_feat_map"],
                text.embeddings,
                core.encoder_refiner.decoder_input_fusion.semantic_branch[0].weight,
                core.encoder_refiner.decoder_input_fusion.detail_branch[0].weight,
            ),
        )
        self.assertTrue(
            all(x.grad is None for x in core.segmentation_head.parameters())
        )
        core.eval()
        with torch.no_grad():
            prediction = core.run_encoder_refiner_from_cache(
                cache, batch, return_debug=True
            )
            cache["chunk_class_counts"] = [2]
            unchunked = core.run_encoder_refiner_from_cache(cache, batch)
        torch.testing.assert_close(
            prediction[OUTPUT_KEYS.final_logits],
            unchunked[OUTPUT_KEYS.final_logits],
            rtol=2e-5,
            atol=2e-6,
        )
        for key in (
            OUTPUT_KEYS.refined_encoder_36,
            OUTPUT_KEYS.refined_fpn_36,
            OUTPUT_KEYS.score_embed_36,
        ):
            self.assertIn(key, prediction)
            self.assertFalse(prediction[key].requires_grad)
        self.assertEqual(prediction[OUTPUT_KEYS.final_logits].shape, (1, 2, 288, 288))

    def test_config_build_chain_rejects_removed_options(self):
        from ovrs_sam3.config_dataclasses import EncoderRefinerConfig, OpenCLIPConfig
        from ovrs_sam3.engine.config import Config
        from ovrs_sam3.model_builder import SAM3ModelBuilder

        for config in (
            "configs/train/isaid_loveda_exp.py",
            "configs/train/isaid_loveda_full.py",
            "configs/test/loveda.py",
        ):
            model = Config.fromfile(ROOT / config)["model"]
            options = SAM3ModelBuilder.validate_encoder_refiner_cfg(
                model["encoder_refiner_cfg"]
            )
            self.assertEqual((options.window_size, options.shift_size), (12, 6))
            OpenCLIPConfig(**model["openclip_cfg"])
        with self.assertRaises(TypeError):
            EncoderRefinerConfig(score_intra_attn_type="window")
        with self.assertRaises(TypeError):
            OpenCLIPConfig(image_intermediate_layers=[7, 15])
        for options in ({"window_size": 13}, {"shift_size": 12}, {"num_heads": 7}):
            with self.assertRaises(ValueError):
                SAM3ModelBuilder.validate_encoder_refiner_cfg(options)

    def test_clip_dense_gradient_boundary_without_middle_outputs(self):
        from ovrs_sam3.models.openclip_image_encoder import OpenCLIPImageEncoder

        class TinyVisual(nn.Module):
            def __init__(self):
                super().__init__()
                self.width = self.output_dim = 8
                self.conv1 = nn.Conv2d(3, 8, 14, stride=14, bias=False)
                self.class_embedding = nn.Parameter(torch.randn(8))
                self.positional_embedding = nn.Parameter(torch.randn(5, 8))
                self.patch_dropout = nn.Identity()
                self.ln_pre, self.ln_post = nn.LayerNorm(8), nn.LayerNorm(8)
                self.proj = nn.Parameter(torch.randn(8, 8))
                block = nn.Module()
                block.ln_1, block.ln_2 = nn.LayerNorm(8), nn.LayerNorm(8)
                block.attn = nn.MultiheadAttention(8, 2)
                block.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
                self.transformer = nn.Module()
                self.transformer.resblocks = nn.ModuleList([block])

        image_encoder = OpenCLIPImageEncoder(TinyVisual())
        raw = [torch.rand(3, 32, 32)]
        image_encoder.visual.requires_grad_(True)
        image_encoder.set_enable_grad(True)
        result = image_encoder.encode_raw_images(raw, torch.device("cpu"))
        self.assertEqual(set(result), {"feat_map"})
        self.assertEqual(result["feat_map"].shape, (1, 8, 36, 36))
        result["feat_map"].square().mean().backward()
        assert_gradients(self, (image_encoder.visual.proj,))
        with torch.no_grad():
            self.assertFalse(
                image_encoder.encode_raw_images(raw, torch.device("cpu"))[
                    "feat_map"
                ].requires_grad
            )
        image_encoder.set_enable_grad(False)
        self.assertFalse(
            image_encoder.encode_raw_images(raw, torch.device("cpu"))[
                "feat_map"
            ].requires_grad
        )


class StreamingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    def test_actual_trainer_joint_proxy_backward_matches_full_loss(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = EncoderRefinerLayer(8, 8, 2, 3, 1, 0)
                self.head = nn.Conv2d(16, 1, 1)

            def build_encoder_refiner_cache(self, batch):
                return {"chunk_class_counts": [2, 1]}

            def run_encoder_refiner_lowres_from_cache(
                self, cache, batch, return_debug=False
            ):
                encoder, fpn = self.layer(*batch.features)
                return {"refined_encoder_36": encoder, "refined_fpn_36": fpn}

            def decode_encoder_refiner_chunk_from_cache(
                self,
                encoder_refiner_cache,
                refined_encoder_36_chunk,
                refined_fpn_36_chunk,
                class_start,
                class_end,
                return_teacher_logits=False,
            ):
                b, p, d, h, w = refined_encoder_36_chunk.shape
                x = torch.cat((refined_encoder_36_chunk, refined_fpn_36_chunk), dim=2)
                logits = self.head(x.reshape(b * p, 2 * d, h, w))
                logits = F.interpolate(
                    logits, (288, 288), mode="bilinear", align_corners=False
                )
                result = {OUTPUT_KEYS.final_logits: logits.reshape(b, p, 288, 288)}
                if return_teacher_logits:
                    result[OUTPUT_KEYS.sam3_teacher_logits] = torch.zeros_like(
                        result[OUTPUT_KEYS.final_logits]
                    )
                return result

        label = torch.randint(0, 2, (2, 6, 6))
        label[:, 0, 0] = 255
        features = [torch.randn(2, 3, 8, 6, 6, requires_grad=True) for _ in range(3)]
        batch = SimpleNamespace(
            features=features,
            find_targets=[SimpleNamespace(semantic_label_map=label)],
            find_metadatas=[
                SimpleNamespace(
                    num_prompts=3, num_classes=2, prompt_to_class_id=[0, 0, 1]
                )
            ],
        )
        model = ToyModel()
        reference = copy.deepcopy(model)
        criterion = SemanticCriterion(
            SemanticCriterionConfig(sam3_mask_distill_weight=0.5)
        )
        # A non-unit scale checks that proxy gradients are not scaled twice.
        scale = 128
        trainer = Trainer.__new__(Trainer)
        trainer.model, trainer.criterion = model, criterion
        trainer.cfg, trainer.device = (
            SimpleNamespace(use_amp=False),
            torch.device("cpu"),
        )
        trainer.scaler = SimpleNamespace(scale=lambda loss: loss * scale)
        stats = trainer._compute_train_loss_streaming(batch)
        expected_features = [x.detach().clone().requires_grad_() for x in features]
        expected_batch = SimpleNamespace(features=expected_features)
        output = reference.run_encoder_refiner_lowres_from_cache({}, expected_batch)
        logits = reference.decode_encoder_refiner_chunk_from_cache(
            {},
            output["refined_encoder_36"],
            output["refined_fpn_36"],
            0,
            3,
            return_teacher_logits=True,
        )
        expected = criterion(logits, {"label_map": label}, torch.tensor([0, 0, 1]), 2)
        (expected["total_loss"] * scale).backward()
        torch.testing.assert_close(stats["total_loss"], expected["total_loss"])
        for actual, wanted in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-4, atol=2e-6)
        for actual, wanted in zip(features, expected_features):
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
