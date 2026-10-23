"""CPU behavioral regressions; no pretrained weights or datasets needed."""

import copy
import importlib
import importlib.util
from pathlib import Path
import sys
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


attention = load("models.encoder_refiner_attention")
EncoderRefinerLayer = attention.EncoderRefinerLayer
ClassConditionedEncoderRefiner = load("models.encoder_refiner").ClassConditionedEncoderRefiner
Sam3Image = load("models.sam3_image").Sam3Image
SAM3ModelBuilder = load("model_builder").SAM3ModelBuilder
SemanticCriterion = load("losses.semantic_criterion").SemanticCriterion
SemanticCriterionConfig = load("config_dataclasses").SemanticCriterionConfig
OUTPUT_KEYS = load("models.task_modes").OUTPUT_KEYS
torch.set_num_threads(1)


def make_criterion(**kwargs):
    values = dict(sam3_mask_distill_weight=0.0, final_dice_weight=0.0)
    values.update(kwargs)
    return SemanticCriterion(SemanticCriterionConfig(**values))


def make_layer(kind="intra", mode="window", presence=True, mask=True, dropout=0.0):
    return EncoderRefinerLayer(
        hidden_dim=16, score_embed_dim=16, num_heads=4, window_size=2, shift_size=1,
        score_attention_type=kind, score_intra_attn_type=mode, feature_intra_attn_type=mode,
        dropout=dropout, inter_presence_bias_enabled=presence, attention_mask_gate_enabled=mask,
    )


def layer_inputs():
    return [torch.randn(1, 2, 16, 2, 2), torch.randn(1, 2, 16, 2, 2),
            torch.tensor([[0.9, 0.1]]), torch.randn(1, 2, 256, 2, 2), torch.randn(1, 2, 256, 2, 2)]


class AttentionModulationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_presence_bias_is_target_class_specific_and_self_is_unchanged(self):
        probabilities = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.8, 1.0]], requires_grad=True)
        bias = attention.presence_log_bias(probabilities, 0.05, 1.0)
        self.assertFalse(bias.requires_grad)
        self.assertTrue(torch.isfinite(bias).all())
        torch.testing.assert_close(bias.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 3))
        torch.testing.assert_close(bias[0, 0, 1], torch.tensor(0.145).log())
        torch.testing.assert_close(bias[0, 1, 0], torch.tensor(0.905).log())
        torch.testing.assert_close(bias[0, 0, 2], torch.tensor(0.05).log())
        self.assertEqual(bias[1, 0, 2].item(), 0)

    def test_presence_bias_matches_renormalized_probability_gating(self):
        logits = torch.randn(2, 3, 3)
        probabilities = torch.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.6]])
        bias = attention.presence_log_bias(probabilities, 0.05, 0.7)
        actual = (logits + bias).softmax(-1)
        weights = logits.softmax(-1) * bias.exp()
        expected = weights / weights.sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected)

    def test_actual_class_attention_uses_correct_image_and_key_axis(self):
        feature = torch.randn(2, 3, 8, 2, 2)
        score = torch.randn_like(feature)
        probabilities = torch.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.6]])
        for stream, value in (("feature", feature), ("score", score)):
            module = attention.ClassScoreAttention(
                hidden_dim=8, score_embed_dim=8, num_heads=2, value_stream=stream,
                dropout=0.0, presence_bias_enabled=True,
            )
            with torch.no_grad():
                for projection in (module.q_proj, module.k_proj):
                    projection.weight.zero_(); projection.bias.zero_()
                for projection in (module.v_proj, module.out_proj):
                    projection.weight.copy_(torch.eye(8)); projection.bias.zero_()
            actual = module(feature, score, presence_probs=probabilities)
            weights = attention.presence_log_bias(probabilities, 0.05, 1.0).softmax(-1)
            expected = torch.einsum("bij,bjdhw->bidhw", weights, value)
            torch.testing.assert_close(actual, expected)

    def test_residual_gate_zero_midpoint_warmup_and_detach(self):
        logits = torch.tensor([-torch.inf, 0., torch.inf], requires_grad=True)
        gate = attention.mask_residual_gate(logits, 1.0, True)
        torch.testing.assert_close(gate, torch.tensor([0., 0.5, 1.0]), rtol=0, atol=0)
        self.assertFalse(gate.requires_grad)
        torch.testing.assert_close(attention.mask_residual_gate(logits, 0.0, True), torch.ones(3))
        torch.testing.assert_close(attention.mask_residual_gate(logits, 0.5, True), torch.tensor([0.5, 0.75, 1.]))
        tiny = attention.mask_residual_gate(torch.tensor(-30.), 1.0, True)
        self.assertGreater(tiny.item(), 0)
        torch.testing.assert_close(tiny, torch.tensor(-30.).sigmoid(), rtol=0, atol=0)
        attached = attention.mask_residual_gate(logits, 1.0, False)
        attached.sum().backward()
        self.assertGreater(logits.grad[1].item(), 0)

    def test_shared_layer_mask_gates_both_attention_updates_by_receiving_class(self):
        class ConstantUpdate(nn.Module):
            def forward(self, feature, **kwargs):
                return torch.ones_like(feature)

        class FixedMask(nn.Module):
            def __init__(self, logits):
                super().__init__(); self.logits = logits; self.calls = 0

            def forward(self, feature, score):
                self.calls += 1
                return self.logits

        logits = torch.tensor([[[[[-torch.inf, 0.], [torch.inf, -torch.inf]]],
                                [[[torch.inf, -torch.inf], [0., torch.inf]]]]])
        gate = logits.sigmoid()
        for kind in ("intra", "inter"):
            for mode, steps in (("window", 2), ("local_3x3", 4)):
                for presence in (False, True):
                    with self.subTest(kind=kind, mode=mode, presence=presence):
                        layer = make_layer(kind, mode, presence=presence)
                        layer.mask_head = FixedMask(logits)
                        layer.class_attn = ConstantUpdate()
                        if mode == "window":
                            layer.window_attn_regular = ConstantUpdate()
                            layer.window_attn_shifted = ConstantUpdate()
                        else:
                            layer.local_attn = nn.ModuleList([ConstantUpdate() for _ in range(steps)])
                        layer._ffn_feature_update = lambda feature: torch.zeros_like(feature)
                        layer._ffn_score_update = lambda score: torch.zeros_like(score)
                        args = layer_inputs()
                        feature, score, masks = layer(*args)
                        feature_steps = 1 if kind == "intra" else steps
                        score_steps = steps if kind == "intra" else 1
                        torch.testing.assert_close(feature, args[0] + feature_steps * gate)
                        torch.testing.assert_close(score, args[1] + score_steps * gate)
                        zero_pixels = (gate == 0).expand_as(feature)
                        torch.testing.assert_close(feature[zero_pixels], args[0][zero_pixels], rtol=0, atol=0)
                        torch.testing.assert_close(score[zero_pixels], args[1][zero_pixels], rtol=0, atol=0)
                        self.assertIs(masks, logits)
                        self.assertEqual(layer.mask_head.calls, 1)

    def test_zero_attention_mask_does_not_gate_ffn_residuals(self):
        class ZeroMask(nn.Module):
            def forward(self, feature, score):
                return torch.full_like(feature[:, :, :1], -torch.inf)

        for kind in ("intra", "inter"):
            layer = make_layer(kind, presence=False)
            layer.mask_head = ZeroMask()
            layer._ffn_feature_update = lambda feature: torch.ones_like(feature)
            layer._ffn_score_update = lambda score: torch.full_like(score, 2.)
            args = layer_inputs(); args[2] = None
            feature, score, _ = layer(*args)
            torch.testing.assert_close(feature, args[0] + 1)
            torch.testing.assert_close(score, args[1] + 2)

    def test_every_intra_substep_gates_only_its_residual_target(self):
        class ConstantUpdate(nn.Module):
            def forward(self, feature, **kwargs):
                return torch.ones_like(feature)

        for kind in ("intra", "inter"):
            for mode, steps in (("window", 2), ("local_3x3", 4)):
                with self.subTest(kind=kind, mode=mode):
                    layer = make_layer(kind=kind, mode=mode, presence=False)
                    if mode == "window":
                        layer.window_attn_regular = ConstantUpdate()
                        layer.window_attn_shifted = ConstantUpdate()
                    else:
                        layer.local_attn = nn.ModuleList([ConstantUpdate() for _ in range(4)])
                    feature = torch.full((1, 2, 16, 2, 2), 2.)
                    score = torch.full_like(feature, 3.)
                    gate = torch.full((1, 2, 1, 2, 2), 0.1)
                    output_feature, output_score = layer._update_intra(
                        feature, score, torch.zeros(1, 2, 256, 2, 2), gate,
                    )
                    torch.testing.assert_close(output_feature, feature + (steps * 0.1 if kind == "inter" else 0))
                    torch.testing.assert_close(output_score, score + (steps * 0.1 if kind == "intra" else 0))

    def test_all_four_ablation_combinations_for_both_layer_kinds_and_spatial_modes(self):
        for presence in (False, True):
            for mask in (False, True):
                for kind in ("intra", "inter"):
                    for mode in ("window", "local_3x3"):
                        with self.subTest(presence=presence, mask=mask, kind=kind, mode=mode):
                            module = make_layer(kind, mode, presence, mask)
                            inputs = layer_inputs()
                            if not presence:
                                inputs[2] = None
                            output_feature, output_score, masks = module(*inputs)
                            loss = output_feature.square().mean() + output_score.square().mean()
                            if mask:
                                self.assertEqual(masks.shape, (1, 2, 1, 2, 2))
                                loss = loss + masks.square().mean()
                            else:
                                self.assertIsNone(masks)
                                self.assertFalse(any("mask_head" in name for name, _ in module.named_parameters()))
                            self.assertEqual(module.class_attn.q_proj.in_features, 32)
                            loss.backward()
                            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters()))

    def test_neutral_settings_match_unmodulated_layer_exactly(self):
        baseline = make_layer(presence=False, mask=False)
        enabled = make_layer(presence=True, mask=True)
        enabled.load_state_dict(baseline.state_dict(), strict=False)
        args = layer_inputs()
        plain = baseline(args[0], args[1], None, *args[3:])
        neutral = enabled(args[0], args[1], torch.ones(1, 2), *args[3:], mask_gate_strength=0.0)
        for actual, expected in zip(neutral[:2], plain[:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gate_detach_keeps_mask_head_on_auxiliary_supervision(self):
        module = make_layer(presence=False, mask=True)
        args = layer_inputs(); args[2] = None
        result = module(*args)
        sum(value.square().mean() for value in result[:2]).backward()
        self.assertTrue(all(parameter.grad is None for parameter in module.mask_head.parameters()))
        module.zero_grad(set_to_none=True)
        mask_logits = module(*args)[2]
        F.binary_cross_entropy_with_logits(mask_logits, torch.ones_like(mask_logits)).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in module.mask_head.parameters()))

    def test_layer_checkpoint_matches_outputs_and_gradients(self):
        from torch.utils.checkpoint import checkpoint
        for presence, mask in ((False, False), (True, True)):
            plain = make_layer("inter", "local_3x3", presence, mask, dropout=0.1)
            checked = copy.deepcopy(plain)
            base = layer_inputs()
            if not presence:
                base[2] = None
            results, gradients = [], []
            for use_checkpoint, module in ((False, plain), (True, checked)):
                args = [None if value is None else value.clone().requires_grad_(i < 2) for i, value in enumerate(base)]
                torch.manual_seed(77)
                result = checkpoint(module, *args, 0.6, use_reentrant=False) if use_checkpoint else module(*args, 0.6)
                sum(value.square().mean() for value in result if value is not None).backward()
                results.append(result)
                gradients.append([parameter.grad for parameter in module.parameters()] + [args[0].grad, args[1].grad])
            for actual, expected in zip(*results):
                if actual is not None:
                    torch.testing.assert_close(actual, expected)
            for actual, expected in zip(*gradients):
                torch.testing.assert_close(actual, expected)

    def test_auxiliary_loss_balances_positive_and_negative_pixels_including_ignore(self):
        criterion = make_criterion(final_balanced_bce_weight=0.0, refiner_mask_bce_weight=1.0)
        labels = torch.tensor([[[0, 255], [255, 255]]])
        masks = torch.full((2, 1, 1, 2, 2), 2., requires_grad=True)
        final = torch.zeros(1, 1, 2, 2, requires_grad=True)
        loss = criterion({OUTPUT_KEYS.final_logits: final, OUTPUT_KEYS.refiner_mask_logits: masks},
                         {"label_map": labels}, torch.tensor([0]), 1)
        expected = (F.softplus(torch.tensor(-2.)) + F.softplus(torch.tensor(2.))) / 2
        torch.testing.assert_close(loss["loss_refiner_mask_bce"], expected)
        loss["total_loss"].backward()
        self.assertLess(masks.grad[0, 0, 0, 0, 0].item(), 0)
        self.assertTrue((masks.grad[0, 0, 0].flatten()[1:] > 0).all())
        self.assertEqual(final.grad.abs().sum().item(), 0)

    def test_auxiliary_all_negative_all_positive_and_all_ignore_are_finite(self):
        criterion = make_criterion(final_balanced_bce_weight=0.0, refiner_mask_bce_weight=1.0)
        for label, sign in ((255, 1), (0, -1)):
            masks = torch.zeros(2, 1, 1, 2, 2, requires_grad=True)
            output = {OUTPUT_KEYS.final_logits: torch.zeros(1, 1, 4, 4, requires_grad=True),
                      OUTPUT_KEYS.refiner_mask_logits: masks}
            loss = criterion(output, {"label_map": torch.full((1, 4, 4), label)}, torch.tensor([0]), 1)
            torch.testing.assert_close(loss["loss_refiner_mask_bce"], torch.tensor(2.).log())
            loss["total_loss"].backward()
            self.assertTrue((sign * masks.grad > 0).all())

    def test_mask_switch_off_has_no_auxiliary_loss(self):
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        losses = make_criterion()({OUTPUT_KEYS.final_logits: logits},
                                 {"label_map": torch.tensor([[[0, 255], [1, 1]]])}, torch.arange(2), 2)
        self.assertEqual(losses["loss_refiner_mask_bce"].item(), 0)
        self.assertEqual(losses["loss_refiner_mask_weighted"].item(), 0)
        losses["total_loss"].backward()
        self.assertEqual(logits.grad[0, :, 0, 1].abs().sum().item(), 0)

    def test_streamed_losses_and_gradients_match_full_for_all_layers_and_aliases(self):
        criterion = make_criterion()
        labels = torch.tensor([[[0, 255], [1, 1]], [[255, 255], [255, 255]]])
        mapping = torch.tensor([0, 1, 0])
        final = torch.randn(2, 3, 2, 2, requires_grad=True)
        masks = torch.randn(4, 2, 3, 1, 1, requires_grad=True)
        outputs = {OUTPUT_KEYS.final_logits: final, OUTPUT_KEYS.refiner_mask_logits: masks}
        full = criterion(outputs, {"label_map": labels}, mapping, 2)
        full["total_loss"].backward()
        expected_grads = [final.grad.clone(), masks.grad.clone()]
        final.grad = masks.grad = None
        context = criterion.prepare_streaming_context({"label_map": labels}, 3, mapping, 2, (2, 2))
        accum = {}
        for start, end in ((0, 1), (1, 3)):
            result = criterion.forward_chunk({OUTPUT_KEYS.final_logits: final[:, start:end],
                                             OUTPUT_KEYS.refiner_mask_logits: masks[:, :, start:end]}, context, start, end)
            result["total_loss"].backward()
            for key, value in result.items():
                accum[key] = accum.get(key, 0) + value.detach()
        for key in full:
            torch.testing.assert_close(accum[key], full[key])
        for actual, expected in zip((final.grad, masks.grad), expected_grads):
            torch.testing.assert_close(actual, expected)

    def test_trainer_combines_mask_and_feature_proxy_gradients_in_one_backward(self):
        Trainer = load("engine.trainer").Trainer

        class Model(nn.Module):
            def __init__(self, masks_enabled):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.4))
                self.decoder_weight = nn.Parameter(torch.tensor(0.7))
                self.mask_weight = nn.Parameter(torch.tensor(0.6)) if masks_enabled else None
                self.steps = []

            def set_training_step(self, step):
                self.steps.append(step)

            def build_encoder_refiner_cache(self, batch):
                return {"chunk_class_counts": [1, 2]}

            def run_encoder_refiner_lowres_from_cache(self, *args, **kwargs):
                features = self.weight * torch.tensor([1., 2., 3.]).view(1, 3, 1, 1, 1)
                result = {"refiner_features_36": features}
                if self.mask_weight is not None:
                    maps = features[..., 0] * self.mask_weight
                    result[OUTPUT_KEYS.refiner_mask_logits] = torch.stack([maps.expand(1, 3, 2, 2), (maps + 0.2).expand(1, 3, 2, 2)])
                return result

            def decode_encoder_refiner_chunk_from_cache(self, refiner_feature_36_chunk, class_start, class_end, **kwargs):
                logits = refiner_feature_36_chunk[..., 0].expand(1, class_end - class_start, 288, 288)
                return {OUTPUT_KEYS.final_logits: logits * self.decoder_weight}

        labels = torch.full((1, 288, 288), 255, dtype=torch.long)
        labels[:, :144, :144] = 0
        labels[:, 144:, 144:] = 1
        batch = types.SimpleNamespace(find_targets=[types.SimpleNamespace(semantic_label_map=labels)],
            find_metadatas=[types.SimpleNamespace(num_prompts=3, num_classes=2, prompt_to_class_id=[0, 1, 0])])
        for masks_enabled in (False, True):
            model = Model(masks_enabled); reference = copy.deepcopy(model)
            trainer = Trainer.__new__(Trainer)
            trainer.model, trainer.criterion = model, make_criterion()
            trainer.cfg = types.SimpleNamespace(use_amp=False)
            trainer.device = torch.device("cpu"); trainer.global_iter = 500
            trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
            actual = trainer._compute_train_loss_streaming(batch)
            lowres = reference.run_encoder_refiner_lowres_from_cache()
            outputs = reference.decode_encoder_refiner_chunk_from_cache(
                refiner_feature_36_chunk=lowres["refiner_features_36"], class_start=0, class_end=3,
            )
            if masks_enabled:
                outputs[OUTPUT_KEYS.refiner_mask_logits] = lowres[OUTPUT_KEYS.refiner_mask_logits]
            expected = trainer.criterion(outputs, {"label_map": labels}, torch.tensor([0, 1, 0]), 2)
            expected["total_loss"].backward()
            self.assertEqual(model.steps, [500])
            for key, value in actual.items():
                torch.testing.assert_close(value, expected[key])
            for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_presence_decoder_outputs_probability_and_is_frozen(self):
        core = Sam3Image.__new__(Sam3Image); nn.Module.__init__(core)
        core.transformer = nn.Module()
        core.transformer.decoder = SAM3ModelBuilder._create_transformer_decoder().eval()
        seen = []
        hook = core.transformer.decoder.register_forward_hook(lambda module, args, result: seen.append(result))
        encoder = {"memory": torch.randn(6, 2, 256), "padding_mask": None,
                   "pos_embed": torch.randn(6, 2, 256), "level_start_index": torch.tensor([0]),
                   "spatial_shapes": torch.tensor([[2, 3]]), "valid_ratios": torch.ones(2, 1, 2)}
        output = core._predict_presence(encoder, torch.randn(4, 2, 256), torch.zeros(2, 4, dtype=torch.bool), 1, 2)
        torch.testing.assert_close(output["presence_probs"], seen[0][2][-1].sigmoid().reshape(1, 2))
        self.assertFalse(output["presence_probs"].requires_grad)
        self.assertEqual(set(output), {"presence_probs"})
        self.assertTrue(all(not parameter.requires_grad for parameter in core.transformer.decoder.parameters()))
        hook.remove()

    def test_disabled_presence_skips_decoder_and_uses_raw_memory_when_enabled(self):
        for enabled in (False, True):
            core = Sam3Image.__new__(Sam3Image); nn.Module.__init__(core)
            core.hidden_dim = 256; core.inter_presence_bias_enabled = enabled
            raw = torch.zeros(72 * 72, 1, 256)
            core._encode_prompt = lambda *args: (torch.zeros(2, 1, 256), torch.zeros(1, 2, dtype=torch.bool), {})
            core._run_full_encoder = lambda **kwargs: {"memory": raw}
            seen = []
            core._predict_presence = lambda encoder, *args: seen.append(encoder["memory"]) or {"presence_probs": torch.tensor([[0.8]])}
            core.segmentation_head = nn.Module()
            core.segmentation_head.apply_prompt_cross_attention = lambda **kwargs: kwargs["encoder_hidden_states"] + 5
            output = core._encode_sam3_chunk({}, None, None, 1, 1)
            self.assertEqual(len(seen), int(enabled))
            self.assertEqual("presence_probs" in output, enabled)
            self.assertEqual(output["encoder_features_72"].mean().item(), 5)
            if enabled:
                self.assertIs(seen[0], raw)

    def test_builder_flags_defaults_validation_and_decoder_construction(self):
        config = SAM3ModelBuilder.build_config()
        self.assertTrue(config.encoder_refiner_cfg.inter_presence_bias_enabled)
        self.assertTrue(config.encoder_refiner_cfg.attention_mask_gate_enabled)
        self.assertEqual(config.encoder_refiner_cfg.attention_mask_gate_warmup_iters, 0)
        self.assertFalse(hasattr(config.encoder_refiner_cfg, "intra_mask_gate_floor"))
        for fields in ({"inter_presence_bias_floor": 0.0}, {"attention_mask_gate_enabled": 1},
                       {"inter_presence_bias_strength": float("nan")}, {"attention_mask_gate_warmup_iters": -1}):
            with self.assertRaises((TypeError, ValueError)):
                SAM3ModelBuilder.build_config(encoder_refiner_cfg=fields)
        with patch.object(SAM3ModelBuilder, "_create_transformer_encoder", return_value=nn.Linear(2, 2)), \
             patch.object(SAM3ModelBuilder, "_create_transformer_decoder") as decoder:
            transformer = SAM3ModelBuilder._create_transformer(False)
            self.assertIsNone(transformer.decoder)
            decoder.assert_not_called()

    def test_full_refiner_warmup_eval_and_disabled_mask_outputs(self):
        class Scores(nn.Module):
            def forward(self, class_names, remoteclip_feat_map):
                return torch.zeros(1, 2, 256, 36, 36), torch.zeros(1, 2, 64, 36, 36), torch.zeros(2, 64, 8)

        class Layer(nn.Module):
            def __init__(self, mask):
                super().__init__(); self.mask = mask
                self.projection = nn.Linear(256, 256); self.strengths = []

            def forward(self, feature_36, score_embed_36, presence_probs, clip_mid_36, sam_fpn_36, mask_gate_strength):
                self.strengths.append(mask_gate_strength)
                updated = self.projection(feature_36.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
                masks = updated.mean(2, keepdim=True) if self.mask else None
                return feature_36 + updated, score_embed_36, masks

        for mask in (False, True):
            module = ClassConditionedEncoderRefiner(
                clip_text_encoder=None, clip_dim=8, clip_native_dim=4, prompt_templates=["{}"] * 64,
                inter_presence_bias_enabled=False, attention_mask_gate_enabled=mask, use_checkpoint=True,
                attention_mask_gate_warmup_iters=1000,
            )
            module.clip_score_embed = Scores()
            module.layers = nn.ModuleList([Layer(mask) for _ in range(4)])
            module.set_training_step(500)
            self.assertEqual(module.mask_gate_progress, 0.5)
            args = (torch.randn(1, 2, 256, 72, 72), torch.zeros(1, 8, 36, 36), ["a", "b"], None,
                    [torch.randn(1, 4, 36, 36)] * 2, (7, 15), torch.randn(1, 256, 72, 72))
            output = module(*args)
            self.assertEqual(OUTPUT_KEYS.refiner_mask_logits in output, mask)
            loss = output["refiner_features_36"].sin().mean()
            if mask:
                self.assertEqual(output[OUTPUT_KEYS.refiner_mask_logits].shape, (4, 1, 2, 36, 36))
                loss = loss + output[OUTPUT_KEYS.refiner_mask_logits].square().mean()
            loss.backward()
            self.assertEqual(module.layers[0].strengths[0], 0.5)
            module.eval()
            with torch.no_grad():
                module(*args)
            self.assertEqual(module.layers[0].strengths[-1], 1.0)
            self.assertFalse(hasattr(module.decoder_input_fusion, "query_cross_attention"))


if __name__ == "__main__":
    unittest.main()
