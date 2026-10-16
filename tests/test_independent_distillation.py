"""CPU regressions without pretrained weights or optional vision packages."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ovrs_sam3", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(spec)
sys.modules.setdefault("ovrs_sam3", package)
from ovrs_sam3.config_dataclasses import SemanticCriterionConfig
from ovrs_sam3.engine.trainer import Trainer
from ovrs_sam3.losses.semantic_criterion import SemanticCriterion
from ovrs_sam3.models.decoder_input_fusion import DecoderInputFusion
from ovrs_sam3.models.encoder_refiner import ClassConditionedEncoderRefiner
from ovrs_sam3.models.encoder_refiner_attention import EncoderRefinerLayer
from ovrs_sam3.models.task_modes import OUTPUT_KEYS


def load_decode_methods():
    """Execute the real methods without importing unused torchvision encoders."""
    tree = ast.parse((ROOT / "models/sam3_image.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sam3Image")
    names = {"_feature_72_to_hidden_states", "decode_independent_distill_pairs_from_cache"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = {"torch": torch, "OUTPUT_KEYS": OUTPUT_KEYS}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "models/sam3_image.py"), "exec"), namespace)
    return namespace["Sam3Image"]


def criterion(weight=0.5, width=1):
    return SemanticCriterion(SemanticCriterionConfig(
        final_balanced_bce_weight=1.0, final_dice_weight=0.0,
        sam3_mask_distill_weight=weight, sam3_mask_distill_boundary_width=width,
    ))


def context_for(crit, labels, mapping):
    return crit.prepare_streaming_context(
        targets={"label_map": labels}, num_prompt_channels=len(mapping),
        prompt_to_class_id=torch.tensor(mapping), num_label_classes=max(mapping) + 1,
        target_hw=labels.shape[-2:],
    )


class ToyModel(nn.Module):
    """Small differentiable model exercising the actual Trainer orchestration."""
    def __init__(self, chunk_size):
        super().__init__()
        self.clip = nn.Parameter(torch.tensor(0.7))
        self.score = nn.Parameter(torch.tensor(0.4))
        self.guidance = nn.Parameter(torch.tensor(0.2))
        self.refiner = nn.Parameter(torch.tensor(0.6))
        self.fusion = nn.Parameter(torch.tensor(0.8))
        self.chunk_size = chunk_size
        self.pairs_seen = []
        self.lowres_calls = 0

    def build_encoder_refiner_cache(self, batch):
        count = batch.find_metadatas[0].num_prompts
        return {"chunk_class_counts": [min(self.chunk_size, count - i) for i in range(0, count, self.chunk_size)]}

    def refine(self, feature, score, guidance):
        # Nonlinear shared parameters and true prompt mixing in the main branch.
        return torch.tanh(feature + self.refiner * (score + score.mean(1, keepdim=True) + guidance))

    def run_encoder_refiner_lowres_from_cache(self, cache, batch, return_debug=False):
        self.lowres_calls += 1
        b, p = 2, batch.find_metadatas[0].num_prompts
        source = torch.arange(1, b * p * 4 + 1).reshape(b, p, 1, 2, 2).float() / 40
        image = self.clip * source
        score = torch.sin(image * self.score)
        guide = self.guidance * image.mean(1)
        inputs = {"initial_feature_36": source, "initial_score_36": score,
                  "clip_guidance_36": guide, "fpn_guidance_36": source.mean(1)}
        main = self.refine(source, score, guide[:, None])
        return {"refiner_features_36": main, "distill_inputs": inputs}

    def logits(self, x):
        b, p = x.shape[:2]
        return F.interpolate(x.reshape(b * p, 1, 2, 2), (288, 288), mode="bilinear", align_corners=False).reshape(b, p, 288, 288) * self.fusion

    def decode_encoder_refiner_chunk_from_cache(self, encoder_refiner_cache, refiner_feature_36_chunk, class_start, class_end):
        return {OUTPUT_KEYS.final_logits: self.logits(refiner_feature_36_chunk)}

    def decode_independent_distill_pairs_from_cache(self, encoder_refiner_cache, distill_inputs, pair_indices):
        self.pairs_seen.extend(pair_indices.tolist())
        i, p = pair_indices.unbind(1)
        feature = distill_inputs["initial_feature_36"][i, p][:, None]
        score = distill_inputs["initial_score_36"][i, p][:, None]
        guide = distill_inputs["clip_guidance_36"][i][:, None]
        independent = self.refine(feature, score, guide)
        return {OUTPUT_KEYS.distill_student_logits: self.logits(independent),
                OUTPUT_KEYS.sam3_teacher_logits: self.logits(feature).detach()}


class IndependentDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masks_aliases_denominator_and_gradient_separation(self):
        crit = criterion()
        labels = torch.full((2, 7, 7), 255, dtype=torch.long)
        labels[0, 2, 2] = 0
        labels[0, 4, 4] = 1
        labels[1, 3, 3] = 1
        ctx = context_for(crit, labels, [0, 0, 1, 2])
        pairs = ctx.presence_target.nonzero()
        self.assertEqual(pairs.tolist(), [[0, 0], [0, 1], [0, 2], [1, 2]])
        student = torch.randn(4, 1, 7, 7, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        main = torch.randn(2, 4, 7, 7, requires_grad=True)
        outputs = {OUTPUT_KEYS.distill_student_logits: student, OUTPUT_KEYS.sam3_teacher_logits: teacher}
        losses = crit.forward_distill_pairs(outputs, ctx, pairs)
        # Independent brute-force masks using Chebyshev distance from class GT.
        masks = []
        for i, p in pairs.tolist():
            cls = [0, 0, 1, 2][p]
            mask = labels[i] != 255
            for y, x in (labels[i] == cls).nonzero().tolist():
                mask[max(0, y-1):y+2, max(0, x-1):x+2] = True
            masks.append(mask)
        masks = torch.stack(masks)[:, None]
        self.assertEqual(ctx.distill_pixel_denom, masks.sum().item())
        expected = (F.binary_cross_entropy_with_logits(student, teacher.detach().sigmoid(), reduction="none") * masks).sum() / masks.sum()
        torch.testing.assert_close(losses["loss_sam3_mask_distill_bce"], expected)
        split = sum(crit.forward_distill_pairs({k: v[j:j+1] for k, v in outputs.items()}, ctx, pairs[j:j+1])["total_loss"] for j in range(4))
        torch.testing.assert_close(losses["total_loss"], split)
        losses["total_loss"].backward()
        self.assertIsNone(main.grad)
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[~masks].count_nonzero().item(), 0)
        main_losses = crit.forward_chunk({OUTPUT_KEYS.final_logits: main}, ctx, 0, 4)
        self.assertEqual(main_losses["loss_sam3_mask_distill_bce"].item(), 0)
        main_losses["total_loss"].backward()
        self.assertGreater(main.grad.abs().sum().item(), 0)

    def test_singleton_refiner_packing_and_checkpoint(self):
        for mode in ("window", "local_3x3"):
            with self.subTest(mode=mode):
                torch.manual_seed(2)
                model = ClassConditionedEncoderRefiner.__new__(ClassConditionedEncoderRefiner)
                nn.Module.__init__(model)
                model.use_checkpoint = True
                model.layers = nn.ModuleList([
                    EncoderRefinerLayer(hidden_dim=256, score_embed_dim=256, num_heads=8,
                        window_size=3, shift_size=1, dropout=0,
                        score_attention_type="intra" if j % 2 == 0 else "inter",
                        score_intra_attn_type=mode, feature_intra_attn_type=mode)
                    for j in range(4)
                ])
                model.feature_output_norm = nn.LayerNorm(256)
                values = [torch.randn(2, 1, 256, 6, 6, requires_grad=True) for _ in range(4)]
                packed = model.refine_streams(*values)[0]
                loss = packed.square().mean() + packed[..., 0, 0].mean()
                loss.backward()
                grads = [x.grad.clone() for x in values]
                params = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                unpacked = [x.detach().requires_grad_() for x in values]
                model.use_checkpoint = False
                single = torch.cat([model.refine_streams(*(x[j:j+1] for x in unpacked))[0] for j in range(2)])
                torch.testing.assert_close(packed, single, atol=2e-6, rtol=2e-5)
                (single.square().mean() + single[..., 0, 0].mean()).backward()
                for want, x in zip(grads, unpacked):
                    torch.testing.assert_close(want, x.grad, atol=2e-6, rtol=2e-4)
                for name, p in model.named_parameters():
                    if name in params:
                        torch.testing.assert_close(params[name], p.grad, atol=3e-6, rtol=2e-4)
                for layer in model.layers:
                    self.assertEqual(layer.class_attn.q_proj.weight.grad.count_nonzero().item(), 0)
                    self.assertEqual(layer.class_attn.k_proj.weight.grad.count_nonzero().item(), 0)
                # Changing another packed pair must leave the first pair unchanged.
                changed = [x.detach().clone() for x in values]
                for x in changed:
                    x[1] += 5
                with torch.no_grad():
                    torch.testing.assert_close(model.refine_streams(*changed)[0][0], packed[0])

    def test_fusion_image_selection_and_checkpoint_gradients(self):
        torch.manual_seed(3)
        model = DecoderInputFusion(use_checkpoint=True).train()
        reference = copy.deepcopy(model)
        reference.use_checkpoint = False
        ids = torch.tensor([1, 0, 1])
        ref = torch.randn(3, 256, 36, 36, requires_grad=True)
        enc = torch.randn(3, 256, 72, 72)
        fpn = torch.randn(2, 256, 72, 72)
        actual = model(ref, enc, fpn, image_ids=ids)
        ref2 = ref.detach().requires_grad_()
        expected = torch.cat([reference(ref2[j:j+1], enc[j:j+1], fpn[i:i+1]) for j, i in enumerate(ids.tolist())])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(ref.grad, ref2.grad, atol=1e-7, rtol=1e-4)
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)

    def test_actual_pair_decode_routes_initial_inputs_and_teacher(self):
        class RecordingRefiner(nn.Module):
            def refine_streams(self, feature, score, clip, fpn):
                self.inputs = (feature, score, clip, fpn)
                return feature + score + clip + fpn, score
            def fuse_decoder_input_chunk(self, refiner_feature_36, original_encoder_feature_72, sam_fpn_72, image_ids):
                self.image_ids = image_ids
                return F.interpolate(refiner_feature_36, (72, 72)) + original_encoder_feature_72
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.weight = nn.Parameter(torch.tensor(1.1), requires_grad=False)
            def forward(self, backbone_feats, image_ids, encoder_hidden_states):
                self.calls.append((torch.is_grad_enabled(), image_ids.clone(), encoder_hidden_states.detach().clone()))
                x = encoder_hidden_states.mean(2).T.reshape(-1, 1, 72, 72)
                return {"semantic_seg": F.interpolate(x * self.weight, (288, 288))}
        model = load_decode_methods()()
        model.encoder_refiner = RecordingRefiner()
        model.segmentation_head = Head()
        cache = {"cross_attended_encoder_features_72": torch.randn(2, 3, 256, 72, 72),
                 "backbone_fpn": [torch.empty(0), torch.empty(0), torch.randn(2, 256, 72, 72)]}
        inputs = {"initial_feature_36": torch.randn(2, 3, 256, 36, 36),
                  "initial_score_36": torch.randn(2, 3, 256, 36, 36, requires_grad=True),
                  "clip_guidance_36": torch.randn(2, 256, 36, 36, requires_grad=True),
                  "fpn_guidance_36": torch.randn(2, 256, 36, 36)}
        pairs = torch.tensor([[1, 2], [0, 1], [1, 0]])
        output = model.decode_independent_distill_pairs_from_cache(cache, inputs, pairs)
        self.assertEqual([x[0] for x in model.segmentation_head.calls], [False, True])
        self.assertEqual(model.encoder_refiner.inputs[0].shape, (3, 1, 256, 36, 36))
        torch.testing.assert_close(model.encoder_refiner.inputs[0][:, 0], inputs["initial_feature_36"][pairs[:, 0], pairs[:, 1]])
        torch.testing.assert_close(model.encoder_refiner.inputs[1][:, 0], inputs["initial_score_36"][pairs[:, 0], pairs[:, 1]])
        self.assertFalse(output[OUTPUT_KEYS.sam3_teacher_logits].requires_grad)
        output[OUTPUT_KEYS.distill_student_logits].square().mean().backward()
        self.assertGreater(inputs["initial_score_36"].grad.abs().sum().item(), 0)
        self.assertGreater(inputs["clip_guidance_36"].grad.abs().sum().item(), 0)
        self.assertIsNone(model.segmentation_head.weight.grad)

    def test_streaming_matches_joint_backward_and_skips_disabled_or_empty(self):
        for chunk in (1, 3):
            for weight, empty in ((0.5, False), (0.0, False), (0.5, True)):
                with self.subTest(chunk=chunk, weight=weight, empty=empty):
                    model = ToyModel(chunk)
                    reference = copy.deepcopy(model)
                    crit = criterion(weight)
                    labels = torch.full((2, 288, 288), 255, dtype=torch.long)
                    if not empty:
                        labels[0, 10:20, 10:20] = 0
                        labels[0, 40:50, 40:50] = 1
                        labels[1, 15:25, 15:25] = 1
                    batch = SimpleNamespace(find_targets=[SimpleNamespace(semantic_label_map=labels)],
                        find_metadatas=[SimpleNamespace(num_prompts=4, num_classes=3, prompt_to_class_id=[0, 0, 1, 2])])
                    trainer = Trainer.__new__(Trainer)
                    trainer.model, trainer.criterion = model, crit
                    trainer.device = torch.device("cpu")
                    trainer.cfg = SimpleNamespace(use_amp=False)
                    # Simulate a fixed AMP scale and catch accidental double scaling.
                    trainer.scaler = SimpleNamespace(scale=lambda loss: loss * 8)
                    actual = trainer._compute_train_loss_streaming(batch)
                    cache = reference.build_encoder_refiner_cache(batch)
                    out = reference.run_encoder_refiner_lowres_from_cache(cache, batch)
                    ctx = context_for(crit, labels, [0, 0, 1, 2])
                    main = reference.logits(out["refiner_features_36"])
                    expected = crit.forward_chunk({OUTPUT_KEYS.final_logits: main}, ctx, 0, 4)["total_loss"]
                    pairs = ctx.presence_target.nonzero()
                    if weight and len(pairs):
                        outputs = reference.decode_independent_distill_pairs_from_cache(cache, out["distill_inputs"], pairs)
                        expected = expected + crit.forward_distill_pairs(outputs, ctx, pairs)["total_loss"]
                    expected.backward()
                    torch.testing.assert_close(actual["total_loss"], expected.detach())
                    for (name, p), q in zip(model.named_parameters(), reference.parameters()):
                        self.assertIsNotNone(p.grad, name)
                        torch.testing.assert_close(p.grad / 8, q.grad, atol=2e-6, rtol=2e-5, msg=name)
                    self.assertEqual(model.lowres_calls, 1)
                    self.assertEqual(model.pairs_seen, pairs.tolist() if weight else [])


if __name__ == "__main__":
    unittest.main()
