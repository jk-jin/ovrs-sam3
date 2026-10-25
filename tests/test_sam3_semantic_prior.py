"""CPU regressions; no SAM3/RemoteCLIP checkpoint or dataset required."""
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
PREFIX = ROOT.name

def module(name):
    return importlib.import_module(f"{PREFIX}.{name}")

Refiner = module("models.encoder_refiner").ClassConditionedEncoderRefiner
Sam3Image = module("models.sam3_image").Sam3Image
Adapter = module("models.adapters.semantic_adapter").SemanticSegAdapter
SegmentationHead = module("models.maskformer_segmentation").UniversalSegmentationHead
KEYS = module("models.task_modes").OUTPUT_KEYS


def make_refiner(checkpoint=True):
    return Refiner(
        clip_text_encoder=nn.Identity(), clip_dim=8, clip_native_dim=8,
        fusion_layers=0, dropout=0, prompt_templates=["{}"] * 64,
        use_checkpoint=checkpoint,
    )


class PriorTests(unittest.TestCase):
    def test_prior_and_fusion_learn_through_frozen_student_head(self):
        torch.manual_seed(5)
        refiner = make_refiner().train()
        frozen_head = nn.Conv2d(256, 1, 1).requires_grad_(False)
        encoder = torch.randn(1, 2, 256, 72, 72)
        sam3_logits = torch.randn(1, 2, 288, 288, requires_grad=True)
        # The full forward keeps the initializer outside SAM3's no_grad.
        with patch.object(refiner.clip_score_embed, "forward", return_value=(
            torch.zeros(1, 2, 256, 36, 36), torch.zeros(1), torch.zeros(1)
        )):
            output = refiner(
                encoder_features_72=encoder,
                sam3_semantic_logits=sam3_logits,
                clip_image_feat_map=torch.zeros(1, 8, 36, 36),
                class_names=["a", "b"],
                clip_mid_features=[torch.zeros(1, 8, 36, 36)] * 2,
                clip_mid_layer_indices=(7, 15),
                sam_fpn_72=torch.zeros(1, 256, 72, 72),
            )["refiner_features_36"]
        self.assertEqual(output.shape, (1, 2, 256, 36, 36))
        F.binary_cross_entropy_with_logits(
            frozen_head(output.flatten(0, 1)), torch.ones(2, 1, 36, 36)
        ).backward()
        for group in (refiner.semantic_prior_encoder, refiner.feature_input_fusion):
            for param in group.parameters():
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(sum(p.grad.abs().sum().item() for p in group.parameters()), 0)
        self.assertIsNone(sam3_logits.grad)
        self.assertIsNone(frozen_head.weight.grad)

    def test_soft_prior_changes_initial_feature(self):
        refiner = make_refiner(False).eval()
        encoder = torch.zeros(1, 1, 256, 72, 72)
        logits = torch.full((1, 1, 288, 288), -4.0)
        changed = logits.clone()
        changed[..., :144, :144] = 4.0
        with torch.no_grad():
            first = refiner.initialize_feature_stream(encoder, logits)
            second = refiner.initialize_feature_stream(encoder, changed)
        self.assertGreater((first - second).abs().max().item(), 1e-3)


class CountingPixels(nn.Module):
    out_dim = 8
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, features):
        self.calls += 1
        return features[-1]


class HeadTests(unittest.TestCase):
    def test_semantic_and_instance_share_one_pixel_pass(self):
        pixels = CountingPixels()
        head = SegmentationHead(8, 1, pixels, no_dec=False).requires_grad_(False).eval()
        hidden = torch.randn(16, 2, 8)
        original = head(
            backbone_feats=[torch.zeros(1, 8, 4, 4)], image_ids=torch.tensor([0, 0]),
            encoder_hidden_states=hidden, obj_queries=torch.randn(2, 3, 8),
        )
        self.assertEqual(pixels.calls, 1)
        self.assertEqual(original["semantic_seg"].shape, (2, 1, 4, 4))
        self.assertEqual(original["pred_masks"].shape, (2, 3, 4, 4))
        student_input = hidden.clone().requires_grad_(True)
        student = head(
            backbone_feats=[torch.zeros(1, 8, 4, 4)], image_ids=torch.tensor([0, 0]),
            encoder_hidden_states=student_input,
        )
        self.assertEqual(pixels.calls, 2)
        self.assertNotIn("pred_masks", student)
        student["semantic_seg"].square().mean().backward()
        self.assertIsNotNone(student_input.grad)
        self.assertTrue(all(p.grad is None for p in head.parameters()))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.metadata = SimpleNamespace(
            num_prompts=3, num_classes=2, prompt_to_class_id=[0, 0, 1],
        )
        self.batch = SimpleNamespace(find_metadatas=[self.metadata])
        self.student = torch.tensor([[[[.2, .8]], [[.3, .1]], [[.1, .4]]]])
        self.original = torch.tensor([[[[.99, .99]], [[.8, .4]], [[.9, .9]]]])
        self.presence = torch.tensor([[.01, .9, .5]])
        self.raw = {
            KEYS.final_logits: torch.logit(self.student),
            KEYS.sam3_semantic_logits: torch.logit(self.original),
            KEYS.sam3_presence_logits: torch.logit(self.presence),
        }

    def test_student_only_alias_merge_then_relative_filter(self):
        outputs = Adapter(.5)(self.raw, self.batch, output_mode="infer")
        torch.testing.assert_close(outputs[KEYS.raw_prompt_score_map], self.student)
        torch.testing.assert_close(outputs[KEYS.raw_final_score_map], torch.tensor([
            [[[.3, .8]], [[.1, .4]]]
        ]))
        torch.testing.assert_close(outputs[KEYS.final_score_map], torch.tensor([
            [[[0., .8]], [[0., .4]]]
        ]))
        self.assertEqual(outputs[KEYS.final_pred].tolist(), [[[0, 0]]])
        torch.testing.assert_close(outputs[KEYS.prompt_logits], self.raw[KEYS.final_logits])
        # Original SAM3 masks/presence are optional diagnostics, not inference inputs.
        student_only = Adapter(.5)(
            {KEYS.final_logits: self.raw[KEYS.final_logits]}, self.batch, output_mode="infer"
        )
        for key in (KEYS.raw_prompt_score_map, KEYS.final_score_map, KEYS.final_pred):
            torch.testing.assert_close(student_only[key], outputs[key])

    def test_training_output_is_unfused_and_needs_no_original_outputs(self):
        outputs = Adapter(.5)(
            {KEYS.final_logits: self.raw[KEYS.final_logits]}, self.batch,
            output_mode="final",
        )
        self.assertIs(outputs[KEYS.final_logits], self.raw[KEYS.final_logits])

    def test_tta_averages_student_probabilities_only(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = Adapter(.5)
            def forward(self, batch):
                flipped = batch.img_batch[0, 0, 0, 0].item() > .5
                student, original = (.9, .1) if flipped else (.1, .9)
                return self.adapter({
                    KEYS.final_logits: torch.logit(torch.full((1, 1, 1, 2), student)),
                    KEYS.sam3_semantic_logits: torch.logit(torch.full((1, 1, 1, 2), original)),
                    KEYS.sam3_presence_logits: torch.logit(torch.tensor([[.99]])),
                }, batch, output_mode="infer")
        batch = SimpleNamespace(
            img_batch=torch.tensor([[[[0., 1.]]]]), raw_images=None,
            find_metadatas=[SimpleNamespace(num_prompts=1, num_classes=1, prompt_to_class_id=[0])],
        )
        output = module("engine.evaluator").inference_with_tta(
            Model(), batch, dict(enabled=True, scales=[1.], flip_modes=["none", "h"], size_divisor=1)
        )
        # Mean(.1, .9), unaffected by original SAM3 probabilities and presence.
        torch.testing.assert_close(
            output[KEYS.raw_prompt_score_map], torch.full((1, 1, 1, 2), .5)
        )


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_embed = nn.Embedding(3, 256)
        self.bbox_embed = nn.Linear(256, 4)
        self.memories = []
    def forward(self, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["apply_dac"] is False
        self.memories.append(kwargs["memory"])
        tgt = kwargs["tgt"]
        n = tgt.shape[1]
        return tgt[None], torch.full((1, 3, n, 4), .5), torch.zeros(1, 1, n), None


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
    def apply_prompt_cross_attention(self, encoder_hidden_states, **kwargs):
        return encoder_hidden_states + 10
    def forward(self, **kwargs):
        hidden = kwargs["encoder_hidden_states"]
        self.calls.append((torch.is_grad_enabled(), hidden))
        n = hidden.shape[1]
        value = hidden.mean(dim=(0, 2)).reshape(n, 1, 1, 1)
        outputs = {"semantic_seg": value.expand(n, 1, 288, 288)}
        if "obj_queries" in kwargs:
            outputs["pred_masks"] = value.expand(n, 3, 288, 288)
        return outputs


class FakeFusion(nn.Module):
    def fuse_decoder_input_chunk(self, refiner_feature_36, **kwargs):
        return F.interpolate(refiner_feature_36, size=(72, 72))


class CacheTests(unittest.TestCase):
    def test_original_before_refiner_and_teacher_reuse_with_uneven_chunks(self):
        core = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(core)
        core.hidden_dim = 256
        core.anchor = nn.Parameter(torch.zeros(()))
        core.prompt_chunk_size = 2
        core.transformer = nn.Module()
        core.transformer.decoder = FakeDecoder()
        core.segmentation_head = FakeHead()
        core.encoder_refiner = FakeFusion()
        core.dot_prod_scoring = lambda hs, *args: torch.zeros(*hs.shape[:-1], 1)
        core.ensure_text_cache = lambda **kwargs: None
        core._slice_text_cache = lambda start, end: {
            "language_features": torch.zeros(2, end - start, 256),
            "language_mask": torch.zeros(end - start, 2, dtype=torch.bool),
        }
        core.backbone = nn.Module()
        core.backbone.forward_image = lambda _: {
            "backbone_fpn": [torch.zeros(2, 256, 1, 1)] * 2 + [torch.zeros(2, 256, 72, 72)]
        }
        core._build_clip_image_cache = lambda **kwargs: {
            "clip_image_feat_map_native": torch.zeros(2, 8, 36, 36),
            KEYS.clip_mid_features: [], "clip_mid_layer_indices": (),
        }
        core._encode_prompt = lambda bb, find, geo: (
            torch.zeros(2, find.img_ids.numel(), 256),
            torch.zeros(find.img_ids.numel(), 2, dtype=torch.bool), bb,
        )
        memories = []
        def encode(**kwargs):
            n = kwargs["find_input"].img_ids.numel()
            memory = torch.full((72 * 72, n, 256), float(len(memories)))
            memories.append(memory)
            return dict(memory=memory, padding_mask=None, pos_embed=torch.zeros_like(memory),
                        level_start_index=torch.tensor([0]), spatial_shapes=torch.tensor([[72,72]]),
                        valid_ratios=torch.ones(n, 1, 2))
        core._run_full_encoder = encode
        batch = SimpleNamespace(
            find_inputs=[None], find_text_batch=["a", "b", "c"],
            img_batch=torch.zeros(2, 3, 1, 1),
        )
        cache = core.build_encoder_refiner_cache(batch)
        self.assertEqual(cache["chunk_class_counts"], [2, 1])
        self.assertEqual(len(core.segmentation_head.calls), 2)
        self.assertEqual(cache[KEYS.sam3_semantic_logits].shape, (2, 3, 288, 288))
        self.assertEqual(cache[KEYS.sam3_presence_logits].shape, (2, 3))
        for raw, received, (grad, post_ca) in zip(
            memories, core.transformer.decoder.memories, core.segmentation_head.calls
        ):
            self.assertIs(raw, received)
            self.assertFalse(grad)
            torch.testing.assert_close(post_ca, raw + 10)
        self.assertFalse(cache[KEYS.sam3_semantic_logits].requires_grad)
        for chunk in cache[KEYS.sam3_instance_outputs]:
            self.assertEqual(chunk["pred_masks"].device.type, "cpu")
            self.assertEqual(chunk["pred_masks"].shape[2], 3)
        for start, end in [(0, 2), (2, 3)]:
            features = torch.randn(2, end - start, 256, 36, 36, requires_grad=True)
            outputs = core.decode_encoder_refiner_chunk_from_cache(
                cache, features, start, end, return_teacher_logits=True,
            )
            torch.testing.assert_close(
                outputs[KEYS.sam3_teacher_logits], cache[KEYS.sam3_semantic_logits][:, start:end]
            )
            self.assertFalse(outputs[KEYS.sam3_teacher_logits].requires_grad)
            outputs[KEYS.final_logits].square().mean().backward()
            self.assertIsNotNone(features.grad)
        self.assertEqual(len(core.segmentation_head.calls), 4)  # two original + two student, never six


class DecoderTests(unittest.TestCase):
    def test_real_six_layer_200_query_decoder_and_scorer(self):
        builder = module("model_builder").SAM3ModelBuilder
        transformer = builder._create_transformer().eval().requires_grad_(False)
        decoder = transformer.decoder
        self.assertEqual(decoder.num_queries, 200)
        self.assertEqual(decoder.num_layers, 6)
        self.assertEqual(decoder.presence_token.num_embeddings, 1)
        with torch.no_grad():
            hs, boxes, presence, _ = decoder(
                tgt=decoder.query_embed.weight[:, None], memory=torch.randn(4, 1, 256),
                pos=torch.zeros(4, 1, 256), spatial_shapes=torch.tensor([[2, 2]]),
                level_start_index=torch.tensor([0]), valid_ratios=torch.ones(1, 1, 2),
                memory_text=torch.randn(3, 1, 256),
                text_attention_mask=torch.zeros(1, 3, dtype=torch.bool), apply_dac=False,
            )
            scores = builder._create_dot_product_scoring().eval()(
                hs.transpose(1, 2), torch.randn(3, 1, 256), torch.zeros(1, 3, dtype=torch.bool)
            )
        self.assertEqual(hs.shape, (6, 200, 1, 256))
        self.assertEqual(boxes.shape, (6, 200, 1, 4))
        self.assertEqual(presence.shape, (6, 1, 1))
        self.assertEqual(scores.shape, (6, 1, 200, 1))
        self.assertTrue(torch.isfinite(scores).all())

    def test_checkpoint_cannot_leave_restored_frozen_modules_random(self):
        core = nn.Module()
        core.transformer = nn.Module()
        core.transformer.decoder = nn.Linear(2, 2)
        core.dot_prod_scoring = nn.Linear(2, 1)
        builder = module("model_builder").SAM3ModelBuilder
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sam3.pt")
            weights = core.state_dict()
            torch.save(weights, path)
            builder._load_checkpoint(core, path)
            del weights["transformer.decoder.weight"]
            torch.save(weights, path)
            with self.assertRaisesRegex(RuntimeError, "required frozen instance weights"):
                builder._load_checkpoint(core, path)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
