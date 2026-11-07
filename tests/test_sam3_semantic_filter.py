"""Inference filtering and previous-commit state_dict compatibility without SAM weights."""

import importlib
import io
import runpy
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_COMMIT = "7b55d320cfabecbf9af333d14c7195fc4de2c8f4"
PACKAGE = "_sam3_filter_test"

# Isolate the tested modules from optional torchvision/backbone dependencies.
# Geometry and VL types are annotations only in the exercised decode path.
for name, path in ((PACKAGE, ROOT), (f"{PACKAGE}.models", ROOT / "models"),
                   (f"{PACKAGE}.engine", ROOT / "engine")):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
for name, symbol in (("geometry_encoders", "Prompt"), ("vl_combiner", "SAM3VLBackbone")):
    module = types.ModuleType(f"{PACKAGE}.models.{name}")
    setattr(module, symbol, nn.Module)
    sys.modules[module.__name__] = module

Sam3Image = importlib.import_module(f"{PACKAGE}.models.sam3_image").Sam3Image
Segmentor = importlib.import_module(f"{PACKAGE}.models.segmentor").SAM3Segmentor
Adapter = importlib.import_module(f"{PACKAGE}.models.adapters.semantic_adapter").SemanticSegAdapter
KEYS = importlib.import_module(f"{PACKAGE}.models.task_modes").OUTPUT_KEYS
inference_with_tta = importlib.import_module(f"{PACKAGE}.engine.evaluator").inference_with_tta


def previous_class(path, class_name):
    source = subprocess.check_output(["git", "show", f"{PREVIOUS_COMMIT}:{path}"], cwd=ROOT, text=True)
    name = f"{PACKAGE}." + path[:-3].replace("/", ".") + "_previous"
    module = types.ModuleType(name)
    module.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = module
    exec(compile(source, f"{PREVIOUS_COMMIT}:{path}", "exec"), module.__dict__)
    return getattr(module, class_name)


def metadata(mapping=(0, 1)):
    return SimpleNamespace(num_prompts=len(mapping), num_classes=max(mapping) + 1,
                           prompt_to_class_id=list(mapping))


class TextEncoder(nn.Module):
    output_dim = 8


class ImageEncoder(nn.Module):
    output_dim = 8
    native_dim = 8


class SemanticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, backbone_feats, image_ids, encoder_hidden_states):
        self.calls.append(torch.is_grad_enabled())
        scores = encoder_hidden_states.mean(dim=(0, 2)) * self.weight
        return {"semantic_seg": scores[:, None, None, None].expand(-1, 1, 288, 288)}


def make_model(threshold=None, core_type=Sam3Image, adapter_type=Adapter, segmentor_type=Segmentor):
    transformer = nn.Linear(4, 4)
    transformer.d_model = 256
    core = core_type(
        backbone=nn.Linear(4, 4), transformer=transformer,
        input_geometry_encoder=nn.Linear(4, 4), segmentation_head=SemanticHead(),
        clip_image_encoder=ImageEncoder(), clip_text_encoder=TextEncoder(),
        openclip_prompt_templates=["a satellite image of {}."] * 64,
        encoder_refiner_dropout=0,
    )
    kwargs = {} if adapter_type is not Adapter else {"sam3_semantic_filter_threshold": threshold}
    model = segmentor_type(core=core, adapter=adapter_type(**kwargs), task_mode="semantic")
    return model.eval()


def cache_and_features():
    original = torch.stack([torch.full((256, 72, 72), x) for x in (-2., 0., 2.)])[None]
    cache = {"cross_attended_encoder_features_72": original,
             "backbone_fpn": [torch.zeros(1), torch.zeros(1), torch.zeros(1, 256, 72, 72)],
             "chunk_class_counts": [1, 2]}
    return cache, torch.randn(1, 3, 256, 36, 36)


class Sam3SemanticFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(11)

    def test_threshold_validation_and_endpoints(self):
        for value in (-.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Adapter(sam3_semantic_filter_threshold=value)
        scores = torch.full((1, 2, 1, 2), .9)
        sam3 = torch.tensor([[[[0., 1.]], [[.5, 1.]]]])
        for threshold in (0., 1.):
            out = Adapter(sam3_semantic_filter_threshold=threshold).build_infer_score_outputs(
                scores, metadata(), sam3)
            torch.testing.assert_close(out[KEYS.final_score_map], scores.masked_fill(sam3 < threshold, 0))

    def test_exact_zero_equality_and_retained_student_probabilities(self):
        student = torch.tensor([[[[.9, .8, .7]], [[.6, .5, .4]]]])
        sam3 = torch.tensor([[[[.1, .2, .9]], [[.3, .05, .2]]]])
        adapter = Adapter(sam3_semantic_filter_threshold=.2)
        out = adapter.build_infer_score_outputs(student, metadata(), sam3)
        expected = torch.tensor([[[[0., .8, .7]], [[.6, 0., .4]]]])
        torch.testing.assert_close(out[KEYS.final_score_map], expected, rtol=0, atol=0)
        torch.testing.assert_close(out[KEYS.raw_prompt_score_map], student)
        torch.testing.assert_close(out[KEYS.raw_final_score_map], student)
        torch.testing.assert_close(out[KEYS.final_pred], expected.argmax(1))

    def test_aliases_merge_in_class_space_before_filtering(self):
        student = torch.tensor([[[[.9, .8]], [[.1, .2]], [[.7, .6]]]])
        sam3 = torch.tensor([[[[.01, .01]], [[.3, .1]], [[.4, .1]]]])
        out = Adapter(sam3_semantic_filter_threshold=.2).build_infer_score_outputs(
            student, metadata((0, 0, 1)), sam3)
        torch.testing.assert_close(out[KEYS.final_score_map], torch.tensor([[[[.9, 0.]], [[.7, 0.]]]]))

    def test_filter_runs_after_existing_relative_filter(self):
        student = torch.tensor([[[[.4, .5, .9]], [[.1, .4, .7]]]])
        sam3 = torch.tensor([[[[.1, .1, .8]], [[.8, .8, .1]]]])
        expected = Adapter(class_relative_prob_thd=.5).build_infer_score_outputs(student, metadata())
        actual = Adapter(class_relative_prob_thd=.5, sam3_semantic_filter_threshold=.2)
        actual = actual.build_infer_score_outputs(student, metadata(), sam3)
        torch.testing.assert_close(actual[KEYS.final_score_map],
                                   expected[KEYS.final_score_map].masked_fill(sam3 < .2, 0))

    def test_missing_or_misaligned_teacher_is_rejected_only_when_enabled(self):
        scores = torch.rand(1, 2, 2, 2)
        enabled = Adapter(sam3_semantic_filter_threshold=.2)
        with self.assertRaises(ValueError):
            enabled.build_infer_score_outputs(scores, metadata())
        with self.assertRaises(ValueError):
            enabled.build_infer_score_outputs(scores, metadata(), torch.rand(1, 1, 2, 2))
        Adapter().build_infer_score_outputs(scores, metadata())

    def test_training_logits_and_gradients_are_unchanged(self):
        logits = torch.randn(1, 2, 2, 2, requires_grad=True)
        adapter = Adapter(sam3_semantic_filter_threshold=.2)
        batch = SimpleNamespace(find_metadatas=[metadata()])
        result = adapter({KEYS.final_logits: logits}, batch, output_mode="final")
        self.assertIs(result[KEYS.final_logits], logits)
        result[KEYS.final_logits].sum().backward()
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))

    def test_chunked_original_sam3_decode_is_optional_and_frozen(self):
        model = make_model(.5)
        cache, features = cache_and_features()
        batch = SimpleNamespace(find_metadatas=[metadata((0, 1, 2))])
        with patch.object(model.core, "run_encoder_refiner_lowres_from_cache",
                          return_value={"refiner_features_36": features}), torch.no_grad():
            out = model.run_encoder_refiner_from_cache(cache, batch)
            expected = torch.tensor([-2., 0., 2.]).sigmoid()[None, :, None, None].expand(1, 3, 288, 288)
            torch.testing.assert_close(out[KEYS.sam3_prompt_score_map], expected)
            self.assertFalse(out[KEYS.sam3_prompt_score_map].requires_grad)
            self.assertEqual(len(model.core.segmentation_head.calls), 4)  # Two chunks, teacher+student.
            model.adapter.sam3_semantic_filter_threshold = None
            model.core.segmentation_head.calls.clear()
            off = model.run_encoder_refiner_from_cache(cache, batch)
            self.assertNotIn(KEYS.sam3_prompt_score_map, off)
            self.assertEqual(len(model.core.segmentation_head.calls), 2)
            torch.testing.assert_close(out[KEYS.final_logits], off[KEYS.final_logits])
        # Even when gradients are enabled, only the student decode records a graph.
        model.core.segmentation_head.calls.clear()
        decoded = model.decode_encoder_refiner_chunk_from_cache(cache, features[:, :1], 0, 1, True)
        self.assertEqual(model.core.segmentation_head.calls, [False, True])
        self.assertFalse(decoded[KEYS.sam3_teacher_logits].requires_grad)
        self.assertTrue(decoded[KEYS.final_logits].requires_grad)

    def test_tta_averages_both_probabilities_before_applying_threshold(self):
        adapter = Adapter(sam3_semantic_filter_threshold=.5)
        class TwoViewModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = adapter
                self.view = 0
            def forward(self, batch):
                student = torch.full((1, 2, 4, 4), .9)
                values = ([.2, .4], [.6, .8])[self.view]
                self.view += 1
                sam3 = torch.tensor(values)[None, :, None, None].expand_as(student)
                out = self.adapter.build_infer_score_outputs(student, batch.find_metadatas[0], sam3)
                out[KEYS.sam3_prompt_score_map] = sam3
                return out
        batch = SimpleNamespace(img_batch=torch.zeros(1, 3, 4, 4), raw_images=None,
                                find_metadatas=[metadata()])
        out = inference_with_tta(TwoViewModel(), batch,
                                 {"enabled": True, "scales": [1.], "flip_modes": ["none", "h"], "size_divisor": 1})
        torch.testing.assert_close(out[KEYS.sam3_prompt_score_map][:, :, 0, 0], torch.tensor([[.4, .6]]))
        torch.testing.assert_close(out[KEYS.final_score_map][:, 0], torch.zeros(1, 4, 4))
        torch.testing.assert_close(out[KEYS.final_score_map][:, 1], torch.full((1, 4, 4), .9))

    def test_previous_commit_state_dict_loads_strictly_and_disabled_outputs_match(self):
        old_core = previous_class("models/sam3_image.py", "Sam3Image")
        old_adapter = previous_class("models/adapters/semantic_adapter.py", "SemanticSegAdapter")
        old_segmentor = previous_class("models/segmentor.py", "SAM3Segmentor")
        previous = make_model(core_type=old_core, adapter_type=old_adapter, segmentor_type=old_segmentor)
        current = make_model(.2)
        previous_state, current_state = previous.state_dict(), current.state_dict()
        self.assertEqual(list(previous_state), list(current_state))
        self.assertEqual({k: v.shape for k, v in previous_state.items()},
                         {k: v.shape for k, v in current_state.items()})
        checkpoint = io.BytesIO()
        torch.save({"model": previous_state}, checkpoint)
        checkpoint.seek(0)
        result = current.load_state_dict(torch.load(checkpoint, weights_only=True)["model"], strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(dict(current.adapter.state_dict()), {})
        cache, features = cache_and_features()
        batch = SimpleNamespace(find_metadatas=[metadata((0, 1, 2))])
        def forward(model):
            with patch.object(model.core, "build_encoder_refiner_cache", return_value=cache), \
                 patch.object(model.core, "run_encoder_refiner_lowres_from_cache",
                              return_value={"refiner_features_36": features}), torch.no_grad():
                return model(batch)
        before = forward(previous)
        current.adapter.sam3_semantic_filter_threshold = None
        after = forward(current)
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)
        current.adapter.sam3_semantic_filter_threshold = .2
        filtered = forward(current)
        torch.testing.assert_close(filtered[KEYS.final_score_map][:, 0], torch.zeros(1, 288, 288))
        torch.testing.assert_close(filtered[KEYS.final_score_map][:, 1:], after[KEYS.final_score_map][:, 1:])

    def test_config_default_is_disabled(self):
        Config = runpy.run_path(str(ROOT / "engine/config.py"))["Config"]
        cfg = Config.fromfile(ROOT / "configs/test/isaid.py")
        self.assertIsNone(cfg["model"]["adapter_cfg"]["sam3_semantic_filter_threshold"])


if __name__ == "__main__":
    unittest.main()
