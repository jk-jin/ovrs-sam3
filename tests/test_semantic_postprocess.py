from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from tests._bootstrap import load_module


adapter_mod = load_module("models.adapters.semantic_adapter")
evaluator_mod = load_module("engine.evaluator")
OUTPUT_KEYS = load_module("models.task_modes").OUTPUT_KEYS


def make_metadata(background_enabled: bool):
    names = ["building", "water"] + (["background"] if background_enabled else [])
    return SimpleNamespace(
        num_prompts=len(names),
        num_classes=len(names),
        prompt_to_class_id=list(range(len(names))),
        class_names=names,
        background_enabled=background_enabled,
        background_class_id=2 if background_enabled else 0,
        eval_num_classes=len(names),
        eval_class_names=names,
    )


class TestPredictionProbabilityFilter(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor([[[[0.1, 0.5]], [[0.4, 0.2]]]])
        self.adapter = adapter_mod.SemanticSegAdapter(
            prediction_prob_threshold=0.5,
        )

    def scores_for(self, background_enabled: bool):
        if background_enabled:
            return torch.cat([self.scores, torch.full_like(self.scores[:, :1], 0.05)], dim=1)
        return self.scores

    def infer(self, background_enabled: bool):
        batch = SimpleNamespace(
            find_metadatas=[make_metadata(background_enabled)],
        )
        return self.adapter(
            raw_outputs={OUTPUT_KEYS.final_logits: self.scores_for(background_enabled).logit()},
            batch=batch,
            output_mode="infer",
        )

    def test_no_background_preserves_low_scores_and_winning_class(self):
        outputs = self.infer(False)
        torch.testing.assert_close(
            outputs[OUTPUT_KEYS.final_score_map], self.scores,
        )
        self.assertEqual(
            outputs[OUTPUT_KEYS.final_pred].tolist(), [[[1, 0]]],
        )

    def test_background_filters_without_shifting_class_ids(self):
        outputs = self.infer(True)
        torch.testing.assert_close(
            outputs[OUTPUT_KEYS.final_score_map],
            torch.tensor([[[[0.0, 0.5]], [[0.0, 0.0]], [[0.0, 0.0]]]]),
        )
        torch.testing.assert_close(
            outputs[OUTPUT_KEYS.raw_final_score_map], self.scores_for(True),
        )
        pred, num_classes, _ = evaluator_mod.build_eval_semantic_pred(
            outputs, make_metadata(True), background_prob_threshold=0.0,
        )
        self.assertEqual(num_classes, 3)
        self.assertEqual(pred.tolist(), [[[2, 0]]])

    def test_different_thresholds_change_background_fallback(self):
        predictions = []
        for threshold in (0.3, 0.5):
            adapter = adapter_mod.SemanticSegAdapter(prediction_prob_threshold=threshold)
            outputs = adapter.build_infer_score_outputs(self.scores_for(True), make_metadata(True))
            pred, _, _ = evaluator_mod.build_eval_semantic_pred(
                outputs, make_metadata(True), background_prob_threshold=0.0,
            )
            predictions.append(pred.tolist())
        self.assertEqual(predictions, [[[[1, 0]]], [[[2, 0]]]])

    def test_none_threshold_disables_prediction_filter(self):
        adapter = adapter_mod.SemanticSegAdapter()
        outputs = adapter.build_infer_score_outputs(
            self.scores_for(True), make_metadata(True),
        )
        torch.testing.assert_close(
            outputs[OUTPUT_KEYS.final_score_map], self.scores_for(True),
        )

    def test_tta_uses_current_dataset_background_setting(self):
        adapter = self.adapter

        class ScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = adapter

            def forward(self, batch):
                return self.adapter.build_infer_score_outputs(
                    batch.img_batch, batch.find_metadatas[0],
                )

        model = ScoreModel()
        for background_enabled in (False, True):
            with self.subTest(background_enabled=background_enabled):
                batch = SimpleNamespace(
                    img_batch=self.scores_for(background_enabled),
                    raw_images=None,
                    find_metadatas=[make_metadata(background_enabled)],
                )
                outputs = evaluator_mod.inference_with_tta(
                    model,
                    batch,
                    dict(
                        enabled=True,
                        scales=[1.0],
                        flip_modes=["none", "h"],
                        size_divisor=1,
                    ),
                )
                expected = self.infer(background_enabled)
                torch.testing.assert_close(
                    outputs[OUTPUT_KEYS.final_score_map],
                    expected[OUTPUT_KEYS.final_score_map],
                )


if __name__ == "__main__":
    unittest.main()
