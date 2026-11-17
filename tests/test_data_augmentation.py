from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests._bootstrap import REPO_ROOT, load_module


transforms = load_module("data.transforms")
dataset_mod = load_module("data.dataset")
config_mod = load_module("engine.config")
criterion_mod = load_module("losses.semantic_criterion")


def sample(height=10, width=10):
    labels = torch.arange(height * width).reshape(height, width).long()
    image = labels.float()[None].expand(3, -1, -1).clone()
    return dict(image=image, raw_image=image.clone(), label_map=labels,
                eval_label_map=labels + 1000, raw_image_original="original")


class TestSpatialAugmentation(unittest.TestCase):
    def setUp(self):
        self.random_state = random.getstate()

    def tearDown(self):
        random.setstate(self.random_state)

    def test_square_crop_then_single_resize_stays_aligned(self):
        source = sample()
        snapshots = {key: value.clone() for key, value in source.items() if torch.is_tensor(value)}
        augment = transforms.RandomResizedCrop((6, 6), scale=(0.6, 1.0), prob=0.7)
        with patch.object(transforms.random, "random", return_value=0.1), \
             patch.object(transforms.random, "uniform", return_value=64.0) as area, \
             patch.object(transforms.random, "randint", side_effect=[1, 2]):
            result = augment(source)
        area.assert_called_once_with(60.0, 100.0)
        expected_image = F.interpolate(source["image"][None, :, 1:9, 2:10], (6, 6),
                                       mode="bilinear", align_corners=False)[0]
        expected_mask = F.interpolate(source["label_map"][None, None, 1:9, 2:10].float(),
                                      (6, 6), mode="nearest")[0, 0].long()
        torch.testing.assert_close(result["image"], expected_image)
        torch.testing.assert_close(result["raw_image"], result["image"])
        torch.testing.assert_close(result["label_map"], expected_mask)
        torch.testing.assert_close(result["eval_label_map"], expected_mask + 1000)
        self.assertEqual(result["img_shape"], (6, 6))
        self.assertEqual(result["raw_image_original"], "original")
        for key, value in snapshots.items():
            torch.testing.assert_close(source[key], value)

    def test_full_view_branch_resizes_without_cropping(self):
        source = sample(8, 12)
        with patch.object(transforms.random, "random", return_value=0.8), \
             patch.object(transforms.random, "uniform") as area:
            result = transforms.RandomResizedCrop((6, 6), prob=0.7)(source)
        area.assert_not_called()
        expected = F.interpolate(source["label_map"][None, None].float(), (6, 6), mode="nearest")[0, 0]
        torch.testing.assert_close(result["label_map"], expected.long())
        torch.testing.assert_close(result["raw_image"], result["image"])

    def test_all_quarter_turns_transform_both_views_and_masks(self):
        source = sample(4, 6)
        for turns in range(4):
            with self.subTest(turns=turns), patch.object(transforms.random, "randrange", return_value=turns):
                result = transforms.RandomRotate90()(source)
                for key in ("image", "raw_image", "label_map", "eval_label_map"):
                    torch.testing.assert_close(result[key], torch.rot90(source[key], turns, (-2, -1)))
                self.assertEqual(result["img_shape"], tuple(result["label_map"].shape))

    def test_rectangular_and_single_pixel_inputs_fit_without_padding(self):
        for h, w in ((2, 20), (1, 1)):
            result = transforms.RandomResizedCrop((8, 8), prob=1.0)(sample(h, w))
            self.assertEqual(result["image"].shape, (3, 8, 8))
            self.assertTrue(set(result["label_map"].flatten().tolist()) <= set(range(h * w)))

    def test_seed_replay_and_training_pipeline_preserve_alignment(self):
        cfg, _ = config_mod.build_effective_config(REPO_ROOT / "configs/experiments/main/dlrsd_loveda_full.py")
        pipeline = dataset_mod._build_transform_from_cfg(cfg["train_dataloader"]["dataset"]["transforms"])
        source = sample(32, 32)
        source["image"] = source["image"] / 1023.0
        source["raw_image"] = source["image"].clone()
        source["label_map"] = source["label_map"] % 17
        source["eval_label_map"] = source["label_map"].clone()
        random.seed(42)
        state = random.getstate()
        first = pipeline(source)
        random.setstate(state)
        second = pipeline(source)
        for key in ("image", "raw_image", "label_map", "eval_label_map"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        torch.testing.assert_close(first["image"], (first["raw_image"] - 0.5) / 0.5)
        torch.testing.assert_close(first["label_map"], first["eval_label_map"])
        self.assertEqual(first["image"].shape, (3, 1008, 1008))
        self.assertTrue(set(first["label_map"].unique().tolist()) <= set(range(17)))

    def test_presence_uses_labels_after_crop(self):
        source = sample(10, 10)
        source["label_map"] = torch.zeros(10, 10, dtype=torch.long)
        source["label_map"][-1] = 1
        with patch.object(transforms.random, "uniform", return_value=64.0), \
             patch.object(transforms.random, "randint", side_effect=[0, 0]):
            result = transforms.RandomResizedCrop((8, 8), prob=1.0)(source)
        context = criterion_mod.SemanticCriterion().prepare_streaming_context(
            {"label_map": result["label_map"][None]}, 2, torch.arange(2), 2, target_hw=(8, 8),
        )
        self.assertEqual(context.presence_target.tolist(), [[True, False]])

    def test_invalid_crop_settings_are_rejected(self):
        for kwargs in (dict(prob=-0.1), dict(prob=float("nan")), dict(prob=True),
                       dict(scale=(0.0, 1.0)), dict(scale=(1.0, 0.6)),
                       dict(scale=(0.6, float("inf"))), dict(size=(0, 1008))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                transforms.RandomResizedCrop(**(dict(size=(1008, 1008)) | kwargs))


if __name__ == "__main__":
    unittest.main()
