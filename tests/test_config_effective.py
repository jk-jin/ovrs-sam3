from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from tests._bootstrap import REPO_ROOT, load_module


config_mod = load_module("engine.config")
dataset_mod = load_module("data.dataset")
evaluator_mod = load_module("engine.evaluator")
templates_mod = load_module("data.prompt_templates")
task_modes = load_module("models.task_modes")


class TestEffectiveConfig(unittest.TestCase):
    def setUp(self):
        self.main_config = (
            REPO_ROOT
            / "configs"
            / "experiments"
            / "main"
            / "dlrsd_loveda_full.py"
        )

    def _load(self):
        cfg, fingerprint = config_mod.build_effective_config(
            self.main_config
        )
        return cfg, fingerprint

    def test_main_entry_values(self):
        cfg, fingerprint = self._load()
        self.assertEqual(cfg["train_cfg"]["monitor"], "semantic.miou")
        self.assertIsNone(cfg["train_cfg"]["val_max_iters"])
        self.assertEqual(cfg["evaluation"]["metric_groups"], [])
        self.assertEqual(cfg["evaluation"]["dataset"], "loveda")
        self.assertEqual(
            len(cfg["model"]["remoteclip"]["prompt_templates"]), 64
        )
        self.assertTrue(cfg["logging"]["wandb"]["enabled"])
        self.assertEqual(cfg["logging"]["wandb"]["mode"], "offline")
        self.assertEqual(
            cfg["model"]["freeze"]["trainable_modules"],
            ["core.feature_aggregation"],
        )
        self.assertEqual(
            cfg["model"]["aggregation"]["num_blocks"], 2
        )
        self.assertEqual(
            cfg["optimizer"]["paramwise_cfg"]["custom_keys"][
                "core.feature_aggregation"
            ]["lr_mult"],
            1.0,
        )
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(cfg["loss"]["present_class_weight"], 0.8)
        self.assertEqual(cfg["loss"]["reference_distill_weight"], 0.5)
        self.assertEqual(cfg["param_scheduler"][1]["min_lr_ratio"], 0.01)
        self.assertNotIn("eta_min", cfg["param_scheduler"][1])

    def test_window_sharing_option_validation_and_fingerprint(self):
        key = "share_window_weights_across_scales"
        original, fingerprint = self._load()
        self.assertIs(original["model"]["aggregation"][key], True)
        separate, other = config_mod.build_effective_config(
            self.main_config, cfg_options=[f"model.aggregation.{key}=False"]
        )
        self.assertIs(separate["model"]["aggregation"][key], False)
        self.assertIs(separate["experiment"]["factors"][key], False)
        self.assertNotEqual(other, fingerprint)
        for invalid in (1, None, "False"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, key):
                separate["model"]["aggregation"][key] = invalid
                config_mod.finalize_config(separate)
        with self.assertRaisesRegex(ValueError, "window_size"):
            config_mod.build_effective_config(
                self.main_config, cfg_options=["model.aggregation.window_size=9"]
            )

    def test_prompt_templates_are_64(self):
        self.assertEqual(
            len(templates_mod.REMOTECLIP_PROMPT_TEMPLATES), 64
        )

    def test_training_augmentation_and_validation_pipeline(self):
        cfg, fingerprint = self._load()
        transforms = cfg["train_dataloader"]["dataset"]["transforms"]
        self.assertEqual([t["type"] for t in transforms], [
            "ToTensor", "ConvertImageDtype", "RandomResizedCrop", "RandomRotate90",
            "ColorAugSSD", "RandomHorizontalFlip", "Normalize",
        ])
        crop = transforms[2]
        self.assertEqual(crop["size"], (1008, 1008))
        self.assertEqual(crop["scale"], (0.6, 1.0))
        self.assertEqual(crop["prob"], 0.7)
        validation = cfg["val_dataloader"]["dataset"]["transforms"]
        self.assertEqual([t["type"] for t in validation], [
            "ToTensor", "ConvertImageDtype", "Resize", "Normalize",
        ])
        changed = copy.deepcopy(cfg)
        changed["train_dataloader"]["dataset"]["transforms"][2]["prob"] = 0.0
        self.assertNotEqual(config_mod.config_fingerprint(changed), fingerprint)

    def test_all_training_entries_use_dlrsd(self):
        entries = list((REPO_ROOT / "configs/experiments").rglob("*.py"))
        for path in entries:
            with self.subTest(path=path.name):
                cfg, _ = config_mod.build_effective_config(path)
                self.assertEqual(cfg["training"]["dataset"], "dlrsd")
        cfg, _ = config_mod.build_effective_config(
            REPO_ROOT / "configs/test/isaid.py"
        )
        self.assertEqual(cfg["evaluation"]["dataset"], "isaid")

    def test_deleted_options_are_rejected(self):
        for option in (
            "model.remoteclip.train_random_rotate=True",
            "loss.present_ignore_suppression_weight=0.0",
            "loss.absent_class_weight=0.2",
            "loss.reference_distill_start_iter=1000",
            "loss.reference_distill_end_iter=12000",
            "loss.reference_distill_start_weight=0.5",
            "loss.reference_distill_end_weight=0.0",
            "loss.reference_boundary_width=3",
            "evaluation.postprocess.reference_prob_threshold=0.1",
            "visualization.save_reference_segmentation=True",
        ):
            with self.subTest(option=option), self.assertRaises(KeyError):
                config_mod.build_effective_config(
                    self.main_config, cfg_options=[option]
                )

    def test_dlrsd_loveda_entry_values(self):
        cfg, fingerprint = config_mod.build_effective_config(
            REPO_ROOT / "configs/experiments/main/dlrsd_loveda_full.py"
        )
        self.assertEqual(cfg["training"]["dataset"], "dlrsd")
        self.assertEqual(cfg["training"]["split"], "train")
        self.assertEqual(
            cfg["training"]["protocol_id"], "dlrsd_ovrsisbench_official_train_v1"
        )
        dataset = cfg["train_dataloader"]["dataset"]
        self.assertEqual(len(dataset["classes"]), 17)
        self.assertEqual(
            dataset["img_dir"], "data/datasets/DLRSD/img_dir/train"
        )
        self.assertEqual(
            dataset["ann_dir"], "data/datasets/DLRSD/ann_dir/train"
        )
        self.assertFalse(dataset["background_cfg"]["enabled"])
        self.assertFalse(dataset["reduce_zero_label"])
        self.assertEqual(
            cfg["runtime"]["work_dir"], "./work_dirs/multiscale_windows/full/seed_42/train"
        )
        self.assertEqual(
            cfg["logging"]["wandb"]["tags"][1], "dlrsd-train"
        )
        self.assertEqual(len(fingerprint), 64)

    def test_dedicated_cli_precedence(self):
        cfg, _ = config_mod.build_effective_config(
            self.main_config,
            cfg_options=["runtime.seed=7", "runtime.device=cpu"],
            seed=9,
            device="cuda",
            work_dir="/tmp/reca_override",
        )
        self.assertEqual(cfg["runtime"]["seed"], 9)
        self.assertEqual(cfg["runtime"]["device"], "cuda")
        self.assertEqual(cfg["runtime"]["work_dir"], "/tmp/reca_override")

    def test_fingerprint_ignores_run_fields_but_tracks_semantics(self):
        cfg, fingerprint = self._load()
        changed = copy.deepcopy(cfg)
        changed["runtime"]["work_dir"] = "/tmp/another_work_dir"
        changed["logging"]["wandb"]["enabled"] = False
        changed["experiment"]["name"] = "another_name"
        self.assertEqual(
            config_mod.config_fingerprint(changed), fingerprint
        )

        operational = copy.deepcopy(cfg)
        operational["train_dataloader"]["num_workers"] = 1
        operational["train_dataloader"]["dataset"]["img_dir"] = "/other"
        operational["train_cfg"]["save_interval"] = 123
        self.assertEqual(
            config_mod.config_fingerprint(operational), fingerprint
        )

        changed["loss"]["reference_distill_weight"] = 0.25
        self.assertNotEqual(
            config_mod.config_fingerprint(changed), fingerprint
        )

        changed = copy.deepcopy(cfg)
        changed["train_dataloader"]["batch_size"] = 8
        self.assertNotEqual(
            config_mod.config_fingerprint(changed), fingerprint
        )

    def test_unknown_root_key_rejected(self):
        cfg = config_mod.Config.fromfile(self.main_config)
        cfg["bogus_section"] = {}
        with self.assertRaises(KeyError):
            config_mod.finalize_config(cfg)

    def test_unknown_cfg_option_rejected(self):
        cfg = config_mod.Config.fromfile(self.main_config)
        with self.assertRaises(KeyError):
            config_mod.apply_cfg_options(
                cfg, ["loss.typo_weight=1.0"]
            )

    def test_scheduler_matches_max_iters(self):
        cfg = config_mod.Config.fromfile(self.main_config)
        cfg["param_scheduler"][1]["T_max"] = 18000
        with self.assertRaises(ValueError):
            config_mod.finalize_config(cfg)

    def test_loss_and_schedule_weights_are_validated_and_fingerprinted(self):
        cfg, fingerprint = self._load()
        for section, key in (("loss", "present_class_weight"),
                             ("schedule", "min_lr_ratio")):
            changed = copy.deepcopy(cfg)
            target = changed["loss"] if section == "loss" else changed["param_scheduler"][1]
            target[key] = 0.5
            self.assertNotEqual(config_mod.config_fingerprint(changed), fingerprint)
            for value in (-0.1, 1.1, float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    target[key] = value
                    config_mod.finalize_config(changed)
        cfg["param_scheduler"][1]["eta_min"] = 1e-6
        with self.assertRaises(ValueError):
            config_mod.finalize_config(cfg)

    def test_fixed_distill_weight_validation_and_fingerprint(self):
        cfg, fingerprint = self._load()
        changed = copy.deepcopy(cfg)
        changed["loss"]["reference_distill_weight"] = 0.25
        config_mod.finalize_config(changed)
        self.assertNotEqual(config_mod.config_fingerprint(changed), fingerprint)
        for value in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                changed["loss"]["reference_distill_weight"] = value
                config_mod.finalize_config(changed)
        # Fixed weights need not be constrained to [0, 1].
        changed["loss"]["reference_distill_weight"] = 2.0
        config_mod.finalize_config(changed)

    def test_inheritance_cycle_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "a.py").write_text(
                '_base_ = ["b.py"]\n', encoding="utf-8"
            )
            (tmp / "b.py").write_text(
                '_base_ = ["a.py"]\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                config_mod.Config.fromfile(tmp / "a.py")


class TestLoveDALabelSpace(unittest.TestCase):
    def _make_dataset(self):
        ds = object.__new__(dataset_mod.OVSemanticSegDataset)
        ds.reduce_zero_label = False
        ds.ignore_index = 255
        ds.background_enabled = True
        ds.background_class_id = 0
        ds.background_exclude_from_forward = True
        return ds

    def test_background_is_not_removed_twice(self):
        ds = self._make_dataset()
        raw = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 255]])

        eval_labels = ds._apply_reduce_zero_label(raw)
        self.assertEqual(
            eval_labels.tolist(),
            [[0, 1, 2, 3, 4, 5, 6, 255]],
        )

        forward_labels = ds._apply_background_exclusion_for_forward(
            eval_labels
        )
        self.assertEqual(
            forward_labels.tolist(),
            [[255, 0, 1, 2, 3, 4, 5, 255]],
        )

    def test_eval_mapping_restores_seven_class_ids(self):
        eval_labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 255]])
        forward_labels = torch.tensor(
            [[255, 0, 1, 2, 3, 4, 5, 255]]
        )
        self.assertEqual(
            forward_labels.tolist()[0][1:7],
            list(range(6)),
        )

        # Build one row: background and six foreground positions plus ignore.
        scores = torch.zeros(1, 6, 8, 1)
        for forward_class in range(6):
            scores[0, forward_class, forward_class + 1, 0] = 0.9

        metadata = type("Metadata", (), {
            "background_enabled": True,
            "background_class_id": 0,
            "background_exclude_from_forward": True,
            "eval_num_classes": 7,
            "eval_class_names": [
                "background",
                "building",
                "road",
                "water",
                "barren",
                "forest",
                "agricultural",
            ],
        })()

        pred_eval, num_classes, names = (
            evaluator_mod.build_eval_semantic_pred(
                outputs={
                    task_modes.OUTPUT_KEYS.final_score_map: scores
                },
                metadata=metadata,
                background_prob_threshold=0.0,
            )
        )
        self.assertEqual(num_classes, 7)
        self.assertEqual(len(names), 7)
        pred_eval_2d = pred_eval.squeeze(-1)
        self.assertEqual(
            pred_eval_2d.tolist(),
            [[0, 1, 2, 3, 4, 5, 6, 0]],
        )
        # All seven valid eval positions match the GT eval labels.
        valid = eval_labels != 255
        self.assertTrue(
            bool((pred_eval_2d[valid] == eval_labels[valid]).all().item())
        )


if __name__ == "__main__":
    unittest.main()
