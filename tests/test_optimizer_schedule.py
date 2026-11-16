from __future__ import annotations

import copy
import runpy
import unittest

import torch

from tests._bootstrap import REPO_ROOT, load_module


optimizer_mod = load_module("engine.optimizer_builder")


class TestRelativeCosineSchedule(unittest.TestCase):
    def _build(self, schedule):
        optimizer = torch.optim.SGD([
            {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": 1e-4},
            {"params": [torch.nn.Parameter(torch.zeros(()))], "lr": 1e-6},
        ])
        scheduler = optimizer_mod.build_scheduler(optimizer, {"param_scheduler": schedule})
        return optimizer, scheduler

    def _advance(self, optimizer, scheduler):
        optimizer.step()
        scheduler.step()
        return [group["lr"] for group in optimizer.param_groups]

    def test_training_schedules_preserve_group_ratio_and_reach_floors(self):
        for name in ("full_20k", "exp_4k"):
            with self.subTest(schedule=name):
                cfg = runpy.run_path(str(REPO_ROOT / f"configs/_base_/schedule/{name}.py"))
                schedule = cfg["param_scheduler"]
                optimizer, scheduler = self._build(schedule)
                self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 1e-7])
                warmup = schedule[0]["total_iters"]
                duration = schedule[1]["T_max"]
                previous = None
                for step in range(1, cfg["train_cfg"]["max_iters"] + 1):
                    lrs = self._advance(optimizer, scheduler)
                    self.assertAlmostEqual(lrs[1] / lrs[0], 0.01, places=12)
                    if step == warmup:
                        self.assertAlmostEqual(lrs[0], 1e-4, places=14)
                        self.assertAlmostEqual(lrs[1], 1e-6, places=14)
                    elif step == warmup + duration // 2:
                        self.assertAlmostEqual(lrs[0], 5.05e-5, places=14)
                        self.assertAlmostEqual(lrs[1], 5.05e-7, places=14)
                    if step > warmup:
                        self.assertLess(lrs[0], previous[0])
                        self.assertLess(lrs[1], previous[1])
                    previous = lrs
                self.assertAlmostEqual(lrs[0], 1e-6, places=14)
                self.assertAlmostEqual(lrs[1], 1e-8, places=14)

    def test_resume_matches_uninterrupted_across_warmup_transition(self):
        schedule = [
            dict(type="LinearLR", start_factor=0.1, total_iters=4, end=4),
            dict(type="RelativeCosineAnnealingLR", T_max=12, min_lr_ratio=0.01, end=16),
        ]
        for checkpoint_step in (0, 2, 3, 4, 5, 10, 15):
            with self.subTest(checkpoint_step=checkpoint_step):
                optimizer, scheduler = self._build(schedule)
                for _ in range(checkpoint_step):
                    self._advance(optimizer, scheduler)
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                scheduler_state = copy.deepcopy(scheduler.state_dict())
                expected = [self._advance(optimizer, scheduler)
                            for _ in range(checkpoint_step, 16)]
                resumed_optimizer, resumed_scheduler = self._build(schedule)
                resumed_optimizer.load_state_dict(optimizer_state)
                resumed_scheduler.load_state_dict(scheduler_state)
                actual = [self._advance(resumed_optimizer, resumed_scheduler)
                          for _ in range(checkpoint_step, 16)]
                self.assertEqual(actual, expected)

    def test_single_cosine_and_invalid_settings(self):
        for ratio in (0.0, 0.01, 1.0):
            optimizer, scheduler = self._build(
                dict(type="RelativeCosineAnnealingLR", T_max=10, min_lr_ratio=ratio)
            )
            for _ in range(11):
                lrs = self._advance(optimizer, scheduler)
            self.assertAlmostEqual(lrs[0], 1e-4 * ratio, places=14)
            self.assertAlmostEqual(lrs[1], 1e-6 * ratio, places=14)
        for ratio in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self._build(dict(type="RelativeCosineAnnealingLR", T_max=10, min_lr_ratio=ratio))


if __name__ == "__main__":
    unittest.main()
