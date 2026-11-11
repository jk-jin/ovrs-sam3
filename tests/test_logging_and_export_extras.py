from __future__ import annotations

import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tests._bootstrap import load_module
from tests.test_experiment_results import (
    choose_record,
    import_baseline,
    load_json,
    parse_csv,
    run_tool,
)


hooks_mod = load_module("engine.experiment_hooks")
runtime_state_mod = load_module("engine.runtime_state")


class FakeRun:
    name = "fake-run"

    def __init__(self, fail_log: bool = False):
        self.fail_log = fail_log
        self.logged = []
        self.defined = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, payload, commit=True):
        if self.fail_log:
            raise RuntimeError("synthetic log failure")
        self.logged.append(payload)

    def finish(self, exit_code=0):
        pass


class FakeTrainer:
    def __init__(self, tmp: Path):
        self.cfg = SimpleNamespace(
            save_dir=str(tmp),
            max_iters=100,
            monitor="semantic.miou",
            monitor_mode="max",
        )
        self.log_state = {
            "data_cycle": 1,
            "iter_time": 0.1,
            "data_time": 0.02,
            "memory_mb": 1000,
            "lrs": [1e-4, 1e-6],
            "log_vars": {
                "total_loss": 1.0,
                "loss_seg_bce": 0.5,
                "loss_reference_distill_weighted": 0.3,
            },
            "extra_log_vars": {"debug/value": 2.0},
        }
        self.raw_cfg_for_logging = {
            "experiment": {
                "name": "synthetic",
                "method_id": "reca_sam3",
            },
            "runtime": {"seed": 42},
        }
        self.experiment_cfg = self.raw_cfg_for_logging["experiment"]
        self.seed = 42
        self.attempt_id = "attempt1"
        self.parent_attempt_id = None
        self.run_id = "run1"
        self.resume_from_step = None
        self.resume_diff = None
        self.config_fingerprint = "f" * 64
        self.global_iter = 0
        self.is_eval_only = False
        self.is_resumed = False


def _fake_wandb(init_run=None, init_raises=False):
    module = types.ModuleType("wandb")

    def _init(**kwargs):
        if init_raises:
            raise RuntimeError("synthetic init failure")
        return init_run or FakeRun()

    module.init = _init
    return module


class TestLoggingHooks(unittest.TestCase):
    def test_jsonl_records_full_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            trainer = FakeTrainer(tmp)
            hook = hooks_mod.MetricsJsonlHook()
            hook.before_run(trainer)
            hook.after_train_iter(trainer, 20, None, {})
            hook.after_val(
                trainer,
                20,
                {
                    "semantic.miou": 0.5,
                    "semantic.iou_class_0": 0.4,
                    "_class_names": ["background"],
                },
            )
            hook.after_run(trainer)

            lines = [
                json.loads(line)
                for line in (tmp / "metrics.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            self.assertEqual(lines[0]["mode"], "meta")
            self.assertEqual(lines[0]["attempt_id"], "attempt1")
            train = lines[1]
            self.assertEqual(train["mode"], "train")
            self.assertEqual(train["loss_seg_bce"], 0.5)
            self.assertEqual(train["lrs"], [1e-4, 1e-6])
            val = lines[2]
            self.assertEqual(val["mode"], "val")
            self.assertEqual(val["semantic.miou"], 0.5)
            self.assertEqual(val["semantic.iou_class_0"], 0.4)
            self.assertNotIn("_class_names", val)

    def test_eval_only_does_not_touch_training_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "metrics.jsonl"
            path.write_text("sentinel\n", encoding="utf-8")
            trainer = FakeTrainer(tmp)
            trainer.is_eval_only = True
            hook = hooks_mod.MetricsJsonlHook()
            hook.before_run(trainer)
            self.assertFalse(hook.enabled)
            self.assertEqual(
                path.read_text(encoding="utf-8"), "sentinel\n"
            )

    def test_wandb_init_failure_does_not_break_training(self):
        old = sys.modules.get("wandb")
        sys.modules["wandb"] = _fake_wandb(init_raises=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                hook = hooks_mod.WandbHook(enabled=True, mode="offline")
                hook.before_run(FakeTrainer(Path(tmp)))
                self.assertFalse(hook.enabled)
                self.assertIsNone(hook._run)
        finally:
            if old is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old

    def test_wandb_log_failure_disables_only_wandb(self):
        old = sys.modules.get("wandb")
        sys.modules["wandb"] = _fake_wandb(FakeRun(fail_log=True))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                trainer = FakeTrainer(Path(tmp))
                hook = hooks_mod.WandbHook(enabled=True, mode="offline")
                hook.before_run(trainer)
                self.assertTrue(hook.enabled)
                hook.after_train_iter(trainer, 20, None, {})
                self.assertFalse(hook.enabled)
        finally:
            if old is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old

    def test_wandb_lifecycle_preserves_rng(self):
        old = sys.modules.get("wandb")
        sys.modules["wandb"] = _fake_wandb()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                trainer = FakeTrainer(Path(tmp))
                random.seed(123)
                np.random.seed(123)
                torch.manual_seed(123)
                before = runtime_state_mod.capture_rng_state()

                hook = hooks_mod.WandbHook(enabled=True, mode="offline")
                hook.before_run(trainer)
                hook.after_train_iter(trainer, 20, None, {})
                hook.after_val(trainer, 20, {"semantic.miou": 0.5})
                hook.after_run(trainer)

                after = runtime_state_mod.capture_rng_state()
                self.assertEqual(before["python"], after["python"])
                self.assertTrue(
                    torch.equal(
                        before["numpy"]["keys"], after["numpy"]["keys"]
                    )
                )
                self.assertEqual(
                    before["numpy"]["position"],
                    after["numpy"]["position"],
                )
                self.assertTrue(
                    torch.equal(
                        before["torch_cpu"], after["torch_cpu"]
                    )
                )
        finally:
            if old is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old


class TestCollectAndCurves(unittest.TestCase):
    def test_collect_experiments_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            output = tmp / "long.csv"
            proc = run_tool(
                "collect_experiments.py",
                [str(imported), "--output", str(output)],
                cwd=tmp,
            )
            self.assertEqual(
                proc.returncode, 0, msg=proc.stdout + proc.stderr
            )
            rows = parse_csv(output)
            self.assertEqual(len(rows), 4)
            self.assertIn("miou_fraction", rows[0])
            values = sorted(float(row["miou_fraction"]) for row in rows)
            self.assertEqual(values, [0.0, 0.68, 0.70, 0.72])

    def test_metrics_and_factor_curves_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            run_dir = tmp / "run"
            run_dir.mkdir()
            events = [
                {
                    "mode": "meta",
                    "event": "before_run",
                    "attempt_id": "a1",
                    "resume_from_step": None,
                },
                {"mode": "val", "iter": 3, "attempt_id": "a1", "semantic.miou": 0.5},
                {"mode": "val", "iter": 6, "attempt_id": "a1", "semantic.miou": 0.6},
                {
                    "mode": "meta",
                    "event": "before_run",
                    "attempt_id": "a2",
                    "resume_from_step": 3,
                },
                {"mode": "val", "iter": 6, "attempt_id": "a2", "semantic.miou": 0.7},
            ]
            (run_dir / "metrics.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            variant_root = tmp / "variants"
            record = choose_record(imported, "method_b")
            record["eval_id"] = "method_b_blocks8"
            record["experiment"]["factors"]["blocks"] = 8
            record["metrics"]["miou"] = 0.75
            (variant_root / "results").mkdir(parents=True)
            (variant_root / "results" / "method_b_blocks8.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )

            spec = {
                "schema_version": 1,
                "tables": [],
                "curves": [
                    {
                        "name": "miou_curve",
                        "type": "metrics",
                        "run": str(run_dir),
                        "y": "semantic.miou",
                        "label": "mIoU",
                        "display_unit": "percent",
                    },
                    {
                        "name": "blocks_curve",
                        "type": "factor",
                        "filter": {
                            "experiment.method_id": "method_b"
                        },
                        "x_factor": "experiment.factors.blocks",
                        "y": "miou",
                        "label": "mIoU",
                        "display_unit": "percent",
                    },
                ],
            }
            spec_path = tmp / "curve_spec.json"
            spec_path.write_text(
                json.dumps(spec, ensure_ascii=False), encoding="utf-8"
            )
            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(variant_root),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(tmp / "out"),
                ],
                cwd=tmp,
            )
            self.assertEqual(
                proc.returncode, 0, msg=proc.stdout + proc.stderr
            )
            self.assertTrue(
                (tmp / "out" / "figures" / "miou_curve.pdf").exists()
            )
            self.assertTrue(
                (tmp / "out" / "figures" / "blocks_curve.pdf").exists()
            )
            report = load_json(tmp / "out" / "export_report.json")
            self.assertEqual(report["curves"]["miou_curve"]["type"], "metrics")
            self.assertEqual(report["curves"]["blocks_curve"]["type"], "factor")


if __name__ == "__main__":
    unittest.main()
