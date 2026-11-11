from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import FIXTURES, REPO_ROOT
from utils.experiment_results import (
    atomic_write_json,
    escape_latex,
    load_json,
    mean_std,
    normalize_record_units,
    validate_record,
)


def run_tool(name: str, args: list[str], cwd: Path):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / name), *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def import_baseline(tmp: Path) -> Path:
    output = tmp / "imported with space"
    proc = run_tool(
        "import_results.py",
        [
            "--input",
            str(FIXTURES / "import" / "baseline.csv"),
            "--mapping",
            str(FIXTURES / "import" / "mapping.json"),
            "--protocol",
            str(FIXTURES / "import" / "protocol.json"),
            "--input-unit",
            "percent",
            "--output-dir",
            str(output),
        ],
        cwd=tmp,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"import failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return output


def result_files(root: Path) -> list[Path]:
    return sorted((root / "results").glob("*.json"))


def choose_record(root: Path, method_id: str) -> dict:
    for path in result_files(root):
        record = load_json(path)
        if record["experiment"]["method_id"] == method_id:
            return copy.deepcopy(record)
    raise AssertionError(f"method {method_id!r} not found in {root}")


def write_record(root: Path, record: dict) -> Path:
    record.pop("_source_path", None)
    return atomic_write_json(
        root / "results" / f"{record['eval_id']}.json", record
    )


def parse_csv(path: Path) -> list[dict]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestResultUtilities(unittest.TestCase):
    def test_mean_std_sample_ddof_one(self):
        mean, std = mean_std([70.0, 72.0], ddof=1)
        self.assertAlmostEqual(mean, 71.0)
        self.assertAlmostEqual(std, math.sqrt(2.0))

        mean, std = mean_std([70.0], ddof=1)
        self.assertAlmostEqual(mean, 70.0)
        self.assertIsNone(std)

    def test_escape_latex(self):
        text = escape_latex("a_b & c% d# $e$ {f} ~g^h \\i")
        self.assertIn(r"a\_b", text)
        self.assertIn(r"\&", text)
        self.assertIn(r"\%", text)
        self.assertIn(r"\#", text)
        self.assertIn(r"\$e\$", text)
        self.assertIn(r"\{f\}", text)
        self.assertIn(r"\textasciitilde{}g", text)
        self.assertIn(r"\textasciicircum{}h", text)
        self.assertIn(r"\textbackslash{}i", text)

    def test_url_encoding_example(self):
        text = escape_latex("tab:main_loveda")
        self.assertEqual(text, r"tab:main\_loveda")

    def test_percent_to_fraction_is_idempotent(self):
        record = {
            "metrics": {
                "unit": "percent",
                "miou": 72.3,
                "macc": 80.0,
                "pixel_acc": 85.0,
                "per_class": [
                    {
                        "class_id": 0,
                        "class_name": "background",
                        "iou": 90.0,
                        "accuracy": 95.0,
                    }
                ],
            }
        }
        normalized = normalize_record_units(record)
        self.assertEqual(normalized["metrics"]["unit"], "fraction")
        self.assertAlmostEqual(normalized["metrics"]["miou"], 0.723)
        self.assertAlmostEqual(
            normalized["metrics"]["per_class"][0]["iou"], 0.9
        )
        again = normalize_record_units(normalized)
        self.assertAlmostEqual(again["metrics"]["miou"], 0.723)


class TestImportExport(unittest.TestCase):
    def test_import_normalizes_and_preserves_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            records = [
                load_json(path) for path in result_files(imported)
            ]
            self.assertEqual(len(records), 4)
            for record in records:
                validate_record(record, require_completed=True)
                self.assertEqual(record["metrics"]["unit"], "fraction")
                self.assertEqual(
                    record["experiment"]["factors"]["blocks"], 4
                )
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["role"], "evaluation")

            method_c = choose_record(imported, "method_c")
            self.assertEqual(method_c["metrics"]["miou"], 0.0)
            self.assertNotIn("per_class", method_c["metrics"])

    def test_export_main_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            output = tmp / "paper generated"
            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--spec",
                    str(FIXTURES / "export" / "spec_main.json"),
                    "--output",
                    str(output),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=proc.stdout + proc.stderr,
            )

            csv_path = output / "data" / "synthetic_main.csv"
            rows = parse_csv(csv_path)
            self.assertEqual(len(rows), 3)
            by_label = {row["row_label"]: row for row in rows}
            method_a = by_label["Method & A_结果"]
            self.assertAlmostEqual(
                float(method_a["miou_mean_percent"]), 71.0
            )
            self.assertAlmostEqual(
                float(method_a["miou_std_percent"]), 1.41
            )
            self.assertEqual(method_a["n_seeds"], "2")
            method_b = by_label["Method B"]
            self.assertAlmostEqual(
                float(method_b["miou_mean_percent"]), 68.0
            )
            self.assertEqual(method_b["miou_std_percent"], "")
            method_c = by_label["Method C"]
            self.assertAlmostEqual(
                float(method_c["miou_mean_percent"]), 0.0
            )

            tex = (output / "tables" / "synthetic_main.tex").read_text(
                encoding="utf-8"
            )
            self.assertIn(r"Method \& A\_结果", tex)
            self.assertIn(r"Synthetic \& test\_结果", tex)
            self.assertIn("71.00", tex)
            self.assertIn(r"$\pm$", tex)
            self.assertIn(r"\label{tab:synthetic_main}", tex)

            report = load_json(output / "export_report.json")
            self.assertFalse(report["draft"])
            self.assertEqual(len(report["adopted"]), 4)
            self.assertTrue(
                (output / "data" / "results_long.csv").exists()
            )

    def test_per_class_missing_strict_then_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            args = [
                "--root",
                str(imported),
                "--spec",
                str(FIXTURES / "export" / "spec_per_class.json"),
                "--output",
                str(tmp / "out"),
                "--tables-only",
            ]
            strict = run_tool(
                "export_paper_results.py", args, cwd=tmp
            )
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn("missing", strict.stderr.lower())

            draft = run_tool(
                "export_paper_results.py",
                args + ["--allow-incomplete"],
                cwd=tmp,
            )
            self.assertEqual(
                draft.returncode,
                0,
                msg=draft.stdout + draft.stderr,
            )
            tex = (
                tmp / "out" / "tables" / "synthetic_per_class.tex"
            ).read_text(encoding="utf-8")
            self.assertIn("% DRAFT", tex)
            self.assertIn("--", tex)
            report = load_json(tmp / "out" / "export_report.json")
            self.assertTrue(report["draft"])

    def test_expected_seed_strict_and_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            args = [
                "--root",
                str(imported),
                "--spec",
                str(FIXTURES / "export" / "spec_missing_seed.json"),
                "--output",
                str(tmp / "out"),
                "--tables-only",
            ]
            strict = run_tool(
                "export_paper_results.py", args, cwd=tmp
            )
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn("missing expected seeds", strict.stderr)

            draft = run_tool(
                "export_paper_results.py",
                args + ["--allow-incomplete"],
                cwd=tmp,
            )
            self.assertEqual(
                draft.returncode,
                0,
                msg=draft.stdout + draft.stderr,
            )
            tex = (
                tmp / "out" / "tables" / "synthetic_missing_seed.tex"
            ).read_text(encoding="utf-8")
            self.assertIn("% DRAFT", tex)

    def test_reported_summary_and_single_run_cannot_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            reported_root = tmp / "reported"
            record = choose_record(imported, "method_a")
            record["eval_id"] = "reported_method_a"
            record["record_type"] = "reported_summary"
            record["experiment"]["seed"] = None
            record["checkpoint"] = None
            record["provenance"]["source"] = "reported"
            record["metrics"] = {
                "unit": "fraction",
                "miou": 0.73,
                "macc": 0.78,
                "pixel_acc": 0.83,
                "std": {"miou": 0.005, "macc": None, "pixel_acc": None},
                "n_seeds": 5,
                "std_definition": "sample",
            }
            write_record(reported_root, record)

            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(reported_root),
                    "--spec",
                    str(FIXTURES / "export" / "spec_main.json"),
                    "--output",
                    str(tmp / "out"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("mixes record types", proc.stderr)

    def test_reported_summary_exports_mean_and_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            reported_root = tmp / "reported"
            record = choose_record(imported, "method_b")
            record["eval_id"] = "reported_method_b"
            record["record_type"] = "reported_summary"
            record["experiment"]["method_id"] = "method_reported"
            record["experiment"]["name"] = "method_reported"
            record["experiment"]["seed"] = None
            record["checkpoint"] = None
            record["provenance"]["source"] = "reported"
            record["metrics"] = {
                "unit": "fraction",
                "miou": 0.70,
                "macc": 0.75,
                "pixel_acc": 0.80,
                "std": {"miou": 0.005, "macc": None, "pixel_acc": None},
                "n_seeds": 5,
                "std_definition": "sample",
            }
            write_record(reported_root, record)

            spec = {
                "schema_version": 1,
                "tables": [{
                    "name": "reported",
                    "filter": {"evaluation.dataset": "loveda"},
                    "rows": [{
                        "label": "Reported",
                        "match": {
                            "experiment.method_id": "method_reported"
                        },
                    }],
                    "metrics": ["miou"],
                    "allowed_varying_fields": [],
                    "factor_columns": [],
                    "display_unit": "percent",
                    "decimals": 2,
                    "aggregate_seeds": True,
                    "std_ddof": 1,
                    "layout": "column",
                    "caption": "Reported synthetic",
                    "label": "tab:reported",
                    "bold_best": False,
                }],
                "curves": [],
            }
            spec_path = tmp / "reported_spec.json"
            spec_path.write_text(
                json.dumps(spec, ensure_ascii=False), encoding="utf-8"
            )
            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(reported_root),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(tmp / "out"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertEqual(
                proc.returncode, 0, msg=proc.stdout + proc.stderr
            )
            rows = parse_csv(tmp / "out" / "data" / "reported.csv")
            self.assertAlmostEqual(
                float(rows[0]["miou_mean_percent"]), 70.0
            )
            self.assertAlmostEqual(
                float(rows[0]["miou_std_percent"]), 0.5
            )
            self.assertEqual(rows[0]["n_seeds"], "5")

    def test_threshold_variants_need_explicit_allowed_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            variants_root = tmp / "thresholds"
            record = choose_record(imported, "method_a")
            record = copy.deepcopy(record)
            record["experiment"]["seed"] = 42
            record["eval_id"] = "method_a_threshold_04"
            record["evaluation"]["postprocess"][
                "prediction_prob_threshold"
            ] = 0.4
            write_record(variants_root, record)

            base_spec = {
                "schema_version": 1,
                "tables": [{
                    "name": "thresholds",
                    "filter": {"evaluation.dataset": "loveda"},
                    "rows": [
                        {
                            "label": "05",
                            "match": {
                                "experiment.method_id": "method_a",
                                "experiment.seed": 42,
                                "evaluation.postprocess.prediction_prob_threshold": 0.5,
                            },
                        },
                        {
                            "label": "04",
                            "match": {
                                "experiment.method_id": "method_a",
                                "experiment.seed": 42,
                                "evaluation.postprocess.prediction_prob_threshold": 0.4,
                            },
                        },
                    ],
                    "metrics": ["miou"],
                    "allowed_varying_fields": [],
                    "factor_columns": [
                        "evaluation.postprocess.prediction_prob_threshold"
                    ],
                    "display_unit": "percent",
                    "decimals": 2,
                    "aggregate_seeds": True,
                    "std_ddof": 1,
                    "layout": "column",
                    "caption": "Threshold variants",
                    "label": "tab:thresholds",
                    "bold_best": False,
                }],
                "curves": [],
            }
            spec_path = tmp / "threshold_spec_strict.json"
            spec_path.write_text(
                json.dumps(base_spec, ensure_ascii=False), encoding="utf-8"
            )
            strict = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(variants_root),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(tmp / "out_strict"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn("allowed_varying_fields", strict.stderr)

            base_spec["tables"][0]["allowed_varying_fields"] = [
                "evaluation.postprocess.prediction_prob_threshold"
            ]
            spec_path.write_text(
                json.dumps(base_spec, ensure_ascii=False), encoding="utf-8"
            )
            allowed = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(variants_root),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(tmp / "out_allowed"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertEqual(
                allowed.returncode, 0,
                msg=allowed.stdout + allowed.stderr,
            )
            rows = parse_csv(
                tmp / "out_allowed" / "data" / "thresholds.csv"
            )
            self.assertEqual(
                [row["evaluation.postprocess.prediction_prob_threshold"] for row in rows],
                ["0.5", "0.4"],
            )

    def test_conflicting_duplicate_eval_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            other_root = tmp / "conflict"
            record = choose_record(imported, "method_b")
            record["metrics"]["miou"] = 0.99
            write_record(other_root, record)

            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(other_root),
                    "--spec",
                    str(FIXTURES / "export" / "spec_main.json"),
                    "--output",
                    str(tmp / "out"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("conflicting duplicate eval_id", proc.stderr)

    def test_identical_duplicate_eval_id_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            duplicate_root = tmp / "duplicate"
            source = choose_record(imported, "method_b")
            write_record(duplicate_root, source)

            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(duplicate_root),
                    "--spec",
                    str(FIXTURES / "export" / "spec_main.json"),
                    "--output",
                    str(tmp / "out"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertEqual(
                proc.returncode, 0, msg=proc.stdout + proc.stderr
            )
            rows = parse_csv(
                tmp / "out" / "data" / "synthetic_main.csv"
            )
            method_b = [
                row for row in rows if row["row_label"] == "Method B"
            ][0]
            self.assertEqual(method_b["n_seeds"], "1")

    def test_same_seed_multiple_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            other_root = tmp / "alt_checkpoint"
            record = choose_record(imported, "method_b")
            record["eval_id"] = "method_b_alt_checkpoint"
            record["checkpoint"]["step"] = 999999
            record["checkpoint"]["path"] = "../iter_0999999.pth"
            write_record(other_root, record)

            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(other_root),
                    "--spec",
                    str(FIXTURES / "export" / "spec_main.json"),
                    "--output",
                    str(tmp / "out"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("multiple candidate", proc.stderr)

    def test_protocol_mismatch_six_class_vs_seven_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imported = import_baseline(tmp)
            six_root = tmp / "six_class"
            record = choose_record(imported, "method_c")
            record["eval_id"] = "method_six"
            record["experiment"]["method_id"] = "method_six"
            record["experiment"]["name"] = "method_six"
            record["evaluation"]["class_ids"] = [0, 1, 2, 3, 4, 5]
            record["evaluation"]["class_names"] = [
                "building",
                "road",
                "water",
                "barren",
                "forest",
                "agricultural",
            ]
            record["evaluation"]["metric_class_ids"] = [
                0, 1, 2, 3, 4, 5
            ]
            write_record(six_root, record)

            spec = {
                "schema_version": 1,
                "tables": [{
                    "name": "mixed",
                    "filter": {"evaluation.dataset": "loveda"},
                    "rows": [
                        {
                            "label": "seven",
                            "match": {
                                "experiment.method_id": "method_c"
                            },
                        },
                        {
                            "label": "six",
                            "match": {
                                "experiment.method_id": "method_six"
                            },
                        },
                    ],
                    "metrics": ["miou"],
                    "allowed_varying_fields": [],
                    "factor_columns": [],
                    "display_unit": "percent",
                    "decimals": 2,
                    "aggregate_seeds": True,
                    "std_ddof": 1,
                    "layout": "column",
                    "caption": "Mixed classes",
                    "label": "tab:mixed",
                    "bold_best": False,
                }],
                "curves": [],
            }
            spec_path = tmp / "mixed_spec.json"
            spec_path.write_text(
                json.dumps(spec, ensure_ascii=False), encoding="utf-8"
            )
            proc = run_tool(
                "export_paper_results.py",
                [
                    "--root",
                    str(imported),
                    "--root",
                    str(six_root),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(tmp / "out"),
                    "--tables-only",
                ],
                cwd=tmp,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("evaluation.class_ids", proc.stderr)

    def test_nan_import_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = tmp / "nan.csv"
            csv_path.write_text(
                "method,variant,seed,miou,macc,pixel_acc\n"
                "bad,full,42,nan,70,80\n",
                encoding="utf-8",
            )
            proc = run_tool(
                "import_results.py",
                [
                    "--input",
                    str(csv_path),
                    "--mapping",
                    str(FIXTURES / "import" / "mapping.json"),
                    "--protocol",
                    str(FIXTURES / "import" / "protocol.json"),
                    "--input-unit",
                    "percent",
                    "--output-dir",
                    str(tmp / "out"),
                ],
                cwd=tmp,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("metrics.miou", proc.stderr)

    def test_no_torch_import_for_export_helpers(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "import utils.experiment_results; "
            "assert 'torch' not in sys.modules, sorted(sys.modules)"
        )
        proc = subprocess.run(
            [sys.executable, "-S", "-c", code],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode, 0, msg=proc.stdout + proc.stderr
        )


if __name__ == "__main__":
    unittest.main()


class TestNativeResultWriter(unittest.TestCase):
    def test_writer_emits_valid_standard_record(self):
        from tests._bootstrap import load_module

        result_writer_mod = load_module("engine.result_writer")

        class FakeEvaluator:
            num_classes = 7

            def per_class_stats(self, class_names):
                return [
                    {
                        "class_id": idx,
                        "class_name": name,
                        "iou": 0.5 + idx * 0.01,
                        "accuracy": 0.6 + idx * 0.01,
                        "intersection": 10.0,
                        "union": 20.0,
                        "target_pixels": 15.0,
                    }
                    for idx, name in enumerate(class_names)
                ]

            def valid_class_counts(self):
                return {
                    "valid_iou_classes": 7,
                    "valid_acc_classes": 7,
                }

        class_names = [
            "background",
            "building",
            "road",
            "water",
            "barren",
            "forest",
            "agricultural",
        ]
        evaluation = {
            "dataset": "loveda",
            "split": "val",
            "data_version": "synthetic-test",
            "protocol_id": "loveda_7class_synthetic_test",
            "ignore_index": 255,
            "background_prob_threshold": 0.0,
            "postprocess": {
                "reference_prob_threshold": 0.1,
                "prediction_prob_threshold": 0.5,
            },
            "tta": {
                "enabled": False,
                "scales": [1.0],
                "flip_modes": ["none"],
                "size_divisor": 14,
            },
            "geometry": {"input_hw": [1008, 1008], "keep_ratio": False},
        }
        summary = {
            "stats": {
                "semantic.miou": 0.5,
                "semantic.macc": 0.6,
                "semantic.pixel_acc": 0.7,
            },
            "evaluator": FakeEvaluator(),
            "class_names": class_names,
            "expected_samples": 10,
            "evaluated_samples": 10,
            "is_full": True,
            "input_hw": (1008, 1008),
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ckpt_path = tmp / "best.pth"
            ckpt_path.write_bytes(b"synthetic checkpoint")
            writer = result_writer_mod.ResultWriter(
                work_dir=tmp,
                experiment={
                    "name": "synthetic",
                    "method_id": "reca_sam3",
                    "method_name": "ReCA-SAM3",
                    "variant_id": "full",
                    "implementation_id": "synthetic-v1",
                    "kind": "main",
                    "factors": {"num_blocks": 4},
                },
                training={
                    "regime": "supervised_adaptation",
                    "dataset": "isaid",
                    "split": "train",
                    "protocol_id": "isaid_train_v1",
                },
                evaluation=evaluation,
                train_cfg={
                    "monitor": "semantic.miou",
                    "monitor_mode": "max",
                },
                dataset_info={
                    "class_names": class_names,
                    "include_background": True,
                    "expected_samples": 10,
                    "sample_list_sha256": "synthetic-list",
                },
                seed=42,
                config_fingerprint="a" * 64,
                git_info={
                    "available": True,
                    "commit": "deadbeef",
                    "dirty": False,
                    "untracked_code": False,
                },
                run_id="synthetic-run",
                attempt_id="attempt123",
            )
            path = writer.write_result(
                role="evaluation",
                summary=summary,
                checkpoint_path=ckpt_path,
                checkpoint_step=2000,
                selection={
                    "metric": "semantic.miou",
                    "mode": "max",
                    "value": 0.5,
                    "is_best": True,
                },
                attempt_id="attempt123",
                resume_from_step=None,
            )
            record = load_json(path)
            validate_record(record, require_completed=True)
            self.assertEqual(record["metrics"]["unit"], "fraction")
            self.assertEqual(record["metrics"]["miou"], 0.5)
            self.assertEqual(record["evaluation"]["expected_samples"], 10)
            self.assertEqual(record["evaluation"]["class_ids"], list(range(7)))
            self.assertEqual(record["checkpoint"]["step"], 2000)
            self.assertEqual(record["checkpoint"]["path"], "../best.pth")
            self.assertEqual(
                record["checkpoint"]["selection"]["value"], 0.5
            )
            self.assertEqual(record["provenance"]["source"], "native")

            partial_summary = dict(summary)
            partial_summary["is_full"] = False
            partial_summary["evaluated_samples"] = 5
            partial_path = writer.write_result(
                role="validation",
                summary=partial_summary,
                checkpoint_path=ckpt_path,
                checkpoint_step=1000,
                selection={
                    "metric": "semantic.miou",
                    "mode": "max",
                    "value": 0.5,
                    "is_best": False,
                },
                attempt_id="attempt123",
                resume_from_step=None,
            )
            partial_record = load_json(partial_path)
            self.assertEqual(partial_record["status"], "partial")


class TestCurveAttemptChain(unittest.TestCase):
    def _module(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "export_mod",
            REPO_ROOT / "tools" / "export_paper_results.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_attempt_chain_drops_rolled_back_points(self):
        module = self._module()
        records = [
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
        effective = module._effective_curve_points(records)
        val_points = [
            (r["iter"], r["semantic.miou"])
            for r in effective if r.get("mode") == "val"
        ]
        self.assertEqual(val_points, [(3, 0.5), (6, 0.7)])
