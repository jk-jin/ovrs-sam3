from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..utils.experiment_results import (
    atomic_write_json,
    make_eval_id,
    sha256_file,
    validate_record,
)


_PATCH_EXCLUDES = (
    ":(exclude)paper/build",
    ":(exclude)paper/generated",
    ":(exclude)work_dirs",
    ":(exclude).agents",
    ":(exclude).idea",
)

_UNTRACKED_IGNORE_PARTS = (
    "__pycache__",
    ".pytest_cache",
    "paper/build",
    "paper/generated",
    "work_dirs",
    ".agents",
    ".idea",
)


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def collect_git_info(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root).resolve()
    result = _run_git(repo_root, ["rev-parse", "HEAD"])
    if result.returncode != 0:
        return {
            "git_available": False,
            "commit": None,
            "dirty": None,
            "untracked_code": None,
        }

    commit = result.stdout.strip()
    diff = _run_git(repo_root, ["diff", "HEAD", "--", *_PATCH_EXCLUDES])
    dirty = diff.returncode != 0 or bool(diff.stdout.strip())

    status = _run_git(repo_root, ["status", "--porcelain"])
    untracked_code = False
    if status.returncode == 0:
        for line in status.stdout.splitlines():
            if not line.startswith("?? "):
                continue
            path = line[3:].strip()
            if not path:
                continue
            suffixes = Path(path).suffix
            if suffixes not in {".py", ".yaml", ".yml", ".json", ".toml", ".cfg"}:
                continue
            normalized = path.replace("\\", "/")
            if any(part in normalized for part in _UNTRACKED_IGNORE_PARTS):
                continue
            untracked_code = True
            break

    return {
        "git_available": True,
        "commit": commit,
        "dirty": bool(dirty),
        "untracked_code": bool(untracked_code),
        "patch_text": diff.stdout if diff.returncode == 0 else "",
    }


def write_code_patch(work_dir: str | Path, git_info: dict) -> Optional[Path]:
    patch_text = git_info.get("patch_text", "")
    if not patch_text:
        return None
    path = Path(work_dir) / "code.patch"
    try:
        path.write_text(patch_text, encoding="utf-8")
    except OSError as exc:
        print(
            f"[warning] Failed to write optional code.patch: {exc}"
        )
        return None
    return path


def build_config_meta(
    *,
    git_info: dict,
    seed: int,
    attempt_id: str,
    run_id: str,
    config_fingerprint: str,
    environment: Optional[dict] = None,
) -> dict:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": int(seed),
        "attempt_id": attempt_id,
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "git": {
            "available": bool(git_info.get("git_available", False)),
            "commit": git_info.get("commit"),
            "dirty": git_info.get("dirty"),
            "untracked_code": git_info.get("untracked_code"),
            "code_patch": (
                "code.patch" if git_info.get("patch_text") else None
            ),
        },
        "environment": environment or {},
    }


def save_effective_config(
    work_dir: str | Path,
    cfg: dict,
    meta: dict,
) -> Path:
    payload = dict(cfg)
    payload["meta"] = meta
    return atomic_write_json(Path(work_dir) / "config.json", payload)


def load_saved_config(work_dir: str | Path) -> Optional[dict]:
    path = Path(work_dir) / "config.json"
    if not path.exists():
        return None
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_tta_views(tta_cfg: dict) -> list[str]:
    if not bool(tta_cfg.get("enabled", False)):
        return ["none"]
    scales = [float(x) for x in tta_cfg.get("scales", [1.0])]
    flips = [str(x) for x in tta_cfg.get("flip_modes", ["none"])]
    return [f"scale={scale:g},flip={flip}" for scale in scales for flip in flips]


class ResultWriter:
    """Write one standard result JSON per completed validation/evaluation."""

    def __init__(
        self,
        *,
        work_dir: str | Path,
        experiment: dict,
        training: dict,
        evaluation: dict,
        train_cfg: dict,
        dataset_info: dict,
        seed: int,
        config_fingerprint: str,
        git_info: dict,
        run_id: str,
        attempt_id: Optional[str],
        config_source_path: Optional[str | Path] = None,
    ):
        self.work_dir = Path(work_dir)
        self.results_dir = self.work_dir / "results"
        self.experiment = dict(experiment)
        self.training = dict(training)
        self.evaluation = dict(evaluation)
        self.train_cfg = dict(train_cfg)
        self.dataset_info = dict(dataset_info)
        self.seed = int(seed)
        self.config_fingerprint = str(config_fingerprint)
        self.git_info = dict(git_info)
        self.run_id = str(run_id)
        self.attempt_id = attempt_id
        self.config_source_path = (
            Path(config_source_path)
            if config_source_path is not None
            else self.work_dir / "config.json"
        )
        self._sha_cache: dict[tuple[str, int], str] = {}

    def _checkpoint_sha256(self, path: Optional[str | Path]) -> Optional[str]:
        if path is None:
            return None
        path = Path(path)
        if not path.is_file():
            return None
        stat = path.stat()
        key = (str(path.resolve()), int(stat.st_mtime_ns))
        cached = self._sha_cache.get(key)
        if cached is not None:
            return cached
        digest = sha256_file(path)
        self._sha_cache[key] = digest
        return digest

    def _build_class_table(
        self,
        evaluator,
        class_names_from_batch,
    ) -> tuple[list[int], list[str], list[dict], dict]:
        evaluator_class_count = int(evaluator.num_classes or 0)
        class_names = class_names_from_batch
        if class_names is None:
            class_names = self.dataset_info.get("class_names", None)
        if class_names is None:
            raise ValueError(
                "Cannot write a result without evaluation class names."
            )
        class_names = [str(x) for x in class_names]

        if evaluator_class_count and len(class_names) != evaluator_class_count:
            raise ValueError(
                "Evaluation class-name count does not match evaluator "
                f"classes: {len(class_names)} vs {evaluator_class_count}."
            )

        per_class = evaluator.per_class_stats(class_names)
        class_ids = [int(row["class_id"]) for row in per_class]
        rows = []
        for row in per_class:
            rows.append({
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "iou": row["iou"],
                "accuracy": row["accuracy"],
                "intersection": row["intersection"],
                "union": row["union"],
                "target_pixels": row["target_pixels"],
            })

        extra = evaluator.valid_class_counts()
        return class_ids, class_names, rows, extra

    def _build_metrics(
        self,
        stats: dict,
        per_class: list[dict],
        valid_counts: dict,
    ) -> dict:
        extra = dict(valid_counts)
        for key, value in stats.items():
            if str(key).startswith("semantic.groups."):
                extra[str(key)] = value

        return {
            "unit": "fraction",
            "miou": stats.get("semantic.miou", None),
            "macc": stats.get("semantic.macc", None),
            "pixel_acc": stats.get("semantic.pixel_acc", None),
            "per_class": per_class,
            "extra": extra,
        }

    def _relative_path(self, path: Optional[str | Path]) -> Optional[str]:
        if path is None:
            return None
        path = Path(path).resolve()
        try:
            return os.path.relpath(path, self.results_dir.resolve())
        except ValueError:
            return str(path)

    def write_result(
        self,
        *,
        role: str,
        summary: dict,
        checkpoint_path: Optional[str | Path],
        checkpoint_step: Optional[int],
        selection: dict,
        attempt_id: Optional[str],
        resume_from_step: Optional[int],
    ) -> Optional[Path]:
        if summary is None:
            raise ValueError("Cannot write a result without validation summary.")

        if role not in {"validation", "evaluation"}:
            raise ValueError(f"Unsupported result role: {role!r}.")

        if checkpoint_step is None:
            raise ValueError(
                "Native result records require an explicit checkpoint step."
            )

        evaluator = summary["evaluator"]
        stats = summary["stats"]
        class_ids, class_names, per_class, valid_counts = (
            self._build_class_table(
                evaluator,
                summary.get("class_names", None),
            )
        )

        expected_samples = summary.get("expected_samples", None)
        if expected_samples is None:
            expected_samples = self.dataset_info.get(
                "expected_samples", None
            )
        evaluated_samples = summary.get("evaluated_samples", None)
        is_full = bool(summary.get("is_full", False))

        input_hw = summary.get("input_hw", None)
        if input_hw is None:
            input_hw = self.evaluation.get("geometry", {}).get(
                "input_hw", [1008, 1008]
            )
        input_hw = [int(x) for x in input_hw]

        selection_value = selection.get("value", None)
        if selection_value is not None:
            selection_value = float(selection_value)
        selection_block = {
            "dataset": selection.get(
                "dataset", self.evaluation.get("dataset")
            ),
            "split": selection.get("split", self.evaluation.get("split")),
            "metric": selection.get("metric", self.train_cfg.get("monitor")),
            "mode": selection.get("mode", self.train_cfg.get("monitor_mode")),
            "value": selection_value,
            "is_best": selection.get("is_best", None),
        }

        metrics = self._build_metrics(stats, per_class, valid_counts)
        tta_cfg = self.evaluation.get("tta", {})
        postprocess = self.evaluation.get("postprocess", {})

        eval_id = make_eval_id(
            role=role,
            dataset=str(self.evaluation.get("dataset", "unknown")),
            split=str(self.evaluation.get("split", "unknown")),
            step=checkpoint_step,
            protocol_id=self.evaluation.get("protocol_id", None),
        )

        record = {
            "schema_version": 1,
            "record_type": "single_run",
            "eval_id": eval_id,
            "status": "completed" if is_full else "partial",
            "role": role,
            "experiment": {
                "run_id": self.run_id,
                "name": self.experiment.get("name"),
                "method_id": self.experiment.get("method_id"),
                "method_name": self.experiment.get("method_name"),
                "variant_id": self.experiment.get("variant_id"),
                "implementation_id": self.experiment.get(
                    "implementation_id"
                ),
                "kind": self.experiment.get("kind"),
                "seed": self.seed,
                "factors": self.experiment.get("factors", {}),
            },
            "training": {
                "regime": self.training.get("regime"),
                "dataset": self.training.get("dataset"),
                "split": self.training.get("split"),
                "protocol_id": self.training.get("protocol_id"),
            },
            "checkpoint": {
                "path": self._relative_path(checkpoint_path),
                "step": int(checkpoint_step),
                "sha256": self._checkpoint_sha256(checkpoint_path),
                "selection": selection_block,
            },
            "evaluation": {
                "dataset": self.evaluation.get("dataset"),
                "split": self.evaluation.get("split"),
                "data_version": self.evaluation.get("data_version"),
                "sample_list_sha256": self.dataset_info.get(
                    "sample_list_sha256", None
                ),
                "expected_samples": (
                    None if expected_samples is None
                    else int(expected_samples)
                ),
                "evaluated_samples": (
                    None if evaluated_samples is None
                    else int(evaluated_samples)
                ),
                "is_full": is_full,
                "protocol_id": self.evaluation.get("protocol_id"),
                "class_ids": class_ids,
                "class_names": class_names,
                "metric_class_ids": list(class_ids),
                "ignore_index": int(
                    self.evaluation.get("ignore_index", 255)
                ),
                "include_background": bool(
                    self.dataset_info.get("include_background", False)
                ),
                "absent_class_policy": (
                    "iou_union_positive_acc_target_positive"
                ),
                "geometry": {
                    "input_hw": input_hw,
                    "keep_ratio": bool(
                        self.evaluation.get("geometry", {}).get(
                            "keep_ratio", False
                        )
                    ),
                },
                "background_prob_threshold": self.evaluation.get(
                    "background_prob_threshold", None
                ),
                "postprocess": {
                    "prediction_prob_threshold": postprocess.get(
                        "prediction_prob_threshold", None
                    ),
                },
                "tta": {
                    "enabled": bool(tta_cfg.get("enabled", False)),
                    "views": build_tta_views(tta_cfg),
                },
            },
            "metrics": metrics,
            "efficiency": None,
            "provenance": {
                "source": "native",
                "git_commit": self.git_info.get("commit"),
                "git_dirty": self.git_info.get("dirty"),
                "git_untracked_code": self.git_info.get(
                    "untracked_code"
                ),
                "config_path": (
                    self._relative_path(self.config_source_path)
                    if self.config_source_path.exists()
                    else None
                ),
                "config_fingerprint": self.config_fingerprint,
                "attempt_id": attempt_id,
                "resume_from_step": resume_from_step,
            },
        }

        validate_record(
            record,
            require_completed=(record["status"] == "completed"),
        )

        path = self.results_dir / f"{eval_id}.json"
        atomic_write_json(path, record)
        print(f"Wrote standard result: {path}")
        return path
