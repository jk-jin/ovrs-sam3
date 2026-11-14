"""Standard experiment-result schema and file helpers.

This module intentionally depends only on the Python standard library so that
result collection, import, and paper export work without torch, CUDA, SAM3,
W&B, or any model package import.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = 1
RESULTS_DIRNAME = "results"

VALID_RECORD_TYPES = {"single_run", "reported_summary"}
VALID_STATUSES = {"planned", "running", "completed", "failed", "partial"}
VALID_ROLES = {"validation", "evaluation"}
VALID_SOURCES = {"native", "external_run", "reported"}
VALID_UNITS = {"fraction", "percent"}
VALID_REGIMES = {"supervised_adaptation", "training_free", "unknown"}
VALID_STD_DEFINITIONS = {"sample", "population", "unspecified"}

_MISSING = object()


class ResultValidationError(ValueError):
    pass


def get_path(obj, path: str, default=_MISSING):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            if default is _MISSING:
                raise KeyError(f"Missing result field path: {path!r}")
            return default
    return cur


def set_path(obj: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if part not in cur:
            cur[part] = {}
        if not isinstance(cur[part], dict):
            raise TypeError(
                f"Cannot set nested field {path!r}: {part!r} is not a dict."
            )
        cur = cur[part]
    cur[parts[-1]] = value


def canonical_json(obj) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json", prefix=".tmp_result_", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=path.suffix or ".txt",
        prefix=".tmp_export_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def uuid_short(length: int = 8) -> str:
    return uuid.uuid4().hex[:length]


def _sanitize_token(value, max_len: int = 48) -> str:
    text = str(value).strip()
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9_.=+-]+", "-", text)
    text = text.strip("-_.")
    if not text:
        text = "unknown"
    return text[:max_len].rstrip("-_.")


def make_eval_id(
    *,
    role: str,
    dataset: str,
    split: str,
    step,
    protocol_id,
    suffix: Optional[str] = None,
) -> str:
    step_token = "unknown" if step is None else str(step)
    return "_".join([
        _sanitize_token(role, 24),
        _sanitize_token(dataset, 32),
        _sanitize_token(split, 24),
        f"step{_sanitize_token(step_token, 24)}",
        _sanitize_token(protocol_id or "protocol-unknown", 48),
        _sanitize_token(suffix or uuid_short(), 16),
    ])


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_number_or_null(value) -> bool:
    return value is None or _is_number(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultValidationError(message)


def _require_keys(obj, keys: Iterable[str], name: str) -> None:
    _require(isinstance(obj, dict), f"{name} must be an object.")
    missing = [key for key in keys if key not in obj]
    _require(not missing, f"{name} is missing required fields: {missing}.")


def _validate_threshold(value, name: str) -> None:
    if value is None:
        return
    _require(
        _is_number(value) and 0.0 <= float(value) <= 1.0,
        f"{name} must be null or a finite value in [0,1].",
    )


def _validate_metrics_payload(metrics: dict, record_type: str) -> None:
    _require_keys(metrics, ["unit"], "metrics")
    unit = metrics["unit"]
    _require(
        unit in VALID_UNITS,
        f"metrics.unit must be one of {sorted(VALID_UNITS)}, got {unit!r}.",
    )

    upper = 1.0 if unit == "fraction" else 100.0
    for key in ("miou", "macc", "pixel_acc"):
        value = metrics.get(key, None)
        if value is None:
            continue
        _require(
            _is_number(value) and 0.0 <= float(value) <= upper,
            f"metrics.{key} must be null or in [0,{upper:g}] for unit "
            f"{unit!r}.",
        )

    per_class = metrics.get("per_class", None)
    if per_class is not None:
        _require(isinstance(per_class, list), "metrics.per_class must be a list.")
        for idx, row in enumerate(per_class):
            name = f"metrics.per_class[{idx}]"
            _require_keys(
                row, ["class_id", "class_name", "iou", "accuracy"], name
            )
            _require(_is_int(row["class_id"]), f"{name}.class_id must be int.")
            _require(
                isinstance(row["class_name"], str),
                f"{name}.class_name must be a string.",
            )
            for metric_name in ("iou", "accuracy"):
                value = row[metric_name]
                if value is None:
                    continue
                _require(
                    _is_number(value)
                    and 0.0 <= float(value) <= upper,
                    f"{name}.{metric_name} must be null or in [0,{upper:g}].",
                )
            for count_name in (
                "intersection", "union", "target_pixels"
            ):
                if count_name not in row:
                    continue
                value = row[count_name]
                _require(
                    _is_number(value) and float(value) >= 0.0,
                    f"{name}.{count_name} must be non-negative.",
                )

    if record_type == "reported_summary":
        std = metrics.get("std", None)
        if std is not None:
            _require(isinstance(std, dict), "metrics.std must be an object.")
            for key, value in std.items():
                if value is None:
                    continue
                _require(
                    _is_number(value)
                    and 0.0 <= float(value) <= upper,
                    f"metrics.std.{key} must be null or in [0,{upper:g}].",
                )
        n_seeds = metrics.get("n_seeds", None)
        if n_seeds is not None:
            _require(
                _is_int(n_seeds) and n_seeds >= 0,
                "metrics.n_seeds must be null or a non-negative int.",
            )
        std_definition = metrics.get("std_definition", None)
        if std_definition is not None:
            _require(
                std_definition in VALID_STD_DEFINITIONS,
                "metrics.std_definition must be sample, population, or "
                "unspecified.",
            )


def _validate_evaluation_block(
    evaluation: dict,
    *,
    native: bool,
) -> None:
    _require_keys(
        evaluation,
        [
            "dataset",
            "split",
            "data_version",
            "sample_list_sha256",
            "expected_samples",
            "evaluated_samples",
            "is_full",
            "protocol_id",
            "class_ids",
            "class_names",
            "metric_class_ids",
            "ignore_index",
            "include_background",
            "absent_class_policy",
            "geometry",
            "background_prob_threshold",
            "postprocess",
            "tta",
        ],
        "evaluation",
    )

    class_ids = evaluation["class_ids"]
    class_names = evaluation["class_names"]
    metric_class_ids = evaluation["metric_class_ids"]
    _require(isinstance(class_ids, list), "evaluation.class_ids must be a list.")
    _require(
        all(_is_int(x) for x in class_ids),
        "evaluation.class_ids must contain ints.",
    )
    _require(
        len(set(class_ids)) == len(class_ids),
        "evaluation.class_ids must be unique.",
    )
    _require(
        isinstance(class_names, list)
        and all(isinstance(x, str) for x in class_names),
        "evaluation.class_names must be a list of strings.",
    )
    _require(
        len(class_ids) == len(class_names),
        "evaluation.class_ids and class_names must have equal length.",
    )
    _require(
        isinstance(metric_class_ids, list)
        and all(_is_int(x) for x in metric_class_ids),
        "evaluation.metric_class_ids must be a list of ints.",
    )
    _require(
        set(metric_class_ids).issubset(set(class_ids)),
        "evaluation.metric_class_ids must be a subset of class_ids.",
    )

    for name in ("expected_samples", "evaluated_samples"):
        value = evaluation[name]
        _require(
            value is None or (_is_int(value) and value >= 0),
            f"evaluation.{name} must be null or a non-negative int.",
        )
    _require(
        isinstance(evaluation["is_full"], bool)
        or (not native and evaluation["is_full"] is None),
        "evaluation.is_full must be bool (or null for unknown external "
        "metadata).",
    )
    _require(
        isinstance(evaluation["include_background"], bool)
        or (
            not native
            and evaluation["include_background"] is None
        ),
        "evaluation.include_background must be bool (or null for unknown "
        "external metadata).",
    )
    _require(
        _is_int(evaluation["ignore_index"])
        or (not native and evaluation["ignore_index"] is None),
        "evaluation.ignore_index must be int (or null for unknown external "
        "metadata).",
    )

    geometry = evaluation["geometry"]
    _require(
        isinstance(geometry, dict) or (not native and geometry is None),
        "evaluation.geometry must be an object (or null for unknown "
        "external metadata).",
    )
    if isinstance(geometry, dict):
        input_hw = geometry.get("input_hw", None)
        if input_hw is not None:
            _require(
                isinstance(input_hw, list)
                and len(input_hw) == 2
                and all(_is_int(x) and x > 0 for x in input_hw),
                "evaluation.geometry.input_hw must be two positive ints.",
            )
        else:
            _require(
                not native,
                "Native records require evaluation.geometry.input_hw.",
            )
        keep_ratio = geometry.get("keep_ratio", None)
        _require(
            isinstance(keep_ratio, bool) or (not native and keep_ratio is None),
            "evaluation.geometry.keep_ratio must be bool (or null for "
            "unknown external metadata).",
        )
    else:
        _require(
            not native,
            "Native records require evaluation.geometry.",
        )

    _validate_threshold(
        evaluation["background_prob_threshold"],
        "evaluation.background_prob_threshold",
    )
    postprocess = evaluation["postprocess"]
    _require(
        isinstance(postprocess, dict),
        "evaluation.postprocess must be an object.",
    )
    _validate_threshold(
        postprocess.get("prediction_prob_threshold", None),
        "evaluation.postprocess.prediction_prob_threshold",
    )

    tta = evaluation["tta"]
    _require(
        isinstance(tta, dict) or (not native and tta is None),
        "evaluation.tta must be an object (or null for unknown external "
        "metadata).",
    )
    if isinstance(tta, dict):
        enabled = tta.get("enabled", None)
        _require(
            isinstance(enabled, bool)
            or (not native and enabled is None),
            "evaluation.tta.enabled must be bool (or null for unknown "
            "external metadata).",
        )
        views = tta.get("views", None)
        _require(
            isinstance(views, list)
            and all(isinstance(x, str) for x in views)
            and (len(views) > 0 or enabled is None),
            "evaluation.tta.views must be a list of strings; it may be "
            "empty only when tta.enabled is null.",
        )
    else:
        _require(
            not native,
            "Native records require evaluation.tta.",
        )


def _validate_checkpoint_block(checkpoint, record_type: str) -> None:
    if checkpoint is None:
        _require(
            record_type == "reported_summary",
            "checkpoint may be null only for reported_summary records.",
        )
        return

    _require(isinstance(checkpoint, dict), "checkpoint must be an object.")
    for key in ("path", "step", "sha256", "selection"):
        _require(key in checkpoint, f"checkpoint.{key} is required.")
    _require(
        checkpoint["path"] is None or isinstance(checkpoint["path"], str),
        "checkpoint.path must be null or a string.",
    )
    _require(
        checkpoint["step"] is None or _is_int(checkpoint["step"]),
        "checkpoint.step must be null or int.",
    )
    _require(
        checkpoint["sha256"] is None or isinstance(checkpoint["sha256"], str),
        "checkpoint.sha256 must be null or a string.",
    )
    selection = checkpoint["selection"]
    _require(isinstance(selection, dict), "checkpoint.selection must be an object.")
    for key in ("dataset", "split", "metric", "mode", "value"):
        _require(key in selection, f"checkpoint.selection.{key} is required.")
    _require(
        selection["mode"] in {None, "max", "min"},
        "checkpoint.selection.mode must be max, min, or null.",
    )
    _require(
        selection["value"] is None or _is_number(selection["value"]),
        "checkpoint.selection.value must be null or a number.",
    )


def validate_record(
    record: dict,
    *,
    require_completed: bool = False,
) -> None:
    _require(isinstance(record, dict), "Result record must be an object.")
    _require_keys(
        record,
        [
            "schema_version",
            "record_type",
            "eval_id",
            "status",
            "role",
            "experiment",
            "training",
            "checkpoint",
            "evaluation",
            "metrics",
            "provenance",
        ],
        "result",
    )

    _require(
        _is_int(record["schema_version"])
        and record["schema_version"] == SCHEMA_VERSION,
        f"Unsupported schema_version={record['schema_version']!r}; "
        f"this exporter supports {SCHEMA_VERSION}.",
    )
    _require(
        isinstance(record["eval_id"], str) and record["eval_id"].strip(),
        "eval_id must be a non-empty string.",
    )
    record_type = record["record_type"]
    _require(
        record_type in VALID_RECORD_TYPES,
        f"Unknown record_type={record_type!r}.",
    )
    status = record["status"]
    _require(
        status in VALID_STATUSES,
        f"Unknown status={status!r}.",
    )
    _require(
        record["role"] in VALID_ROLES,
        f"Unknown role={record['role']!r}.",
    )
    if require_completed:
        _require(
            status == "completed",
            f"Record {record.get('eval_id')!r} has status={status!r}, "
            "expected completed.",
        )

    experiment = record["experiment"]
    _require_keys(
        experiment,
        [
            "run_id",
            "name",
            "method_id",
            "method_name",
            "variant_id",
            "implementation_id",
            "kind",
            "seed",
            "factors",
        ],
        "experiment",
    )
    _require(
        experiment["seed"] is None or _is_int(experiment["seed"]),
        "experiment.seed must be null or int.",
    )
    _require(
        isinstance(experiment["factors"], dict),
        "experiment.factors must be an object.",
    )
    for key, value in experiment["factors"].items():
        _require(
            value is None
            or isinstance(value, (bool, int, float, str)),
            f"experiment.factors.{key} must be a scalar or null.",
        )

    training = record["training"]
    _require_keys(
        training,
        ["regime", "dataset", "split", "protocol_id"],
        "training",
    )
    _require(
        training["regime"] in VALID_REGIMES,
        f"Unknown training.regime={training['regime']!r}.",
    )

    provenance = record["provenance"]
    _require(isinstance(provenance, dict), "provenance must be an object.")
    _require(
        provenance.get("source", None) in VALID_SOURCES,
        "provenance.source must be native, external_run, or reported.",
    )
    native = provenance.get("source") == "native"

    _validate_checkpoint_block(record["checkpoint"], record_type)
    _validate_evaluation_block(
        record["evaluation"], native=native
    )
    _validate_metrics_payload(record["metrics"], record_type)

    eval_ids = record["evaluation"]["class_ids"]
    eval_names = record["evaluation"]["class_names"]
    id_to_name = {
        int(class_id): str(name)
        for class_id, name in zip(eval_ids, eval_names)
    }
    for idx, row in enumerate(
        record["metrics"].get("per_class", []) or []
    ):
        class_id = int(row["class_id"])
        _require(
            class_id in id_to_name,
            f"metrics.per_class[{idx}].class_id={class_id} is not in "
            "evaluation.class_ids.",
        )
        _require(
            str(row["class_name"]) == id_to_name[class_id],
            f"metrics.per_class[{idx}].class_name={row['class_name']!r} "
            f"does not match evaluation.class_names[{class_id}]="
            f"{id_to_name[class_id]!r}.",
        )

    if status == "completed":
        _require(
            any(
                record["metrics"].get(key, None) is not None
                for key in ("miou", "macc", "pixel_acc")
            ),
            "Completed records require at least one overall metric.",
        )
        _require(
            len(record["evaluation"]["class_ids"]) > 0
            and len(record["evaluation"]["class_names"]) > 0,
            "Completed records require an explicit evaluation class set.",
        )

    if (
        status == "completed"
        and provenance.get("source") == "native"
    ):
        _require(
            len(record["evaluation"]["class_ids"]) > 0,
            "Completed native records require a non-empty class set.",
        )
        _require(
            len(record["evaluation"]["metric_class_ids"]) > 0,
            "Completed native records require non-empty metric_class_ids.",
        )
        _require(
            _is_int(experiment["seed"]),
            "Completed native records require experiment.seed.",
        )
        _require(
            _is_int(record["checkpoint"]["step"]),
            "Completed native records require checkpoint.step.",
        )
        _require(
            record["evaluation"]["is_full"] is not None,
            "Completed native records require evaluation.is_full.",
        )
        _require(
            _is_int(record["evaluation"]["expected_samples"]),
            "Completed native records require evaluation.expected_samples.",
        )
        _require(
            _is_int(record["evaluation"]["evaluated_samples"]),
            "Completed native records require evaluation.evaluated_samples.",
        )


def normalize_record_units(record: dict) -> dict:
    """Convert percent metrics to fraction exactly once."""
    metrics = record.get("metrics", None)
    if not isinstance(metrics, dict):
        return record
    unit = metrics.get("unit", None)
    if unit not in VALID_UNITS:
        raise ResultValidationError(
            "metrics.unit is required and must be 'fraction' or 'percent'."
        )
    if unit == "fraction":
        return record

    def _convert(value):
        if value is None:
            return None
        return float(value) / 100.0

    for key in ("miou", "macc", "pixel_acc", "extra"):
        if key == "extra":
            continue
        if key in metrics:
            metrics[key] = _convert(metrics[key])

    if isinstance(metrics.get("std", None), dict):
        metrics["std"] = {
            k: _convert(v) for k, v in metrics["std"].items()
        }

    per_class = metrics.get("per_class", None)
    if isinstance(per_class, list):
        for row in per_class:
            for key in ("iou", "accuracy"):
                if key in row:
                    row[key] = _convert(row[key])

    metrics["unit"] = "fraction"
    return record


def result_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    results_dir = root / RESULTS_DIRNAME
    if results_dir.is_dir():
        return sorted(results_dir.glob("*.json"))
    return sorted(root.rglob(f"{RESULTS_DIRNAME}/*.json"))


def scan_results(
    roots: Iterable[str | Path],
) -> tuple[list[dict], list[str]]:
    """Scan standard result JSON files, validate, normalize, and dedupe."""
    records_by_id: dict[str, dict] = {}
    errors: list[str] = []
    seen_paths = set()

    for root in roots:
        for path in result_files(root):
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                record = load_json(path)
                record = normalize_record_units(record)
                validate_record(record)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue

            eval_id = record.get("eval_id", None)
            if not eval_id:
                errors.append(f"{path}: missing eval_id.")
                continue

            existing = records_by_id.get(eval_id)
            if existing is not None:
                existing_payload = {
                    k: v for k, v in existing.items()
                    if not str(k).startswith("_")
                }
                if canonical_json(existing_payload) == canonical_json(record):
                    continue
                errors.append(
                    f"{path}: conflicting duplicate eval_id={eval_id!r}."
                )
                continue

            record["_source_path"] = str(path)
            records_by_id[eval_id] = record

    records = sorted(
        records_by_id.values(),
        key=lambda r: (
            str(get_path(r, "evaluation.dataset", "")),
            str(get_path(r, "evaluation.split", "")),
            str(get_path(r, "experiment.method_id", "")),
            str(get_path(r, "experiment.variant_id", "")),
            -1 if get_path(r, "experiment.seed", None) is None
            else int(get_path(r, "experiment.seed")),
            str(r.get("eval_id", "")),
        ),
    )
    return records, errors


def mean_std(values: list[float], ddof: int = 1) -> tuple[Optional[float], Optional[float]]:
    clean = [float(x) for x in values if x is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) <= ddof:
        return mean, None
    variance = sum((x - mean) ** 2 for x in clean) / (len(clean) - ddof)
    return mean, math.sqrt(variance)


def escape_latex(text: object) -> str:
    value = str(text)
    placeholders = {
        "\x00": r"\textbackslash{}",
        "\x01": r"\textasciitilde{}",
        "\x02": r"\textasciicircum{}",
    }
    value = value.replace("\\", "\x00")
    value = value.replace("~", "\x01")
    value = value.replace("^", "\x02")
    for old, new in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        value = value.replace(old, new)
    for old, new in placeholders.items():
        value = value.replace(old, new)
    return value


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".csv", prefix=".tmp_csv_", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def format_scalar(value, decimals: int) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{decimals}f}"


def _json_cell(value) -> str:
    if value is None:
        return ""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def record_to_long_row(record: dict) -> dict:
    """Flatten one standard record into a long-table CSV row."""
    experiment = record["experiment"]
    evaluation = record["evaluation"]
    metrics = record["metrics"]
    checkpoint = record.get("checkpoint", None) or {}
    selection = checkpoint.get("selection", {}) or {}
    provenance = record.get("provenance", {}) or {}

    return {
        "eval_id": record.get("eval_id"),
        "record_type": record.get("record_type"),
        "role": record.get("role"),
        "status": record.get("status"),
        "method_id": experiment.get("method_id"),
        "method_name": experiment.get("method_name"),
        "variant_id": experiment.get("variant_id"),
        "implementation_id": experiment.get("implementation_id"),
        "kind": experiment.get("kind"),
        "seed": experiment.get("seed"),
        "factors": _json_cell(experiment.get("factors", {})),
        "training_regime": get_path(record, "training.regime", None),
        "training_dataset": get_path(record, "training.dataset", None),
        "training_split": get_path(record, "training.split", None),
        "training_protocol_id": get_path(
            record, "training.protocol_id", None
        ),
        "eval_dataset": evaluation.get("dataset"),
        "eval_split": evaluation.get("split"),
        "data_version": evaluation.get("data_version"),
        "eval_protocol_id": evaluation.get("protocol_id"),
        "is_full": evaluation.get("is_full"),
        "expected_samples": evaluation.get("expected_samples"),
        "evaluated_samples": evaluation.get("evaluated_samples"),
        "class_names": _json_cell(evaluation.get("class_names", [])),
        "background_prob_threshold": evaluation.get(
            "background_prob_threshold"
        ),
        "prediction_prob_threshold": get_path(
            record,
            "evaluation.postprocess.prediction_prob_threshold",
            None,
        ),
        "tta_enabled": get_path(
            record, "evaluation.tta.enabled", None
        ),
        "tta_views": _json_cell(
            get_path(record, "evaluation.tta.views", [])
        ),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_path": checkpoint.get("path"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "selection_dataset": selection.get("dataset"),
        "selection_split": selection.get("split"),
        "selection_metric": selection.get("metric"),
        "selection_mode": selection.get("mode"),
        "selection_value": selection.get("value"),
        "miou_fraction": metrics.get("miou"),
        "macc_fraction": metrics.get("macc"),
        "pixel_acc_fraction": metrics.get("pixel_acc"),
        "std_fraction": _json_cell(metrics.get("std")),
        "n_seeds": metrics.get("n_seeds"),
        "std_definition": metrics.get("std_definition"),
        "per_class": _json_cell(metrics.get("per_class", [])),
        "source": provenance.get("source"),
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
        "config_path": provenance.get("config_path"),
        "result_path": record.get("_source_path", ""),
        "notes": provenance.get("notes"),
    }
