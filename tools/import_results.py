from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_results import (  # noqa: E402
    atomic_write_json,
    canonical_json,
    get_path,
    load_json,
    make_eval_id,
    normalize_record_units,
    set_path,
    sha256_file,
    validate_record,
)


_COMPLEX_PATHS = (
    "evaluation.class_ids",
    "evaluation.class_names",
    "evaluation.metric_class_ids",
    "evaluation.geometry",
    "evaluation.tta",
    "evaluation.postprocess",
    "metrics.per_class",
    "metrics.std",
    "metrics.extra",
    "experiment.factors",
)

_BOOL_PATHS = {
    "evaluation.is_full",
    "evaluation.include_background",
    "evaluation.tta.enabled",
    "visualization.enabled",
}

_INT_PATHS = {
    "experiment.seed",
    "checkpoint.step",
    "evaluation.expected_samples",
    "evaluation.evaluated_samples",
    "evaluation.ignore_index",
    "metrics.n_seeds",
}

_FLOAT_PATHS = {
    "metrics.miou",
    "metrics.macc",
    "metrics.pixel_acc",
    "checkpoint.selection.value",
    "evaluation.background_prob_threshold",
    "evaluation.postprocess.reference_prob_threshold",
    "evaluation.postprocess.prediction_prob_threshold",
}

_STRING_PATHS = {
    "eval_id",
    "record_type",
    "status",
    "role",
    "experiment.run_id",
    "experiment.name",
    "experiment.method_id",
    "experiment.method_name",
    "experiment.variant_id",
    "experiment.implementation_id",
    "experiment.kind",
    "training.regime",
    "training.dataset",
    "training.split",
    "training.protocol_id",
    "checkpoint.path",
    "checkpoint.sha256",
    "checkpoint.selection.dataset",
    "checkpoint.selection.split",
    "checkpoint.selection.metric",
    "checkpoint.selection.mode",
    "evaluation.dataset",
    "evaluation.split",
    "evaluation.data_version",
    "evaluation.sample_list_sha256",
    "evaluation.protocol_id",
    "evaluation.absent_class_policy",
    "metrics.unit",
    "metrics.std_definition",
    "provenance.source",
    "provenance.original_file",
    "provenance.source_url",
    "provenance.doi",
    "provenance.citation",
    "provenance.notes",
    "provenance.table",
    "provenance.page",
}


class ImportErrorList(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def _is_complex(path: str) -> bool:
    return any(path == item or path.startswith(item + ".") for item in _COMPLEX_PATHS)


def _parse_bool(text: str, path: str):
    lowered = text.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "1":
        return True
    if lowered == "0":
        return False
    raise ValueError(
        f"{path}: expected true/false (or 1/0), got {text!r}."
    )


def _parse_scalar(text: str):
    raw = text.strip()
    if raw == "":
        return None
    lowered = raw.lower()
    if lowered in ("null", "none"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def coerce_csv_value(path: str, text: str):
    raw = text.strip()
    if raw == "":
        return None
    if _is_complex(path):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}: expected JSON value in CSV cell, got {raw!r}: {exc}"
            ) from exc
    if path in _BOOL_PATHS:
        return _parse_bool(raw, path)
    if path in _INT_PATHS:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{path}: expected an integer, got {raw!r}."
            ) from exc
    if path in _FLOAT_PATHS or path.startswith("metrics.std."):
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{path}: expected a number, got {raw!r}."
            ) from exc
        return value
    if path in _STRING_PATHS or path.startswith("experiment.factors."):
        if path.startswith("experiment.factors."):
            return _parse_scalar(raw)
        return raw
    return _parse_scalar(raw)


class RecordBuilder:
    _UNSET = object()

    def __init__(self):
        self.record: dict = {}
        self.origins: dict[str, str] = {}

    def put(self, path: str, value, origin: str) -> None:
        existing = get_path(self.record, path, default=self._UNSET)
        if existing is not self._UNSET:
            if existing != value:
                raise ValueError(
                    f"Conflicting value for {path!r}: "
                    f"{self.origins.get(path, 'earlier')}={existing!r} vs "
                    f"{origin}={value!r}."
                )
            return
        set_path(self.record, path, value)
        self.origins[path] = origin

    def merge_dict(self, obj: dict, origin: str) -> None:
        if not isinstance(obj, dict):
            raise TypeError(f"{origin} must provide a JSON object.")
        for key, value in obj.items():
            if key == "schema_version":
                continue
            if isinstance(value, dict):
                self._merge_nested(key, value, origin)
            else:
                self.put(key, value, origin)

    def _merge_nested(self, prefix: str, obj: dict, origin: str) -> None:
        for key, value in obj.items():
            path = f"{prefix}.{key}"
            if isinstance(value, dict):
                self._merge_nested(path, value, origin)
            else:
                self.put(path, value, origin)


def _apply_defaults(
    builder: RecordBuilder,
    *,
    default_source: str,
    input_unit: str | None,
) -> None:
    had_checkpoint = "checkpoint" in builder.record
    defaults = {
        "schema_version": 1,
        "experiment": {
            "run_id": None,
            "name": None,
            "method_id": None,
            "method_name": None,
            "variant_id": "unspecified",
            "implementation_id": "unspecified",
            "kind": "unspecified",
            "seed": None,
            "factors": {},
        },
        "training": {
            "regime": "unknown",
            "dataset": None,
            "split": None,
            "protocol_id": None,
        },
        "checkpoint": {
            "path": None,
            "step": None,
            "sha256": None,
            "selection": {
                "dataset": None,
                "split": None,
                "metric": None,
                "mode": None,
                "value": None,
            },
        },
        "evaluation": {
            "dataset": None,
            "split": None,
            "data_version": None,
            "sample_list_sha256": None,
            "expected_samples": None,
            "evaluated_samples": None,
            "is_full": None,
            "protocol_id": None,
            "class_ids": [],
            "class_names": [],
            "metric_class_ids": [],
            "ignore_index": None,
            "include_background": None,
            "absent_class_policy": "unknown",
            "geometry": None,
            "background_prob_threshold": None,
            "postprocess": {
                "reference_prob_threshold": None,
                "prediction_prob_threshold": None,
            },
            "tta": {
                "enabled": None,
                "views": [],
            },
        },
        "metrics": {
            "unit": input_unit,
            "miou": None,
            "macc": None,
            "pixel_acc": None,
            "per_class": [],
        },
        "provenance": {
            "source": default_source,
        },
    }
    for key, value in defaults.items():
        if key not in builder.record:
            builder.record[key] = value
    for path in (
        "experiment",
        "training",
        "checkpoint",
        "evaluation",
        "metrics",
        "provenance",
    ):
        for key, value in defaults[path].items():
            if key not in builder.record[path]:
                builder.record[path][key] = value

    if (
        builder.record.get("record_type") == "reported_summary"
        and not had_checkpoint
    ):
        builder.record["checkpoint"] = None

    if not builder.record["metrics"].get("unit", None):
        if input_unit is None:
            raise ValueError(
                "metrics.unit is missing. Pass --input-unit for CSV input "
                "or declare metrics.unit in the standard JSON."
            )
        builder.record["metrics"]["unit"] = input_unit


def _finalize_record(
    builder: RecordBuilder,
    *,
    input_path: Path,
    mapping_path: Path | None,
    protocol_path: Path | None,
) -> dict:
    record = builder.record

    record["schema_version"] = 1
    if not record.get("eval_id", None):
        record["eval_id"] = make_eval_id(
            role=str(record.get("role", "evaluation")),
            dataset=str(get_path(record, "evaluation.dataset", "unknown")),
            split=str(get_path(record, "evaluation.split", "unknown")),
            step=get_path(record, "checkpoint.step", None),
            protocol_id=get_path(
                record, "evaluation.protocol_id", "protocol-unknown"
            ),
        )

    if record.get("status", None) is None:
        metrics = record.get("metrics", {})
        has_metric = any(
            metrics.get(key, None) is not None
            for key in ("miou", "macc", "pixel_acc")
        )
        record["status"] = "completed" if has_metric else "planned"

    metrics = record["metrics"]
    if not metrics.get("per_class", None):
        metrics.pop("per_class", None)

    if not record["evaluation"].get("class_ids", None) and record[
        "evaluation"
    ].get("class_names", None):
        class_names = list(record["evaluation"]["class_names"])
        record["evaluation"]["class_ids"] = list(range(len(class_names)))
        if not record["evaluation"].get("metric_class_ids", None):
            record["evaluation"]["metric_class_ids"] = list(
                record["evaluation"]["class_ids"]
            )

    provenance = record["provenance"]
    provenance["original_file"] = str(input_path)
    provenance["original_sha256"] = sha256_file(input_path)
    if mapping_path is not None:
        provenance["mapping_file"] = str(mapping_path)
        provenance["mapping_sha256"] = sha256_file(mapping_path)
    if protocol_path is not None:
        provenance["protocol_file"] = str(protocol_path)
        provenance["protocol_sha256"] = sha256_file(protocol_path)
    provenance["imported_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )

    if not record["experiment"].get("name", None):
        record["experiment"]["name"] = (
            record["experiment"].get("method_id")
            or "imported"
        )
    if not record["experiment"].get("method_name", None):
        record["experiment"]["method_name"] = (
            record["experiment"].get("method_id")
            or "Imported method"
        )
    if not record["experiment"].get("run_id", None):
        record["experiment"]["run_id"] = record["eval_id"]

    return record


def _load_json_mapping(path: Path) -> dict:
    mapping = load_json(path)
    if not isinstance(mapping, dict):
        raise TypeError(f"Mapping file must contain a JSON object: {path}")
    if mapping.get("schema_version", 1) != 1:
        raise ValueError(
            f"Unsupported mapping schema_version in {path}: "
            f"{mapping.get('schema_version')!r}"
        )
    columns = mapping.get("columns", {})
    constants = mapping.get("constants", {})
    if not isinstance(columns, dict):
        raise TypeError("Mapping 'columns' must be an object.")
    if not isinstance(constants, dict):
        raise TypeError("Mapping 'constants' must be an object.")
    return mapping


def _load_protocol(path: Path | None) -> dict | None:
    if path is None:
        return None
    protocol = load_json(path)
    if not isinstance(protocol, dict):
        raise TypeError(f"Protocol file must contain a JSON object: {path}")
    if protocol.get("schema_version", 1) != 1:
        raise ValueError(
            f"Unsupported protocol schema_version in {path}: "
            f"{protocol.get('schema_version')!r}"
        )
    return protocol


def _apply_mapping_constants(
    builder: RecordBuilder,
    mapping: dict,
) -> None:
    for path, value in mapping.get("constants", {}).items():
        builder.put(path, value, "mapping.constants")


def _build_from_source_object(
    *,
    source_obj: dict,
    mapping: dict,
    protocol: dict | None,
    input_unit: str | None,
    default_source: str,
    input_path: Path,
    mapping_path: Path,
    protocol_path: Path | None,
    origin_prefix: str,
) -> dict:
    builder = RecordBuilder()
    if protocol is not None:
        builder.merge_dict(protocol, "protocol")
    _apply_mapping_constants(builder, mapping)

    for target, source_path in mapping.get("columns", {}).items():
        value = get_path(source_obj, source_path)
        builder.put(target, value, f"{origin_prefix}:{source_path}")

    _apply_defaults(
        builder,
        default_source=default_source,
        input_unit=input_unit,
    )

    declared_unit = get_path(builder.record, "metrics.unit", None)
    if declared_unit is None:
        raise ValueError("metrics.unit is missing after applying the mapping.")
    if input_unit is not None and str(declared_unit) != str(input_unit):
        raise ValueError(
            "metrics.unit conflicts with --input-unit: "
            f"{declared_unit!r} vs {input_unit!r}."
        )

    record = _finalize_record(
        builder,
        input_path=input_path,
        mapping_path=mapping_path,
        protocol_path=protocol_path,
    )
    record = normalize_record_units(record)
    validate_record(record)
    return record


def _build_from_csv_row(
    *,
    row: dict,
    header: list[str],
    mapping: dict,
    protocol: dict | None,
    input_unit: str,
    input_path: Path,
    mapping_path: Path,
    protocol_path: Path | None,
    row_number: int,
) -> dict:
    builder = RecordBuilder()
    if protocol is not None:
        builder.merge_dict(protocol, "protocol")
    _apply_mapping_constants(builder, mapping)

    mapped_source_columns = set()
    for target, source_column in mapping.get("columns", {}).items():
        if source_column not in header:
            raise ValueError(
                f"mapping column {source_column!r} for target {target!r} "
                f"is missing from the CSV header."
            )
        mapped_source_columns.add(source_column)
        raw = row.get(source_column, "")
        if raw is None or str(raw).strip() == "":
            continue
        builder.put(
            target,
            coerce_csv_value(target, str(raw)),
            f"csv row {row_number} column {source_column!r}",
        )

    for column in header:
        if not column.startswith("factor."):
            continue
        if column in mapped_source_columns:
            continue
        factor_name = column[len("factor."):]
        if not factor_name:
            raise ValueError(
                f"CSV column {column!r} has an empty factor name."
            )
        raw = row.get(column, "")
        value = None if str(raw).strip() == "" else _parse_scalar(
            str(raw)
        )
        builder.put(
            f"experiment.factors.{factor_name}",
            value,
            f"csv row {row_number} factor column {column!r}",
        )

    _apply_defaults(
        builder,
        default_source="external_run",
        input_unit=input_unit,
    )
    declared_unit = get_path(builder.record, "metrics.unit", None)
    if str(declared_unit) != str(input_unit):
        raise ValueError(
            "metrics.unit conflicts with --input-unit: "
            f"{declared_unit!r} vs {input_unit!r}."
        )

    record = _finalize_record(
        builder,
        input_path=input_path,
        mapping_path=mapping_path,
        protocol_path=protocol_path,
    )
    record = normalize_record_units(record)
    validate_record(record)
    return record


def _load_standard_json(path: Path, input_unit: str | None) -> list[dict]:
    payload = load_json(path)
    items = payload if isinstance(payload, list) else [payload]
    records = []
    for idx, record in enumerate(items):
        if not isinstance(record, dict):
            raise ValueError(
                f"Standard JSON item {idx} must be an object."
            )
        metrics = record.get("metrics", None)
        if not isinstance(metrics, dict):
            raise ValueError(
                f"Standard JSON item {idx} is missing metrics."
            )
        declared_unit = metrics.get("unit", None)
        if declared_unit is None:
            if input_unit is None:
                raise ValueError(
                    "metrics.unit must be declared in standard JSON input."
                )
            metrics["unit"] = input_unit
        elif input_unit is not None and str(declared_unit) != str(input_unit):
            raise ValueError(
                "metrics.unit conflicts with --input-unit: "
                f"{declared_unit!r} vs {input_unit!r}."
            )
        record = normalize_record_units(record)
        validate_record(record)
        if not record.get("eval_id", None):
            raise ValueError(
                f"Standard JSON item {idx} is missing eval_id."
            )
        records.append(record)
    return records


def _safe_eval_filename(eval_id: str) -> str:
    token = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in str(eval_id)
    ).strip("._-")
    return token or "imported_result"


def _write_imported(
    records: list[dict],
    output_dir: Path,
) -> list[Path]:
    results_dir = output_dir / "results"
    planned = []
    seen = set()
    for record in records:
        eval_id = record["eval_id"]
        if eval_id in seen:
            raise ValueError(f"Duplicate eval_id in import batch: {eval_id}")
        seen.add(eval_id)

        path = results_dir / f"{_safe_eval_filename(eval_id)}.json"
        if path.exists():
            existing = load_json(path)
            if canonical_json(existing) == canonical_json(record):
                print(f"Unchanged existing result: {path}")
                continue
            raise FileExistsError(
                f"Refusing to overwrite conflicting imported result: {path}"
            )
        planned.append((path, record))

    written = []
    for path, record in planned:
        atomic_write_json(path, record)
        written.append(path)
        print(f"Wrote imported result: {path}")
    return written


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Import external CSV/JSON experiment results into the standard "
            "single-run JSON schema."
        )
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="JSON mapping: target field paths -> source columns/paths.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        help="JSON partial standard record with shared metadata.",
    )
    parser.add_argument(
        "--input-unit",
        choices=["fraction", "percent"],
        default=None,
        help=(
            "Unit of the input metrics. Required for CSV; optional for JSON "
            "when metrics.unit is already declared."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    mapping_path = Path(args.mapping).resolve() if args.mapping else None
    protocol_path = Path(args.protocol).resolve() if args.protocol else None
    output_dir = Path(args.output_dir).resolve()

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        protocol = _load_protocol(protocol_path)

        if input_path.suffix.lower() == ".csv":
            if mapping_path is None:
                raise ValueError("CSV import requires --mapping.")
            if args.input_unit is None:
                raise ValueError(
                    "CSV import requires --input-unit (fraction or percent)."
                )
            mapping = _load_json_mapping(mapping_path)

            with input_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                header = list(reader.fieldnames or [])
                if not header:
                    raise ValueError("CSV input has no header row.")
                records = [
                    _build_from_csv_row(
                        row=row,
                        header=header,
                        mapping=mapping,
                        protocol=protocol,
                        input_unit=args.input_unit,
                        input_path=input_path,
                        mapping_path=mapping_path,
                        protocol_path=protocol_path,
                        row_number=idx,
                    )
                    for idx, row in enumerate(reader, start=2)
                ]
        elif input_path.suffix.lower() == ".json":
            if mapping_path is not None:
                mapping = _load_json_mapping(mapping_path)
                payload = load_json(input_path)
                items = payload if isinstance(payload, list) else [payload]
                records = []
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"JSON input item {idx} must be an object."
                        )
                    try:
                        records.append(
                            _build_from_source_object(
                                source_obj=item,
                                mapping=mapping,
                                protocol=protocol,
                                input_unit=args.input_unit,
                                default_source="external_run",
                                input_path=input_path,
                                mapping_path=mapping_path,
                                protocol_path=protocol_path,
                                origin_prefix=f"json item {idx}",
                            )
                        )
                    except (KeyError, ValueError, TypeError) as exc:
                        raise ValueError(
                            f"JSON input item {idx}: {exc}"
                        ) from exc
            else:
                records = _load_standard_json(
                    input_path, args.input_unit
                )
        else:
            raise ValueError(
                f"Unsupported input extension {input_path.suffix!r}; "
                "expected .csv or .json."
            )

        if not records:
            raise ValueError("No records were imported.")

        _write_imported(records, output_dir)
        print(f"Imported {len(records)} record(s).")

    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
