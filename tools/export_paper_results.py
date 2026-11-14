from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_results import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    escape_latex,
    format_scalar,
    get_path,
    load_json,
    mean_std,
    read_jsonl,
    record_to_long_row,
    scan_results,
    sha256_file,
    write_csv,
)


FIXED_FIELDS = [
    "evaluation.dataset",
    "evaluation.split",
    "evaluation.data_version",
    "evaluation.protocol_id",
    "evaluation.class_ids",
    "evaluation.class_names",
    "evaluation.metric_class_ids",
    "evaluation.ignore_index",
    "evaluation.include_background",
    "evaluation.absent_class_policy",
    "evaluation.expected_samples",
    "evaluation.evaluated_samples",
    "evaluation.sample_list_sha256",
    "evaluation.geometry.input_hw",
    "evaluation.geometry.keep_ratio",
    "evaluation.background_prob_threshold",
    "evaluation.postprocess.prediction_prob_threshold",
    "evaluation.tta.enabled",
    "evaluation.tta.views",
    "training.regime",
    "training.dataset",
    "training.split",
    "training.protocol_id",
]

TABLE_KEYS = {
    "name",
    "filter",
    "rows",
    "metrics",
    "allowed_varying_fields",
    "display_unit",
    "decimals",
    "aggregate_seeds",
    "std_ddof",
    "expected_seeds",
    "factor_columns",
    "layout",
    "caption",
    "label",
    "bold_best",
}

CURVE_KEYS = {
    "name",
    "type",
    "run",
    "runs",
    "filter",
    "match",
    "x_factor",
    "y",
    "label",
    "smooth_window",
    "multi_seed",
    "group_by",
    "display_unit",
    "save_png",
}

REGIME_LABELS = {
    "supervised_adaptation": "有监督适配",
    "training_free": "免训练",
    "unknown": "未知",
}


class ExportError(ValueError):
    pass


def _sanitize_filename(name: str) -> str:
    text = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in str(name)
    ).strip("_")
    if not text:
        raise ExportError("Table/curve name must not be empty.")
    return text


def _matches(record: dict, match: dict) -> bool:
    for path, expected in match.items():
        actual = get_path(record, path, None)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _is_allowed(path: str, allowed: list[str]) -> bool:
    for item in allowed:
        if path == item:
            return True
        if path.startswith(item + ".") or item.startswith(path + "."):
            return True
    return False


def _record_is_formal(record: dict, allow_incomplete: bool) -> bool:
    if allow_incomplete:
        return (
            record.get("status") in {"completed", "partial"}
            and record.get("role") == "evaluation"
        )
    return (
        record.get("status") == "completed"
        and record.get("role") == "evaluation"
        and get_path(record, "evaluation.is_full", None) is True
    )


def _resolve_metric(record: dict, metric: str):
    metrics = record["metrics"]
    if metric in {"miou", "macc", "pixel_acc"}:
        return metrics.get(metric, None)

    per_class = metrics.get("per_class", []) or []
    name = None
    field = None
    if metric.startswith("iou."):
        field = "iou"
        name = metric[len("iou."):]
    elif metric.startswith("accuracy."):
        field = "accuracy"
        name = metric[len("accuracy."):]
    elif metric.startswith("per_class.") and metric.endswith(".iou"):
        name = metric[len("per_class."):-len(".iou")]
        field = "iou"
    elif metric.startswith("per_class.") and metric.endswith(".accuracy"):
        name = metric[len("per_class."):-len(".accuracy")]
        field = "accuracy"
    else:
        raise ExportError(
            f"Unsupported table metric {metric!r}. Use miou, macc, "
            "pixel_acc, iou.<class_name>, or accuracy.<class_name>."
        )

    for row in per_class:
        if str(row.get("class_name", "")) == name:
            return row.get(field, None)
    return None


def _resolve_metric_std(record: dict, metric: str):
    if metric in {"miou", "macc", "pixel_acc"}:
        std = record["metrics"].get("std", None)
        if isinstance(std, dict):
            return std.get(metric, None)
    return None


def _metric_display_name(metric: str) -> str:
    if metric == "miou":
        return "mIoU"
    if metric == "macc":
        return "mAcc"
    if metric == "pixel_acc":
        return "Pixel Acc"
    if metric.startswith("iou."):
        return f"IoU-{metric[len('iou.'):]}"
    if metric.startswith("accuracy."):
        return f"Acc-{metric[len('accuracy.'):]}"
    if metric.startswith("per_class.") and metric.endswith(".iou"):
        return f"IoU-{metric[len('per_class.'):-len('.iou')]}"
    if metric.startswith("per_class.") and metric.endswith(".accuracy"):
        return f"Acc-{metric[len('per_class.'):-len('.accuracy')]}"
    return metric


def _to_display(value, unit: str):
    if value is None:
        return None
    return float(value) * 100.0 if unit == "percent" else float(value)


def _unit_suffix(unit: str) -> str:
    return "percent" if unit == "percent" else "fraction"


def _check_table_spec(table: dict) -> None:
    if not isinstance(table, dict):
        raise ExportError("Each table spec must be an object.")
    unknown = sorted(set(table) - TABLE_KEYS)
    if unknown:
        raise ExportError(f"Unknown table keys: {unknown}.")
    name = str(table.get("name", "")).strip()
    if not name:
        raise ExportError("Table name must not be empty.")
    if not isinstance(table.get("filter", {}), dict):
        raise ExportError(f"Table {name!r}: filter must be an object.")
    if not isinstance(table.get("rows", None), list) or not table["rows"]:
        raise ExportError(f"Table {name!r}: rows must be a non-empty list.")
    for idx, row in enumerate(table["rows"]):
        if not isinstance(row, dict) or "label" not in row:
            raise ExportError(
                f"Table {name!r}: rows[{idx}] must contain a label."
            )
        if not isinstance(row.get("match", {}), dict):
            raise ExportError(
                f"Table {name!r}: rows[{idx}].match must be an object."
            )
    metrics = table.get("metrics", None)
    if not isinstance(metrics, list) or not metrics:
        raise ExportError(f"Table {name!r}: metrics must be a non-empty list.")
    if not isinstance(table.get("allowed_varying_fields", []), list):
        raise ExportError(
            f"Table {name!r}: allowed_varying_fields must be a list."
        )
    if table.get("display_unit", "percent") not in {"fraction", "percent"}:
        raise ExportError(
            f"Table {name!r}: display_unit must be fraction or percent."
        )
    decimals = table.get("decimals", 2)
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ExportError(f"Table {name!r}: decimals must be a non-negative int.")
    if not isinstance(table.get("aggregate_seeds", True), bool):
        raise ExportError(f"Table {name!r}: aggregate_seeds must be bool.")
    if table.get("std_ddof", 1) != 1:
        raise ExportError(
            f"Table {name!r}: std_ddof must be 1 (sample standard deviation)."
        )
    expected = table.get("expected_seeds", None)
    if expected is not None and (
        not isinstance(expected, list)
        or any(
            isinstance(x, bool) or not isinstance(x, int) for x in expected
        )
    ):
        raise ExportError(
            f"Table {name!r}: expected_seeds must be a list of ints."
        )
    factors = table.get("factor_columns", [])
    if not isinstance(factors, list) or any(
        not isinstance(x, str) for x in factors
    ):
        raise ExportError(
            f"Table {name!r}: factor_columns must be a list of field paths."
        )
    if table.get("layout", "column") not in {"column", "wide"}:
        raise ExportError(
            f"Table {name!r}: layout must be 'column' or 'wide'."
        )
    if not isinstance(table.get("caption", ""), str) or not str(
        table.get("caption", "")
    ).strip():
        raise ExportError(f"Table {name!r}: caption must be a non-empty string.")
    if not isinstance(table.get("label", ""), str) or not str(
        table.get("label", "")
    ).strip():
        raise ExportError(f"Table {name!r}: label must be a non-empty string.")
    if not isinstance(table.get("bold_best", False), bool):
        raise ExportError(f"Table {name!r}: bold_best must be bool.")


def _check_curve_spec(curve: dict) -> None:
    if not isinstance(curve, dict):
        raise ExportError("Each curve spec must be an object.")
    unknown = sorted(set(curve) - CURVE_KEYS)
    if unknown:
        raise ExportError(f"Unknown curve keys: {unknown}.")
    name = str(curve.get("name", "")).strip()
    if not name:
        raise ExportError("Curve name must not be empty.")
    curve_type = curve.get("type", "metrics")
    if curve_type not in {"metrics", "factor"}:
        raise ExportError(
            f"Curve {name!r}: type must be 'metrics' or 'factor'."
        )
    if curve_type == "metrics":
        if not curve.get("run", None) and not curve.get("runs", None):
            raise ExportError(
                f"Curve {name!r}: metrics curve requires run or runs."
            )
        if not curve.get("y", None):
            raise ExportError(f"Curve {name!r}: y is required.")
    else:
        if not curve.get("x_factor", None):
            raise ExportError(f"Curve {name!r}: x_factor is required.")
        if not curve.get("y", None):
            raise ExportError(f"Curve {name!r}: y is required.")
    if curve.get("multi_seed", "separate") not in {
        "separate",
        "mean_band",
    }:
        raise ExportError(
            f"Curve {name!r}: multi_seed must be separate or mean_band."
        )
    smooth = curve.get("smooth_window", None)
    if smooth is not None and (
        isinstance(smooth, bool)
        or not isinstance(smooth, int)
        or smooth <= 1
    ):
        raise ExportError(
            f"Curve {name!r}: smooth_window must be null or an int > 1."
        )


def _check_protocol_compatibility(
    records: list[dict],
    allowed: list[str],
    strict: bool,
    warnings: list[str],
    table_name: str,
) -> None:
    if not records:
        return

    for path in FIXED_FIELDS:
        values = [get_path(record, path, None) for record in records]
        distinct = []
        for value in values:
            if value not in distinct:
                distinct.append(value)

        if len(distinct) > 1:
            if _is_allowed(path, allowed):
                continue
            raise ExportError(
                f"Table {table_name!r}: protocol field {path!r} differs "
                f"across selected records: {distinct!r}. Add it to "
                "allowed_varying_fields only if the table intentionally "
                "compares this difference."
            )

        if distinct and distinct[0] is None:
            message = (
                f"Table {table_name!r}: protocol field {path!r} is "
                "unknown (null) for selected records."
            )
            if _is_allowed(path, allowed):
                continue
            if strict:
                raise ExportError(message + " Use --allow-incomplete for a draft.")
            warnings.append(message)


def _check_full_sample_counts(
    records: list[dict],
    strict: bool,
    warnings: list[str],
    table_name: str,
) -> None:
    for record in records:
        if get_path(record, "evaluation.is_full", None) is not True:
            continue
        expected = get_path(
            record, "evaluation.expected_samples", None
        )
        evaluated = get_path(
            record, "evaluation.evaluated_samples", None
        )
        if expected is None or evaluated is None:
            continue
        if int(expected) != int(evaluated):
            message = (
                f"Table {table_name!r}: record {record.get('eval_id')!r} "
                f"is_full=true but expected_samples={expected} and "
                f"evaluated_samples={evaluated} differ."
            )
            if strict:
                raise ExportError(
                    message + " Use --allow-incomplete for a draft."
                )
            warnings.append(message)


def _check_native_class_coverage(
    records: list[dict],
    strict: bool,
    warnings: list[str],
    table_name: str,
) -> None:
    for record in records:
        if get_path(record, "provenance.source", None) != "native":
            continue
        metric_ids = get_path(
            record, "evaluation.metric_class_ids", []
        )
        valid = get_path(
            record, "metrics.extra.valid_iou_classes", None
        )
        if valid is None:
            continue
        if int(valid) != len(metric_ids):
            message = (
                f"Table {table_name!r}: native record "
                f"{record.get('eval_id')!r} has valid_iou_classes={valid} "
                f"but metric_class_ids has {len(metric_ids)} classes."
            )
            if strict:
                raise ExportError(
                    message + " Use --allow-incomplete for a draft."
                )
            warnings.append(message)


def _row_identity_check(records: list[dict], table_name: str, label: str) -> None:
    for path in (
        "experiment.method_id",
        "experiment.variant_id",
        "experiment.implementation_id",
        "experiment.factors",
    ):
        raw_values = [get_path(record, path, None) for record in records]
        values = {
            canonical_json(value)
            if isinstance(value, (dict, list)) else value
            for value in raw_values
        }
        if len(values) > 1:
            raise ExportError(
                f"Table {table_name!r} row {label!r}: {path} differs "
                f"within the row: {sorted(map(str, values))}."
            )


def _factor_value(
    records: list[dict],
    path: str,
    table_name: str,
    label: str,
):
    values = {get_path(record, path, None) for record in records}
    if len(values) > 1:
        raise ExportError(
            f"Table {table_name!r} row {label!r}: factor column {path!r} "
            "differs across seeds; aggregate_seeds cannot collapse it."
        )
    return values.pop() if values else None


def _row_constant_value(
    records: list[dict],
    path: str,
    table_name: str,
    label: str,
):
    raw_values = [get_path(record, path, None) for record in records]
    keys = {
        canonical_json(value)
        if isinstance(value, (dict, list)) else value
        for value in raw_values
    }
    if len(keys) > 1:
        raise ExportError(
            f"Table {table_name!r} row {label!r}: allowed field {path!r} "
            "differs across seeds."
        )
    return raw_values[0] if raw_values else None


def _single_run_values(
    records: list[dict],
    metric: str,
    aggregate: bool,
    strict: bool,
    warnings: list[str],
    table_name: str,
    label: str,
) -> list[dict]:
    by_seed = defaultdict(list)
    for record in records:
        seed = get_path(record, "experiment.seed", None)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ExportError(
                f"Table {table_name!r} row {label!r}: single_run record "
                f"{record.get('eval_id')!r} has no integer seed."
            )
        by_seed[seed].append(record)

    for seed, seed_records in by_seed.items():
        if len(seed_records) > 1:
            raise ExportError(
                f"Table {table_name!r} row {label!r}: seed {seed} has "
                "multiple candidate evaluations; select an explicit "
                "eval_id or checkpoint step before export."
            )

    out = []
    for seed in sorted(by_seed):
        record = by_seed[seed][0]
        value = _resolve_metric(record, metric)
        if value is None and strict:
            raise ExportError(
                f"Table {table_name!r} row {label!r}: metric {metric!r} "
                f"is missing for seed {seed}. Use --allow-incomplete for "
                "a draft."
            )
        out.append({
            "seed": seed,
            "value": value,
            "std": None,
            "n": 1,
            "record_type": "single_run",
            "source": get_path(record, "provenance.source", None),
            "regime": get_path(record, "training.regime", None),
        })

    if not out:
        return [{
            "seed": None,
            "value": None,
            "std": None,
            "n": 0,
            "record_type": "single_run",
            "source": None,
            "regime": None,
        }]

    if aggregate:
        mean, std = mean_std(
            [row["value"] for row in out if row["value"] is not None],
            ddof=1,
        )
        return [{
            "seed": None,
            "value": mean,
            "std": std,
            "n": len(out),
            "record_type": "single_run",
            "source": out[0]["source"],
            "regime": out[0]["regime"],
        }]

    return out


def _reported_values(
    records: list[dict],
    metric: str,
    strict: bool,
    table_name: str,
    label: str,
) -> list[dict]:
    if len(records) != 1:
        raise ExportError(
            f"Table {table_name!r} row {label!r}: reported_summary rows "
            "must select exactly one record. Do not mix or average "
            "reported means/stds with raw seeds."
        )
    record = records[0]
    value = _resolve_metric(record, metric)
    std = _resolve_metric_std(record, metric)
    if value is None and strict:
        raise ExportError(
            f"Table {table_name!r} row {label!r}: reported metric "
            f"{metric!r} is missing."
        )
    return [{
        "seed": None,
        "value": value,
        "std": std,
        "n": record["metrics"].get("n_seeds", None),
        "record_type": "reported_summary",
        "source": get_path(record, "provenance.source", None),
        "regime": get_path(record, "training.regime", None),
    }]


def _build_table(
    table: dict,
    records: list[dict],
    strict: bool,
    warnings: list[str],
    report: dict,
) -> dict:
    name = table["name"]
    filter_spec = table.get("filter", {})
    allowed = list(table.get("allowed_varying_fields", []))
    display_unit = table.get("display_unit", "percent")
    decimals = int(table.get("decimals", 2))
    aggregate = bool(table.get("aggregate_seeds", True))
    expected_seeds = table.get("expected_seeds", None)
    factor_columns = list(table.get("factor_columns", []))
    allowed_columns_spec = [
        path for path in allowed
        if path not in factor_columns and path != "training.regime"
    ]
    metrics = list(table["metrics"])

    base_records = [
        record for record in records
        if _matches(record, filter_spec)
    ]
    if not base_records:
        raise ExportError(
            f"Table {name!r}: filter matched no formal results."
        )

    selected_for_protocol = []
    display_rows = []
    adopted_ids = []

    for row_spec in table["rows"]:
        label = str(row_spec["label"])
        row_records = [
            record for record in base_records
            if _matches(record, row_spec.get("match", {}))
        ]
        if not row_records:
            if strict:
                raise ExportError(
                    f"Table {name!r} row {label!r}: match selected no "
                    "records."
                )
            warnings.append(
                f"Table {name!r} row {label!r}: match selected no records."
            )
            display_rows.append({
                "label": label,
                "factors": {
                    path: None for path in factor_columns
                },
                "allowed_values": {
                    path: None for path in allowed_columns_spec
                },
                "metrics": {
                    metric: {
                        "value": None,
                        "std": None,
                        "n": 0,
                    }
                    for metric in metrics
                },
                "seed": None,
                "regime": None,
                "source": None,
            })
            continue

        record_types = {record.get("record_type") for record in row_records}
        if len(record_types) > 1:
            raise ExportError(
                f"Table {name!r} row {label!r}: mixes record types "
                f"{sorted(record_types)}. Do not average reported summaries "
                "with raw single runs."
            )
        record_type = record_types.pop()
        if record_type == "reported_summary" and expected_seeds is not None:
            raise ExportError(
                f"Table {name!r} row {label!r}: expected_seeds is not "
                "applicable to reported_summary."
            )

        _row_identity_check(row_records, name, label)
        selected_for_protocol.extend(row_records)
        adopted_ids.extend(
            record.get("eval_id") for record in row_records
        )

        factors = {
            path: _factor_value(row_records, path, name, label)
            for path in factor_columns
        }

        if record_type == "reported_summary":
            per_metric = {
                metric: _reported_values(
                    row_records, metric, strict, name, label
                )[0]
                for metric in metrics
            }
            display_rows.append({
                "label": label,
                "factors": factors,
                "allowed_values": {
                    path: _row_constant_value(
                        row_records, path, name, label
                    )
                    for path in allowed_columns_spec
                },
                "metrics": per_metric,
                "seed": None,
                "regime": get_path(row_records[0], "training.regime", None),
                "source": get_path(
                    row_records[0], "provenance.source", None
                ),
            })
            continue

        seeds = sorted({
            get_path(record, "experiment.seed", None)
            for record in row_records
        }, key=lambda x: (-1 if x is None else x))
        if expected_seeds is not None:
            expected_set = set(expected_seeds)
            actual_set = set(seeds)
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            if missing and strict:
                raise ExportError(
                    f"Table {name!r} row {label!r}: missing expected "
                    f"seeds {missing}."
                )
            if missing:
                warnings.append(
                    f"Table {name!r} row {label!r}: missing expected "
                    f"seeds {missing}."
                )
            if extra:
                raise ExportError(
                    f"Table {name!r} row {label!r}: unexpected seeds "
                    f"{extra} outside expected_seeds {expected_seeds}."
                )

        per_metric_rows = {
            metric: _single_run_values(
                row_records,
                metric,
                aggregate,
                strict,
                warnings,
                name,
                label,
            )
            for metric in metrics
        }

        if not aggregate:
            seed_values = sorted({
                row["seed"] for rows in per_metric_rows.values()
                for row in rows
                if row["seed"] is not None
            })
            for seed in seed_values:
                display_rows.append({
                    "label": f"{label} (seed={seed})",
                    "factors": factors,
                    "allowed_values": {
                        path: _row_constant_value(
                            row_records, path, name, label
                        )
                        for path in allowed_columns_spec
                    },
                    "metrics": {
                        metric: next(
                            (
                                row for row in per_metric_rows[metric]
                                if row["seed"] == seed
                            ),
                            {
                                "value": None,
                                "std": None,
                                "n": 1,
                                "source": None,
                                "regime": None,
                            },
                        )
                        for metric in metrics
                    },
                    "seed": seed,
                    "regime": get_path(
                        row_records[0], "training.regime", None
                    ),
                    "source": get_path(
                        row_records[0], "provenance.source", None
                    ),
                })
            continue

        display_rows.append({
            "label": label,
            "factors": factors,
            "allowed_values": {
                path: _row_constant_value(
                    row_records, path, name, label
                )
                for path in allowed_columns_spec
            },
            "metrics": {
                metric: per_metric_rows[metric][0]
                for metric in metrics
            },
            "seed": None,
            "regime": get_path(row_records[0], "training.regime", None),
            "source": get_path(row_records[0], "provenance.source", None),
        })

    allowed_columns = []
    for path in allowed_columns_spec:
        values = {
            canonical_json(row["allowed_values"].get(path))
            if isinstance(row["allowed_values"].get(path), (dict, list))
            else row["allowed_values"].get(path)
            for row in display_rows
        }
        if len(values) > 1:
            allowed_columns.append(path)

    _check_protocol_compatibility(
        selected_for_protocol, allowed, strict, warnings, name
    )
    _check_native_class_coverage(
        selected_for_protocol, strict, warnings, name
    )
    _check_full_sample_counts(
        selected_for_protocol, strict, warnings, name
    )

    report["tables"][name] = {
        "rows": len(display_rows),
        "adopted_eval_ids": sorted(
            {x for x in adopted_ids if x is not None}
        ),
        "aggregate_seeds": aggregate,
        "std_ddof": 1,
        "display_unit": display_unit,
        "metrics": metrics,
        "allowed_varying_fields": allowed,
    }

    return {
        "spec": table,
        "unit": display_unit,
        "decimals": decimals,
        "metrics": metrics,
        "factor_columns": factor_columns,
        "allowed_columns": allowed_columns,
        "rows": display_rows,
    }


def _best_flags(table_result: dict) -> dict:
    spec = table_result["spec"]
    if not spec.get("bold_best", False):
        return {}

    scores = defaultdict(list)
    for row_idx, row in enumerate(table_result["rows"]):
        key_group = (
            row.get("regime"),
            row.get("source"),
        )
        for metric in table_result["metrics"]:
            value = row["metrics"][metric].get("value", None)
            if value is None:
                continue
            scores[(key_group, metric)].append(
                (float(value), row_idx)
            )

    best = {}
    for (group, metric), values in scores.items():
        values.sort(key=lambda x: x[0], reverse=True)
        best[(group, metric)] = values[0][1]
    return best


def _latex_metric_cell(
    row_value: dict,
    unit: str,
    decimals: int,
    bold: bool,
) -> str:
    mean = _to_display(row_value.get("value", None), unit)
    std = _to_display(row_value.get("std", None), unit)
    if mean is None:
        return "--"
    if std is None:
        text = format_scalar(mean, decimals)
    else:
        text = (
            f"{format_scalar(mean, decimals)} $\\pm$ "
            f"{format_scalar(std, decimals)}"
        )
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def _display_column_names(paths: list[str]) -> dict[str, str]:
    leaves = [path.split(".")[-1] for path in paths]
    counts = {leaf: leaves.count(leaf) for leaf in leaves}
    return {
        path: (leaf if counts[leaf] == 1 else path)
        for path, leaf in zip(paths, leaves)
    }


def _latex_label(value: str) -> str:
    text = str(value)
    cleaned = "".join(
        ch if ch.isalnum() or ch in ":_.-" else "-"
        for ch in text
    )
    return cleaned or "tab:generated"


def _render_latex(table_result: dict, draft: bool) -> str:
    spec = table_result["spec"]
    unit = table_result["unit"]
    decimals = table_result["decimals"]
    metrics = table_result["metrics"]
    factor_columns = table_result["factor_columns"]
    allowed_columns = table_result.get("allowed_columns", [])
    rows = table_result["rows"]

    regimes = {row.get("regime") for row in rows}
    sources = {row.get("source") for row in rows}
    show_regime = len(regimes) > 1 or (regimes and None in regimes)
    show_source = len(sources) > 1

    best = _best_flags(table_result)

    col_count = (
        1
        + (1 if show_regime else 0)
        + (1 if show_source else 0)
        + len(factor_columns)
        + len(allowed_columns)
        + len(metrics)
        + 1
    )
    colspec = "l" + "c" * (col_count - 1)
    layout = spec.get("layout", "column")
    env = "table*" if layout == "wide" else "table"
    width = "\\textwidth" if layout == "wide" else "\\columnwidth"

    lines = []
    if draft:
        lines.append("% DRAFT: incomplete results; not a formal paper table.")
    lines.extend([
        f"\\begin{{{env}}}[t]",
        "    \\centering",
        f"    \\caption{{{escape_latex(spec.get('caption', ''))}}}",
        f"    \\label{{{_latex_label(spec.get('label', ''))}}}",
        f"    \\resizebox{{{width}}}{{!}}{{%",
        f"    \\begin{{tabular}}{{{colspec}}}",
        "        \\toprule",
    ])

    header_cells = ["方法/设置"]
    if show_regime:
        header_cells.append("训练条件")
    if show_source:
        header_cells.append("来源")
    display_names = _display_column_names(
        factor_columns + allowed_columns
    )
    header_cells.extend(
        escape_latex(display_names[path]) for path in factor_columns
    )
    header_cells.extend(
        escape_latex(display_names[path]) for path in allowed_columns
    )
    for metric in metrics:
        header_cells.append(
            f"{escape_latex(_metric_display_name(metric))} "
            f"({escape_latex(unit)})"
        )
    header_cells.append("Seeds")
    lines.append("        " + " & ".join(header_cells) + " \\\\")
    lines.append("        \\midrule")

    for row_idx, row in enumerate(rows):
        cells = [escape_latex(row["label"])]
        if show_regime:
            cells.append(
                escape_latex(
                    REGIME_LABELS.get(
                        row.get("regime"), str(row.get("regime"))
                    )
                )
            )
        if show_source:
            cells.append(escape_latex(row.get("source", "--")))
        for path in factor_columns:
            value = row["factors"].get(path, None)
            cells.append(
                "--" if value is None else escape_latex(value)
            )
        for path in allowed_columns:
            value = row["allowed_values"].get(path, None)
            if value is None:
                cells.append("--")
            else:
                cells.append(escape_latex(value))
        group = (row.get("regime"), row.get("source"))
        for metric in metrics:
            is_best = best.get((group, metric)) == row_idx
            cells.append(
                _latex_metric_cell(
                    row["metrics"][metric],
                    unit,
                    decimals,
                    is_best,
                )
            )
        n = row["metrics"][metrics[0]].get("n", None) if metrics else None
        cells.append("--" if n is None else str(int(n)))
        lines.append("        " + " & ".join(cells) + " \\\\")

    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "    }",
        f"\\end{{{env}}}",
        "",
    ])
    return "\n".join(lines)


def _table_csv_rows(
    table_result: dict,
    draft: bool = False,
) -> tuple[list[str], list[dict]]:
    unit = table_result["unit"]
    suffix = _unit_suffix(unit)
    metrics = table_result["metrics"]
    factor_columns = table_result["factor_columns"]
    allowed_columns = table_result.get("allowed_columns", [])
    rows = table_result["rows"]
    decimals = table_result["decimals"]

    regimes = {row.get("regime") for row in rows}
    sources = {row.get("source") for row in rows}
    show_regime = len(regimes) > 1 or (regimes and None in regimes)
    show_source = len(sources) > 1

    fieldnames = ["row_label"]
    if draft:
        fieldnames.append("draft")
    if show_regime:
        fieldnames.append("training_regime")
    if show_source:
        fieldnames.append("source")
    fieldnames.extend(factor_columns)
    fieldnames.extend(allowed_columns)
    for metric in metrics:
        fieldnames.append(f"{metric}_mean_{suffix}")
        fieldnames.append(f"{metric}_std_{suffix}")
    fieldnames.append("n_seeds")

    csv_rows = []
    for row in rows:
        out = {"row_label": row["label"]}
        if draft:
            out["draft"] = True
        if show_regime:
            out["training_regime"] = row.get("regime")
        if show_source:
            out["source"] = row.get("source")
        for path in factor_columns:
            out[path] = row["factors"].get(path, None)
        for path in allowed_columns:
            out[path] = row["allowed_values"].get(path, None)
        for metric in metrics:
            value = row["metrics"][metric]
            mean = _to_display(value.get("value", None), unit)
            std = _to_display(value.get("std", None), unit)
            out[f"{metric}_mean_{suffix}"] = (
                None if mean is None else round(mean, decimals)
            )
            out[f"{metric}_std_{suffix}"] = (
                None if std is None else round(std, decimals)
            )
        out["n_seeds"] = row["metrics"][metrics[0]].get("n", None) if metrics else None
        csv_rows.append(out)

    return fieldnames, csv_rows


def _resolve_run_dir(
    spec_dir: Path,
    roots: list[Path],
    value: str,
) -> Path:
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(spec_dir / path)
        for root in roots:
            candidates.append(root / path)
            candidates.append(root / "runs" / path)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "metrics.jsonl").exists():
            return candidate.resolve()
    raise ExportError(
        f"Cannot resolve metrics run {value!r}. Tried: "
        f"{[str(x) for x in candidates]}"
    )


def _effective_curve_points(records: list[dict]) -> list[dict]:
    meta_events = [
        record for record in records
        if record.get("mode") == "meta"
        and record.get("event") == "before_run"
    ]
    attempts = []
    seen = set()
    for event in meta_events:
        attempt_id = event.get("attempt_id", None)
        key = attempt_id or f"_implicit_{len(attempts)}"
        if key in seen:
            continue
        seen.add(key)
        attempts.append({
            "attempt_id": attempt_id,
            "resume_from_step": event.get("resume_from_step", None),
        })

    data_records = [
        record for record in records
        if record.get("mode") in {"train", "val"}
    ]

    if not attempts:
        return sorted(
            data_records,
            key=lambda r: int(r.get("iter", 0)),
        )

    by_attempt = defaultdict(list)
    for record in data_records:
        by_attempt[record.get("attempt_id", None)].append(record)

    effective = []
    for idx, attempt in enumerate(attempts):
        start = attempt.get("resume_from_step", None)
        start = 0 if start is None else int(start)
        next_start = None
        if idx + 1 < len(attempts):
            next_start = attempts[idx + 1].get("resume_from_step", None)
            if next_start is None:
                next_start = None
        attempt_records = by_attempt.get(attempt["attempt_id"], [])
        for record in attempt_records:
            step = record.get("iter", None)
            if not isinstance(step, int):
                continue
            if step <= start:
                continue
            if (
                next_start is not None
                and step > int(next_start)
            ):
                continue
            effective.append(record)

    effective.sort(key=lambda r: int(r.get("iter", 0)))
    return effective


def _smooth(values: list[float], window: int) -> list[float]:
    if window is None or window <= 1:
        return values
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _collect_metrics_series(
    records: list[dict],
    y_key: str,
    smooth_window: int | None,
) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    key_candidates = [y_key]
    if y_key.startswith("val/") or y_key.startswith("train/"):
        key_candidates.append(y_key.split("/", 1)[1])
    for record in records:
        value = None
        for key in key_candidates:
            if key in record:
                value = record[key]
                break
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        steps.append(int(record.get("iter", 0)))
        values.append(float(value))
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    values = [values[i] for i in order]
    return steps, _smooth(values, smooth_window)


def _factor_series(
    records: list[dict],
    curve: dict,
    strict: bool,
) -> list[dict]:
    filter_spec = curve.get("filter", {})
    match = curve.get("match", {})
    x_factor = curve["x_factor"]
    y_metric = curve["y"]
    group_by = list(curve.get("group_by", []))

    selected = [
        record for record in records
        if _matches(record, filter_spec) and _matches(record, match)
    ]
    if not selected:
        raise ExportError(
            f"Curve {curve['name']!r}: selector matched no records."
        )

    series = defaultdict(list)
    for record in selected:
        x = get_path(record, x_factor, None)
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise ExportError(
                f"Curve {curve['name']!r}: factor {x_factor!r} must be "
                f"numeric, got {x!r}."
            )
        y = _resolve_metric(record, y_metric)
        if y is None:
            if strict:
                raise ExportError(
                    f"Curve {curve['name']!r}: metric {y_metric!r} "
                    f"missing for {record.get('eval_id')!r}."
                )
            continue

        factors = get_path(record, "experiment.factors", {}) or {}
        varying = sorted(
            key for key in factors
            if key != x_factor.split(".")[-1]
        )
        if not group_by:
            group_fields = varying
        else:
            group_fields = [
                path.split(".")[-1]
                for path in group_by
                if path.startswith("experiment.factors.")
            ]
        group_key = (
            get_path(record, "experiment.method_id", None),
            get_path(record, "experiment.variant_id", None),
            get_path(record, "experiment.seed", None),
            get_path(record, "training.regime", None),
            tuple((name, factors.get(name)) for name in group_fields),
        )
        series[group_key].append((float(x), float(y)))

    return [
        {
            "key": key,
            "points": sorted(points, key=lambda p: p[0]),
        }
        for key, points in sorted(series.items(), key=lambda kv: str(kv[0]))
    ]


def _plot_metric_curve(
    curve: dict,
    points_by_run: list[tuple[str, list[int], list[float]]],
    output_dir: Path,
    unit: str,
    report: dict,
    draft: bool = False,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = curve["name"]
    multi = curve.get("multi_seed", "separate")
    save_png = bool(curve.get("save_png", True))
    smooth = curve.get("smooth_window", None)
    y_label = curve.get("label", curve.get("y", "metric"))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if multi == "mean_band" and len(points_by_run) > 1:
        common_steps = set(points_by_run[0][1])
        for _, steps, _ in points_by_run[1:]:
            common_steps &= set(steps)
        common = sorted(common_steps)
        value_maps = [
            dict(zip(steps, values)) for _, steps, values in points_by_run
        ]
        means = []
        stds = []
        for step in common:
            values = [float(m[step]) for m in value_maps]
            mean, std = mean_std(values, ddof=1)
            means.append(
                mean * 100.0 if unit == "percent" else mean
            )
            stds.append(
                None if std is None else (
                    std * 100.0 if unit == "percent" else std
                )
            )
        ax.plot(common, means, label=curve.get("label", name))
        low = [
            m - s for m, s in zip(means, stds) if s is not None
        ]
        high = [
            m + s for m, s in zip(means, stds) if s is not None
        ]
        if low:
            valid_steps = [
                step for step, s in zip(common, stds) if s is not None
            ]
            ax.fill_between(
                valid_steps, low, high, alpha=0.2
            )
    else:
        for label, steps, values in points_by_run:
            display = [
                v * 100.0 if unit == "percent" else v for v in values
            ]
            ax.plot(steps, display, label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel(y_label if unit != "percent" else f"{y_label} (%)")
    if smooth:
        ax.set_title(f"Smoothed (moving average, window={smooth})")
    if len(points_by_run) > 1 or multi == "mean_band":
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    if draft:
        fig.text(
            0.5, 0.005, "DRAFT — incomplete data",
            ha="center", va="bottom", fontsize=8, color="red",
        )
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    pdf_path = figures_dir / f"{_sanitize_filename(name)}.pdf"
    fig.savefig(pdf_path)
    outputs.append(pdf_path)
    if save_png:
        png_path = figures_dir / f"{_sanitize_filename(name)}.png"
        fig.savefig(png_path, dpi=160)
        outputs.append(png_path)
    plt.close(fig)

    report["curves"][name] = {
        "type": "metrics",
        "runs": [label for label, _, _ in points_by_run],
        "multi_seed": multi,
        "smooth_window": smooth,
        "outputs": [str(path) for path in outputs],
    }
    return outputs


def _plot_factor_curve(
    curve: dict,
    series: list[dict],
    output_dir: Path,
    unit: str,
    report: dict,
    draft: bool = False,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = curve["name"]
    save_png = bool(curve.get("save_png", True))
    x_label = curve["x_factor"].split(".")[-1]
    y_label = curve.get("label", curve.get("y", "metric"))
    multi = curve.get("multi_seed", "separate")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    if multi == "mean_band":
        grouped = defaultdict(list)
        for item in series:
            key = item["key"]
            group_key = (key[0], key[1], key[3], key[4])
            grouped[group_key].append(item["points"])
        for key, point_lists in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            common_x = set(point_lists[0][i][0] for i in range(len(point_lists[0])))
            for points in point_lists[1:]:
                common_x &= set(x for x, _ in points)
            xs = sorted(common_x)
            maps = [dict(points) for points in point_lists]
            means = []
            stds = []
            for x in xs:
                values = [float(m[x]) for m in maps]
                mean, std = mean_std(values, ddof=1)
                means.append(mean * 100.0 if unit == "percent" else mean)
                stds.append(
                    None if std is None else (
                        std * 100.0 if unit == "percent" else std
                    )
                )
            label = str(key[0]) + ("/" + str(key[1]) if len(key) > 1 else "")
            ax.plot(xs, means, label=label)
            valid_x = [x for x, s in zip(xs, stds) if s is not None]
            low = [m - s for m, s in zip(means, stds) if s is not None]
            high = [m + s for m, s in zip(means, stds) if s is not None]
            if valid_x:
                ax.fill_between(valid_x, low, high, alpha=0.2)
    else:
        for item in series:
            xs = [p[0] for p in item["points"]]
            ys = [
                p[1] * 100.0 if unit == "percent" else p[1]
                for p in item["points"]
            ]
            key = item["key"]
            label = f"{key[0]}/{key[1]}"
            if key[2] is not None:
                label += f"/seed{key[2]}"
            ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label if unit != "percent" else f"{y_label} (%)")
    if len(series) > 1:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    if draft:
        fig.text(
            0.5, 0.005, "DRAFT — incomplete data",
            ha="center", va="bottom", fontsize=8, color="red",
        )
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    pdf_path = figures_dir / f"{_sanitize_filename(name)}.pdf"
    fig.savefig(pdf_path)
    outputs.append(pdf_path)
    if save_png:
        png_path = figures_dir / f"{_sanitize_filename(name)}.png"
        fig.savefig(png_path, dpi=160)
        outputs.append(png_path)
    plt.close(fig)

    report["curves"][name] = {
        "type": "factor",
        "x_factor": curve["x_factor"],
        "y": curve["y"],
        "multi_seed": multi,
        "outputs": [str(path) for path in outputs],
    }
    return outputs


def _export_curves(
    curves: list[dict],
    records: list[dict],
    roots: list[Path],
    spec_dir: Path,
    output_dir: Path,
    unit_default: str,
    strict: bool,
    draft: bool,
    report: dict,
) -> None:
    for curve in curves:
        unit = curve.get("display_unit", unit_default)
        if unit not in {"fraction", "percent"}:
            raise ExportError(
                f"Curve {curve['name']!r}: display_unit must be "
                "fraction or percent."
            )

        if curve.get("type", "metrics") == "metrics":
            run_values = curve.get("runs", None)
            if run_values is None:
                run_values = [curve.get("run")]
            if not isinstance(run_values, list):
                raise ExportError(
                    f"Curve {curve['name']!r}: runs must be a list."
                )
            points_by_run = []
            for run_value in run_values:
                run_dir = _resolve_run_dir(
                    spec_dir, roots, str(run_value)
                )
                jsonl_records = read_jsonl(run_dir / "metrics.jsonl")
                effective = _effective_curve_points(jsonl_records)
                filtered = [
                    record for record in effective
                    if record.get("mode") in {"train", "val"}
                ]
                steps, values = _collect_metrics_series(
                    filtered,
                    str(curve["y"]),
                    curve.get("smooth_window", None),
                )
                if not steps:
                    raise ExportError(
                        f"Curve {curve['name']!r}: no points for key "
                        f"{curve['y']!r} in {run_dir}."
                    )
                base_label = str(curve.get("label", run_dir.name))
                if len(run_values) > 1:
                    base_label = f"{base_label} {Path(str(run_value)).name}"
                points_by_run.append(
                    (base_label, steps, values)
                )
            _plot_metric_curve(
                curve, points_by_run, output_dir, unit, report,
                draft=draft,
            )
        else:
            series = _factor_series(records, curve, strict)
            _plot_factor_curve(
                curve, series, output_dir, unit, report,
                draft=draft,
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export paper tables and curves from standard result JSON and "
            "metrics.jsonl files."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        type=str,
        help=(
            "Work directory root or result file. Repeat --root to collect "
            "multiple method directories."
        ),
    )
    parser.add_argument("--spec", required=True, type=str)
    parser.add_argument(
        "--output", type=str, default="paper/generated"
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Export tables and CSV only; do not import matplotlib.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Produce a clearly marked draft with missing/incomplete data. "
            "Strict mode is the default."
        ),
    )
    args = parser.parse_args()

    strict = not args.allow_incomplete
    draft = bool(args.allow_incomplete)
    spec_path = Path(args.spec).resolve()
    spec_dir = spec_path.parent
    output_dir = Path(args.output).resolve()
    roots = [Path(root).resolve() for root in args.root]

    warnings: list[str] = []
    report = {
        "schema_version": 1,
        "status": "running",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "draft": draft,
        "roots": [str(root) for root in roots],
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path) if spec_path.exists() else None,
        "tables": {},
        "curves": {},
        "adopted": [],
        "excluded": [],
        "warnings": warnings,
        "outputs": [],
    }

    try:
        if not spec_path.exists():
            raise ExportError(f"Spec file not found: {spec_path}")
        spec = load_json(spec_path)
        if not isinstance(spec, dict):
            raise ExportError("Spec file must contain a JSON object.")
        if spec.get("schema_version", 1) != 1:
            raise ExportError(
                "Unsupported spec schema_version: "
                f"{spec.get('schema_version')!r}"
            )
        tables = spec.get("tables", [])
        curves = spec.get("curves", [])
        if not isinstance(tables, list) or not isinstance(curves, list):
            raise ExportError("Spec 'tables' and 'curves' must be lists.")

        table_names = set()
        table_files = set()
        labels = set()
        curve_names = set()
        curve_files = set()
        for table in tables:
            _check_table_spec(table)
            name = table["name"]
            label = table.get("label", "")
            if name in table_names:
                raise ExportError(f"Duplicate table name: {name!r}")
            if label and label in labels:
                raise ExportError(f"Duplicate LaTeX label: {label!r}")
            filename = _sanitize_filename(name)
            if filename in table_files:
                raise ExportError(
                    f"Table names collide after filename sanitization: "
                    f"{name!r}"
                )
            table_names.add(name)
            table_files.add(filename)
            if label:
                labels.add(label)
        for curve in curves:
            _check_curve_spec(curve)
            if curve["name"] in curve_names:
                raise ExportError(
                    f"Duplicate curve name: {curve['name']!r}"
                )
            filename = _sanitize_filename(curve["name"])
            if filename in curve_files:
                raise ExportError(
                    "Curve names collide after filename sanitization: "
                    f"{curve['name']!r}"
                )
            curve_names.add(curve["name"])
            curve_files.add(filename)

        records, scan_errors = scan_results(roots)
        if scan_errors:
            if strict:
                raise ExportError(
                    "Result scan errors:\n" + "\n".join(scan_errors)
                )
            warnings.extend(scan_errors)
        if not records:
            raise ExportError("No standard result records found.")

        formal = [
            record for record in records
            if _record_is_formal(record, args.allow_incomplete)
        ]
        skipped = [
            record for record in records if record not in formal
        ]
        report["excluded"] = [
            {
                "eval_id": record.get("eval_id"),
                "path": record.get("_source_path"),
                "reason": (
                    "not completed evaluation"
                    if record.get("role") != "evaluation"
                    else "not full/complete evaluation"
                ),
            }
            for record in skipped
        ]

        table_results = []
        for table in tables:
            table_results.append(
                _build_table(table, formal, strict, warnings, report)
            )

        # Long table (all valid standard records).
        long_rows = [record_to_long_row(record) for record in records]
        long_path = output_dir / "data" / "results_long.csv"
        if long_rows:
            write_csv(long_path, list(long_rows[0].keys()), long_rows)
            report["outputs"].append(str(long_path))

        for table_result in table_results:
            table = table_result["spec"]
            name = _sanitize_filename(table["name"])
            fields, rows = _table_csv_rows(table_result, draft=draft)
            csv_path = output_dir / "data" / f"{name}.csv"
            write_csv(csv_path, fields, rows)
            report["outputs"].append(str(csv_path))

            tex = _render_latex(table_result, draft)
            tex_path = output_dir / "tables" / f"{name}.tex"
            atomic_write_text(tex_path, tex)
            report["outputs"].append(str(tex_path))

        if not args.tables_only:
            _export_curves(
                curves,
                formal,
                roots,
                spec_dir,
                output_dir,
                "percent",
                strict,
                draft,
                report,
            )

        if tables and not curves:
            adopted_ids = set()
            for table_report in report["tables"].values():
                adopted_ids.update(
                    table_report.get("adopted_eval_ids", [])
                )
            adopted_records = [
                record for record in formal
                if record.get("eval_id") in adopted_ids
            ]
        else:
            adopted_records = formal

        report["adopted"] = [
            {
                "eval_id": record.get("eval_id"),
                "path": record.get("_source_path"),
                "sha256": (
                    sha256_file(record["_source_path"])
                    if record.get("_source_path")
                    and Path(record["_source_path"]).is_file()
                    else None
                ),
                "method_id": get_path(
                    record, "experiment.method_id", None
                ),
                "variant_id": get_path(
                    record, "experiment.variant_id", None
                ),
                "seed": get_path(record, "experiment.seed", None),
                "checkpoint_step": get_path(
                    record, "checkpoint.step", None
                ),
                "source": get_path(
                    record, "provenance.source", None
                ),
            }
            for record in adopted_records
        ]
        report["warnings"] = warnings
        report["draft"] = draft
        report["status"] = "completed"

        report_path = output_dir / "export_report.json"
        atomic_write_json(report_path, report)
        print(f"Wrote export report: {report_path}")
        if draft:
            print(
                "DRAFT export: incomplete data or metadata is present. "
                "Do not use these tables as final paper results."
            )

    except ExportError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        failure_report = {
            "schema_version": 1,
            "status": "failed",
            "draft": True,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "roots": [str(root) for root in roots],
            "spec_path": str(spec_path),
            "error": str(exc),
            "warnings": warnings,
        }
        try:
            atomic_write_json(
                output_dir / "export_report.json", failure_report
            )
        except Exception:
            pass
        raise SystemExit(2)


if __name__ == "__main__":
    main()
