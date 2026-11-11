from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_results import (  # noqa: E402
    get_path,
    record_to_long_row,
    scan_results,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect standard results/*.json records into a long CSV. "
            "Training-process validation records are excluded by default."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=str,
        help="work_dirs root(s) or result JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/generated/data/results_long.csv",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Include role='validation' records.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Include records with is_full=false or null.",
    )
    args = parser.parse_args()

    records, errors = scan_results(args.roots)
    if errors:
        print("Result scan errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(2)

    selected = []
    excluded = 0
    for record in records:
        status = record.get("status")
        if status not in {"completed", "partial"}:
            excluded += 1
            continue
        if status == "partial" and not args.allow_partial:
            excluded += 1
            continue
        if (
            record.get("role") != "evaluation"
            and not args.include_validation
        ):
            excluded += 1
            continue
        is_full = get_path(record, "evaluation.is_full", None)
        if is_full is not True and not args.allow_partial:
            excluded += 1
            continue
        selected.append(record)

    if not selected:
        print(
            "No matching completed evaluation results were found. "
            f"Scanned {len(records)} record(s), excluded {excluded}.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    rows = [record_to_long_row(record) for record in selected]
    fieldnames = list(rows[0].keys())
    output = Path(args.output)
    write_csv(output, fieldnames, rows)
    print(
        f"Wrote {output} with {len(rows)} result(s); "
        f"excluded {excluded} record(s)."
    )


if __name__ == "__main__":
    main()
