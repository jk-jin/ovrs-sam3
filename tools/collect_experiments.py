from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path):
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


def pick_best_val(records, monitor: str, mode: str):
    vals = [
        r for r in records
        if r.get("mode") == "val" and isinstance(r.get(monitor), (int, float))
    ]
    if not vals:
        return None

    reverse = mode == "max"
    return sorted(vals, key=lambda r: float(r[monitor]), reverse=reverse)[0]


def pick_last_train(records):
    vals = [r for r in records if r.get("mode") == "train"]
    if not vals:
        return None
    return sorted(vals, key=lambda r: int(r.get("iter", -1)))[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="work_dirs root or a directory containing experiments")
    parser.add_argument("--monitor", type=str, default="semantic.miou")
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"])
    parser.add_argument("--output", type=str, default="experiment_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    exp_dirs = []

    if (root / "metrics.jsonl").exists():
        exp_dirs = [root]
    else:
        exp_dirs = sorted([p for p in root.rglob("*") if (p / "metrics.jsonl").exists()])

    rows = []
    for exp_dir in exp_dirs:
        records = read_jsonl(exp_dir / "metrics.jsonl")
        best_val = pick_best_val(records, args.monitor, args.mode)
        last_train = pick_last_train(records)

        row = {
            "exp_dir": str(exp_dir),
            "best_iter": None,
            f"best_{args.monitor}": None,
            "last_iter": None,
            "last_total_loss": None,
            "last_memory_mb": None,
        }

        if best_val is not None:
            row["best_iter"] = best_val.get("iter")
            row[f"best_{args.monitor}"] = best_val.get(args.monitor)

            for k, v in best_val.items():
                if isinstance(v, (int, float)) and k not in row:
                    row[f"val/{k}"] = v

        if last_train is not None:
            row["last_iter"] = last_train.get("iter")
            row["last_total_loss"] = last_train.get("total_loss")
            row["last_memory_mb"] = last_train.get("memory_mb")

            for k, v in last_train.items():
                if isinstance(v, (int, float)) and k not in row:
                    row[f"train/{k}"] = v

        rows.append(row)

    if not rows:
        print(f"No metrics.jsonl found under {root}")
        return

    all_keys = []
    for row in rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)

    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path} with {len(rows)} experiments.")


if __name__ == "__main__":
    main()
