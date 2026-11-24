"""Configuration-driven, process-isolated paper training and validation.

This module intentionally has no model/GPU imports: planning is usable before
installing training dependencies or copying datasets and checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pprint
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..engine.config import Config, build_effective_config, config_fingerprint, finalize_config
from .experiment_results import atomic_write_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def token(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"Invalid directory identifier: {value!r}")
    return value


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(f"{key} = {pprint.pformat(value, sort_dicts=False)}" for key, value in cfg.items()) + "\n")


def load_suite(path):
    cfg = Config.fromfile(resolve(path))["suite"]
    entries = cfg["experiments"]
    names = [token(e["variant"]) for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate experiment variant in suite.")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be nonempty and unique.")
    return cfg


def train_jobs(suite, only=None):
    root = resolve(suite["root"])
    jobs = []
    for entry in suite["experiments"]:
        variant = entry["variant"]
        if only and variant not in only:
            continue
        for seed in suite["seeds"]:
            work = root / variant / f"seed_{int(seed)}" / "train"
            cfg, fingerprint = build_effective_config(resolve(entry["config"]), seed=int(seed), work_dir=str(work))
            if cfg["experiment"]["variant_id"] != variant or cfg["model"]["pipeline"].startswith("sam3_"):
                raise ValueError(f"Training entry does not describe trainable variant {variant}")
            jobs.append(dict(key=f"{variant}:{seed}", variant=variant, seed=int(seed), mode="train",
                             work=str(work), cfg=cfg, signature=fingerprint,
                             plan=str(root / "plans" / variant / f"seed_{seed}" / "train.py")))
    return jobs


def evaluation_config(source, target, work):
    """Retain the trained architecture/loss/identity; replace only evaluation data."""
    cfg = copy.deepcopy(source)
    cfg.pop("meta", None)
    target_cfg, _ = build_effective_config(resolve(f"configs/test/{token(target)}.py"))
    cfg["val_dataloader"] = target_cfg["val_dataloader"]
    cfg["evaluation"] = target_cfg["evaluation"]
    cfg["runtime"]["work_dir"] = str(work)
    cfg["model"]["eval_mode"] = True
    cfg["logging"]["wandb"]["mode"] = "offline"
    return finalize_config(cfg)


def validation_jobs(suite, only=None, dry_run=False):
    root = resolve(suite["root"]) / "validation" / token(suite["validation_name"])
    entries = {e["variant"]: e for e in suite["experiments"]}
    jobs, pending, selected = [], [], []
    seen = set()
    for choice in suite.get("selections", []):
        variant, seed = token(choice["variant"]), int(choice["seed"])
        if only and variant not in only:
            continue
        if variant not in entries or (variant, seed) in seen:
            raise ValueError(f"Unknown or duplicate checkpoint selection: {variant}:{seed}")
        seen.add((variant, seed))
        if not choice.get("checkpoint"):
            pending.append(f"{variant}:{seed}")
            continue
        checkpoint = resolve(choice["checkpoint"])
        # config.json is the actual training snapshot saved next to the weight.
        source = read_json(checkpoint.parent / "config.json")
        if source is None or not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint and its training config.json are required: {checkpoint}")
        if source["experiment"]["variant_id"] != variant or source["runtime"]["seed"] != seed:
            raise ValueError(f"Checkpoint identity does not match {variant}:{seed}")
        source_cfg = dict(source)
        meta = source_cfg.pop("meta", {})
        if meta.get("config_fingerprint") != config_fingerprint(source_cfg):
            raise ValueError(f"Training config fingerprint is inconsistent: {checkpoint.parent}")
        selected.append((entries[variant], seed, checkpoint, source, False))
    for entry in suite.get("baselines", []):
        if only and entry["variant"] not in only:
            continue
        source, _ = build_effective_config(resolve(entry["config"]))
        if entry.get("checkpoint"):
            source["model"]["checkpoint_path"] = str(resolve(entry["checkpoint"]))
            source["model"]["load_from_hf"] = False
        if (not source["model"]["pipeline"].startswith("sam3_")
                or source["experiment"]["variant_id"] != entry["variant"]):
            raise ValueError("Evaluation-only baseline must match its original SAM3 variant.")
        checkpoint = resolve(source["model"]["checkpoint_path"])
        if not dry_run and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for seed in suite["seeds"]:
            cfg = copy.deepcopy(source)
            cfg["runtime"]["seed"] = int(seed)
            selected.append((entry, int(seed), checkpoint, cfg, True))
    digests = {}
    for entry, seed, checkpoint, source, pretrained in selected:
        variant = token(entry["variant"])
        if checkpoint not in digests:
            digests[checkpoint] = sha256_file(checkpoint) if checkpoint.is_file() else "unavailable-dry-run"
        datasets = suite["dataset_groups"][entry["datasets"]]
        if len(set(datasets)) != len(datasets):
            raise ValueError("Duplicate evaluation dataset.")
        for dataset in datasets:
            work = root / variant / f"seed_{seed}" / token(dataset)
            cfg = evaluation_config(source, dataset, work)
            identity = dict(config=config_fingerprint(cfg), checkpoint_sha256=digests[checkpoint])
            signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            jobs.append(dict(key=f"{variant}:{seed}:{dataset}", variant=variant, seed=seed, dataset=dataset,
                             eval_dataset=cfg["evaluation"]["dataset"],
                             eval_protocol_id=cfg["evaluation"]["protocol_id"],
                             mode="validate", work=str(work), cfg=cfg, signature=signature,
                             checkpoint=str(checkpoint), checkpoint_sha256=digests[checkpoint], pretrained=pretrained,
                             source_config=None if pretrained else str(checkpoint.parent / "config.json"),
                             plan=str(work / "evaluation.py")))
    if len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate validation output directories.")
    return jobs, pending


def command_for(job, device, resume_from=None):
    cmd = [sys.executable, str(REPO_ROOT / "tools/train.py"), job["plan"], "--device", device]
    if job["mode"] == "validate":
        cmd += ["--eval-only"]
        if job["pretrained"]:
            cmd += ["--eval-iter", "0"]
        else:
            cmd += ["--load-model-from", job["checkpoint"]]
    elif resume_from:
        cmd += ["--resume-from", str(resume_from)]
    return cmd


def execute_job(job, device, resume=False, run_process=subprocess.run):
    work = Path(job["work"])
    state_path = work / "suite_status.json"
    old = read_json(state_path)
    if old and old.get("signature") != job["signature"]:
        raise ValueError(f"Changed experiment/checkpoint in existing directory; choose a new root or validation_name: {work}")
    if old and old.get("status") == "completed":
        artifacts = ([work / "latest.pth", work / f"iter_{job['cfg']['train_cfg']['max_iters']:07d}.pth"]
                     if job["mode"] == "train" else list((work / "results").glob("*.json")))
        if resume and artifacts and all(p.is_file() for p in artifacts):
            return dict(key=job["key"], status="skipped_completed", work=str(work))
        raise FileExistsError(f"Completed output already exists; use --resume to skip: {work}")
    resume_from = None
    if job["mode"] == "train" and (work / "config.json").exists():
        saved = read_json(work / "config.json")
        if saved.get("meta", {}).get("config_fingerprint") != job["signature"]:
            raise ValueError(f"Training fingerprint changed: {work}")
        resume_from = work / "latest.pth"
        if not resume or not resume_from.is_file():
            raise ValueError(f"Existing training artifacts require --resume and latest.pth, or a new root: {work}")
    if old and not resume:
        raise FileExistsError(f"Existing attempt requires --resume: {work}")
    work.mkdir(parents=True, exist_ok=True)
    write_config(Path(job["plan"]), job["cfg"])
    state = {k: v for k, v in job.items() if k != "cfg"}
    state.update(status="running", device=device)
    atomic_write_json(state_path, state)
    command = command_for(job, device, resume_from)
    with (work / "console.log").open("a") as output:
        process = run_process(command, cwd=REPO_ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
    state.update(returncode=process.returncode, status="completed" if process.returncode == 0 else "failed")
    if state["status"] == "completed":
        if job["mode"] == "train":
            required = [work / "latest.pth", work / f"iter_{job['cfg']['train_cfg']['max_iters']:07d}.pth"]
        else:
            required = list((work / "results").glob("*.json"))
        if not required or not all(p.is_file() for p in required):
            state.update(status="failed", error="Worker exited without required checkpoint/result artifacts.")
    atomic_write_json(state_path, state)
    return {k: state[k] for k in ("key", "status", "work", "returncode")}


def execute_jobs(jobs, devices, resume=False):
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be nonempty and unique (one worker per device).")
    waiting = queue.Queue()
    for job in jobs:
        waiting.put(job)
    def worker(device):
        results = []
        while True:
            try:
                job = waiting.get_nowait()
            except queue.Empty:
                return results
            print(f"[{device}] {job['key']}", flush=True)
            try:
                result = execute_job(job, device, resume)
            except Exception as exc:
                result = dict(key=job["key"], status="failed", work=job["work"], error=str(exc))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        return [result for group in pool.map(worker, devices) for result in group]


def write_checkpoint_index(suite):
    index, selections = [], []
    for job in train_jobs(suite):
        work = Path(job["work"])
        files = sorted(str(p) for p in work.glob("*.pth"))
        index.append(dict(variant=job["variant"], seed=job["seed"], config=str(work / "config.json"), checkpoints=files))
        selections.append(dict(variant=job["variant"], seed=job["seed"], checkpoint=None))
    root = resolve(suite["root"])
    atomic_write_json(root / "checkpoints.json", index)
    template = copy.deepcopy(suite)
    template.update(validation_name="selected_weights", selections=selections, baselines=[
        dict(variant=name, config=f"configs/baselines/{name}.py", datasets="main")
        for name in ("semantic_only", "semantic_presence")
    ])
    # Do not overwrite a user's edited selections when resuming training.
    path = root / "validation_template.py"
    if not path.exists():
        write_config(path, dict(suite=template))


def write_validation_summary(root, jobs):
    rows = []
    for job in jobs:
        state = read_json(Path(job["work"]) / "suite_status.json") or {}
        if state.get("status") != "completed" or state.get("signature") != job["signature"]:
            continue
        for path in sorted((Path(job["work"]) / "results").glob("*.json")):
            record = read_json(path)
            metrics = record.get("metrics", {})
            rows.append(dict(variant=job["variant"], seed=job["seed"], dataset=job["dataset"],
                             eval_dataset=job["eval_dataset"], eval_protocol_id=job["eval_protocol_id"],
                             checkpoint=job["checkpoint"], checkpoint_sha256=job["checkpoint_sha256"],
                             miou=metrics.get("miou"), macc=metrics.get("macc"), result=str(path)))
    atomic_write_json(root / "metrics_summary.json", rows)
    with (root / "metrics_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "seed", "dataset", "eval_dataset", "eval_protocol_id", "checkpoint", "checkpoint_sha256", "miou", "macc", "result"])
        writer.writeheader()
        writer.writerows(rows)


def select_jobs(suite, only=None, datasets=None, checkpoint=None, seed=None):
    """Apply CLI choices before planning; saved training configs remain authoritative."""
    suite = copy.deepcopy(suite)
    if seed is not None:
        suite["seeds"] = [seed]
        suite["selections"] = [r for r in suite.get("selections", []) if r["seed"] == seed]
    targets = {p.stem for p in (REPO_ROOT / "configs/test").glob("*.py")}
    if datasets:
        if len(set(datasets)) != len(datasets) or set(datasets) - targets:
            raise ValueError(f"Unknown or duplicate evaluation targets. Choose from {sorted(targets)}")
        suite["dataset_groups"]["selected"] = datasets
        for entry in suite["experiments"] + suite.get("baselines", []):
            entry["datasets"] = "selected"
    if checkpoint:
        if not only or len(only) != 1:
            raise ValueError("--checkpoint requires exactly one variant in --only.")
        variant = only[0]
        baseline = next((r for r in suite.get("baselines", []) if r["variant"] == variant), None)
        if baseline is not None:
            baseline["checkpoint"] = checkpoint
        else:
            if seed is None and len(suite["seeds"]) != 1:
                raise ValueError("Select --seed for a multi-seed suite.")
            suite["selections"] = [dict(variant=variant, seed=suite["seeds"][0], checkpoint=checkpoint)]
    return suite


def main(mode):
    parser = argparse.ArgumentParser(description=f"Batch {mode} for paper experiments")
    parser.add_argument("config")
    parser.add_argument("--dry-run", action="store_true", help="print plans without creating outputs or starting workers")
    parser.add_argument("--resume", action="store_true", help="skip completed matching jobs and retry interrupted jobs")
    parser.add_argument("--only", nargs="+", help="variant names to run")
    parser.add_argument("--devices", nargs="+", help="one independent worker per device, e.g. cuda:0 cuda:1")
    parser.add_argument("--seed", type=int, help="run one seed")
    if mode == "validate":
        parser.add_argument("--datasets", nargs="+", help="evaluation target names from configs/test")
        parser.add_argument("--checkpoint", help="explicit weight for exactly one --only variant")
        parser.add_argument("--validation-name", help="separate output namespace for a new weight selection")
    args = parser.parse_args()
    suite = load_suite(args.config)
    valid = {e["variant"] for e in suite["experiments"] + suite.get("baselines", [])}
    if args.only and set(args.only) - valid:
        parser.error(f"Unknown variants: {sorted(set(args.only) - valid)}")
    try:
        suite = select_jobs(suite, args.only, getattr(args, "datasets", None),
                            getattr(args, "checkpoint", None), args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if getattr(args, "validation_name", None):
        suite["validation_name"] = token(args.validation_name)
    devices = args.devices or suite["devices"]
    if mode == "train":
        jobs, pending = train_jobs(suite, args.only), []
        summary_root = resolve(suite["root"])
    else:
        jobs, pending = validation_jobs(suite, args.only, args.dry_run)
        summary_root = resolve(suite["root"]) / "validation" / token(suite["validation_name"])
    if args.dry_run:
        plans = [{k: v for k, v in job.items() if k != "cfg"} | dict(
            command=command_for(job, devices[0]), pipeline=job["cfg"]["model"]["pipeline"],
            batch_size=job["cfg"].get("train_dataloader", {}).get("batch_size"),
            lr=job["cfg"].get("optimizer", {}).get("lr"), max_iters=job["cfg"]["train_cfg"]["max_iters"])
            for job in jobs]
        print(json.dumps(dict(jobs=plans, pending=pending), ensure_ascii=False, indent=2))
        return 0
    if pending:
        print("No checkpoint selected (pending): " + ", ".join(pending), flush=True)
    results = execute_jobs(jobs, devices, args.resume)
    summary_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_root / f"{mode}_summary.json", dict(jobs=results, pending=pending))
    if mode == "train":
        write_checkpoint_index(suite)
    else:
        write_validation_summary(summary_root, jobs)
    return int(any(r["status"] == "failed" for r in results))
