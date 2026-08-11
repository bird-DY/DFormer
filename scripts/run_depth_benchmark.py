#!/usr/bin/env python3
"""Run a resumable, multi-condition depth-corruption benchmark."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from depth_benchmark_protocol import get_protocol


PARAMETER_FLAGS = {
    "missing_rate": "--missing_rate",
    "noise_std": "--noise_std",
    "shift_x": "--shift_x",
    "shift_y": "--shift_y",
    "block_size": "--block_size",
    "block_count": "--block_count",
    "blur_kernel": "--blur_kernel",
    "depth_scale": "--depth_scale",
    "outlier_rate": "--outlier_rate",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        choices=("pilot", "representative", "core", "extended"),
        default="core",
    )
    parser.add_argument("--checkpoint-label", default="checkpoint")
    parser.add_argument("--config", default="local_configs.NYUDepthv2.DFormerv2_S")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--reserve-memory-mb", default=4096, type=int)
    parser.add_argument("--val-batch-size", default=1, type=int)
    parser.add_argument("--eval-scales", nargs="+", default=("1.0",))
    parser.add_argument("--flip", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--rerun", action="store_true", help="rerun conditions that already have a valid report")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running evaluation")
    return parser.parse_args()


def _values_match(reported, expected):
    if isinstance(expected, float):
        try:
            return abs(float(reported) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return reported == expected


def _latest_valid_report(condition_dir, condition, args):
    candidates = sorted(condition_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    checkpoint = args.checkpoint.resolve()
    for path in candidates:
        try:
            with path.open(encoding="utf-8") as input_file:
                data = json.load(input_file)
            summary = data["summary"]
            reported_condition = summary.get("condition_id", summary.get("depth_corruption"))
            reported_checkpoint = Path(summary["checkpoint"]).resolve()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        expected_fields = {
            "corruption_seed": args.seed,
            "depth_corruption": condition["depth_corruption"],
            **condition["parameters"],
        }
        fields_match = all(
            key in summary and _values_match(summary[key], value)
            for key, value in expected_fields.items()
        )
        if (
            reported_condition == condition["condition_id"]
            and reported_checkpoint == checkpoint
            and fields_match
        ):
            return path
    return None


def _write_manifest(path, args, conditions):
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "protocol": args.protocol,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint": str(args.checkpoint.resolve()),
        "git_commit": git_commit,
        "config": args.config,
        "corruption_seed": args.seed,
        "eval_scales": [float(scale) for scale in args.eval_scales],
        "flip": args.flip,
        "amp": args.amp,
        "sliding_window": args.sliding,
        "conditions": conditions,
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)


def _build_command(args, condition, condition_dir):
    command = [
        args.python,
        "-u",
        "utils/eval.py",
        f"--config={args.config}",
        "--gpus=1",
        f"--val_batch_size={args.val_batch_size}",
        f"--gpu_memory_reserve_mb={args.reserve_memory_mb}",
        "--no-syncbn",
        "--amp" if args.amp else "--no-amp",
        "--no-compile",
        "--no-mst",
        "--sliding" if args.sliding else "--no-sliding",
        "--flip" if args.flip else "--no-flip",
        f"--corruption_seed={args.seed}",
        f"--continue_fpath={args.checkpoint.resolve()}",
        f"--report_dir={condition_dir.resolve()}",
        f"--condition_id={condition['condition_id']}",
        f"--benchmark_protocol={args.protocol}",
        f"--depth_corruption={condition['depth_corruption']}",
        "--eval_scales",
        *[str(scale) for scale in args.eval_scales],
    ]
    for parameter, value in condition["parameters"].items():
        command.append(f"{PARAMETER_FLAGS[parameter]}={value}")
    return command


def _run_and_tee(command, log_path, env):
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    args.checkpoint = args.checkpoint.expanduser()
    args.report_root = args.report_root.expanduser().resolve()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.reserve_memory_mb < 0:
        raise ValueError("--reserve-memory-mb must be non-negative")
    if args.val_batch_size < 1:
        raise ValueError("--val-batch-size must be positive")
    if any(float(scale) <= 0 for scale in args.eval_scales):
        raise ValueError("--eval-scales values must be greater than zero")

    conditions = get_protocol(args.protocol)
    args.report_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.report_root / "depth_benchmark_manifest.json", args, conditions)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = 0
    skipped = 0
    for index, condition in enumerate(conditions, start=1):
        condition_id = condition["condition_id"]
        condition_dir = args.report_root / condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)
        existing = _latest_valid_report(condition_dir, condition, args)
        if existing is not None and not args.rerun:
            skipped += 1
            print(f"[{index}/{len(conditions)}] SKIP {condition_id}: {existing}", flush=True)
            continue

        command = _build_command(args, condition, condition_dir)
        print(f"\n[{index}/{len(conditions)}] RUN {condition_id}: {condition['label']}", flush=True)
        print("Command:", shlex.join(command), flush=True)
        if args.dry_run:
            continue

        return_code = _run_and_tee(command, condition_dir / "eval.log", env)
        if return_code != 0:
            raise RuntimeError(
                f"Condition {condition_id} failed with exit code {return_code}. "
                f"Resume the same command after fixing the issue; completed conditions will be skipped."
            )
        completed += 1

    if args.dry_run:
        print(f"\nDry run completed for {len(conditions)} conditions.")
        return

    summary_command = [args.python, "scripts/summarize_depth_benchmark.py", str(args.report_root)]
    subprocess.run(summary_command, check=True, env=env)
    print(
        f"\nDepth benchmark completed: {args.report_root} "
        f"(new={completed}, skipped={skipped}, total={len(conditions)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
