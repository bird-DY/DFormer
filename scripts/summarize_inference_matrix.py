#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from tabulate import tabulate


WEIGHT_ORDER = ("author", "reproduced")
STRATEGY_ORDER = ("A_ss_noflip", "B_ss_flip", "C_ms_noflip", "D_ms_flip")


def load_report(report_dir):
    candidates = sorted(report_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    report_path = candidates[-1]
    with report_path.open("r", encoding="utf-8") as report_file:
        payload = json.load(report_file)
    return report_path, payload["summary"]


def main():
    parser = argparse.ArgumentParser(description="Summarize the NYUv2 inference-strategy matrix.")
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows = []
    missing = []
    for weight_name in WEIGHT_ORDER:
        for strategy in STRATEGY_ORDER:
            loaded = load_report(args.report_root / weight_name / strategy)
            if loaded is None:
                missing.append(f"{weight_name}/{strategy}")
                continue
            report_path, summary = loaded
            rows.append(
                {
                    "weight": weight_name,
                    "strategy": strategy,
                    "eval_scales": ",".join(str(scale) for scale in summary["eval_scales"]),
                    "flip": bool(summary["flip"]),
                    "sliding_window": bool(summary["sliding_window"]),
                    "amp": bool(summary["amp"]),
                    "miou_percent": float(summary["miou_percent"]),
                    "pixel_accuracy_percent": float(summary["pixel_accuracy_percent"]),
                    "elapsed_seconds": float(summary["elapsed_seconds"]),
                    "images_per_second": float(summary["images_per_second"]),
                    "peak_gpu_memory_mb": float(summary["peak_gpu_memory_mb"]),
                    "checkpoint": summary["checkpoint"],
                    "report_json": str(report_path.resolve()),
                }
            )

    if missing and not args.allow_incomplete:
        raise SystemExit("Missing reports: " + ", ".join(missing))
    if not rows:
        raise SystemExit(f"No evaluation reports found under {args.report_root}")

    args.report_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_root / "inference_matrix_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["weight"], row["strategy"]): row for row in rows}
    matrix_rows = []
    for weight_name in WEIGHT_ORDER:
        matrix_rows.append(
            [weight_name]
            + [
                lookup.get((weight_name, strategy), {}).get("miou_percent", "-")
                for strategy in STRATEGY_ORDER
            ]
        )

    print("\nInference strategy mIoU matrix (%)")
    print(tabulate(matrix_rows, headers=["weight", *STRATEGY_ORDER], tablefmt="github"))
    print("\nDetailed results")
    print(
        tabulate(
            [
                [
                    row["weight"],
                    row["strategy"],
                    row["miou_percent"],
                    row["elapsed_seconds"],
                    row["images_per_second"],
                    row["peak_gpu_memory_mb"],
                ]
                for row in rows
            ],
            headers=["weight", "strategy", "mIoU (%)", "seconds", "images/s", "peak GPU MiB"],
            tablefmt="github",
        )
    )
    if missing:
        print("\nIncomplete matrix; missing: " + ", ".join(missing))
    print(f"\nSummary CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
