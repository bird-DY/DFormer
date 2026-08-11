#!/usr/bin/env python3
"""Summarize a formal multi-severity depth-corruption benchmark."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_json(path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _latest_report(condition_dir, condition_id):
    candidates = sorted(condition_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            report = _load_json(path)
            summary = report["summary"]
            reported_condition = summary.get("condition_id", summary.get("depth_corruption"))
            if reported_condition == condition_id and "per_class" in report:
                return path, report
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None, None


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(report_root, require_complete=True):
    report_root = report_root.resolve()
    manifest_path = report_root / "depth_benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)

    reports = {}
    report_paths = {}
    missing = []
    for condition in manifest["conditions"]:
        condition_id = condition["condition_id"]
        report_path, report = _latest_report(report_root / condition_id, condition_id)
        if report is None:
            missing.append(condition_id)
        else:
            reports[condition_id] = report
            report_paths[condition_id] = report_path

    if missing and require_complete:
        raise RuntimeError("Benchmark is incomplete; missing conditions: " + ", ".join(missing))
    if "clean" not in reports:
        raise RuntimeError("A completed clean condition is required")

    clean_miou = float(reports["clean"]["summary"]["miou_percent"])
    clean_classes = {
        int(row["class_id"]): row for row in reports["clean"]["per_class"]
    }
    summary_rows = []
    per_class_rows = []
    family_mious = defaultdict(list)
    family_drops = defaultdict(list)

    for condition in manifest["conditions"]:
        condition_id = condition["condition_id"]
        if condition_id not in reports:
            continue
        report = reports[condition_id]
        summary = report["summary"]
        miou = float(summary["miou_percent"])
        absolute_drop = clean_miou - miou
        relative_drop = 100.0 * absolute_drop / clean_miou if clean_miou else 0.0
        family = condition["family"]
        if family != "clean":
            family_mious[family].append(miou)
            family_drops[family].append(absolute_drop)

        summary_rows.append(
            {
                "condition_id": condition_id,
                "family": family,
                "severity": condition["severity"],
                "label": condition["label"],
                "depth_corruption": condition["depth_corruption"],
                "parameters_json": json.dumps(condition["parameters"], sort_keys=True),
                "miou_percent": round(miou, 2),
                "absolute_miou_drop": round(absolute_drop, 2),
                "relative_miou_drop_percent": round(relative_drop, 2),
                "mean_accuracy_percent": summary["mean_accuracy_percent"],
                "pixel_accuracy_percent": summary["pixel_accuracy_percent"],
                "mean_f1_percent": summary["mean_f1_percent"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "images_per_second": summary["images_per_second"],
                "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
                "report_json": str(report_paths[condition_id]),
            }
        )

        for class_result in report["per_class"]:
            class_id = int(class_result["class_id"])
            clean_iou = float(clean_classes[class_id]["iou_percent"])
            condition_iou = float(class_result["iou_percent"])
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_result["class_name"],
                    "condition_id": condition_id,
                    "family": family,
                    "severity": condition["severity"],
                    "iou_percent": round(condition_iou, 2),
                    "clean_iou_percent": round(clean_iou, 2),
                    "iou_drop": round(clean_iou - condition_iou, 2),
                    "precision_percent": class_result["precision_percent"],
                    "recall_percent": class_result["recall_percent"],
                    "f1_percent": class_result["f1_percent"],
                    "support_pixels": class_result["support_pixels"],
                }
            )

    family_rows = []
    for family in family_mious:
        values = family_mious[family]
        drops = family_drops[family]
        family_rows.append(
            {
                "family": family,
                "num_conditions": len(values),
                "mean_miou_percent": round(sum(values) / len(values), 2),
                "worst_miou_percent": round(min(values), 2),
                "best_miou_percent": round(max(values), 2),
                "mean_absolute_drop": round(sum(drops) / len(drops), 2),
                "worst_absolute_drop": round(max(drops), 2),
            }
        )

    corrupted_rows = [row for row in summary_rows if row["family"] != "clean"]
    condition_mean = (
        sum(float(row["miou_percent"]) for row in corrupted_rows) / len(corrupted_rows)
        if corrupted_rows
        else None
    )
    family_macro = (
        sum(float(row["mean_miou_percent"]) for row in family_rows) / len(family_rows)
        if family_rows
        else None
    )
    worst = min(corrupted_rows, key=lambda row: float(row["miou_percent"])) if corrupted_rows else None
    aggregate = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "checkpoint_label": manifest["checkpoint_label"],
        "checkpoint": manifest["checkpoint"],
        "clean_miou_percent": round(clean_miou, 2),
        "num_completed_conditions": len(summary_rows),
        "num_expected_conditions": len(manifest["conditions"]),
        "missing_conditions": missing,
        "condition_mean_corrupted_miou_percent": round(condition_mean, 2) if condition_mean is not None else None,
        "family_macro_corrupted_miou_percent": round(family_macro, 2) if family_macro is not None else None,
        "condition_mean_relative_robustness_percent": (
            round(100.0 * condition_mean / clean_miou, 2)
            if condition_mean is not None and clean_miou
            else None
        ),
        "worst_condition": worst,
        "families": family_rows,
        "conditions": summary_rows,
    }

    summary_path = report_root / "depth_benchmark_summary.csv"
    family_path = report_root / "depth_benchmark_family_summary.csv"
    per_class_path = report_root / "depth_benchmark_per_class.csv"
    aggregate_path = report_root / "depth_benchmark_aggregate.json"
    _write_csv(summary_path, summary_rows)
    _write_csv(family_path, family_rows)
    _write_csv(per_class_path, per_class_rows)
    with aggregate_path.open("w", encoding="utf-8") as output_file:
        json.dump(aggregate, output_file, ensure_ascii=False, indent=2)

    print("\nFormal depth benchmark summary")
    print("condition_id, mIoU, absolute drop, relative drop")
    for row in summary_rows:
        print(
            f"{row['condition_id']}, {float(row['miou_percent']):.2f}, "
            f"{float(row['absolute_miou_drop']):.2f}, "
            f"{float(row['relative_miou_drop_percent']):.2f}%"
        )
    print(f"\nCondition summary: {summary_path}")
    print(f"Family summary:    {family_path}")
    print(f"Per-class summary: {per_class_path}")
    print(f"Aggregate JSON:    {aggregate_path}")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summarize(args.report_root, require_complete=not args.allow_incomplete)


if __name__ == "__main__":
    main()
