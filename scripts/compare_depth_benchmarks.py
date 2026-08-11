#!/usr/bin/env python3
"""Compare two formal depth-corruption benchmark reports condition by condition."""

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--reference-label", default="author")
    parser.add_argument("--candidate-label", default="reproduced")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    reference_root = args.reference_root.resolve()
    candidate_root = args.candidate_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = {
        row["condition_id"]: row
        for row in _read_csv(reference_root / "depth_benchmark_summary.csv")
    }
    candidate_rows = {
        row["condition_id"]: row
        for row in _read_csv(candidate_root / "depth_benchmark_summary.csv")
    }
    common_conditions = [condition for condition in reference_rows if condition in candidate_rows]
    if "clean" not in common_conditions:
        raise RuntimeError("Both benchmarks must contain a clean condition")

    summary_rows = []
    for condition_id in common_conditions:
        reference = reference_rows[condition_id]
        candidate = candidate_rows[condition_id]
        reference_miou = float(reference["miou_percent"])
        candidate_miou = float(candidate["miou_percent"])
        reference_drop = float(reference["absolute_miou_drop"])
        candidate_drop = float(candidate["absolute_miou_drop"])
        summary_rows.append(
            {
                "condition_id": condition_id,
                "family": reference["family"],
                "severity": reference["severity"],
                "reference_label": args.reference_label,
                "candidate_label": args.candidate_label,
                "reference_miou_percent": round(reference_miou, 2),
                "candidate_miou_percent": round(candidate_miou, 2),
                "candidate_minus_reference_miou": round(candidate_miou - reference_miou, 2),
                "reference_absolute_drop": round(reference_drop, 2),
                "candidate_absolute_drop": round(candidate_drop, 2),
                "candidate_minus_reference_drop": round(candidate_drop - reference_drop, 2),
                "candidate_robustness_advantage": round(reference_drop - candidate_drop, 2),
            }
        )

    reference_classes = {
        (int(row["class_id"]), row["condition_id"]): row
        for row in _read_csv(reference_root / "depth_benchmark_per_class.csv")
    }
    candidate_classes = {
        (int(row["class_id"]), row["condition_id"]): row
        for row in _read_csv(candidate_root / "depth_benchmark_per_class.csv")
    }
    class_rows = []
    for key, reference in reference_classes.items():
        if key not in candidate_classes or key[1] not in common_conditions:
            continue
        candidate = candidate_classes[key]
        reference_iou = float(reference["iou_percent"])
        candidate_iou = float(candidate["iou_percent"])
        reference_drop = float(reference["iou_drop"])
        candidate_drop = float(candidate["iou_drop"])
        class_rows.append(
            {
                "class_id": key[0],
                "class_name": reference["class_name"],
                "condition_id": key[1],
                "family": reference["family"],
                "severity": reference["severity"],
                "reference_iou_percent": round(reference_iou, 2),
                "candidate_iou_percent": round(candidate_iou, 2),
                "candidate_minus_reference_iou": round(candidate_iou - reference_iou, 2),
                "reference_iou_drop": round(reference_drop, 2),
                "candidate_iou_drop": round(candidate_drop, 2),
                "candidate_robustness_advantage": round(reference_drop - candidate_drop, 2),
            }
        )

    reference_families = {
        row["family"]: row
        for row in _read_csv(reference_root / "depth_benchmark_family_summary.csv")
    }
    candidate_families = {
        row["family"]: row
        for row in _read_csv(candidate_root / "depth_benchmark_family_summary.csv")
    }
    family_rows = []
    for family, reference in reference_families.items():
        if family not in candidate_families:
            continue
        candidate = candidate_families[family]
        reference_mean = float(reference["mean_miou_percent"])
        candidate_mean = float(candidate["mean_miou_percent"])
        reference_drop = float(reference["mean_absolute_drop"])
        candidate_drop = float(candidate["mean_absolute_drop"])
        family_rows.append(
            {
                "family": family,
                "reference_mean_miou_percent": round(reference_mean, 2),
                "candidate_mean_miou_percent": round(candidate_mean, 2),
                "candidate_minus_reference_miou": round(candidate_mean - reference_mean, 2),
                "reference_mean_drop": round(reference_drop, 2),
                "candidate_mean_drop": round(candidate_drop, 2),
                "candidate_robustness_advantage": round(reference_drop - candidate_drop, 2),
            }
        )

    summary_path = output_dir / "depth_benchmark_checkpoint_comparison.csv"
    family_path = output_dir / "depth_benchmark_family_comparison.csv"
    class_path = output_dir / "depth_benchmark_per_class_comparison.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(family_path, family_rows)
    _write_csv(class_path, class_rows)

    aggregate = {
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "num_common_conditions": len(common_conditions),
        "conditions": summary_rows,
        "families": family_rows,
    }
    aggregate_path = output_dir / "depth_benchmark_comparison.json"
    with aggregate_path.open("w", encoding="utf-8") as output_file:
        json.dump(aggregate, output_file, ensure_ascii=False, indent=2)

    print("\nDepth benchmark checkpoint comparison")
    print("condition_id, reference mIoU, candidate mIoU, reference drop, candidate drop")
    for row in summary_rows:
        print(
            f"{row['condition_id']}, {row['reference_miou_percent']:.2f}, "
            f"{row['candidate_miou_percent']:.2f}, {row['reference_absolute_drop']:.2f}, "
            f"{row['candidate_absolute_drop']:.2f}"
        )
    print(f"\nCondition comparison: {summary_path}")
    print(f"Family comparison:    {family_path}")
    print(f"Per-class comparison: {class_path}")
    print(f"Aggregate JSON:        {aggregate_path}")


if __name__ == "__main__":
    main()
