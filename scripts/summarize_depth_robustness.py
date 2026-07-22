import argparse
import csv
import json
from pathlib import Path


CONDITION_ORDER = ["clean", "random_missing", "gaussian_noise", "shift", "zero"]


def load_reports(report_root):
    reports = {}
    for path in report_root.rglob("*.json"):
        if path.name == "depth_robustness_aggregate.json":
            continue
        with path.open(encoding="utf-8") as report_file:
            data = json.load(report_file)
        if "summary" not in data or "per_class" not in data:
            continue
        condition = data["summary"].get("depth_corruption")
        if condition in CONDITION_ORDER:
            reports[condition] = data
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()

    report_root = args.report_root.resolve()
    reports = load_reports(report_root)
    if "clean" not in reports:
        raise RuntimeError(f"No clean report found under {report_root}")

    clean_miou = float(reports["clean"]["summary"]["miou_percent"])
    summary_rows = []
    corrupted_mious = []
    for condition in CONDITION_ORDER:
        if condition not in reports:
            continue
        summary = reports[condition]["summary"]
        miou = float(summary["miou_percent"])
        absolute_drop = clean_miou - miou
        relative_drop = 100.0 * absolute_drop / clean_miou if clean_miou else 0.0
        if condition != "clean":
            corrupted_mious.append(miou)
        summary_rows.append(
            {
                "condition": condition,
                "miou_percent": round(miou, 2),
                "absolute_miou_drop": round(absolute_drop, 2),
                "relative_miou_drop_percent": round(relative_drop, 2),
                "mean_accuracy_percent": summary["mean_accuracy_percent"],
                "pixel_accuracy_percent": summary["pixel_accuracy_percent"],
                "mean_f1_percent": summary["mean_f1_percent"],
                "images_per_second": summary["images_per_second"],
                "peak_gpu_memory_mb": summary["peak_gpu_memory_mb"],
            }
        )

    summary_csv = report_root / "depth_robustness_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    class_lookup = {
        condition: {row["class_id"]: row for row in report["per_class"]}
        for condition, report in reports.items()
    }
    per_class_rows = []
    for clean_class in reports["clean"]["per_class"]:
        class_id = clean_class["class_id"]
        row = {
            "class_id": class_id,
            "class_name": clean_class["class_name"],
        }
        clean_iou = float(clean_class["iou_percent"])
        for condition in CONDITION_ORDER:
            if condition not in class_lookup:
                continue
            condition_iou = float(class_lookup[condition][class_id]["iou_percent"])
            row[f"{condition}_iou_percent"] = round(condition_iou, 2)
            row[f"{condition}_iou_drop"] = round(clean_iou - condition_iou, 2)
        per_class_rows.append(row)

    per_class_csv = report_root / "depth_robustness_per_class_iou.csv"
    with per_class_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(per_class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_class_rows)

    aggregate = {
        "clean_miou_percent": round(clean_miou, 2),
        "mean_corrupted_miou_percent": (
            round(sum(corrupted_mious) / len(corrupted_mious), 2) if corrupted_mious else None
        ),
        "conditions": summary_rows,
    }
    aggregate_json = report_root / "depth_robustness_aggregate.json"
    with aggregate_json.open("w", encoding="utf-8") as output_file:
        json.dump(aggregate, output_file, ensure_ascii=False, indent=2)

    print("\nDepth robustness summary")
    print("condition, mIoU, absolute drop, relative drop")
    for row in summary_rows:
        print(
            f"{row['condition']}, {row['miou_percent']:.2f}, "
            f"{row['absolute_miou_drop']:.2f}, {row['relative_miou_drop_percent']:.2f}%"
        )
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Per-class CSV: {per_class_csv}")
    print(f"Aggregate JSON: {aggregate_json}")


if __name__ == "__main__":
    main()
