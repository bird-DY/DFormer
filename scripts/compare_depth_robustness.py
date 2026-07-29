#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from tabulate import tabulate


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def main():
    parser = argparse.ArgumentParser(description="Compare author and reproduced depth-robustness runs.")
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument("--reproduced-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    author_root = args.author_root.resolve()
    reproduced_root = args.reproduced_root.resolve()
    output_dir = (args.output_dir or reproduced_root.parent / "depth_robustness_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    author_summary = {row["condition"]: row for row in read_csv(author_root / "depth_robustness_summary.csv")}
    reproduced_summary = {
        row["condition"]: row for row in read_csv(reproduced_root / "depth_robustness_summary.csv")
    }
    conditions = [condition for condition in author_summary if condition in reproduced_summary]
    if "clean" not in conditions:
        raise RuntimeError("Both runs must contain the clean condition")

    summary_rows = []
    for condition in conditions:
        author = author_summary[condition]
        reproduced = reproduced_summary[condition]
        author_miou = float(author["miou_percent"])
        reproduced_miou = float(reproduced["miou_percent"])
        author_drop = float(author["absolute_miou_drop"])
        reproduced_drop = float(reproduced["absolute_miou_drop"])
        summary_rows.append(
            {
                "condition": condition,
                "author_miou_percent": round(author_miou, 2),
                "reproduced_miou_percent": round(reproduced_miou, 2),
                "reproduced_minus_author_miou": round(reproduced_miou - author_miou, 2),
                "author_absolute_drop": round(author_drop, 2),
                "reproduced_absolute_drop": round(reproduced_drop, 2),
                "reproduced_minus_author_drop": round(reproduced_drop - author_drop, 2),
                "author_relative_drop_percent": float(author["relative_miou_drop_percent"]),
                "reproduced_relative_drop_percent": float(reproduced["relative_miou_drop_percent"]),
            }
        )

    summary_path = output_dir / "depth_robustness_checkpoint_comparison.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    author_classes = {int(row["class_id"]): row for row in read_csv(author_root / "depth_robustness_per_class_iou.csv")}
    reproduced_classes = {
        int(row["class_id"]): row for row in read_csv(reproduced_root / "depth_robustness_per_class_iou.csv")
    }
    class_rows = []
    for class_id, author in author_classes.items():
        reproduced = reproduced_classes[class_id]
        for condition in conditions:
            author_iou = float(author[f"{condition}_iou_percent"])
            reproduced_iou = float(reproduced[f"{condition}_iou_percent"])
            author_drop = float(author[f"{condition}_iou_drop"])
            reproduced_drop = float(reproduced[f"{condition}_iou_drop"])
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": author["class_name"],
                    "condition": condition,
                    "author_iou_percent": round(author_iou, 2),
                    "reproduced_iou_percent": round(reproduced_iou, 2),
                    "reproduced_minus_author_iou": round(reproduced_iou - author_iou, 2),
                    "author_iou_drop": round(author_drop, 2),
                    "reproduced_iou_drop": round(reproduced_drop, 2),
                    "reproduced_minus_author_drop": round(reproduced_drop - author_drop, 2),
                }
            )

    class_path = output_dir / "depth_robustness_per_class_comparison.csv"
    with class_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)

    print("\nDepth robustness checkpoint comparison")
    print(
        tabulate(
            [
                [
                    row["condition"],
                    row["author_miou_percent"],
                    row["reproduced_miou_percent"],
                    row["author_absolute_drop"],
                    row["reproduced_absolute_drop"],
                ]
                for row in summary_rows
            ],
            headers=["condition", "author mIoU", "reproduced mIoU", "author drop", "reproduced drop"],
            tablefmt="github",
        )
    )
    print(f"\nSummary comparison: {summary_path}")
    print(f"Per-class comparison: {class_path}")


if __name__ == "__main__":
    main()
