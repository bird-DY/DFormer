#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from tabulate import tabulate


WEIGHTS = ("author", "reproduced")
STRATEGIES = ("A_ss_noflip", "B_ss_flip", "C_ms_noflip", "D_ms_flip")


def load_latest_report(report_dir):
    reports = sorted(report_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(f"No JSON report found under {report_dir}")
    with reports[-1].open(encoding="utf-8") as report_file:
        return json.load(report_file)


def main():
    parser = argparse.ArgumentParser(description="Analyze per-class IoU for the 2x4 inference matrix.")
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    report_root = args.report_root.resolve()

    reports = {
        (weight, strategy): load_latest_report(report_root / weight / strategy)
        for weight in WEIGHTS
        for strategy in STRATEGIES
    }
    class_lookup = {
        key: {int(row["class_id"]): row for row in report["per_class"]}
        for key, report in reports.items()
    }
    reference = reports[("author", "A_ss_noflip")]["per_class"]

    long_rows = []
    comparison_rows = []
    for reference_class in reference:
        class_id = int(reference_class["class_id"])
        class_name = reference_class["class_name"]
        row = {"class_id": class_id, "class_name": class_name}

        for weight in WEIGHTS:
            for strategy in STRATEGIES:
                values = class_lookup[(weight, strategy)][class_id]
                iou = float(values["iou_percent"])
                row[f"{weight}_{strategy}_iou"] = round(iou, 2)
                long_rows.append(
                    {
                        "weight": weight,
                        "strategy": strategy,
                        "class_id": class_id,
                        "class_name": class_name,
                        "iou_percent": round(iou, 2),
                        "precision_percent": values["precision_percent"],
                        "recall_percent": values["recall_percent"],
                        "f1_percent": values["f1_percent"],
                        "support_pixels": values["support_pixels"],
                    }
                )

        for strategy in STRATEGIES:
            row[f"reproduced_minus_author_{strategy}"] = round(
                row[f"reproduced_{strategy}_iou"] - row[f"author_{strategy}_iou"], 2
            )
        for weight in WEIGHTS:
            baseline = row[f"{weight}_A_ss_noflip_iou"]
            row[f"{weight}_flip_gain"] = round(row[f"{weight}_B_ss_flip_iou"] - baseline, 2)
            row[f"{weight}_multiscale_gain"] = round(
                row[f"{weight}_C_ms_noflip_iou"] - baseline, 2
            )
            row[f"{weight}_full_tta_gain"] = round(row[f"{weight}_D_ms_flip_iou"] - baseline, 2)
        comparison_rows.append(row)

    long_path = report_root / "inference_matrix_per_class_long.csv"
    with long_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    comparison_path = report_root / "inference_matrix_per_class_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    top_checkpoint_gains = sorted(
        comparison_rows,
        key=lambda row: row["reproduced_minus_author_D_ms_flip"],
        reverse=True,
    )[:10]
    print("\nTop reproduced-checkpoint gains under multi-scale flip inference")
    print(
        tabulate(
            [
                [
                    row["class_name"],
                    row["author_D_ms_flip_iou"],
                    row["reproduced_D_ms_flip_iou"],
                    row["reproduced_minus_author_D_ms_flip"],
                ]
                for row in top_checkpoint_gains
            ],
            headers=["class", "author IoU", "reproduced IoU", "gain"],
            tablefmt="github",
        )
    )
    largest_declines = sorted(
        comparison_rows,
        key=lambda row: row["reproduced_minus_author_D_ms_flip"],
    )[:10]
    print("\nLargest reproduced-checkpoint declines under multi-scale flip inference")
    print(
        tabulate(
            [
                [
                    row["class_name"],
                    row["author_D_ms_flip_iou"],
                    row["reproduced_D_ms_flip_iou"],
                    row["reproduced_minus_author_D_ms_flip"],
                ]
                for row in largest_declines
            ],
            headers=["class", "author IoU", "reproduced IoU", "gain"],
            tablefmt="github",
        )
    )
    top_tta_gains = sorted(
        comparison_rows,
        key=lambda row: row["reproduced_full_tta_gain"],
        reverse=True,
    )[:10]
    print("\nClasses benefiting most from full TTA on the reproduced checkpoint")
    print(
        tabulate(
            [
                [
                    row["class_name"],
                    row["reproduced_A_ss_noflip_iou"],
                    row["reproduced_D_ms_flip_iou"],
                    row["reproduced_full_tta_gain"],
                ]
                for row in top_tta_gains
            ],
            headers=["class", "single-scale IoU", "MS+flip IoU", "TTA gain"],
            tablefmt="github",
        )
    )
    d_gains = [row["reproduced_minus_author_D_ms_flip"] for row in comparison_rows]
    print(
        "\nD-protocol class summary: "
        f"improved={sum(gain > 0 for gain in d_gains)}, "
        f"unchanged={sum(gain == 0 for gain in d_gains)}, "
        f"declined={sum(gain < 0 for gain in d_gains)}, "
        f"improved_by_at_least_1={sum(gain >= 1 for gain in d_gains)}"
    )
    print(f"\nLong-form CSV: {long_path}")
    print(f"Comparison CSV: {comparison_path}")


if __name__ == "__main__":
    main()
