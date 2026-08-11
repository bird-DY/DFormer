import json
import tempfile
import unittest
from pathlib import Path

from scripts.depth_benchmark_protocol import get_protocol
from scripts.summarize_depth_benchmark import summarize


def _report(condition_id, corruption, miou, class_ious):
    return {
        "summary": {
            "condition_id": condition_id,
            "depth_corruption": corruption,
            "miou_percent": miou,
            "mean_accuracy_percent": 70.0,
            "pixel_accuracy_percent": 80.0,
            "mean_f1_percent": 65.0,
            "elapsed_seconds": 10.0,
            "images_per_second": 2.0,
            "peak_gpu_memory_mb": 500.0,
        },
        "per_class": [
            {
                "class_id": index,
                "class_name": f"class_{index}",
                "iou_percent": value,
                "precision_percent": value + 1,
                "recall_percent": value + 2,
                "f1_percent": value + 3,
                "support_pixels": 100,
            }
            for index, value in enumerate(class_ious)
        ],
    }


class DepthBenchmarkProtocolTest(unittest.TestCase):
    def test_protocol_sizes_and_unique_ids(self):
        expected_sizes = {"pilot": 5, "representative": 6, "core": 18, "extended": 26}
        for name, expected_size in expected_sizes.items():
            conditions = get_protocol(name)
            condition_ids = [condition["condition_id"] for condition in conditions]
            self.assertEqual(len(conditions), expected_size)
            self.assertEqual(len(set(condition_ids)), expected_size)
            self.assertEqual(condition_ids[0], "clean")

    def test_extended_protocol_contains_core(self):
        core_ids = {condition["condition_id"] for condition in get_protocol("core")}
        extended_ids = {condition["condition_id"] for condition in get_protocol("extended")}
        self.assertTrue(core_ids.issubset(extended_ids))

    def test_summarizer_computes_condition_and_family_metrics(self):
        conditions = [
            {
                "condition_id": "clean",
                "family": "clean",
                "severity": 0,
                "label": "Clean",
                "depth_corruption": "clean",
                "parameters": {},
            },
            {
                "condition_id": "random_missing_r050",
                "family": "random_missing",
                "severity": 0.5,
                "label": "Random missing 50%",
                "depth_corruption": "random_missing",
                "parameters": {"missing_rate": 0.5},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = {
                "protocol": "test",
                "checkpoint_label": "test-weight",
                "checkpoint": "/tmp/test.pth",
                "conditions": conditions,
            }
            (root / "depth_benchmark_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for condition_id, corruption, miou, class_ious in (
                ("clean", "clean", 60.0, [50.0, 70.0]),
                ("random_missing_r050", "random_missing", 50.0, [40.0, 60.0]),
            ):
                condition_dir = root / condition_id
                condition_dir.mkdir()
                (condition_dir / "report.json").write_text(
                    json.dumps(_report(condition_id, corruption, miou, class_ious)),
                    encoding="utf-8",
                )

            aggregate = summarize(root)

            self.assertEqual(aggregate["clean_miou_percent"], 60.0)
            self.assertEqual(aggregate["condition_mean_corrupted_miou_percent"], 50.0)
            self.assertEqual(aggregate["condition_mean_relative_robustness_percent"], 83.33)
            self.assertTrue((root / "depth_benchmark_summary.csv").is_file())
            self.assertTrue((root / "depth_benchmark_per_class.csv").is_file())


if __name__ == "__main__":
    unittest.main()
