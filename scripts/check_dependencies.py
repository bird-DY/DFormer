#!/usr/bin/env python3
"""Run pip check while accepting MMCV's headless OpenCV equivalent."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys


HEADLESS_EQUIVALENT_MESSAGES = {
    "mmcv-full 1.7.2 requires opencv-python, which is not installed.",
    "mmengine 0.10.4 requires opencv-python, which is not installed.",
}


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    ignored_messages = HEADLESS_EQUIVALENT_MESSAGES.intersection(lines)
    if ignored_messages:
        import cv2

        headless_version = importlib.metadata.version("opencv-python-headless")
        module_matches_distribution = (
            cv2.__version__ == headless_version
            or headless_version.startswith(cv2.__version__ + ".")
        )
        if not module_matches_distribution:
            raise RuntimeError(
                "OpenCV module/distribution mismatch: "
                f"cv2={cv2.__version__}, opencv-python-headless={headless_version}"
            )
        for message in ignored_messages:
            lines.remove(message)
        print(
            "Accepted OpenMMLab's opencv-python metadata requirement because "
            f"opencv-python-headless {headless_version} is installed."
        )

    if lines:
        raise RuntimeError("Dependency conflicts detected:\n  " + "\n  ".join(lines))
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    print("Dependency metadata check passed")


if __name__ == "__main__":
    main()
