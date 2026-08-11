#!/usr/bin/env python3
"""Canonical condition definitions for the NYUv2 depth-robustness benchmark."""

import argparse
import json
from copy import deepcopy


def _condition(condition_id, family, severity, label, corruption, **parameters):
    return {
        "condition_id": condition_id,
        "family": family,
        "severity": severity,
        "label": label,
        "depth_corruption": corruption,
        "parameters": parameters,
    }


PILOT_CONDITIONS = [
    _condition("clean", "clean", 0, "Clean", "clean"),
    _condition(
        "random_missing_r030",
        "random_missing",
        0.30,
        "Random missing 30%",
        "random_missing",
        missing_rate=0.30,
    ),
    _condition(
        "gaussian_noise_s003",
        "gaussian_noise",
        0.03,
        "Gaussian noise sigma=0.03",
        "gaussian_noise",
        noise_std=0.03,
    ),
    _condition("shift_x04", "shift", 4, "Horizontal shift 4 px", "shift", shift_x=4, shift_y=0),
    _condition("zero", "missing_modality", 1, "All depth set to zero", "zero"),
]


REPRESENTATIVE_CONDITIONS = [
    _condition("clean", "clean", 0, "Clean", "clean"),
    _condition(
        "random_missing_r030",
        "random_missing",
        0.30,
        "Random missing 30%",
        "random_missing",
        missing_rate=0.30,
    ),
    _condition(
        "random_missing_r050",
        "random_missing",
        0.50,
        "Random missing 50%",
        "random_missing",
        missing_rate=0.50,
    ),
    _condition(
        "gaussian_noise_s005",
        "gaussian_noise",
        0.05,
        "Gaussian noise sigma=0.05",
        "gaussian_noise",
        noise_std=0.05,
    ),
    _condition("shift_x08", "shift", 8, "Horizontal shift 8 px", "shift", shift_x=8, shift_y=0),
    _condition("zero", "missing_modality", 1, "All depth set to zero", "zero"),
]


CORE_CONDITIONS = [
    _condition("clean", "clean", 0, "Clean", "clean"),
    *[
        _condition(
            f"random_missing_r{int(rate * 100):03d}",
            "random_missing",
            rate,
            f"Random missing {int(rate * 100)}%",
            "random_missing",
            missing_rate=rate,
        )
        for rate in (0.10, 0.30, 0.50)
    ],
    *[
        _condition(
            f"block_missing_s{size:03d}",
            "block_missing",
            size,
            f"One missing block {size}x{size}",
            "block_missing",
            block_size=size,
            block_count=1,
        )
        for size in (64, 128, 192)
    ],
    *[
        _condition(
            f"gaussian_noise_s{int(sigma * 100):03d}",
            "gaussian_noise",
            sigma,
            f"Gaussian noise sigma={sigma:.2f}",
            "gaussian_noise",
            noise_std=sigma,
        )
        for sigma in (0.01, 0.03, 0.05)
    ],
    *[
        _condition(
            f"gaussian_blur_k{kernel:02d}",
            "gaussian_blur",
            kernel,
            f"Gaussian blur kernel={kernel}",
            "gaussian_blur",
            blur_kernel=kernel,
        )
        for kernel in (3, 5, 9)
    ],
    *[
        _condition(
            f"shift_x{pixels:02d}",
            "shift",
            pixels,
            f"Horizontal shift {pixels} px",
            "shift",
            shift_x=pixels,
            shift_y=0,
        )
        for pixels in (2, 4, 8)
    ],
    _condition("zero", "missing_modality", 1, "All depth set to zero", "zero"),
    _condition("mean_fill", "missing_modality", 2, "All depth set to image mean", "mean_fill"),
]


EXTENDED_ONLY_CONDITIONS = [
    _condition(
        "random_missing_r070",
        "random_missing",
        0.70,
        "Random missing 70%",
        "random_missing",
        missing_rate=0.70,
    ),
    _condition(
        "gaussian_noise_s010",
        "gaussian_noise",
        0.10,
        "Gaussian noise sigma=0.10",
        "gaussian_noise",
        noise_std=0.10,
    ),
    _condition("shift_x16", "shift", 16, "Horizontal shift 16 px", "shift", shift_x=16, shift_y=0),
    _condition("depth_scale_x080", "depth_scale", 0.20, "Depth scale x0.80", "depth_scale", depth_scale=0.80),
    _condition("depth_scale_x120", "depth_scale", 0.20, "Depth scale x1.20", "depth_scale", depth_scale=1.20),
    *[
        _condition(
            f"random_outlier_r{int(rate * 100):03d}",
            "random_outlier",
            rate,
            f"Random depth outliers {int(rate * 100)}%",
            "random_outlier",
            outlier_rate=rate,
        )
        for rate in (0.01, 0.03, 0.05)
    ],
]


def get_protocol(name):
    if name == "pilot":
        return deepcopy(PILOT_CONDITIONS)
    if name == "representative":
        return deepcopy(REPRESENTATIVE_CONDITIONS)
    if name == "core":
        return deepcopy(CORE_CONDITIONS)
    if name == "extended":
        conditions = deepcopy(CORE_CONDITIONS)
        insertion_order = {
            "random_missing": 4,
            "gaussian_noise": 10,
            "shift": 16,
        }
        for condition in EXTENDED_ONLY_CONDITIONS:
            family = condition["family"]
            if family in insertion_order:
                conditions.insert(insertion_order[family], deepcopy(condition))
                for key in insertion_order:
                    if insertion_order[key] >= insertion_order[family] and key != family:
                        insertion_order[key] += 1
            else:
                conditions.append(deepcopy(condition))
        return conditions
    raise ValueError(f"Unknown protocol: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("pilot", "representative", "core", "extended"),
        default="core",
    )
    args = parser.parse_args()
    print(json.dumps(get_protocol(args.protocol), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
