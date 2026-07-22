import hashlib

import numpy as np


DEPTH_CORRUPTIONS = ("clean", "random_missing", "gaussian_noise", "shift", "zero")


class DepthCorruptor:
    """Apply a deterministic corruption to an unnormalized single-channel depth image."""

    def __init__(
        self,
        name="clean",
        seed=12345,
        missing_rate=0.3,
        noise_std=0.03,
        shift_x=4,
        shift_y=0,
    ):
        if name not in DEPTH_CORRUPTIONS:
            raise ValueError(f"Unknown depth corruption: {name}")
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be in [0, 1]")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")

        self.name = name
        self.seed = int(seed)
        self.missing_rate = float(missing_rate)
        self.noise_std = float(noise_std)
        self.shift_x = int(shift_x)
        self.shift_y = int(shift_y)

    def _rng(self, sample_id):
        key = f"{self.name}:{self.seed}:{sample_id}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, byteorder="little", signed=False))

    @staticmethod
    def _shift_with_zeros(depth, shift_x, shift_y):
        height, width = depth.shape
        output = np.zeros_like(depth)

        source_x_start = max(0, -shift_x)
        source_x_end = min(width, width - shift_x)
        source_y_start = max(0, -shift_y)
        source_y_end = min(height, height - shift_y)

        if source_x_start >= source_x_end or source_y_start >= source_y_end:
            return output

        target_x_start = source_x_start + shift_x
        target_x_end = source_x_end + shift_x
        target_y_start = source_y_start + shift_y
        target_y_end = source_y_end + shift_y
        output[target_y_start:target_y_end, target_x_start:target_x_end] = depth[
            source_y_start:source_y_end,
            source_x_start:source_x_end,
        ]
        return output

    def __call__(self, depth, sample_id):
        if depth.ndim != 2:
            raise ValueError(f"DepthCorruptor expects a 2D image, got shape {depth.shape}")
        if self.name == "clean":
            return depth
        if self.name == "zero":
            return np.zeros_like(depth)
        if self.name == "shift":
            return self._shift_with_zeros(depth, self.shift_x, self.shift_y)

        valid_mask = depth > 0
        valid_indices = np.flatnonzero(valid_mask)
        rng = self._rng(sample_id)

        if self.name == "random_missing":
            output = depth.copy()
            missing_count = int(round(self.missing_rate * len(valid_indices)))
            if missing_count > 0:
                missing_indices = rng.choice(valid_indices, size=missing_count, replace=False)
                output.flat[missing_indices] = 0
            return output

        if self.name == "gaussian_noise":
            output = depth.astype(np.float32, copy=True)
            noise = rng.normal(0.0, self.noise_std * 255.0, size=depth.shape)
            output[valid_mask] += noise[valid_mask]
            output = np.clip(np.rint(output), 0.0, 255.0)
            output[~valid_mask] = 0.0
            return output.astype(depth.dtype)

        raise AssertionError(f"Unhandled depth corruption: {self.name}")


def build_depth_corruptor(args):
    return DepthCorruptor(
        name=args.depth_corruption,
        seed=args.corruption_seed,
        missing_rate=args.missing_rate,
        noise_std=args.noise_std,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
    )
