import hashlib

import cv2
import numpy as np


DEPTH_CORRUPTIONS = (
    "clean",
    "random_missing",
    "block_missing",
    "gaussian_noise",
    "gaussian_blur",
    "shift",
    "zero",
    "mean_fill",
    "depth_scale",
    "random_outlier",
)


TRAINING_DEPTH_CORRUPTION_PROFILES = {
    "representative": (
        ("random_missing_r030", "random_missing", {"missing_rate": 0.30}),
        ("random_missing_r050", "random_missing", {"missing_rate": 0.50}),
        ("gaussian_noise_s003", "gaussian_noise", {"noise_std": 0.03}),
        ("gaussian_noise_s005", "gaussian_noise", {"noise_std": 0.05}),
        ("shift_x04", "shift", {"shift_x": 4, "shift_y": 0}),
        ("shift_x08", "shift", {"shift_x": 8, "shift_y": 0}),
        ("zero", "zero", {}),
    ),
}


class DepthCorruptor:
    """Apply a deterministic corruption to an unnormalized single-channel depth image.

    NYUv2 depth images in this repository are read as 8-bit grayscale images. Noise
    magnitudes are therefore expressed relative to the 255-value input range. Invalid
    source pixels are represented by zero and are preserved by corruptions other than
    complete-modality replacements (``zero`` and ``mean_fill``).
    """

    def __init__(
        self,
        name="clean",
        seed=12345,
        missing_rate=0.3,
        noise_std=0.03,
        shift_x=4,
        shift_y=0,
        block_size=128,
        block_count=1,
        blur_kernel=5,
        depth_scale=1.0,
        outlier_rate=0.03,
    ):
        if name not in DEPTH_CORRUPTIONS:
            raise ValueError(f"Unknown depth corruption: {name}")
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be in [0, 1]")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if block_count < 1:
            raise ValueError("block_count must be a positive integer")
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if depth_scale <= 0.0:
            raise ValueError("depth_scale must be greater than zero")
        if not 0.0 <= outlier_rate <= 1.0:
            raise ValueError("outlier_rate must be in [0, 1]")

        self.name = name
        self.seed = int(seed)
        self.missing_rate = float(missing_rate)
        self.noise_std = float(noise_std)
        self.shift_x = int(shift_x)
        self.shift_y = int(shift_y)
        self.block_size = int(block_size)
        self.block_count = int(block_count)
        self.blur_kernel = int(blur_kernel)
        self.depth_scale = float(depth_scale)
        self.outlier_rate = float(outlier_rate)

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

    @staticmethod
    def _cast_like(values, reference):
        if np.issubdtype(reference.dtype, np.integer):
            limits = np.iinfo(reference.dtype)
            values = np.clip(np.rint(values), limits.min, limits.max)
        return values.astype(reference.dtype)

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

        if self.name == "mean_fill":
            if not np.any(valid_mask):
                return np.zeros_like(depth)
            mean_depth = float(depth[valid_mask].mean())
            return self._cast_like(np.full(depth.shape, mean_depth, dtype=np.float32), depth)

        if self.name == "random_missing":
            output = depth.copy()
            missing_count = int(round(self.missing_rate * len(valid_indices)))
            if missing_count > 0:
                missing_indices = rng.choice(valid_indices, size=missing_count, replace=False)
                output.flat[missing_indices] = 0
            return output

        if self.name == "block_missing":
            output = depth.copy()
            height, width = output.shape
            block_height = min(self.block_size, height)
            block_width = min(self.block_size, width)
            for _ in range(self.block_count):
                top = int(rng.integers(0, height - block_height + 1))
                left = int(rng.integers(0, width - block_width + 1))
                output[top : top + block_height, left : left + block_width] = 0
            return output

        if self.name == "gaussian_noise":
            output = depth.astype(np.float32, copy=True)
            noise = rng.normal(0.0, self.noise_std * 255.0, size=depth.shape)
            output[valid_mask] += noise[valid_mask]
            output[~valid_mask] = 0.0
            return self._cast_like(output, depth)

        if self.name == "gaussian_blur":
            blurred_depth = cv2.GaussianBlur(
                depth.astype(np.float32),
                (self.blur_kernel, self.blur_kernel),
                sigmaX=0,
                borderType=cv2.BORDER_REFLECT_101,
            )
            blurred_validity = cv2.GaussianBlur(
                valid_mask.astype(np.float32),
                (self.blur_kernel, self.blur_kernel),
                sigmaX=0,
                borderType=cv2.BORDER_REFLECT_101,
            )
            output = np.zeros_like(blurred_depth)
            np.divide(
                blurred_depth,
                blurred_validity,
                out=output,
                where=blurred_validity > np.finfo(np.float32).eps,
            )
            output[~valid_mask] = 0.0
            return self._cast_like(output, depth)

        if self.name == "depth_scale":
            output = depth.astype(np.float32, copy=True)
            output[valid_mask] *= self.depth_scale
            output[~valid_mask] = 0.0
            return self._cast_like(output, depth)

        if self.name == "random_outlier":
            output = depth.copy()
            outlier_count = int(round(self.outlier_rate * len(valid_indices)))
            if outlier_count > 0:
                outlier_indices = rng.choice(valid_indices, size=outlier_count, replace=False)
                if np.issubdtype(depth.dtype, np.integer):
                    upper_bound = int(np.iinfo(depth.dtype).max)
                else:
                    upper_bound = 255
                values = rng.integers(1, upper_bound + 1, size=outlier_count)
                output.flat[outlier_indices] = values.astype(depth.dtype)
            return output

        raise AssertionError(f"Unhandled depth corruption: {self.name}")


class RandomTrainingDepthAugmentor:
    """Randomly corrupt training depth while preserving clean RGB and labels.

    A dedicated RNG makes the augmentation sequence repeatable for a fixed seed.
    The current NYUv2 configs use ``num_workers=0``, so the RNG has a single,
    deterministic owner throughout training.
    """

    def __init__(self, probability=0.5, profile="representative", seed=12345):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("training depth corruption probability must be in [0, 1]")
        if profile not in TRAINING_DEPTH_CORRUPTION_PROFILES:
            raise ValueError(f"Unknown training depth corruption profile: {profile}")

        self.probability = float(probability)
        self.profile = profile
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._conditions = []
        for condition_id, name, parameters in TRAINING_DEPTH_CORRUPTION_PROFILES[profile]:
            self._conditions.append(
                (
                    condition_id,
                    DepthCorruptor(name=name, seed=self.seed, **parameters),
                )
            )

    @property
    def condition_ids(self):
        return tuple(condition_id for condition_id, _ in self._conditions)

    def __call__(self, depth, sample_id):
        if self._rng.random() >= self.probability:
            return depth

        condition_index = int(self._rng.integers(0, len(self._conditions)))
        condition_id, corruptor = self._conditions[condition_index]
        draw_id = int(self._rng.integers(0, np.iinfo(np.int64).max))
        augmented_sample_id = f"train:{condition_id}:{sample_id}:{draw_id}"
        return corruptor(depth, sample_id=augmented_sample_id)


def build_depth_corruptor(args):
    return DepthCorruptor(
        name=args.depth_corruption,
        seed=args.corruption_seed,
        missing_rate=args.missing_rate,
        noise_std=args.noise_std,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        block_size=args.block_size,
        block_count=args.block_count,
        blur_kernel=args.blur_kernel,
        depth_scale=args.depth_scale,
        outlier_rate=args.outlier_rate,
    )
