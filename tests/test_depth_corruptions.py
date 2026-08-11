import unittest

import numpy as np

from utils.dataloader.depth_corruptions import DepthCorruptor


class DepthCorruptorTest(unittest.TestCase):
    def test_random_missing_is_exact_and_deterministic(self):
        depth = np.full((10, 10), 100, dtype=np.uint8)
        corruptor = DepthCorruptor("random_missing", seed=7, missing_rate=0.3)

        first = corruptor(depth, sample_id="sample-a")
        second = corruptor(depth, sample_id="sample-a")

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(np.count_nonzero(first == 0), 30)
        self.assertTrue(np.all(depth == 100))

    def test_gaussian_noise_preserves_invalid_pixels_and_dtype(self):
        depth = np.full((16, 16), 128, dtype=np.uint8)
        depth[0, 0] = 0
        corruptor = DepthCorruptor("gaussian_noise", seed=11, noise_std=0.03)

        output = corruptor(depth, sample_id="sample-b")

        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output[0, 0], 0)
        self.assertFalse(np.array_equal(output[1:, 1:], depth[1:, 1:]))

    def test_shift_uses_zero_padding(self):
        depth = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
        corruptor = DepthCorruptor("shift", shift_x=1, shift_y=0)

        output = corruptor(depth, sample_id="sample-c")

        expected = np.array([[0, 1, 2], [0, 4, 5], [0, 7, 8]], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_zero_removes_all_depth(self):
        depth = np.full((4, 5), 200, dtype=np.uint8)
        output = DepthCorruptor("zero")(depth, sample_id="sample-d")
        self.assertEqual(np.count_nonzero(output), 0)

    def test_block_missing_is_deterministic_and_has_requested_area(self):
        depth = np.full((20, 30), 100, dtype=np.uint8)
        corruptor = DepthCorruptor("block_missing", seed=13, block_size=8, block_count=1)

        first = corruptor(depth, sample_id="sample-e")
        second = corruptor(depth, sample_id="sample-e")

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(np.count_nonzero(first == 0), 64)

    def test_gaussian_blur_preserves_dtype_and_changes_depth_edges(self):
        depth = np.full((11, 11), 50, dtype=np.uint8)
        depth[5, 5] = 250
        output = DepthCorruptor("gaussian_blur", blur_kernel=5)(depth, sample_id="sample-f")

        self.assertEqual(output.dtype, np.uint8)
        self.assertLess(output[5, 5], depth[5, 5])
        self.assertGreater(output[5, 4], depth[5, 4])

    def test_mean_fill_removes_spatial_geometry(self):
        depth = np.array([[0, 10], [20, 30]], dtype=np.uint8)
        output = DepthCorruptor("mean_fill")(depth, sample_id="sample-g")

        self.assertTrue(np.all(output == 20))

    def test_depth_scale_preserves_invalid_pixels_and_clips(self):
        depth = np.array([[0, 100], [200, 250]], dtype=np.uint8)
        output = DepthCorruptor("depth_scale", depth_scale=1.2)(depth, sample_id="sample-h")

        expected = np.array([[0, 120], [240, 255]], dtype=np.uint8)
        self.assertTrue(np.array_equal(output, expected))

    def test_random_outlier_count_and_determinism(self):
        depth = np.full((10, 10), 100, dtype=np.uint8)
        corruptor = DepthCorruptor("random_outlier", seed=17, outlier_rate=0.1)

        first = corruptor(depth, sample_id="sample-i")
        second = corruptor(depth, sample_id="sample-i")

        self.assertTrue(np.array_equal(first, second))
        # A generated outlier can theoretically equal 100, so inspect selected changes
        # conservatively while ensuring the corruption is active.
        self.assertGreater(np.count_nonzero(first != depth), 0)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            DepthCorruptor("gaussian_blur", blur_kernel=4)
        with self.assertRaises(ValueError):
            DepthCorruptor("block_missing", block_size=0)
        with self.assertRaises(ValueError):
            DepthCorruptor("depth_scale", depth_scale=0)
        with self.assertRaises(ValueError):
            DepthCorruptor("random_outlier", outlier_rate=1.1)


if __name__ == "__main__":
    unittest.main()
