from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.navdp.trajectory_smoothing import smooth_trajectory_xy


class TrajectorySmoothingTests(unittest.TestCase):
    def test_smoothing_preserves_shape_endpoints_and_deviation_limit(self) -> None:
        x = np.linspace(0.1, 1.0, 21)
        macro_curve = 0.05 * np.sin(np.linspace(0.0, np.pi, 21))
        pointwise_zigzag = 0.012 * np.where(np.arange(21) % 2 == 0, 1.0, -1.0)
        y = macro_curve + pointwise_zigzag
        raw = np.column_stack((x, y))

        result = smooth_trajectory_xy(raw)

        self.assertTrue(result.applied)
        self.assertEqual(result.reason, "smoothed")
        self.assertEqual(result.trajectory_xy.shape, raw.shape)
        np.testing.assert_allclose(result.trajectory_xy[0], raw[0], atol=0.0)
        np.testing.assert_allclose(result.trajectory_xy[-1], raw[-1], atol=0.0)
        self.assertLessEqual(result.max_deviation_m, 0.03 + 1e-12)
        self.assertLess(
            result.mean_abs_corner_after_rad,
            result.mean_abs_corner_before_rad,
        )

    def test_short_trajectory_falls_back_unchanged(self) -> None:
        raw = np.asarray([[0.1, 0.0], [0.2, 0.1], [0.3, 0.0]])

        result = smooth_trajectory_xy(raw)

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "trajectory_too_short")
        np.testing.assert_array_equal(result.trajectory_xy, raw)

    def test_repeated_trajectory_falls_back_unchanged(self) -> None:
        raw = np.repeat(np.asarray([[0.2, -0.1]]), repeats=7, axis=0)

        result = smooth_trajectory_xy(raw)

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "insufficient_unique_points")
        np.testing.assert_array_equal(result.trajectory_xy, raw)

    def test_non_finite_input_falls_back_unchanged(self) -> None:
        raw = np.zeros((7, 2), dtype=float)
        raw[3, 1] = np.nan

        result = smooth_trajectory_xy(raw)

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "non_finite_input")
        self.assertTrue(np.isnan(result.trajectory_xy[3, 1]))

    def test_non_numeric_input_reports_fallback(self) -> None:
        result = smooth_trajectory_xy([["bad", "data"]])

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "non_numeric_input")
        self.assertEqual(result.trajectory_xy.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
