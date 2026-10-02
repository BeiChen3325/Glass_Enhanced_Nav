from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.control import Mpc_controller


def _test_trajectory() -> np.ndarray:
    x = np.linspace(0.05, 1.0, 21)
    y = 0.12 * np.sin(np.linspace(0.0, np.pi, 21))
    return np.column_stack((x, y))


class MpcContinuityTests(unittest.TestCase):
    def test_legacy_profile_matches_prechange_baseline(self) -> None:
        controller = Mpc_controller(_test_trajectory())

        controls, _ = controller.solve(np.asarray([0.0, 0.0, 0.0]))

        self.assertAlmostEqual(controls[0, 0], 0.25473621519721174, places=6)
        self.assertAlmostEqual(controls[0, 1], 0.330512700494592, places=6)

    def test_explicit_disabled_continuity_matches_legacy(self) -> None:
        legacy = Mpc_controller(_test_trajectory())
        explicit = Mpc_controller(
            _test_trajectory(),
            angular_delta_weight=0.0,
            max_angular_accel_radps2=None,
        )

        legacy_controls, _ = legacy.solve(np.asarray([0.0, 0.0, 0.0]))
        explicit_controls, _ = explicit.solve(np.asarray([0.0, 0.0, 0.0]))

        np.testing.assert_allclose(explicit_controls, legacy_controls, atol=1e-7)

    def test_continuous_profile_limits_all_yaw_rate_steps(self) -> None:
        previous_w = -0.12
        controller = Mpc_controller(
            _test_trajectory(),
            angular_delta_weight=1.0,
            max_angular_accel_radps2=0.6,
        )

        controls, _ = controller.solve(
            np.asarray([0.0, 0.0, 0.0]),
            previous_w=previous_w,
        )

        yaw_rates = controls[:, 1]
        deltas = np.diff(np.concatenate(([previous_w], yaw_rates)))
        self.assertLessEqual(float(np.max(np.abs(deltas))), 0.060001)

    def test_invalid_continuity_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Mpc_controller(_test_trajectory(), angular_delta_weight=-1.0)
        with self.assertRaises(ValueError):
            Mpc_controller(_test_trajectory(), max_angular_accel_radps2=0.0)


if __name__ == "__main__":
    unittest.main()
