from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.safety.depth_stop import (
    choose_depth_recovery_scan_direction,
    should_run_depth_recovery_scan,
)


class DepthRecoveryScanTests(unittest.TestCase):
    def test_scan_replaces_stationary_hold_while_depth_is_blocked(self) -> None:
        self.assertTrue(
            should_run_depth_recovery_scan(
                enabled=True,
                allow_control_recovery=True,
                blocked=True,
                safe_curve_escape_active=False,
            )
        )

    def test_scan_stops_as_soon_as_depth_clears(self) -> None:
        self.assertFalse(
            should_run_depth_recovery_scan(
                enabled=True,
                allow_control_recovery=True,
                blocked=False,
                safe_curve_escape_active=False,
            )
        )

    def test_scan_prefers_trajectory_side_without_v4_clearance(self) -> None:
        self.assertEqual(
            choose_depth_recovery_scan_direction(
                left_clearance_m=None,
                right_clearance_m=None,
                trajectory_end_y_m=-0.30,
            ),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
