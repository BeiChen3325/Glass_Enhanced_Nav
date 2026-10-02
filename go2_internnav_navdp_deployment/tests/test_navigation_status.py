from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.ros2.navigation_status import (
    decode_navigation_status,
    effective_stop_cause,
    encode_navigation_status,
)


class NavigationStatusTests(unittest.TestCase):
    def test_goal_reached_payload_keeps_distance(self) -> None:
        encoded = encode_navigation_status(
            "goal_reached",
            "pointgoal_reached",
            goal_mode="pointgoal",
            distance_to_goal_m=0.42,
            timestamp_unix_s=123.0,
        )
        self.assertEqual(json.loads(encoded)["timestamp_unix_s"], 123.0)
        decoded = decode_navigation_status(encoded)
        self.assertEqual(decoded["state"], "goal_reached")
        self.assertEqual(decoded["reason"], "pointgoal_reached")
        self.assertAlmostEqual(decoded["distance_to_goal_m"], 0.42)

    def test_goal_reason_explains_timeout_stopmove(self) -> None:
        self.assertEqual(
            effective_stop_cause(
                manual_override_latched=False,
                cmd_timed_out=True,
                navigation_state="goal_reached",
                navigation_reason="pointgoal_reached",
            ),
            "pointgoal_reached",
        )

    def test_obstacle_and_manual_override_are_distinct(self) -> None:
        self.assertEqual(
            effective_stop_cause(
                manual_override_latched=False,
                cmd_timed_out=True,
                navigation_state="obstacle_blocked",
                navigation_reason="close_depth_fraction",
            ),
            "close_depth_fraction",
        )
        self.assertEqual(
            effective_stop_cause(
                manual_override_latched=True,
                cmd_timed_out=True,
                navigation_state="goal_reached",
                navigation_reason="pointgoal_reached",
            ),
            "manual_override",
        )

    def test_malformed_payload_is_safe(self) -> None:
        self.assertEqual(
            decode_navigation_status("not-json"),
            {"state": "unknown", "reason": "malformed_status_payload"},
        )


if __name__ == "__main__":
    unittest.main()
