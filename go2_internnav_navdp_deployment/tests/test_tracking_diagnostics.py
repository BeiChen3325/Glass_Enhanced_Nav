from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.analysis.tracking_diagnostics import (
    TrackingDiagnosticsRecorder,
    generate_tracking_profile_analysis,
)


class TrackingDiagnosticsTests(unittest.TestCase):
    def test_record_and_generate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            recorder = TrackingDiagnosticsRecorder(run_dir)
            raw = np.asarray([[0.1, 0.0], [0.2, 0.02], [0.3, 0.0]])
            effective = np.asarray([[0.1, 0.0], [0.2, 0.01], [0.3, 0.0]])
            recorder.write_reference(
                request_sequence=0,
                response_idx=-1,
                tracking_profile="smooth_continuous",
                raw_local_xy=raw,
                effective_local_xy=effective,
                smoothing_applied=True,
                smoothing_reason="smoothed",
                max_deviation_m=0.01,
                mean_abs_corner_before_rad=0.2,
                mean_abs_corner_after_rad=0.1,
                total_abs_turn_before_rad=0.4,
                total_abs_turn_after_rad=0.2,
                goal={"x": 5.0, "y": 0.0},
            )
            recorder.write_control(
                request_count=1,
                tracking_profile="smooth_continuous",
                odom_x=1.0,
                odom_y=2.0,
                odom_yaw=0.1,
                mpc_v=0.3,
                mpc_w=0.15,
                previous_emitted_v=0.2,
                previous_emitted_w=0.1,
                emitted_v=0.25,
                emitted_w=0.14,
                depth_forward_blocked=False,
            )
            recorder.close()

            artifacts = generate_tracking_profile_analysis(run_dir)

            for path in artifacts.values():
                self.assertTrue(path.exists(), path)
            with artifacts["control_csv"].open(
                "r", encoding="utf-8", newline=""
            ) as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["tracking_profile"], "smooth_continuous")


if __name__ == "__main__":
    unittest.main()
