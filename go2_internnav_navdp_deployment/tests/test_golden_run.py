from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from go2_internnav.golden_run import (
    GoldenRunFormatError,
    GoldenRunResponseRecorder,
)


class GoldenRunResponseRecorderTests(unittest.TestCase):
    def test_metadata_is_written_at_record_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            with GoldenRunResponseRecorder(path) as recorder:
                recorder.write(
                    reset=True,
                    idx=-1,
                    response={"trajectory": [[0.0, 0.0, 0.0]]},
                    elapsed_s=0.25,
                    server_url="http://127.0.0.1:5802/eval_navdp",
                    goal_mode="pointgoal",
                    goal={"x": 3.0, "y": 0.0},
                    metadata={
                        "inference_odom": [1.0, 2.0, 0.5],
                        "scenario_id": "regression",
                    },
                )

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["inference_odom"], [1.0, 2.0, 0.5])
            self.assertEqual(record["scenario_id"], "regression")
            self.assertEqual(record["response"]["trajectory"][0][0], 0.0)

    def test_metadata_cannot_overwrite_core_record_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            with GoldenRunResponseRecorder(path) as recorder:
                with self.assertRaises(GoldenRunFormatError):
                    recorder.write(
                        reset=False,
                        idx=0,
                        response={"trajectory": [[0.0, 0.0, 0.0]]},
                        elapsed_s=0.1,
                        server_url="http://127.0.0.1:5802/eval_navdp",
                        metadata={"response": {"trajectory": []}},
                    )


if __name__ == "__main__":
    unittest.main()
