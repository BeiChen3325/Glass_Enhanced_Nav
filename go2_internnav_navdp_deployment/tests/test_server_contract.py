from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

SERVER_PATH = PROJECT_ROOT / "scripts" / "http_internvla_server_w_navdp_isolated.py"
SPEC = importlib.util.spec_from_file_location("go2_navdp_server_contract", SERVER_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def _rgb_jpeg() -> io.BytesIO:
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8), mode="RGB").save(
        buffer, format="JPEG"
    )
    buffer.seek(0)
    return buffer


def _depth_png() -> io.BytesIO:
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 12), 15000, dtype=np.uint16)).save(
        buffer, format="PNG"
    )
    buffer.seek(0)
    return buffer


def _alignment_npy() -> io.BytesIO:
    buffer = io.BytesIO()
    np.save(buffer, np.full((8, 12), 1.5, dtype=np.float32), allow_pickle=False)
    buffer.seek(0)
    return buffer


class ServerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_run_policy = SERVER.run_policy
        SERVER.request_idx = 0
        SERVER.navdp_navigator = None
        SERVER.args = argparse.Namespace(
            model_mode="navdp",
            stop_threshold=-3.0,
            checkpoint=Path("/tmp/fake.ckpt"),
            return_glass_debug_depths=False,
        )

        def fake_run_policy(**_kwargs):
            trajectory = np.asarray(
                [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.1, 0.0]]],
                dtype=np.float32,
            )
            return trajectory, trajectory[:, None], np.asarray([[0.75]]), {
                "ransac": [{"success": True, "fallback_used": False}],
                "relative_depth": np.ones((1, 2, 2), dtype=np.float32),
                "aligned_depth": np.ones((1, 2, 2), dtype=np.float32) * 2,
                "clipped_enhanced_depth": np.ones((1, 2, 2), dtype=np.float32) * 3,
            }

        SERVER.run_policy = fake_run_policy
        self.client = SERVER.app.test_client()

    def tearDown(self) -> None:
        SERVER.run_policy = self.original_run_policy

    def _post(self, payload, *, alignment=False):
        data = {
            "json": json.dumps(payload),
            "image": (_rgb_jpeg(), "rgb.jpg"),
            "depth": (_depth_png(), "depth.png"),
        }
        if alignment:
            data["alignment_depth"] = (_alignment_npy(), "alignment_depth.npy")
        return self.client.post("/eval_navdp", data=data)

    def test_navdp_nogoal_response_identifies_model(self) -> None:
        response = self._post(
            {"model_mode": "navdp", "goal_mode": "nogoal", "reset": True}
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["model_mode"], "navdp")
        self.assertEqual(np.asarray(body["trajectory"]).shape, (3, 3))

    def test_client_server_model_mismatch_is_rejected_before_inference(self) -> None:
        response = self._post(
            {"model_mode": "x-navdp", "goal_mode": "pointgoal", "goal": {"x": 1, "y": 0}}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("client requested", response.get_json()["error"])

    def test_x_navdp_rejects_unreleased_imagegoal(self) -> None:
        SERVER.args.model_mode = "x-navdp"
        response = self._post({"model_mode": "x-navdp", "goal_mode": "imagegoal"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("only pointgoal", response.get_json()["error"])

    def test_glass_mode_requires_float_depth_and_returns_ransac(self) -> None:
        SERVER.args.model_mode = "x-navdp-glass"
        payload = {
            "model_mode": "x-navdp-glass",
            "goal_mode": "pointgoal",
            "goal": {"x": 1.0, "y": 0.0},
            "inference_odom": [0.0, 0.0, 0.0],
        }
        missing = self._post(payload)
        self.assertEqual(missing.status_code, 400)
        self.assertIn("alignment_depth.npy", missing.get_json()["error"])

        response = self._post(payload, alignment=True)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ransac"][0]["success"])
        self.assertNotIn("relative_depth", body)
        self.assertNotIn("aligned_depth", body)
        self.assertNotIn("clipped_enhanced_depth", body)

        SERVER.args.return_glass_debug_depths = True
        debug_response = self._post(payload, alignment=True)
        self.assertEqual(debug_response.status_code, 200)
        debug_body = debug_response.get_json()
        self.assertEqual(debug_body["trajectory"], body["trajectory"])
        self.assertEqual(debug_body["all_trajectory"], body["all_trajectory"])
        self.assertEqual(debug_body["all_values"], body["all_values"])
        self.assertEqual(debug_body["ransac"], body["ransac"])
        self.assertEqual(np.asarray(debug_body["relative_depth"]).shape, (1, 2, 2))
        self.assertEqual(np.asarray(debug_body["aligned_depth"]).shape, (1, 2, 2))
        self.assertEqual(
            np.asarray(debug_body["clipped_enhanced_depth"]).shape, (1, 2, 2)
        )


if __name__ == "__main__":
    unittest.main()
