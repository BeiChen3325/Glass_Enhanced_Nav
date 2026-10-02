from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_go2_navdp_stack.py"
MODULE_NAME = "go2_navdp_stack_launcher_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
STACK = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = STACK
SPEC.loader.exec_module(STACK)


class StackLauncherTests(unittest.TestCase):
    def test_default_all_mode_is_motion_safe_dry_run(self) -> None:
        args = STACK.parse_args([])
        self.assertEqual(args.components, "all")
        self.assertEqual(args.run_mode, "dry-run")
        command = STACK.client_command(args, Path("/tmp/responses.jsonl"))
        shell_script = command[-1]
        self.assertIn("--dry-run", shell_script)
        self.assertNotIn("--enable-sport-publish", shell_script)

    def test_real_navigation_requires_explicit_confirmation(self) -> None:
        with self.assertRaises(SystemExit):
            STACK.parse_args(["--components", "navigation", "--run-mode", "real"])

    def test_x_navdp_rejects_non_pointgoal(self) -> None:
        with self.assertRaises(SystemExit):
            STACK.parse_args(
                ["--model-mode", "x-navdp", "--goal-mode", "nogoal"]
            )

    def test_real_safety_and_client_share_limits(self) -> None:
        args = STACK.parse_args(
            [
                "--components",
                "navigation",
                "--run-mode",
                "real",
                "--confirm-real-motion",
                "--max-linear-speed",
                "0.35",
                "--max-angular-speed",
                "0.45",
            ]
        )
        safety = STACK.safety_bridge_command(args)[-1]
        client = STACK.client_command(args, Path("/tmp/responses.jsonl"))[-1]
        self.assertIn("--enable-sport-publish", safety)
        self.assertIn("--i-understand-this-can-move-the-robot", safety)
        self.assertIn("--max-vx 0.35", safety)
        self.assertIn("--max-wz 0.45", safety)
        self.assertIn("--max-linear-speed 0.35", client)
        self.assertIn("--max-angular-speed 0.45", client)
        self.assertNotIn("--dry-run", client)


if __name__ == "__main__":
    unittest.main()
