from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.navdp.model_profiles import (
    NAVDP_MODE,
    X_NAVDP_GLASS_MODE,
    X_NAVDP_MODE,
    default_checkpoint,
    normalize_model_mode,
    requires_alignment_depth,
    validate_model_goal_mode,
)


class ModelProfileTests(unittest.TestCase):
    def test_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_model_mode("cross_modal"), NAVDP_MODE)
        self.assertEqual(normalize_model_mode("xnavdp"), X_NAVDP_MODE)
        self.assertEqual(normalize_model_mode("glassrecon"), X_NAVDP_GLASS_MODE)

    def test_released_task_heads_are_enforced(self) -> None:
        for goal_mode in ("nogoal", "pointgoal", "imagegoal"):
            validate_model_goal_mode(NAVDP_MODE, goal_mode)
        validate_model_goal_mode(X_NAVDP_MODE, "pointgoal")
        with self.assertRaisesRegex(ValueError, "only pointgoal"):
            validate_model_goal_mode(X_NAVDP_MODE, "imagegoal")

    def test_checkpoint_and_alignment_contract(self) -> None:
        root = Path("/tmp/NavDP")
        self.assertEqual(
            default_checkpoint(root, NAVDP_MODE),
            root / "model" / "navdp-cross-modal.ckpt",
        )
        self.assertEqual(
            default_checkpoint(root, X_NAVDP_GLASS_MODE),
            root / "model" / "x-navdp_posttrain.ckpt",
        )
        self.assertFalse(requires_alignment_depth(X_NAVDP_MODE))
        self.assertTrue(requires_alignment_depth(X_NAVDP_GLASS_MODE))


if __name__ == "__main__":
    unittest.main()
