"""Model-mode contracts shared by the Go2 NavDP server and client."""

from __future__ import annotations

from pathlib import Path


NAVDP_MODE = "navdp"
X_NAVDP_MODE = "x-navdp"
X_NAVDP_GLASS_MODE = "x-navdp-glass"

SUPPORTED_MODEL_MODES = frozenset(
    {NAVDP_MODE, X_NAVDP_MODE, X_NAVDP_GLASS_MODE}
)
POINTGOAL_ONLY_MODEL_MODES = frozenset({X_NAVDP_MODE, X_NAVDP_GLASS_MODE})


def normalize_model_mode(value: str) -> str:
    """Return one canonical deployment model mode or raise a clear error."""
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "baseline": NAVDP_MODE,
        "cross-modal": NAVDP_MODE,
        "xnavdp": X_NAVDP_MODE,
        "glass": X_NAVDP_GLASS_MODE,
        "glassrecon": X_NAVDP_GLASS_MODE,
        "xnavdp-glass": X_NAVDP_GLASS_MODE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_MODEL_MODES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_MODES))
        raise ValueError(f"unsupported model_mode {value!r}; expected one of: {supported}")
    return normalized


def validate_model_goal_mode(model_mode: str, goal_mode: str) -> None:
    """Enforce the task heads actually released by each checkpoint family."""
    normalized_model = normalize_model_mode(model_mode)
    normalized_goal = str(goal_mode).strip().lower()
    if normalized_model in POINTGOAL_ONLY_MODEL_MODES and normalized_goal != "pointgoal":
        raise ValueError(
            f"{normalized_model} supports only pointgoal in the current NavDP tree; "
            f"got goal_mode={goal_mode!r}"
        )


def is_x_navdp_mode(model_mode: str) -> bool:
    return normalize_model_mode(model_mode) in POINTGOAL_ONLY_MODEL_MODES


def requires_alignment_depth(model_mode: str) -> bool:
    return normalize_model_mode(model_mode) == X_NAVDP_GLASS_MODE


def default_checkpoint(navdp_root: Path, model_mode: str) -> Path:
    """Resolve the repository-owned checkpoint for a deployment mode."""
    normalized = normalize_model_mode(model_mode)
    filename = (
        "navdp-cross-modal.ckpt"
        if normalized == NAVDP_MODE
        else "x-navdp_posttrain.ckpt"
    )
    return Path(navdp_root) / "model" / filename


def default_dav2_checkpoint(navdp_root: Path) -> Path:
    return Path(navdp_root) / "model" / "depth_anything_v2_vits.pth"
