"""Shared NavDP deployment semantics for the HTTP server and ROS 2 client."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SUPPORTED_GOAL_MODES = frozenset({"pointgoal", "nogoal", "imagegoal"})


def normalize_goal_mode(value: Any) -> str:
    """Return a supported lowercase NavDP goal mode."""
    if not isinstance(value, str):
        raise ValueError("goal_mode must be a string")
    mode = value.strip().casefold()
    if mode not in SUPPORTED_GOAL_MODES:
        raise ValueError(
            "goal_mode must be one of: "
            + ", ".join(sorted(SUPPORTED_GOAL_MODES))
        )
    return mode


def normalize_navdp_trajectory(value: Any) -> list[list[float]]:
    """Normalize a single-env NavDP trajectory to ``[[x, y, z], ...]``.

    Native NavDP returns either ``[N, 3]`` for one selected trajectory or
    ``[1, N, 3]`` for a singleton batch. The ROS client consumes x/y and
    ignores z, but the server response is kept as a three-column trajectory.
    """
    try:
        trajectory = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must be numeric") from exc

    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape [N, 2+] or [1, N, 2+]")
    if trajectory.shape[0] == 0:
        raise ValueError("trajectory must not be empty")
    if trajectory.shape[1] < 2:
        raise ValueError("trajectory must contain at least x and y columns")
    if not np.all(np.isfinite(trajectory[:, :2])):
        raise ValueError("trajectory x/y values must be finite")

    if trajectory.shape[1] == 2:
        z = np.zeros((trajectory.shape[0], 1), dtype=float)
        trajectory = np.concatenate((trajectory[:, :2], z), axis=1)
    else:
        trajectory = trajectory[:, :3]
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("trajectory values must be finite")

    return trajectory.astype(float).tolist()


def _odom_values(odom: Any) -> tuple[float, float, float]:
    values = (float(odom[0]), float(odom[1]), float(odom[2]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("odom values must be finite")
    return values


def local_pointgoal_to_world(
    local_xy: tuple[float, float] | list[float] | np.ndarray,
    odom: tuple[float, float, float] | list[float] | np.ndarray,
) -> tuple[float, float]:
    """Anchor a base_link-relative point goal in the world frame."""
    local_x = float(local_xy[0])
    local_y = float(local_xy[1])
    if not math.isfinite(local_x) or not math.isfinite(local_y):
        raise ValueError("pointgoal values must be finite")
    x, y, yaw = _odom_values(odom)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        x + cos_yaw * local_x - sin_yaw * local_y,
        y + sin_yaw * local_x + cos_yaw * local_y,
    )


def world_pointgoal_to_local(
    world_xy: tuple[float, float] | list[float] | np.ndarray,
    odom: tuple[float, float, float] | list[float] | np.ndarray,
) -> tuple[float, float]:
    """Express a fixed world point goal in the current base_link frame."""
    world_x = float(world_xy[0])
    world_y = float(world_xy[1])
    if not math.isfinite(world_x) or not math.isfinite(world_y):
        raise ValueError("pointgoal values must be finite")
    x, y, yaw = _odom_values(odom)
    dx = world_x - x
    dy = world_y - y
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
    )


def pointgoal_distance_to_odom(
    world_xy: tuple[float, float] | list[float] | np.ndarray,
    odom: tuple[float, float, float] | list[float] | np.ndarray,
) -> float:
    """Return planar distance from current odom pose to a fixed world goal."""
    world_x = float(world_xy[0])
    world_y = float(world_xy[1])
    x, y, _ = _odom_values(odom)
    return math.hypot(world_x - x, world_y - y)
