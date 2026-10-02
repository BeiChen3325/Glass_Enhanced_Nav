"""Depth-image based emergency stop checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthSafetyConfig:
    """Configuration for conservative forward-speed blocking."""

    block_distance_m: float = 0.30
    clear_distance_m: float = 0.40
    min_block_duration_s: float = 0.5
    min_valid_depth_m: float = 0.05
    min_valid_fraction: float = 0.05
    min_close_fraction: float = 0.015
    roi_x_min: float = 0.15
    roi_x_max: float = 0.85
    roi_y_min: float = 0.35
    roi_y_max: float = 0.75


@dataclass(frozen=True)
class DepthSafetyStatus:
    should_block_forward: bool
    reason: str
    threshold_m: float | None
    min_depth_m: float | None
    p05_depth_m: float | None
    close_fraction: float
    valid_fraction: float
    roi_shape: tuple[int, int]


@dataclass(frozen=True)
class DepthBlockLatchState:
    """Temporal state for the minimum-duration forward-motion block."""

    blocked: bool = False
    blocked_since_s: float | None = None


DEFAULT_DEPTH_SAFETY_CONFIG = DepthSafetyConfig()

DEFAULT_SAFE_CURVE_ESCAPE_MAX_SPEED_MPS = 0.06
DEFAULT_SAFE_CURVE_ESCAPE_MIN_YAW_RATE_RADPS = 0.10
DEFAULT_SAFE_CURVE_ESCAPE_MIN_DEPTH_M = 0.30
DEFAULT_SAFE_CURVE_ESCAPE_MAX_SAFETY_AGE_S = 6.0
DEFAULT_DEPTH_RECOVERY_SCAN_YAW_RATE_RADPS = 0.25
DEFAULT_DEPTH_RECOVERY_SCAN_REVERSE_INTERVAL_S = 4.0
DEFAULT_DEPTH_RECOVERY_CLEARANCE_MARGIN_M = 0.05


def choose_depth_recovery_scan_direction(
    *,
    left_clearance_m: float | None,
    right_clearance_m: float | None,
    trajectory_end_y_m: float | None,
    previous_direction: int = 1,
    clearance_margin_m: float = DEFAULT_DEPTH_RECOVERY_CLEARANCE_MARGIN_M,
) -> int:
    """Choose +1/left or -1/right for a bounded in-place recovery scan."""

    if not np.isfinite(clearance_margin_m) or clearance_margin_m < 0.0:
        raise ValueError("clearance_margin_m must be finite and non-negative")
    fallback = 1 if previous_direction not in (-1, 1) else int(previous_direction)
    left_valid = left_clearance_m is not None and np.isfinite(left_clearance_m)
    right_valid = right_clearance_m is not None and np.isfinite(right_clearance_m)
    if left_valid and right_valid:
        difference = float(left_clearance_m) - float(right_clearance_m)
        if abs(difference) >= clearance_margin_m:
            return 1 if difference > 0.0 else -1
    if trajectory_end_y_m is not None and np.isfinite(trajectory_end_y_m):
        if abs(float(trajectory_end_y_m)) >= clearance_margin_m:
            return 1 if float(trajectory_end_y_m) > 0.0 else -1
    return fallback


def should_run_depth_recovery_scan(
    *,
    enabled: bool,
    allow_control_recovery: bool,
    blocked: bool,
    safe_curve_escape_active: bool,
) -> bool:
    """Return whether normal MPC yaw should be replaced by an in-place scan."""

    return bool(
        enabled
        and allow_control_recovery
        and blocked
        and not safe_curve_escape_active
    )


def block_positive_forward_velocity(
    vx: float,
    vyaw: float,
    *,
    blocked: bool,
) -> tuple[float, float]:
    """Zero positive forward velocity while preserving commanded rotation."""

    filtered_vx = 0.0 if blocked and vx > 0.0 else float(vx)
    return filtered_vx, float(vyaw)


def limit_forward_velocity_for_safe_curve(
    vx: float,
    vyaw: float,
    *,
    blocked: bool,
    enabled: bool,
    selected_trajectory_safe: bool,
    selected_trajectory_safety_age_s: float | None,
    min_depth_m: float | None,
    max_escape_speed_mps: float = DEFAULT_SAFE_CURVE_ESCAPE_MAX_SPEED_MPS,
    min_yaw_rate_radps: float = DEFAULT_SAFE_CURVE_ESCAPE_MIN_YAW_RATE_RADPS,
    min_depth_for_escape_m: float = DEFAULT_SAFE_CURVE_ESCAPE_MIN_DEPTH_M,
    max_safety_age_s: float = DEFAULT_SAFE_CURVE_ESCAPE_MAX_SAFETY_AGE_S,
) -> tuple[float, float, bool]:
    """Permit a tightly bounded arc only when model and sensor safety agree.

    The normal depth stop remains authoritative unless every escape gate passes:
    the selected model trajectory passed the footprint checker, that verdict is
    fresh, the command is meaningfully turning, and the current minimum depth
    retains a separate hard floor.  When forward speed is capped, angular speed
    is scaled by the same factor so the planned curvature is not tightened.
    """

    values = (
        vx,
        vyaw,
        max_escape_speed_mps,
        min_yaw_rate_radps,
        min_depth_for_escape_m,
        max_safety_age_s,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("safe-curve velocity inputs and limits must be finite")
    if max_escape_speed_mps < 0.0:
        raise ValueError("max_escape_speed_mps must be non-negative")
    if min_yaw_rate_radps < 0.0:
        raise ValueError("min_yaw_rate_radps must be non-negative")
    if min_depth_for_escape_m < 0.0:
        raise ValueError("min_depth_for_escape_m must be non-negative")
    if max_safety_age_s < 0.0:
        raise ValueError("max_safety_age_s must be non-negative")

    if not blocked or vx <= 0.0:
        return float(vx), float(vyaw), False

    safety_age_valid = (
        selected_trajectory_safety_age_s is not None
        and np.isfinite(selected_trajectory_safety_age_s)
        and 0.0 <= selected_trajectory_safety_age_s <= max_safety_age_s
    )
    depth_valid = (
        min_depth_m is not None
        and np.isfinite(min_depth_m)
        and min_depth_m >= min_depth_for_escape_m
    )
    escape_allowed = (
        enabled
        and selected_trajectory_safe
        and safety_age_valid
        and depth_valid
        and abs(vyaw) >= min_yaw_rate_radps
    )
    if escape_allowed:
        limited_vx = min(float(vx), max_escape_speed_mps)
        speed_scale = limited_vx / float(vx)
        return limited_vx, float(vyaw) * speed_scale, True
    return 0.0, float(vyaw), False


def update_depth_block_latch(
    *,
    detection_blocked: bool,
    previous: DepthBlockLatchState,
    now_s: float,
    min_block_duration_s: float,
) -> DepthBlockLatchState:
    """Apply an immediate-on, minimum-duration latch to a depth detection.

    The latch only controls positive forward velocity.  Callers remain free to
    publish angular velocity while ``blocked`` is true.
    """

    if not np.isfinite(now_s):
        raise ValueError("now_s must be finite")
    if not np.isfinite(min_block_duration_s) or min_block_duration_s < 0.0:
        raise ValueError("min_block_duration_s must be finite and non-negative")

    if detection_blocked:
        blocked_since_s = previous.blocked_since_s
        if not previous.blocked or blocked_since_s is None:
            blocked_since_s = float(now_s)
        return DepthBlockLatchState(True, blocked_since_s)

    if not previous.blocked:
        return DepthBlockLatchState(False, None)

    blocked_since_s = previous.blocked_since_s
    if blocked_since_s is None:
        blocked_since_s = float(now_s)
    elapsed_s = max(0.0, float(now_s) - blocked_since_s)
    if elapsed_s < min_block_duration_s:
        return DepthBlockLatchState(True, blocked_since_s)
    return DepthBlockLatchState(False, None)


def assess_depth_safety_stop(
    depth_m: np.ndarray | None,
    config: DepthSafetyConfig = DEFAULT_DEPTH_SAFETY_CONFIG,
    currently_blocked: bool = False,
) -> DepthSafetyStatus:
    """Return whether a depth image should block positive forward speed."""

    threshold_m = config.clear_distance_m if currently_blocked else config.block_distance_m

    if depth_m is None:
        return DepthSafetyStatus(False, "missing_depth", threshold_m, None, None, 0.0, 0.0, (0, 0))

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
        return DepthSafetyStatus(
            False,
            "invalid_depth_shape",
            threshold_m,
            None,
            None,
            0.0,
            0.0,
            (0, 0),
        )

    height, width = depth.shape
    x0 = int(round(width * config.roi_x_min))
    x1 = int(round(width * config.roi_x_max))
    y0 = int(round(height * config.roi_y_min))
    y1 = int(round(height * config.roi_y_max))
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))

    roi = depth[y0:y1, x0:x1]
    valid_mask = np.isfinite(roi) & (roi >= config.min_valid_depth_m)
    valid_fraction = float(np.count_nonzero(valid_mask) / roi.size)
    if valid_fraction < config.min_valid_fraction:
        return DepthSafetyStatus(
            False,
            "insufficient_valid_depth",
            threshold_m,
            None,
            None,
            0.0,
            valid_fraction,
            roi.shape,
        )

    valid_depth = roi[valid_mask]
    min_depth_m = float(np.min(valid_depth))
    p05_depth_m = float(np.percentile(valid_depth, 5))
    close_fraction = float(np.count_nonzero(valid_depth <= threshold_m) / roi.size)
    should_block_forward = close_fraction >= config.min_close_fraction
    reason = "too_close" if should_block_forward else "clear"
    return DepthSafetyStatus(
        should_block_forward,
        reason,
        threshold_m,
        min_depth_m,
        p05_depth_m,
        close_fraction,
        valid_fraction,
        roi.shape,
    )
