"""Conservative smoothing for NavDP MPC reference trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class TrajectorySmoothingConfig:
    """Parameters for endpoint-preserving Savitzky-Golay smoothing."""

    window_length: int = 7
    polynomial_order: int = 2
    max_deviation_m: float = 0.03
    duplicate_epsilon_m: float = 1e-6


@dataclass(frozen=True)
class TrajectorySmoothingResult:
    """A smoothed reference and metrics used by analysis logging."""

    trajectory_xy: np.ndarray
    applied: bool
    reason: str
    max_deviation_m: float
    mean_abs_corner_before_rad: float
    mean_abs_corner_after_rad: float
    total_abs_turn_before_rad: float
    total_abs_turn_after_rad: float


DEFAULT_TRAJECTORY_SMOOTHING_CONFIG = TrajectorySmoothingConfig()


def _turn_metrics(points_xy: np.ndarray, epsilon_m: float) -> tuple[float, float]:
    if points_xy.ndim != 2 or points_xy.shape[0] < 3:
        return 0.0, 0.0

    segments = np.diff(points_xy[:, :2], axis=0)
    segment_norms = np.linalg.norm(segments, axis=1)
    valid_segments = segments[segment_norms > epsilon_m]
    if valid_segments.shape[0] < 2:
        return 0.0, 0.0

    headings = np.arctan2(valid_segments[:, 1], valid_segments[:, 0])
    heading_changes = (np.diff(headings) + np.pi) % (2.0 * np.pi) - np.pi
    absolute_changes = np.abs(heading_changes)
    return float(np.mean(absolute_changes)), float(np.sum(absolute_changes))


def _result_without_smoothing(
    trajectory_xy: np.ndarray,
    reason: str,
    config: TrajectorySmoothingConfig,
) -> TrajectorySmoothingResult:
    mean_corner, total_turn = _turn_metrics(
        trajectory_xy,
        config.duplicate_epsilon_m,
    )
    return TrajectorySmoothingResult(
        trajectory_xy=trajectory_xy.copy(),
        applied=False,
        reason=reason,
        max_deviation_m=0.0,
        mean_abs_corner_before_rad=mean_corner,
        mean_abs_corner_after_rad=mean_corner,
        total_abs_turn_before_rad=total_turn,
        total_abs_turn_after_rad=total_turn,
    )


def smooth_trajectory_xy(
    value: Any,
    config: TrajectorySmoothingConfig = DEFAULT_TRAJECTORY_SMOOTHING_CONFIG,
) -> TrajectorySmoothingResult:
    """Smooth an ``[N, 2+]`` trajectory without moving either endpoint.

    Invalid or undersized inputs are returned unchanged with ``applied=False``.
    The caller remains responsible for rejecting an invalid raw model response.
    """

    try:
        raw_value = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return _result_without_smoothing(
            np.empty((0, 2), dtype=float),
            "non_numeric_input",
            config,
        )

    if raw_value.ndim != 2 or raw_value.shape[1] < 2:
        fallback = (
            raw_value[:, :2].copy()
            if raw_value.ndim == 2
            else np.empty((0, 2), dtype=float)
        )
        return _result_without_smoothing(fallback, "invalid_shape", config)

    trajectory_xy = raw_value[:, :2].copy()
    if not np.all(np.isfinite(trajectory_xy)):
        return _result_without_smoothing(trajectory_xy, "non_finite_input", config)

    if config.window_length < 3 or config.window_length % 2 == 0:
        return _result_without_smoothing(trajectory_xy, "invalid_window_length", config)
    if config.polynomial_order < 0 or config.polynomial_order >= config.window_length:
        return _result_without_smoothing(trajectory_xy, "invalid_polynomial_order", config)
    if config.max_deviation_m < 0.0:
        return _result_without_smoothing(trajectory_xy, "invalid_max_deviation", config)
    if trajectory_xy.shape[0] < config.window_length:
        return _result_without_smoothing(trajectory_xy, "trajectory_too_short", config)

    segment_norms = np.linalg.norm(np.diff(trajectory_xy, axis=0), axis=1)
    if np.count_nonzero(segment_norms > config.duplicate_epsilon_m) < 2:
        return _result_without_smoothing(trajectory_xy, "insufficient_unique_points", config)

    try:
        smoothed = np.column_stack(
            [
                savgol_filter(
                    trajectory_xy[:, axis],
                    config.window_length,
                    config.polynomial_order,
                    mode="interp",
                )
                for axis in range(2)
            ]
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return _result_without_smoothing(trajectory_xy, "savgol_failure", config)

    if not np.all(np.isfinite(smoothed)):
        return _result_without_smoothing(trajectory_xy, "non_finite_output", config)

    smoothed[0] = trajectory_xy[0]
    smoothed[-1] = trajectory_xy[-1]

    displacement = smoothed - trajectory_xy
    displacement_norm = np.linalg.norm(displacement, axis=1)
    if config.max_deviation_m == 0.0:
        smoothed = trajectory_xy.copy()
    else:
        scale = np.minimum(
            1.0,
            config.max_deviation_m / np.maximum(displacement_norm, 1e-12),
        )
        smoothed = trajectory_xy + displacement * scale[:, None]
    smoothed[0] = trajectory_xy[0]
    smoothed[-1] = trajectory_xy[-1]

    final_deviation = np.linalg.norm(smoothed - trajectory_xy, axis=1)
    mean_before, total_before = _turn_metrics(
        trajectory_xy,
        config.duplicate_epsilon_m,
    )
    mean_after, total_after = _turn_metrics(
        smoothed,
        config.duplicate_epsilon_m,
    )
    return TrajectorySmoothingResult(
        trajectory_xy=smoothed,
        applied=True,
        reason="smoothed",
        max_deviation_m=float(np.max(final_deviation, initial=0.0)),
        mean_abs_corner_before_rad=mean_before,
        mean_abs_corner_after_rad=mean_after,
        total_abs_turn_before_rad=total_before,
        total_abs_turn_after_rad=total_after,
    )
