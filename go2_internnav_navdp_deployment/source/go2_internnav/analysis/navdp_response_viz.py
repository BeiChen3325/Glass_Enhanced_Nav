"""Generate PNG and CSV diagnostics from recorded NavDP JSONL responses."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_COLUMNS = [
    "request_sequence",
    "idx",
    "goal_mode",
    "goal_x",
    "goal_y",
    "elapsed_s",
    "candidate_count",
    "selected_end_x",
    "selected_end_y",
    "selected_end_yaw",
    "selected_endpoint_norm_m",
    "selected_path_length_m",
    "selected_max_abs_y",
    "top_candidate_index",
    "top_candidate_value",
    "min_value",
    "max_value",
    "selected_matches_top_candidate",
]


def generate_navdp_response_analysis(
    responses_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Create trajectory CSV, overview PNG, and per-frame PNGs from JSONL records."""

    responses_file = Path(responses_path).expanduser().resolve()
    if output_dir is None:
        analysis_dir = responses_file.parent
    else:
        analysis_dir = Path(output_dir).expanduser().resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = analysis_dir / "trajectory_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(responses_file)
    rows: list[dict[str, Any]] = []
    for frame_number, record in enumerate(records):
        row, frame_data = _summarize_record(record)
        rows.append(row)
        _plot_frame(frame_data, frame_dir / f"idx_{frame_number:04d}.png")

    summary_path = analysis_dir / "trajectory_summary.csv"
    _write_summary_csv(summary_path, rows)

    overview_path = analysis_dir / "trajectory_overview.png"
    _plot_overview(rows, records, overview_path)

    return {
        "summary_csv": summary_path,
        "overview_png": overview_path,
        "frame_dir": frame_dir,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _response_from_record(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response", record)
    return response if isinstance(response, dict) else {}


def _as_trajectory(value: Any) -> np.ndarray:
    try:
        trajectory = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 3), dtype=float)
    if trajectory.ndim == 3 and trajectory.shape[0] == 1:
        trajectory = trajectory[0]
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
        return np.empty((0, 3), dtype=float)
    if not np.all(np.isfinite(trajectory[:, :2])):
        return np.empty((0, 3), dtype=float)
    if trajectory.shape[1] == 2:
        z = np.zeros((trajectory.shape[0], 1), dtype=float)
        trajectory = np.concatenate((trajectory[:, :2], z), axis=1)
    else:
        trajectory = trajectory[:, :3]
    trajectory[~np.isfinite(trajectory)] = np.nan
    return trajectory


def _as_candidate_trajectories(value: Any) -> np.ndarray:
    try:
        trajectories = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 0, 3), dtype=float)

    if trajectories.ndim == 4 and trajectories.shape[0] >= 1:
        trajectories = trajectories[0]
    elif trajectories.ndim == 2:
        trajectories = trajectories[np.newaxis, :, :]
    if trajectories.ndim != 3 or trajectories.shape[0] == 0 or trajectories.shape[2] < 2:
        return np.empty((0, 0, 3), dtype=float)
    if trajectories.shape[2] == 2:
        z = np.zeros((*trajectories.shape[:2], 1), dtype=float)
        trajectories = np.concatenate((trajectories[:, :, :2], z), axis=2)
    else:
        trajectories = trajectories[:, :, :3]
    trajectories[~np.isfinite(trajectories)] = np.nan
    return trajectories


def _as_values(value: Any) -> np.ndarray:
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0,), dtype=float)
    if values.ndim == 2 and values.shape[0] >= 1:
        values = values[0]
    if values.ndim != 1:
        return np.empty((0,), dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def _goal_from_record(record: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    goal = record.get("goal")
    if not isinstance(goal, dict):
        goal = response.get("goal")
    return goal if isinstance(goal, dict) else {}


def _summarize_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    response = _response_from_record(record)
    selected = _as_trajectory(response.get("trajectory"))
    candidates = _as_candidate_trajectories(response.get("all_trajectory"))
    values = _as_values(response.get("all_values"))
    goal = _goal_from_record(record, response)

    selected_end = selected[-1] if selected.size else np.array([math.nan, math.nan, math.nan])
    selected_norm = _endpoint_norm(selected)
    selected_path_len = _path_length(selected)
    selected_max_abs_y = _max_abs_y(selected)

    top_idx = _top_index(values)
    top_value = float(values[top_idx]) if top_idx is not None else math.nan
    min_value = _nanmin(values)
    max_value = _nanmax(values)
    selected_matches_top = _selected_matches_top_candidate(selected, candidates, top_idx)

    row = {
        "request_sequence": record.get("request_sequence", ""),
        "idx": record.get("idx", response.get("idx", "")),
        "goal_mode": record.get("goal_mode", response.get("goal_mode", "")),
        "goal_x": goal.get("x", ""),
        "goal_y": goal.get("y", ""),
        "elapsed_s": _number_or_blank(record.get("elapsed_s", response.get("elapsed_s"))),
        "candidate_count": int(candidates.shape[0]) if candidates.ndim == 3 else 0,
        "selected_end_x": _number_or_blank(selected_end[0]),
        "selected_end_y": _number_or_blank(selected_end[1]),
        "selected_end_yaw": _number_or_blank(selected_end[2]),
        "selected_endpoint_norm_m": _number_or_blank(selected_norm),
        "selected_path_length_m": _number_or_blank(selected_path_len),
        "selected_max_abs_y": _number_or_blank(selected_max_abs_y),
        "top_candidate_index": "" if top_idx is None else int(top_idx),
        "top_candidate_value": _number_or_blank(top_value),
        "min_value": _number_or_blank(min_value),
        "max_value": _number_or_blank(max_value),
        "selected_matches_top_candidate": selected_matches_top,
    }

    return row, {
        "record": record,
        "response": response,
        "selected": selected,
        "candidates": candidates,
        "values": values,
        "goal": goal,
        "row": row,
    }


def _top_index(values: np.ndarray) -> int | None:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    return int(np.nanargmax(values))


def _nanmin(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if values.size and np.any(np.isfinite(values)) else math.nan


def _nanmax(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if values.size and np.any(np.isfinite(values)) else math.nan


def _endpoint_norm(trajectory: np.ndarray) -> float:
    if trajectory.size == 0:
        return math.nan
    return float(np.linalg.norm(trajectory[-1, :2]))


def _path_length(trajectory: np.ndarray) -> float:
    if trajectory.shape[0] < 2:
        return math.nan
    deltas = np.diff(trajectory[:, :2], axis=0)
    return float(np.nansum(np.linalg.norm(deltas, axis=1)))


def _max_abs_y(trajectory: np.ndarray) -> float:
    if trajectory.size == 0:
        return math.nan
    return float(np.nanmax(np.abs(trajectory[:, 1])))


def _selected_matches_top_candidate(
    selected: np.ndarray,
    candidates: np.ndarray,
    top_idx: int | None,
) -> str:
    if selected.size == 0 or candidates.size == 0 or top_idx is None:
        return ""
    if top_idx < 0 or top_idx >= candidates.shape[0]:
        return ""
    candidate = candidates[top_idx]
    if candidate.shape[0] == 0:
        return ""
    return str(bool(np.allclose(selected[-1, :2], candidate[-1, :2], atol=1e-3, equal_nan=False)))


def _number_or_blank(value: Any) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return number


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def _plot_frame(frame_data: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    selected = frame_data["selected"]
    candidates = frame_data["candidates"]
    values = frame_data["values"]
    goal = frame_data["goal"]
    row = frame_data["row"]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.scatter([0.0], [0.0], marker="o", color="black", s=28, label="robot")

    value_mappable = None
    if candidates.size:
        finite_values = values.size == candidates.shape[0] and np.any(np.isfinite(values))
        if finite_values:
            norm = plt.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
            cmap = plt.get_cmap("viridis")
            segments = [traj[:, :2] for traj in candidates if traj.shape[0] >= 2]
            segment_values = [
                values[i] for i, traj in enumerate(candidates) if traj.shape[0] >= 2
            ]
            if segments:
                lines = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=norm,
                    linewidths=1.4,
                    alpha=0.75,
                )
                lines.set_array(np.asarray(segment_values, dtype=float))
                ax.add_collection(lines)
                value_mappable = lines
        else:
            for candidate in candidates:
                if candidate.shape[0] >= 2:
                    ax.plot(candidate[:, 0], candidate[:, 1], color="0.7", linewidth=1.0, alpha=0.55)

    top_idx = row.get("top_candidate_index")
    if top_idx != "" and candidates.size:
        top_idx_int = int(top_idx)
        if 0 <= top_idx_int < candidates.shape[0]:
            top = candidates[top_idx_int]
            ax.plot(top[:, 0], top[:, 1], color="#f28e2b", linewidth=2.4, linestyle="--", label="top critic")

    if selected.size:
        ax.plot(selected[:, 0], selected[:, 1], color="#111111", linewidth=3.0, label="selected")
        ax.scatter([selected[-1, 0]], [selected[-1, 1]], color="#111111", s=34)

    if "x" in goal and "y" in goal:
        try:
            ax.scatter([float(goal["x"])], [float(goal["y"])], marker="*", color="#d62728", s=120, label="local goal")
        except (TypeError, ValueError):
            pass

    _set_equal_limits(ax, selected, candidates, goal)
    ax.set_title(
        "idx={idx} goal={goal_mode} top={top} y_end={y_end}".format(
            idx=row.get("idx", ""),
            goal_mode=row.get("goal_mode", ""),
            top=_format_title_value(row.get("top_candidate_value")),
            y_end=_format_title_value(row.get("selected_end_y")),
        )
    )
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    if value_mappable is not None:
        fig.colorbar(value_mappable, ax=ax, label="critic value")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_overview(rows: list[dict[str, Any]], records: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.0), constrained_layout=True)

    if not rows:
        for ax in axes:
            ax.axis("off")
        axes[1].text(
            0.5,
            0.5,
            "No NavDP responses recorded.",
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return

    cmap = plt.get_cmap("plasma")
    colors = cmap(np.linspace(0.05, 0.95, len(records)))
    ax_paths = axes[0]
    for color, record in zip(colors, records):
        selected = _as_trajectory(_response_from_record(record).get("trajectory"))
        if selected.size:
            ax_paths.plot(selected[:, 0], selected[:, 1], color=color, linewidth=1.8, alpha=0.8)
    ax_paths.scatter([0.0], [0.0], color="black", s=24, label="robot")
    ax_paths.axhline(0.0, color="0.85", linewidth=0.8)
    ax_paths.axvline(0.0, color="0.85", linewidth=0.8)
    ax_paths.set_aspect("equal", adjustable="box")
    ax_paths.set_title("Selected local trajectories over time")
    ax_paths.set_xlabel("local x (m)")
    ax_paths.set_ylabel("local y (m)")
    ax_paths.grid(True, alpha=0.25)

    indices = np.arange(len(rows))
    end_y = _row_numbers(rows, "selected_end_y")
    max_abs_y = _row_numbers(rows, "selected_max_abs_y")
    axes[1].plot(indices, end_y, marker="o", label="selected end y")
    axes[1].plot(indices, max_abs_y, marker=".", label="selected max |y|")
    axes[1].axhline(0.0, color="0.75", linewidth=0.8)
    axes[1].set_title("Selected trajectory lateral offset")
    axes[1].set_xlabel("response sequence")
    axes[1].set_ylabel("meters")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    min_value = _row_numbers(rows, "min_value")
    max_value = _row_numbers(rows, "max_value")
    top_value = _row_numbers(rows, "top_candidate_value")
    axes[2].plot(indices, max_value, marker="o", label="max critic")
    axes[2].plot(indices, top_value, marker=".", linestyle="--", label="top critic")
    axes[2].plot(indices, min_value, marker=".", label="min critic")
    axes[2].set_title("Critic values")
    axes[2].set_xlabel("response sequence")
    axes[2].set_ylabel("critic")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    fig.savefig(path, dpi=140)
    plt.close(fig)


def _row_numbers(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, math.nan)))
        except (TypeError, ValueError):
            values.append(math.nan)
    return np.asarray(values, dtype=float)


def _set_equal_limits(
    ax: Any,
    selected: np.ndarray,
    candidates: np.ndarray,
    goal: dict[str, Any],
) -> None:
    points = [np.array([[0.0, 0.0]], dtype=float)]
    if selected.size:
        points.append(selected[:, :2])
    if candidates.size:
        points.append(candidates[:, :, :2].reshape(-1, 2))
    if "x" in goal and "y" in goal:
        try:
            points.append(np.array([[float(goal["x"]), float(goal["y"])]], dtype=float))
        except (TypeError, ValueError):
            pass

    stacked = np.vstack(points)
    stacked = stacked[np.all(np.isfinite(stacked), axis=1)]
    if stacked.size == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        return

    min_xy = stacked.min(axis=0)
    max_xy = stacked.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    radius = max(float(np.max(max_xy - min_xy)) / 2.0, 0.5) + 0.25
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def _format_title_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.3f}"
