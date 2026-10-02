"""Record and render diagnostics for NavDP trajectory tracking profiles."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_JSONL_NAME = "tracking_references.jsonl"
CONTROL_CSV_NAME = "control_trace.csv"

CONTROL_COLUMNS = [
    "timestamp_unix_s",
    "monotonic_s",
    "request_count",
    "tracking_profile",
    "odom_x",
    "odom_y",
    "odom_yaw",
    "mpc_v",
    "mpc_w",
    "previous_emitted_v",
    "previous_emitted_w",
    "emitted_v",
    "emitted_w",
    "depth_forward_blocked",
]

REFERENCE_SUMMARY_COLUMNS = [
    "request_sequence",
    "response_idx",
    "tracking_profile",
    "smoothing_applied",
    "smoothing_reason",
    "point_count",
    "max_deviation_m",
    "mean_abs_corner_before_deg",
    "mean_abs_corner_after_deg",
    "total_abs_turn_before_deg",
    "total_abs_turn_after_deg",
]


class TrackingDiagnosticsRecorder:
    """Thread-safe writers for per-plan references and per-control-step traces."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.reference_path = self.run_dir / REFERENCE_JSONL_NAME
        self.control_path = self.run_dir / CONTROL_CSV_NAME
        self._reference_file = self.reference_path.open("w", encoding="utf-8")
        self._control_file = self.control_path.open("w", encoding="utf-8", newline="")
        self._control_writer = csv.DictWriter(
            self._control_file,
            fieldnames=CONTROL_COLUMNS,
        )
        self._control_writer.writeheader()
        self._lock = threading.Lock()
        self._closed = False

    def write_reference(
        self,
        *,
        request_sequence: int,
        response_idx: int,
        tracking_profile: str,
        raw_local_xy: np.ndarray,
        effective_local_xy: np.ndarray,
        smoothing_applied: bool,
        smoothing_reason: str,
        max_deviation_m: float,
        mean_abs_corner_before_rad: float,
        mean_abs_corner_after_rad: float,
        total_abs_turn_before_rad: float,
        total_abs_turn_after_rad: float,
        goal: dict[str, Any] | None,
    ) -> None:
        record = {
            "request_sequence": int(request_sequence),
            "response_idx": int(response_idx),
            "timestamp_unix_s": time.time(),
            "tracking_profile": tracking_profile,
            "smoothing_applied": bool(smoothing_applied),
            "smoothing_reason": smoothing_reason,
            "max_deviation_m": float(max_deviation_m),
            "mean_abs_corner_before_rad": float(mean_abs_corner_before_rad),
            "mean_abs_corner_after_rad": float(mean_abs_corner_after_rad),
            "total_abs_turn_before_rad": float(total_abs_turn_before_rad),
            "total_abs_turn_after_rad": float(total_abs_turn_after_rad),
            "raw_local_xy": np.asarray(raw_local_xy, dtype=float).tolist(),
            "effective_local_xy": np.asarray(effective_local_xy, dtype=float).tolist(),
            "goal": goal,
        }
        with self._lock:
            if self._closed:
                return
            self._reference_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._reference_file.flush()

    def write_control(self, **values: Any) -> None:
        row = {column: values.get(column, "") for column in CONTROL_COLUMNS}
        row["timestamp_unix_s"] = values.get("timestamp_unix_s", time.time())
        row["monotonic_s"] = values.get("monotonic_s", time.monotonic())
        with self._lock:
            if self._closed:
                return
            self._control_writer.writerow(row)
            self._control_file.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._reference_file.close()
            self._control_file.close()


def _load_reference_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _write_reference_summary(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=REFERENCE_SUMMARY_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "request_sequence": record.get("request_sequence", ""),
                    "response_idx": record.get("response_idx", ""),
                    "tracking_profile": record.get("tracking_profile", ""),
                    "smoothing_applied": record.get("smoothing_applied", False),
                    "smoothing_reason": record.get("smoothing_reason", ""),
                    "point_count": len(record.get("effective_local_xy", [])),
                    "max_deviation_m": record.get("max_deviation_m", 0.0),
                    "mean_abs_corner_before_deg": np.degrees(
                        record.get("mean_abs_corner_before_rad", 0.0)
                    ),
                    "mean_abs_corner_after_deg": np.degrees(
                        record.get("mean_abs_corner_after_rad", 0.0)
                    ),
                    "total_abs_turn_before_deg": np.degrees(
                        record.get("total_abs_turn_before_rad", 0.0)
                    ),
                    "total_abs_turn_after_deg": np.degrees(
                        record.get("total_abs_turn_after_rad", 0.0)
                    ),
                }
            )


def _plot_reference_diagnostics(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_dir = run_dir / "tracking_reference_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for frame_number, record in enumerate(records):
        raw_xy = np.asarray(record.get("raw_local_xy", []), dtype=float)
        effective_xy = np.asarray(record.get("effective_local_xy", []), dtype=float)
        figure, axis = plt.subplots(figsize=(7, 7))
        if raw_xy.ndim == 2 and raw_xy.shape[1] >= 2:
            axis.plot(raw_xy[:, 0], raw_xy[:, 1], "--", color="tab:gray", label="raw")
        if effective_xy.ndim == 2 and effective_xy.shape[1] >= 2:
            axis.plot(
                effective_xy[:, 0],
                effective_xy[:, 1],
                color="tab:blue",
                linewidth=2,
                label="MPC reference",
            )
        goal = record.get("goal")
        if isinstance(goal, dict) and "x" in goal and "y" in goal:
            axis.scatter([goal["x"]], [goal["y"]], marker="*", s=120, color="tab:red", label="local goal")
        axis.scatter([0.0], [0.0], color="black", s=30, label="robot")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("local x (m)")
        axis.set_ylabel("local y (m)")
        axis.set_title(
            f"profile={record.get('tracking_profile', '')} "
            f"idx={record.get('response_idx', '')} "
            f"max_shift={record.get('max_deviation_m', 0.0):.3f} m"
        )
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(frame_dir / f"idx_{frame_number:04d}.png", dpi=140)
        plt.close(figure)

    overview_path = run_dir / "tracking_reference_overview.png"
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sequences = [record.get("request_sequence", index) for index, record in enumerate(records)]
    before = [np.degrees(record.get("mean_abs_corner_before_rad", 0.0)) for record in records]
    after = [np.degrees(record.get("mean_abs_corner_after_rad", 0.0)) for record in records]
    shifts = [record.get("max_deviation_m", 0.0) for record in records]
    axes[0].plot(sequences, before, marker="o", label="raw mean corner")
    axes[0].plot(sequences, after, marker="o", label="effective mean corner")
    axes[0].set_ylabel("degrees")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(sequences, shifts, marker="o", color="tab:orange")
    axes[1].axhline(0.03, linestyle="--", color="tab:red", label="3 cm limit")
    axes[1].set_xlabel("request sequence")
    axes[1].set_ylabel("max shift (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    figure.tight_layout()
    figure.savefig(overview_path, dpi=140)
    plt.close(figure)
    return overview_path, frame_dir


def _plot_control_trace(run_dir: Path, control_path: Path) -> Path | None:
    if not control_path.is_file():
        return None
    with control_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = np.asarray([float(row["timestamp_unix_s"]) for row in rows])
    elapsed = timestamps - timestamps[0]
    mpc_v = np.asarray([float(row["mpc_v"]) for row in rows])
    mpc_w = np.asarray([float(row["mpc_w"]) for row in rows])
    emitted_v = np.asarray([float(row["emitted_v"]) for row in rows])
    emitted_w = np.asarray([float(row["emitted_w"]) for row in rows])
    blocked = np.asarray(
        [str(row["depth_forward_blocked"]).lower() in {"1", "true"} for row in rows],
        dtype=bool,
    )
    odom_x = np.asarray([float(row["odom_x"]) for row in rows])
    odom_y = np.asarray([float(row["odom_y"]) for row in rows])

    output_path = run_dir / "control_trace.png"
    figure, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(elapsed, mpc_v, label="MPC v")
    axes[0].plot(elapsed, emitted_v, label="emitted v")
    axes[0].fill_between(elapsed, 0, 1, where=blocked, alpha=0.15, transform=axes[0].get_xaxis_transform(), label="depth block")
    axes[0].set_ylabel("m/s")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(elapsed, mpc_w, label="MPC w")
    axes[1].plot(elapsed, emitted_w, label="emitted w")
    axes[1].axhline(0.4, linestyle="--", color="tab:red", alpha=0.5)
    axes[1].axhline(-0.4, linestyle="--", color="tab:red", alpha=0.5)
    axes[1].set_ylabel("rad/s")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(odom_x, odom_y)
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].set_xlabel("world x (m)")
    axes[2].set_ylabel("world y (m)")
    axes[2].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def generate_tracking_profile_analysis(run_dir: str | Path) -> dict[str, Path]:
    """Generate CSV and PNG artifacts from tracking diagnostic logs."""

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    reference_path = resolved_run_dir / REFERENCE_JSONL_NAME
    control_path = resolved_run_dir / CONTROL_CSV_NAME
    records = _load_reference_records(reference_path)

    summary_path = resolved_run_dir / "tracking_reference_summary.csv"
    _write_reference_summary(summary_path, records)
    overview_path, frame_dir = _plot_reference_diagnostics(resolved_run_dir, records)
    control_plot = _plot_control_trace(resolved_run_dir, control_path)

    artifacts = {
        "reference_jsonl": reference_path,
        "control_csv": control_path,
        "summary_csv": summary_path,
        "overview_png": overview_path,
        "frame_dir": frame_dir,
    }
    if control_plot is not None:
        artifacts["control_png"] = control_plot
    return artifacts
