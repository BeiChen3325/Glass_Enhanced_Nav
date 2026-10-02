#!/usr/bin/env python3
"""Render every successful NavDP planning response as an inspectable frame."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
DEPLOYMENT_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(DEPLOYMENT_ROOT / "source"))

from go2_internnav.navdp.trajectory_smoothing import (  # noqa: E402
    DEFAULT_TRAJECTORY_SMOOTHING_CONFIG,
    smooth_trajectory_xy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one image per successful response in a NavDP JSONL log."
    )
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory. A trajectory_sequence_YYYYMMDD_HHMMSS directory "
            "is always created below it. Defaults to the response directory."
        ),
    )
    parser.add_argument(
        "--no-gif", action="store_true", help="Skip the animated GIF export."
    )
    parser.add_argument("--gif-fps", type=float, default=3.0)
    parser.add_argument(
        "--max-gif-frames",
        type=int,
        default=300,
        help="Skip GIF creation above this frame count; PNG and HTML are unaffected.",
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
        if not isinstance(record.get("response"), dict):
            continue
        if "trajectory" not in record["response"]:
            continue
        records.append(record)
    if not records:
        raise ValueError("no successful trajectory responses found")
    return records


def local_to_world(points_xy: np.ndarray, odom: np.ndarray) -> np.ndarray:
    yaw = float(odom[2])
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    return points_xy @ rotation.T + odom[:2]


def flatten_candidates(value: object) -> np.ndarray:
    candidates = np.asarray(value, dtype=float)
    if candidates.ndim == 4 and candidates.shape[0] == 1:
        candidates = candidates[0]
    if candidates.ndim != 3 or candidates.shape[2] < 2:
        raise ValueError(f"invalid all_trajectory shape: {candidates.shape}")
    return candidates


def flatten_values(value: object) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("empty Critic values")
    return values


def smooth_selected(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if trajectory.ndim != 2 or trajectory.shape[0] < 4 or trajectory.shape[1] < 2:
        raise ValueError(f"invalid selected trajectory shape: {trajectory.shape}")
    dropped = trajectory[3:, :2]
    smoothed = smooth_trajectory_xy(
        dropped, DEFAULT_TRAJECTORY_SMOOTHING_CONFIG
    ).trajectory_xy
    return dropped, smoothed


def temporal_metrics(record: dict) -> dict:
    response = record["response"]
    temporal = response.get("temporal_input")
    return temporal if isinstance(temporal, dict) else {}


def candidate_safe_mask(response: dict, count: int) -> np.ndarray:
    safety = response.get("candidate_safety")
    if isinstance(safety, dict) and "candidate_safe" in safety:
        mask = np.asarray(safety["candidate_safe"], dtype=bool).reshape(-1)
        if mask.size == count:
            return mask
    return np.ones(count, dtype=bool)


def frame_metadata(record: dict, frame_file: str) -> dict:
    response = record["response"]
    temporal = temporal_metrics(record)
    selected = int(response["final_selection_index"])
    values = flatten_values(response["all_values"])
    trajectory = np.asarray(response["trajectory"], dtype=float)
    endpoint = trajectory[-1, :2]
    return {
        "file": frame_file,
        "request_sequence": int(record.get("request_sequence", -1)),
        "anchor_frame_id": record.get("anchor_frame_id"),
        "selected_index": selected,
        "critic_value": float(values[selected]),
        "safe_candidate_count": int(response.get("safe_candidate_count", -1)),
        "source_lag_frames": response.get("client_source_lag_frames"),
        "elapsed_s": float(record.get("elapsed_s", math.nan)),
        "temporal_status": temporal.get("status", "unknown"),
        "endpoint_xy": [float(endpoint[0]), float(endpoint[1])],
    }


def style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold", fontsize=11)
    axis.grid(True, color="#dfe5ec", linewidth=0.7, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)


def render_frame(
    records: list[dict],
    record_index: int,
    output_path: Path,
    dpi: int,
) -> None:
    record = records[record_index]
    response = record["response"]
    temporal = temporal_metrics(record)
    candidates = flatten_candidates(response["all_trajectory"])
    values = flatten_values(response["all_values"])
    selected = int(response["final_selection_index"])
    safe = candidate_safe_mask(response, len(candidates))
    raw_trajectory = np.asarray(response["trajectory"], dtype=float)
    dropped_xy, smoothed_xy = smooth_selected(raw_trajectory)
    odom = np.asarray(record["inference_odom"], dtype=float)
    odom_all = np.asarray(
        [item["inference_odom"] for item in records[: record_index + 1]], dtype=float
    )

    goal_local = np.asarray(
        [record["goal"]["x"], record["goal"]["y"]], dtype=float
    )
    goal_world = local_to_world(goal_local[np.newaxis], odom)[0]
    selected_world = local_to_world(smoothed_xy, odom)

    figure = plt.figure(figsize=(14, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.55, 1.0))
    ax_local = figure.add_subplot(grid[0, 0])
    ax_world = figure.add_subplot(grid[0, 1])
    ax_critic = figure.add_subplot(grid[1, :])

    sequence = int(record.get("request_sequence", record_index))
    figure.suptitle(
        f"NavDP planning frame — request_sequence={sequence}",
        fontsize=16,
        fontweight="bold",
    )

    style_axis(ax_local, "A. All candidates and selected local trajectory")
    for candidate_index, candidate in enumerate(candidates):
        is_selected = candidate_index == selected
        ax_local.plot(
            candidate[:, 0],
            candidate[:, 1],
            color="#d1495b" if is_selected else "#cbd3dc",
            linewidth=3.0 if is_selected else 1.0,
            alpha=1.0 if is_selected else 0.75,
            zorder=3 if is_selected else 1,
        )
        ax_local.text(
            candidate[-1, 0],
            candidate[-1, 1],
            str(candidate_index),
            fontsize=7,
            color="#374151",
        )
    ax_local.plot(
        dropped_xy[:, 0],
        dropped_xy[:, 1],
        "--",
        color="#ff8c42",
        linewidth=1.8,
        label="selected raw after drop-3",
        zorder=4,
    )
    ax_local.plot(
        smoothed_xy[:, 0],
        smoothed_xy[:, 1],
        color="#0057b8",
        linewidth=3.0,
        label="smoothed MPC reference",
        zorder=5,
    )
    ax_local.scatter([0], [0], marker="^", s=90, color="black", label="robot")
    ax_local.add_patch(
        Rectangle(
            (-0.35, -0.215),
            0.70,
            0.43,
            fill=False,
            edgecolor="#111827",
            linewidth=1.5,
        )
    )
    goal_norm = max(float(np.linalg.norm(goal_local)), 1e-9)
    arrow_end = goal_local / goal_norm * 1.1
    ax_local.add_patch(
        FancyArrowPatch(
            (0, 0),
            tuple(arrow_end),
            arrowstyle="-|>",
            mutation_scale=16,
            color="#f2a900",
            linewidth=2.2,
            label="direction to goal",
        )
    )
    local_xy = np.concatenate([candidates[:, :, :2].reshape(-1, 2), [[0.0, 0.0]]])
    x_margin = max(0.18, 0.08 * float(np.ptp(local_xy[:, 0])))
    y_margin = max(0.18, 0.15 * float(np.ptp(local_xy[:, 1])))
    ax_local.set_xlim(float(local_xy[:, 0].min()) - x_margin, float(local_xy[:, 0].max()) + x_margin)
    ax_local.set_ylim(float(local_xy[:, 1].min()) - y_margin, float(local_xy[:, 1].max()) + y_margin)
    ax_local.set_xlabel("forward x [m]")
    ax_local.set_ylabel("left y [m] (negative = right)")
    ax_local.set_aspect("equal", adjustable="box")
    ax_local.legend(loc="best", fontsize=8)

    style_axis(ax_world, "B. Sampled odom and current plan in world frame")
    ax_world.plot(
        odom_all[:, 0],
        odom_all[:, 1],
        color="#2a9d8f",
        linewidth=2.5,
        marker="o",
        markersize=3,
        label="sampled odom",
    )
    ax_world.scatter(
        [odom_all[0, 0]], [odom_all[0, 1]], s=70, color="#157f3b", label="start"
    )
    ax_world.scatter(
        [odom[0]], [odom[1]], s=90, marker="X", color="#c93a3a", label="anchor"
    )
    ax_world.plot(
        selected_world[:, 0],
        selected_world[:, 1],
        color="#0057b8",
        linewidth=3.0,
        label="current MPC reference",
    )
    ax_world.scatter(
        [goal_world[0]],
        [goal_world[1]],
        marker="*",
        s=190,
        color="#f2a900",
        edgecolor="black",
        label="goal",
    )
    ax_world.arrow(
        odom[0],
        odom[1],
        0.18 * math.cos(odom[2]),
        0.18 * math.sin(odom[2]),
        width=0.008,
        color="#111827",
    )
    ax_world.set_xlabel("world x [m]")
    ax_world.set_ylabel("world y [m]")
    ax_world.set_aspect("equal", adjustable="datalim")
    ax_world.legend(loc="best", fontsize=8)

    style_axis(ax_critic, "C. Critic values, safety filtering and frame health")
    colors = np.where(safe, "#2a9d8f", "#d96c75").astype(object)
    colors[selected] = "#f2a900"
    bars = ax_critic.bar(
        np.arange(len(values)), values, color=colors.tolist(), edgecolor="#374151"
    )
    ax_critic.axhline(
        -3.0, color="#6b7280", linestyle="--", linewidth=1.2, label="stop threshold"
    )
    ax_critic.set_xticks(np.arange(len(values)))
    ax_critic.set_xlabel("candidate index")
    ax_critic.set_ylabel("Critic value")
    for index, (bar, value) in enumerate(zip(bars, values)):
        ax_critic.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7,
            rotation=90,
        )

    lag = response.get("client_source_lag_frames")
    stats = (
        f"selected={selected}  critic={values[selected]:.3f}  "
        f"safe={int(safe.sum())}/{len(safe)}  "
        f"collision_rejected={response.get('collision_rejection_count', '?')}\n"
        f"temporal={temporal.get('status', 'unknown')}  "
        f"source_fps={temporal.get('source_fps_hz', math.nan):.2f}  "
        f"source_lag={lag} frames  "
        f"RGB_dup={temporal.get('rgb_duplicate_ratio', math.nan):.3f}  "
        f"depth_dup={temporal.get('depth_duplicate_ratio', math.nan):.3f}  "
        f"latency={float(record.get('elapsed_s', math.nan)):.3f}s\n"
        f"anchor_frame={record.get('anchor_frame_id')}  "
        f"odom=({odom[0]:.3f}, {odom[1]:.3f}, {odom[2]:.3f})  "
        f"raw_endpoint=({raw_trajectory[-1, 0]:.3f}, {raw_trajectory[-1, 1]:.3f})"
    )
    ax_critic.text(
        0.99,
        0.96,
        stats,
        transform=ax_critic.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#9aa5b1",
            "alpha": 0.94,
        },
    )
    ax_critic.legend(loc="lower left", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def write_html(output_dir: Path, frames: list[dict]) -> Path:
    frames_json = json.dumps(frames, ensure_ascii=False)
    html_path = output_dir / "index.html"
    html_path.write_text(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NavDP 逐规划帧轨迹</title>
  <style>
    body { margin: 0; background: #10141b; color: #edf2f7; font-family: sans-serif; }
    main { max-width: 1500px; margin: auto; padding: 16px; }
    h1 { font-size: 22px; margin: 0 0 12px; }
    #view { width: 100%; display: block; background: white; border-radius: 8px; }
    .controls { display: grid; grid-template-columns: auto auto 1fr auto; gap: 10px; align-items: center; margin: 12px 0; }
    button { padding: 8px 14px; border: 0; border-radius: 6px; cursor: pointer; }
    input[type=range] { width: 100%; }
    #details { color: #cbd5e1; font-family: monospace; white-space: pre-wrap; }
  </style>
</head>
<body><main>
  <h1>NavDP 每次成功推理的规划轨迹</h1>
  <img id="view" alt="planning frame">
  <div class="controls">
    <button id="prev">上一帧</button><button id="play">播放</button>
    <input id="slider" type="range" min="0" step="1"><span id="counter"></span>
  </div>
  <div id="details"></div>
</main>
<script>
const frames = __FRAMES__;
const image = document.getElementById('view');
const slider = document.getElementById('slider');
const counter = document.getElementById('counter');
const details = document.getElementById('details');
const playButton = document.getElementById('play');
let index = 0, timer = null;
slider.max = String(frames.length - 1);
function show(next) {
  index = Math.max(0, Math.min(frames.length - 1, next));
  const frame = frames[index];
  image.src = frame.file;
  slider.value = String(index);
  counter.textContent = `${index + 1}/${frames.length}`;
  details.textContent = JSON.stringify(frame, null, 2);
}
function stop() { if (timer) clearInterval(timer); timer = null; playButton.textContent = '播放'; }
function toggle() {
  if (timer) { stop(); return; }
  playButton.textContent = '暂停';
  timer = setInterval(() => show((index + 1) % frames.length), 400);
}
document.getElementById('prev').onclick = () => show(index - 1);
playButton.onclick = toggle;
slider.oninput = () => { stop(); show(Number(slider.value)); };
document.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') show(index - 1);
  if (event.key === 'ArrowRight') show(index + 1);
  if (event.key === ' ') { event.preventDefault(); toggle(); }
});
show(0);
</script></body></html>
""".replace("__FRAMES__", frames_json),
        encoding="utf-8",
    )
    return html_path


def write_gif(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    if fps <= 0.0:
        raise ValueError("GIF FPS must be positive")
    images = []
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    adaptive_palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE")
    for path in frame_paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((1200, 760), resampling)
            images.append(
                image.convert("P", palette=adaptive_palette, colors=256)
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, int(round(1000.0 / fps))),
        loop=0,
        optimize=False,
    )


def main() -> int:
    args = parse_args()
    records = load_records(args.responses)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or args.responses.resolve().parent
    output_dir = output_root / f"trajectory_sequence_{timestamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"trajectory_sequence_{timestamp}_{suffix:02d}"
        suffix += 1
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    metadata: list[dict] = []
    for index, record in enumerate(records):
        sequence = int(record.get("request_sequence", index))
        frame_path = frames_dir / f"plan_{index:04d}_seq_{sequence:04d}.png"
        render_frame(records, index, frame_path, int(args.dpi))
        frame_paths.append(frame_path)
        metadata.append(frame_metadata(record, f"frames/{frame_path.name}"))
        print(f"rendered {index + 1}/{len(records)}: {frame_path.name}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_path = write_html(output_dir, metadata)
    gif_output = output_dir / "trajectory_sequence.gif"
    gif_enabled = not args.no_gif and len(records) <= int(args.max_gif_frames)
    if gif_enabled:
        write_gif(frame_paths, gif_output, float(args.gif_fps))
    elif not args.no_gif:
        print(
            f"gif_skipped=frame_count_{len(records)}_exceeds_{args.max_gif_frames}",
            flush=True,
        )

    print(f"records={len(records)}")
    print(f"frames_dir={frames_dir}")
    print(f"html={html_path}")
    print(f"manifest={manifest_path}")
    if gif_enabled:
        print(f"gif={gif_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

