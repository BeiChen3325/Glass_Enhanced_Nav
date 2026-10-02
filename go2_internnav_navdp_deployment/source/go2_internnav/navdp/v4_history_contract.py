"""Strict validation for stateless NavDP V4 real-sensor snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from typing import BinaryIO, Literal, Optional

import numpy as np


HistoryMode = Literal["pointgoal", "imagegoal", "nogoal"]
REQUIRED_FIELDS = (
    "env_ids",
    "episode_ids",
    "rgb_history",
    "depth_history_u16",
    "odom_history",
    "rgb_frame_ids",
    "depth_frame_ids",
    "rgb_timestamps",
    "depth_timestamps",
    "rgb_target_timestamps",
    "depth_target_timestamps",
    "clock_domain",
    "source_fps_hz",
    "max_selection_error_s",
    "rgb_duplicate_ratio",
    "depth_duplicate_ratio",
)
RGB_TARGET_STEPS = np.full(7, 4.0 / 30.0, dtype=np.float64)
DEPTH_TARGET_STEPS = np.full(7, 1.0 / 30.0, dtype=np.float64)


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class V4HistoryRequest:
    env_ids: np.ndarray
    episode_ids: np.ndarray
    rgb_history: np.ndarray
    depth_history_u16: np.ndarray
    odom_history: np.ndarray
    rgb_frame_ids: np.ndarray
    depth_frame_ids: np.ndarray
    rgb_timestamps: np.ndarray
    depth_timestamps: np.ndarray
    rgb_target_timestamps: np.ndarray
    depth_target_timestamps: np.ndarray
    clock_domain: str
    source_fps_hz: float
    max_selection_error_s: float
    rgb_duplicate_ratio: float
    depth_duplicate_ratio: float
    camera_intrinsic: Optional[np.ndarray]
    point_goals: Optional[np.ndarray]
    image_goals: Optional[np.ndarray]

    @property
    def batch_size(self) -> int:
        return int(self.env_ids.shape[0])

    @property
    def depth_history_m(self) -> np.ndarray:
        return self.depth_history_u16.astype(np.float32) / 10_000.0

    @property
    def anchor_frame_ids(self) -> np.ndarray:
        return self.depth_frame_ids[:, -1]

    @property
    def anchor_timestamps(self) -> np.ndarray:
        return self.depth_timestamps[:, -1]

    @property
    def degraded_temporal_input(self) -> bool:
        return bool(
            self.source_fps_hz < 27.0
            or self.rgb_duplicate_ratio > 0.0
            or self.depth_duplicate_ratio > 0.0
        )


def _read_npz(stream: BinaryIO | bytes) -> dict[str, np.ndarray]:
    if isinstance(stream, bytes):
        buffer = io.BytesIO(stream)
    else:
        buffer = io.BytesIO(stream.read())
    try:
        with np.load(buffer, allow_pickle=False) as archive:
            missing = [name for name in REQUIRED_FIELDS if name not in archive]
            if missing:
                raise ValueError(
                    "history NPZ is missing required fields: " + ", ".join(missing)
                )
            return {name: archive[name] for name in archive.files}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid history NPZ: {exc}") from exc


def _scalar_metric(
    values: dict[str, np.ndarray], name: str, *, minimum: float, maximum: float
) -> float:
    array = np.asarray(values[name], dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"{name} must contain one value")
    value = float(array.reshape(-1)[0])
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    return value


def load_v4_history_request(
    stream: BinaryIO | bytes,
    mode: HistoryMode,
    *,
    odom_jump_translation_m: float = 0.75,
    odom_jump_yaw_rad: float = math.pi / 4.0,
    maximum_selection_error_s: float = 0.050,
) -> V4HistoryRequest:
    if mode not in ("pointgoal", "imagegoal", "nogoal"):
        raise ValueError(f"unsupported V4 history mode: {mode}")
    values = _read_npz(stream)
    env_ids = np.asarray(values["env_ids"])
    episode_ids = np.asarray(values["episode_ids"])
    rgb = np.asarray(values["rgb_history"])
    depth = np.asarray(values["depth_history_u16"])
    odom = np.asarray(values["odom_history"], dtype=np.float64)
    rgb_ids = np.asarray(values["rgb_frame_ids"])
    depth_ids = np.asarray(values["depth_frame_ids"])
    rgb_times = np.asarray(values["rgb_timestamps"], dtype=np.float64)
    depth_times = np.asarray(values["depth_timestamps"], dtype=np.float64)
    rgb_targets = np.asarray(values["rgb_target_timestamps"], dtype=np.float64)
    depth_targets = np.asarray(values["depth_target_timestamps"], dtype=np.float64)

    if env_ids.ndim != 1 or env_ids.size != 1 or not np.issubdtype(
        env_ids.dtype, np.integer
    ):
        raise ValueError("real V4 env_ids must be integer [1]")
    env_ids = env_ids.astype(np.int64, copy=False)
    if int(env_ids[0]) != 0:
        raise ValueError("real V4 env_id must be 0")
    if episode_ids.shape != (1,) or not np.issubdtype(
        episode_ids.dtype, np.integer
    ):
        raise ValueError("episode_ids must be integer [1]")
    episode_ids = episode_ids.astype(np.int64, copy=False)
    if int(episode_ids[0]) < 0:
        raise ValueError("episode_id must be non-negative")

    if rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[:2] != (1, 8) or rgb.shape[-1] != 3:
        raise ValueError("rgb_history must be uint8 [1,8,H,W,3]")
    if depth.dtype != np.uint16 or depth.ndim != 4 or depth.shape[:2] != (1, 8):
        raise ValueError("depth_history_u16 must be uint16 [1,8,H,W]")
    if depth.shape[2:4] != rgb.shape[2:4]:
        raise ValueError("RGB/depth spatial shapes must match")
    if odom.shape != (1, 8, 3) or not np.all(np.isfinite(odom)):
        raise ValueError("odom_history must be finite [1,8,3]")

    for name, array in (("rgb_frame_ids", rgb_ids), ("depth_frame_ids", depth_ids)):
        if array.shape != (1, 8) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be integer [1,8]")
        if np.any(np.diff(array, axis=1) < 0):
            raise ValueError(f"{name} must be non-decreasing")
    rgb_ids = rgb_ids.astype(np.int64, copy=False)
    depth_ids = depth_ids.astype(np.int64, copy=False)
    if int(rgb_ids[0, -1]) != int(depth_ids[0, -1]):
        raise ValueError("RGB/depth histories must share the anchor frame")

    for name, array in (
        ("rgb_timestamps", rgb_times),
        ("depth_timestamps", depth_times),
        ("rgb_target_timestamps", rgb_targets),
        ("depth_target_timestamps", depth_targets),
    ):
        if array.shape != (1, 8) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite [1,8]")
    if np.any(np.diff(rgb_times, axis=1) < 0.0) or np.any(
        np.diff(depth_times, axis=1) < 0.0
    ):
        raise ValueError("source timestamps must be non-decreasing")
    if not np.allclose(np.diff(rgb_targets, axis=1)[0], RGB_TARGET_STEPS, atol=1.0e-6):
        raise ValueError("RGB target timestamps must use 30 FPS stride 4")
    if not np.allclose(np.diff(depth_targets, axis=1)[0], DEPTH_TARGET_STEPS, atol=1.0e-6):
        raise ValueError("depth target timestamps must use consecutive 30 FPS targets")
    if not math.isclose(float(rgb_times[0, -1]), float(depth_times[0, -1]), abs_tol=1.0e-6):
        raise ValueError("RGB/depth histories must share the anchor time")
    if not math.isclose(float(rgb_targets[0, -1]), float(depth_targets[0, -1]), abs_tol=1.0e-6):
        raise ValueError("RGB/depth target histories must share the anchor time")

    clock_values = np.asarray(values["clock_domain"])
    if clock_values.size != 1:
        raise ValueError("clock_domain must contain one value")
    clock_domain = str(clock_values.reshape(-1)[0])
    if clock_domain not in ("ros_header", "monotonic_receipt"):
        raise ValueError("clock_domain is invalid")
    source_fps = _scalar_metric(values, "source_fps_hz", minimum=0.1, maximum=240.0)
    declared_error = _scalar_metric(
        values,
        "max_selection_error_s",
        minimum=0.0,
        maximum=maximum_selection_error_s,
    )
    rgb_duplicate_ratio = _scalar_metric(
        values, "rgb_duplicate_ratio", minimum=0.0, maximum=1.0
    )
    depth_duplicate_ratio = _scalar_metric(
        values, "depth_duplicate_ratio", minimum=0.0, maximum=1.0
    )
    observed_error = float(
        max(
            np.abs(rgb_times - rgb_targets).max(),
            np.abs(depth_times - depth_targets).max(),
        )
    )
    if observed_error > declared_error + 1.0e-6:
        raise ValueError("declared max_selection_error_s is smaller than observed")
    observed_rgb_duplicates = float(np.mean(np.diff(rgb_ids[0]) == 0))
    observed_depth_duplicates = float(np.mean(np.diff(depth_ids[0]) == 0))
    if not math.isclose(observed_rgb_duplicates, rgb_duplicate_ratio, abs_tol=1.0e-6):
        raise ValueError("rgb_duplicate_ratio does not match frame ids")
    if not math.isclose(observed_depth_duplicates, depth_duplicate_ratio, abs_tol=1.0e-6):
        raise ValueError("depth_duplicate_ratio does not match frame ids")

    translations = np.linalg.norm(np.diff(odom[:, :, :2], axis=1), axis=-1)
    yaw_deltas = np.abs(_wrap_angle(np.diff(odom[:, :, 2], axis=1)))
    if np.any(translations > odom_jump_translation_m) or np.any(
        yaw_deltas > odom_jump_yaw_rad
    ):
        raise ValueError("odom_history contains a discontinuity")

    point_goals = None
    image_goals = None
    camera_intrinsic = None
    if "camera_intrinsic" in values:
        camera_intrinsic = np.asarray(values["camera_intrinsic"], dtype=np.float64)
        if camera_intrinsic.shape != (3, 3) or not np.all(np.isfinite(camera_intrinsic)):
            raise ValueError("camera_intrinsic must be finite [3,3]")
    if mode == "pointgoal":
        if "point_goals" not in values:
            raise ValueError("pointgoal history NPZ requires point_goals")
        point_goals = np.asarray(values["point_goals"], dtype=np.float32)
        if point_goals.shape != (1, 2) or not np.all(np.isfinite(point_goals)):
            raise ValueError("point_goals must be finite [1,2]")
    elif mode == "imagegoal":
        if "image_goals" not in values:
            raise ValueError("imagegoal history NPZ requires image_goals")
        image_goals = np.asarray(values["image_goals"])
        if image_goals.dtype != np.uint8 or image_goals.ndim != 4 or image_goals.shape[0] != 1 or image_goals.shape[-1] != 3:
            raise ValueError("image_goals must be uint8 [1,H,W,3]")

    return V4HistoryRequest(
        env_ids=env_ids,
        episode_ids=episode_ids,
        rgb_history=rgb,
        depth_history_u16=depth,
        odom_history=odom,
        rgb_frame_ids=rgb_ids,
        depth_frame_ids=depth_ids,
        rgb_timestamps=rgb_times,
        depth_timestamps=depth_times,
        rgb_target_timestamps=rgb_targets,
        depth_target_timestamps=depth_targets,
        clock_domain=clock_domain,
        source_fps_hz=source_fps,
        max_selection_error_s=declared_error,
        rgb_duplicate_ratio=rgb_duplicate_ratio,
        depth_duplicate_ratio=depth_duplicate_ratio,
        camera_intrinsic=camera_intrinsic,
        point_goals=point_goals,
        image_goals=image_goals,
    )
