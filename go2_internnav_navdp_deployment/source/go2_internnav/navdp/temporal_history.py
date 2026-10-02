"""Bounded real-robot sensor history for stateless NavDP V4 inference."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import threading
from typing import Optional

import numpy as np


TARGET_FPS = 30.0
RGB_TARGET_OFFSETS = np.arange(28, -1, -4, dtype=np.float64) / TARGET_FPS
DEPTH_TARGET_OFFSETS = np.arange(7, -1, -1, dtype=np.float64) / TARGET_FPS
DEFAULT_CAPACITY = 32
DEFAULT_ODOM_CAPACITY = 512
DEFAULT_MAX_FRAME_SELECTION_ERROR_S = 0.045
DEFAULT_MAX_ODOM_AGE_S = 0.030
ODOM_JUMP_TRANSLATION_M = 0.75
ODOM_JUMP_YAW_RAD = math.pi / 4.0


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def ros_stamp_seconds(message: object) -> float:
    """Extract a finite positive ROS header stamp, or return NaN."""

    try:
        stamp = message.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) / 1.0e9
    except (AttributeError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) and value > 0.0 else math.nan


@dataclass(frozen=True)
class OdomMatch:
    pose: np.ndarray
    error_s: float


class BoundedOdomHistory:
    """Preallocated odometry history addressable in two independent clocks."""

    def __init__(self, capacity: int = DEFAULT_ODOM_CAPACITY) -> None:
        if int(capacity) < 8:
            raise ValueError("odom capacity must be at least 8")
        self.capacity = int(capacity)
        self._header_times = np.full(self.capacity, np.nan, dtype=np.float64)
        self._receipt_times = np.full(self.capacity, np.nan, dtype=np.float64)
        self._poses = np.empty((self.capacity, 3), dtype=np.float64)
        self._valid = np.zeros(self.capacity, dtype=bool)
        self._write_index = 0
        self._size = 0
        self._lock = threading.RLock()

    @property
    def memory_bytes(self) -> int:
        return int(
            self._header_times.nbytes
            + self._receipt_times.nbytes
            + self._poses.nbytes
            + self._valid.nbytes
        )

    def append(
        self,
        *,
        header_timestamp: float,
        receipt_timestamp: float,
        pose: np.ndarray,
    ) -> None:
        receipt_timestamp = float(receipt_timestamp)
        header_timestamp = float(header_timestamp)
        pose = np.asarray(pose, dtype=np.float64)
        if not math.isfinite(receipt_timestamp):
            raise ValueError("receipt timestamp must be finite")
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("odom pose must be finite [3]")
        if not math.isfinite(header_timestamp) or header_timestamp <= 0.0:
            header_timestamp = math.nan
        with self._lock:
            slot = self._write_index
            self._header_times[slot] = header_timestamp
            self._receipt_times[slot] = receipt_timestamp
            self._poses[slot] = pose
            self._valid[slot] = True
            self._write_index = (slot + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def nearest(self, timestamp: float, clock_domain: str) -> Optional[OdomMatch]:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            return None
        if clock_domain not in ("ros_header", "monotonic_receipt"):
            raise ValueError(f"unsupported clock domain: {clock_domain}")
        with self._lock:
            source = (
                self._header_times
                if clock_domain == "ros_header"
                else self._receipt_times
            )
            candidates = np.flatnonzero(self._valid & np.isfinite(source))
            if candidates.size == 0:
                return None
            errors = np.abs(source[candidates] - timestamp)
            best_offset = int(np.argmin(errors))
            slot = int(candidates[best_offset])
            return OdomMatch(
                pose=self._poses[slot].copy(),
                error_s=float(errors[best_offset]),
            )


@dataclass(frozen=True)
class ClockResolution:
    clock_domain: str
    timestamp: float
    odom: np.ndarray
    odom_error_s: float
    domain_changed: bool


class SensorClockResolver:
    """Choose one clock for RGB, depth and odometry without ever mixing them."""

    def __init__(
        self,
        *,
        header_confirmations: int = 3,
        header_fallback_failures: int = 30,
        max_odom_age_s: float = DEFAULT_MAX_ODOM_AGE_S,
    ) -> None:
        if int(header_confirmations) <= 0:
            raise ValueError("header confirmations must be positive")
        if int(header_fallback_failures) <= 0:
            raise ValueError("header fallback failures must be positive")
        self.header_confirmations = int(header_confirmations)
        self.header_fallback_failures = int(header_fallback_failures)
        self.max_odom_age_s = float(max_odom_age_s)
        self.clock_domain = "pending"
        self._header_successes = 0
        self._header_failures = 0
        self._last_header_timestamp = -math.inf

    def resolve(
        self,
        *,
        camera_header_timestamp: float,
        receipt_timestamp: float,
        odom_history: BoundedOdomHistory,
    ) -> Optional[ClockResolution]:
        header = float(camera_header_timestamp)
        receipt = float(receipt_timestamp)
        header_valid = (
            math.isfinite(header)
            and header > 0.0
            and header > self._last_header_timestamp
        )
        header_match = (
            odom_history.nearest(header, "ros_header") if header_valid else None
        )
        header_usable = (
            header_match is not None
            and header_match.error_s <= self.max_odom_age_s
        )

        if self.clock_domain == "pending":
            if header_usable:
                self._last_header_timestamp = header
                self._header_successes += 1
                self._header_failures = 0
                if self._header_successes < self.header_confirmations:
                    return None
                self.clock_domain = "ros_header"
                return ClockResolution(
                    "ros_header",
                    header,
                    header_match.pose,
                    header_match.error_s,
                    True,
                )
            self._header_successes = 0
            self._header_failures += 1
            if self._header_failures < self.header_fallback_failures:
                return None
            self.clock_domain = "monotonic_receipt"
            receipt_match = odom_history.nearest(receipt, self.clock_domain)
            if receipt_match is None or receipt_match.error_s > self.max_odom_age_s:
                return None
            return ClockResolution(
                self.clock_domain,
                receipt,
                receipt_match.pose,
                receipt_match.error_s,
                True,
            )

        if self.clock_domain == "ros_header":
            if header_usable:
                self._last_header_timestamp = header
                return ClockResolution(
                    "ros_header",
                    header,
                    header_match.pose,
                    header_match.error_s,
                    False,
                )
            self.clock_domain = "monotonic_receipt"
            receipt_match = odom_history.nearest(receipt, self.clock_domain)
            if receipt_match is None or receipt_match.error_s > self.max_odom_age_s:
                return None
            return ClockResolution(
                self.clock_domain,
                receipt,
                receipt_match.pose,
                receipt_match.error_s,
                True,
            )

        receipt_match = odom_history.nearest(receipt, self.clock_domain)
        if receipt_match is None or receipt_match.error_s > self.max_odom_age_s:
            return None
        return ClockResolution(
            self.clock_domain,
            receipt,
            receipt_match.pose,
            receipt_match.error_s,
            False,
        )


@dataclass(frozen=True)
class RealSensorSnapshot:
    episode_id: int
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

    @property
    def anchor_frame_id(self) -> int:
        return int(self.depth_frame_ids[-1])

    @property
    def anchor_timestamp(self) -> float:
        return float(self.depth_timestamps[-1])

    @property
    def anchor_odom(self) -> np.ndarray:
        return self.odom_history[-1].copy()

    @property
    def degraded_temporal_input(self) -> bool:
        return bool(
            self.source_fps_hz < 27.0
            or self.rgb_duplicate_ratio > 0.0
            or self.depth_duplicate_ratio > 0.0
        )

    def as_npz_fields(
        self,
        *,
        point_goal: Optional[np.ndarray] = None,
        image_goal: Optional[np.ndarray] = None,
        camera_intrinsic: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        fields = {
            "env_ids": np.asarray([0], dtype=np.int64),
            "episode_ids": np.asarray([self.episode_id], dtype=np.int64),
            "rgb_history": self.rgb_history[np.newaxis],
            "depth_history_u16": self.depth_history_u16[np.newaxis],
            "odom_history": self.odom_history[np.newaxis],
            "rgb_frame_ids": self.rgb_frame_ids[np.newaxis],
            "depth_frame_ids": self.depth_frame_ids[np.newaxis],
            "rgb_timestamps": self.rgb_timestamps[np.newaxis],
            "depth_timestamps": self.depth_timestamps[np.newaxis],
            "rgb_target_timestamps": self.rgb_target_timestamps[np.newaxis],
            "depth_target_timestamps": self.depth_target_timestamps[np.newaxis],
            "clock_domain": np.asarray([self.clock_domain]),
            "source_fps_hz": np.asarray([self.source_fps_hz], dtype=np.float64),
            "max_selection_error_s": np.asarray(
                [self.max_selection_error_s], dtype=np.float64
            ),
            "rgb_duplicate_ratio": np.asarray(
                [self.rgb_duplicate_ratio], dtype=np.float64
            ),
            "depth_duplicate_ratio": np.asarray(
                [self.depth_duplicate_ratio], dtype=np.float64
            ),
        }
        if point_goal is not None:
            point_goal = np.asarray(point_goal, dtype=np.float32).reshape(1, 2)
            fields["point_goals"] = point_goal
        if image_goal is not None:
            image_goal = np.asarray(image_goal, dtype=np.uint8)
            if image_goal.ndim != 3 or image_goal.shape[-1] != 3:
                raise ValueError("image goal must be uint8 [H,W,3]")
            fields["image_goals"] = image_goal[np.newaxis]
        if camera_intrinsic is not None:
            camera_intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
            if camera_intrinsic.shape != (3, 3) or not np.all(
                np.isfinite(camera_intrinsic)
            ):
                raise ValueError("camera intrinsic must be finite [3,3]")
            fields["camera_intrinsic"] = camera_intrinsic
        return fields

    def to_npz_bytes(
        self,
        *,
        point_goal: Optional[np.ndarray] = None,
        image_goal: Optional[np.ndarray] = None,
        camera_intrinsic: Optional[np.ndarray] = None,
    ) -> io.BytesIO:
        stream = io.BytesIO()
        np.savez(
            stream,
            **self.as_npz_fields(
                point_goal=point_goal,
                image_goal=image_goal,
                camera_intrinsic=camera_intrinsic,
            ),
        )
        stream.seek(0)
        return stream


class RealSensorRingBuffer:
    """Fixed-capacity synchronized RGB/depth/odom history for one robot."""

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        max_frame_selection_error_s: float = DEFAULT_MAX_FRAME_SELECTION_ERROR_S,
    ) -> None:
        if int(capacity) < 29:
            raise ValueError("sensor capacity must be at least 29")
        self.capacity = int(capacity)
        self.max_frame_selection_error_s = float(max_frame_selection_error_s)
        self._lock = threading.RLock()
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._odom = np.empty((self.capacity, 3), dtype=np.float64)
        self._frame_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._timestamps = np.full(self.capacity, np.nan, dtype=np.float64)
        self._valid = np.zeros(self.capacity, dtype=bool)
        self._write_index = 0
        self._size = 0
        self._next_frame_id = 0
        self._episode_id = 0
        self._clock_domain: Optional[str] = None
        self._last_odom: Optional[np.ndarray] = None
        self.last_reset_reason = "initial"

    @property
    def episode_id(self) -> int:
        with self._lock:
            return int(self._episode_id)

    @property
    def memory_bytes(self) -> int:
        with self._lock:
            arrays = (
                self._rgb,
                self._depth,
                self._odom,
                self._frame_ids,
                self._timestamps,
                self._valid,
            )
            return int(sum(value.nbytes for value in arrays if value is not None))

    def _invalidate(self, reason: str, *, release_images: bool = False) -> None:
        self._valid.fill(False)
        self._write_index = 0
        self._size = 0
        self._episode_id += 1
        self._clock_domain = None
        self._last_odom = None
        self.last_reset_reason = str(reason)
        if release_images:
            self._rgb = None
            self._depth = None

    def reset(self, reason: str = "explicit_reset") -> int:
        with self._lock:
            self._invalidate(reason)
            return int(self._episode_id)

    def _allocate(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        if depth.shape != (height, width):
            raise ValueError("RGB/depth shape mismatch")
        self._rgb = np.empty(
            (self.capacity, height, width, 3), dtype=np.uint8
        )
        self._depth = np.empty(
            (self.capacity, height, width), dtype=np.uint16
        )

    def append(
        self,
        *,
        timestamp: float,
        clock_domain: str,
        rgb: np.ndarray,
        depth_u16: np.ndarray,
        odom: np.ndarray,
    ) -> int:
        timestamp = float(timestamp)
        if clock_domain not in ("ros_header", "monotonic_receipt"):
            raise ValueError("invalid clock domain")
        rgb = np.asarray(rgb)
        depth_u16 = np.asarray(depth_u16)
        odom = np.asarray(odom, dtype=np.float64)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("RGB must be uint8 [H,W,3]")
        if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
            raise ValueError("depth must be uint16 [H,W]")
        if odom.shape != (3,) or not np.all(np.isfinite(odom)):
            raise ValueError("odom must be finite [3]")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        with self._lock:
            if self._rgb is None:
                self._allocate(rgb, depth_u16)
            elif (
                rgb.shape != self._rgb.shape[1:]
                or depth_u16.shape != self._depth.shape[1:]
            ):
                self._invalidate("resolution_change", release_images=True)
                self._allocate(rgb, depth_u16)

            if self._clock_domain is not None and self._clock_domain != clock_domain:
                self._invalidate("clock_domain_change")
            self._clock_domain = clock_domain

            if self._size:
                previous_slot = (self._write_index - 1) % self.capacity
                if timestamp <= float(self._timestamps[previous_slot]):
                    self._invalidate("timestamp_not_increasing")
                    self._clock_domain = clock_domain

            if self._last_odom is not None:
                translation = float(np.linalg.norm(odom[:2] - self._last_odom[:2]))
                yaw_delta = abs(_wrap_angle(odom[2] - self._last_odom[2]))
                if (
                    translation > ODOM_JUMP_TRANSLATION_M
                    or yaw_delta > ODOM_JUMP_YAW_RAD
                ):
                    self._invalidate("odom_jump")
                    self._clock_domain = clock_domain

            slot = self._write_index
            assert self._rgb is not None and self._depth is not None
            self._rgb[slot] = rgb
            self._depth[slot] = depth_u16
            self._odom[slot] = odom
            self._frame_ids[slot] = self._next_frame_id
            self._timestamps[slot] = timestamp
            self._valid[slot] = True
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            self._last_odom = odom.copy()
            self._write_index = (slot + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            return int(frame_id)

    def _ordered_slots(self) -> np.ndarray:
        if self._size == 0:
            return np.empty(0, dtype=np.int64)
        start = (self._write_index - self._size) % self.capacity
        return (start + np.arange(self._size, dtype=np.int64)) % self.capacity

    @staticmethod
    def _nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.asarray(
            [int(np.argmin(np.abs(times - target))) for target in targets],
            dtype=np.int64,
        )

    def snapshot(self) -> Optional[RealSensorSnapshot]:
        with self._lock:
            if self._size < 2 or self._rgb is None or self._depth is None:
                return None
            slots = self._ordered_slots()
            times = self._timestamps[slots]
            anchor = float(times[-1])
            rgb_targets = anchor - RGB_TARGET_OFFSETS
            depth_targets = anchor - DEPTH_TARGET_OFFSETS
            if float(times[0]) > float(rgb_targets[0]) + self.max_frame_selection_error_s:
                return None
            rgb_offsets = self._nearest_indices(times, rgb_targets)
            depth_offsets = self._nearest_indices(times, depth_targets)
            rgb_errors = np.abs(times[rgb_offsets] - rgb_targets)
            depth_errors = np.abs(times[depth_offsets] - depth_targets)
            maximum_error = float(max(rgb_errors.max(), depth_errors.max()))
            if maximum_error > self.max_frame_selection_error_s:
                return None
            rgb_slots = slots[rgb_offsets]
            depth_slots = slots[depth_offsets]
            frame_ids = self._frame_ids[slots]
            intervals = np.diff(times)
            positive_intervals = intervals[intervals > 0.0]
            source_fps = (
                0.0
                if positive_intervals.size == 0
                else float(1.0 / np.median(positive_intervals))
            )
            rgb_ids = self._frame_ids[rgb_slots].copy()
            depth_ids = self._frame_ids[depth_slots].copy()
            return RealSensorSnapshot(
                episode_id=int(self._episode_id),
                rgb_history=self._rgb[rgb_slots].copy(),
                depth_history_u16=self._depth[depth_slots].copy(),
                odom_history=self._odom[depth_slots].copy(),
                rgb_frame_ids=rgb_ids,
                depth_frame_ids=depth_ids,
                rgb_timestamps=self._timestamps[rgb_slots].copy(),
                depth_timestamps=self._timestamps[depth_slots].copy(),
                rgb_target_timestamps=rgb_targets.copy(),
                depth_target_timestamps=depth_targets.copy(),
                clock_domain=str(self._clock_domain),
                source_fps_hz=source_fps,
                max_selection_error_s=maximum_error,
                rgb_duplicate_ratio=float(np.mean(np.diff(rgb_ids) == 0)),
                depth_duplicate_ratio=float(np.mean(np.diff(depth_ids) == 0)),
            )
