"""Bandwidth-bounded stateful V4 frame transport primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .temporal_history import RealSensorRingBuffer, RealSensorSnapshot


STATEFUL_TRANSPORT_NAME = "stateful_delta_v1"
STATEFUL_TARGET_SIZE = 224
STATEFUL_JPEG_QUALITY = 80
STATEFUL_DEPTH_QUANTIZATION_M = 0.005
HTTP_DEPTH_UNITS_PER_METER = 10_000.0
STATEFUL_DEPTH_QUANTIZATION_HTTP_UNITS = int(
    round(STATEFUL_DEPTH_QUANTIZATION_M * HTTP_DEPTH_UNITS_PER_METER)
)
STATEFUL_FRAME_BATCH_ENDPOINT = "/v4/stateful/frames"
STATEFUL_MAX_FRAME_BATCH_SIZE = 4
DEFAULT_STATEFUL_SNAPSHOT_TTL_S = 0.75
DEFAULT_STATEFUL_SNAPSHOT_CACHE_SIZE = 32


@dataclass(frozen=True)
class EncodedStatefulFrame:
    """One geometry-preserving, compressed RGB-D/odometry transport frame."""

    rgb_jpeg: bytes
    depth_png: bytes
    camera_intrinsic: np.ndarray
    width: int
    height: int

    @property
    def payload_bytes(self) -> int:
        return len(self.rgb_jpeg) + len(self.depth_png)


@dataclass(frozen=True)
class StatefulSnapshotRecord:
    """One immutable, versioned server snapshot selected for inference."""

    version: int
    snapshot: RealSensorSnapshot
    source_anchor_frame_id: int
    created_monotonic: float


def _letterbox_geometry(
    width: int,
    height: int,
    target_size: int = STATEFUL_TARGET_SIZE,
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0 or target_size <= 0:
        raise ValueError("frame and target dimensions must be positive")
    scale = target_size / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    pad_left = (target_size - resized_width) // 2
    pad_top = (target_size - resized_height) // 2
    return resized_width, resized_height, pad_left, pad_top


def letterbox_intrinsic(
    intrinsic: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_size: int = STATEFUL_TARGET_SIZE,
) -> np.ndarray:
    """Map a source pinhole K through the exact letterbox resize."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("camera intrinsic must be finite [3,3]")
    resized_width, resized_height, pad_left, pad_top = _letterbox_geometry(
        int(source_width), int(source_height), int(target_size)
    )
    scale_x = resized_width / float(source_width)
    scale_y = resized_height / float(source_height)
    mapped = intrinsic.copy()
    mapped[0, 0] *= scale_x
    mapped[1, 1] *= scale_y
    mapped[0, 2] = mapped[0, 2] * scale_x + pad_left
    mapped[1, 2] = mapped[1, 2] * scale_y + pad_top
    return mapped


def _letterbox_array(
    value: np.ndarray,
    *,
    target_size: int = STATEFUL_TARGET_SIZE,
) -> np.ndarray:
    height, width = value.shape[:2]
    resized_width, resized_height, pad_left, pad_top = _letterbox_geometry(
        width, height, target_size
    )
    resized = cv2.resize(
        value,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_right = target_size - resized_width - pad_left
    pad_bottom = target_size - resized_height - pad_top
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def encode_stateful_frame(
    rgb: np.ndarray,
    depth_u16: np.ndarray,
    intrinsic: np.ndarray,
) -> EncodedStatefulFrame:
    """Letterbox and compress one aligned RGB-D frame for transport."""
    rgb = np.asarray(rgb)
    depth_u16 = np.asarray(depth_u16)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB must be uint8 [H,W,3]")
    if depth_u16.dtype != np.uint16 or depth_u16.shape != rgb.shape[:2]:
        raise ValueError("depth must be aligned uint16 [H,W]")

    height, width = rgb.shape[:2]
    encoded_rgb = _letterbox_array(rgb)
    encoded_depth = _letterbox_array(depth_u16)
    quantized_depth = np.rint(
        encoded_depth.astype(np.float32)
        / float(STATEFUL_DEPTH_QUANTIZATION_HTTP_UNITS)
    ).astype(np.uint16)

    rgb_ok, rgb_buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(encoded_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, STATEFUL_JPEG_QUALITY],
    )
    depth_ok, depth_buffer = cv2.imencode(
        ".png",
        quantized_depth,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not rgb_ok or not depth_ok:
        raise ValueError("failed to encode stateful RGB-D frame")
    mapped_intrinsic = letterbox_intrinsic(
        intrinsic,
        source_width=width,
        source_height=height,
    )
    return EncodedStatefulFrame(
        rgb_jpeg=rgb_buffer.tobytes(),
        depth_png=depth_buffer.tobytes(),
        camera_intrinsic=mapped_intrinsic,
        width=STATEFUL_TARGET_SIZE,
        height=STATEFUL_TARGET_SIZE,
    )


def decode_stateful_frame(
    rgb_jpeg: bytes,
    depth_png: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one transport frame into RGB and 0.1 mm HTTP depth units."""
    rgb_bgr = cv2.imdecode(
        np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    quantized_depth = cv2.imdecode(
        np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if rgb_bgr is None or quantized_depth is None:
        raise ValueError("invalid stateful RGB or depth image")
    if rgb_bgr.shape != (STATEFUL_TARGET_SIZE, STATEFUL_TARGET_SIZE, 3):
        raise ValueError("stateful RGB must decode to 224x224x3")
    if (
        quantized_depth.shape != (STATEFUL_TARGET_SIZE, STATEFUL_TARGET_SIZE)
        or quantized_depth.dtype != np.uint16
    ):
        raise ValueError("stateful depth must decode to uint16 224x224")
    restored_depth = np.clip(
        quantized_depth.astype(np.uint32)
        * STATEFUL_DEPTH_QUANTIZATION_HTTP_UNITS,
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), restored_depth


class StatefulV4FrameSession:
    """Server-owned rolling history with immutable, expiring snapshots."""

    def __init__(
        self,
        session_id: str,
        camera_intrinsic: np.ndarray,
        *,
        capacity: int = 64,
        max_frame_selection_error_s: float = 0.045,
        snapshot_ttl_s: float = DEFAULT_STATEFUL_SNAPSHOT_TTL_S,
        snapshot_cache_size: int = DEFAULT_STATEFUL_SNAPSHOT_CACHE_SIZE,
    ) -> None:
        self.session_id = str(session_id)
        self.camera_intrinsic = np.asarray(
            camera_intrinsic, dtype=np.float64
        ).copy()
        if self.camera_intrinsic.shape != (3, 3) or not np.all(
            np.isfinite(self.camera_intrinsic)
        ):
            raise ValueError("session camera intrinsic must be finite [3,3]")
        self.buffer = RealSensorRingBuffer(
            capacity=capacity,
            max_frame_selection_error_s=max_frame_selection_error_s,
        )
        self.snapshot_ttl_s = float(snapshot_ttl_s)
        if self.snapshot_ttl_s <= 0.0:
            raise ValueError("stateful snapshot TTL must be positive")
        if int(snapshot_cache_size) <= 0:
            raise ValueError("stateful snapshot cache size must be positive")
        self._lock = threading.RLock()
        self.source_episode_id: Optional[int] = None
        self.last_source_frame_id = -1
        self._source_ids: dict[int, int] = {}
        self.accepted_frames = 0
        self.accepted_payload_bytes = 0
        self._snapshot_records: deque[StatefulSnapshotRecord] = deque(
            maxlen=int(snapshot_cache_size)
        )
        self._next_snapshot_version = 1
        self._degraded_snapshot_rejections = 0

    def _clear_snapshot_cache_locked(self) -> None:
        self._snapshot_records.clear()

    def _cache_current_snapshot_locked(self) -> None:
        snapshot = self.buffer.snapshot()
        if snapshot is None:
            return
        # Stateful inference is only allowed to observe a complete, unique
        # temporal sample.  A transient jittered frame must not replace the
        # last nominal snapshot; source-lag validation on inference bounds how
        # long that retained snapshot remains usable.
        if snapshot.degraded_temporal_input:
            self._degraded_snapshot_rejections += 1
            return
        source_anchor = self._source_ids.get(snapshot.anchor_frame_id)
        if source_anchor is None:
            raise RuntimeError("stateful source/internal frame mapping is missing")
        if (
            self._snapshot_records
            and self._snapshot_records[-1].source_anchor_frame_id == source_anchor
        ):
            return
        self._snapshot_records.append(
            StatefulSnapshotRecord(
                version=self._next_snapshot_version,
                snapshot=snapshot,
                source_anchor_frame_id=int(source_anchor),
                created_monotonic=time.monotonic(),
            )
        )
        self._next_snapshot_version += 1

    def _append_locked(
        self,
        *,
        source_episode_id: int,
        source_frame_id: int,
        timestamp: float,
        clock_domain: str,
        odom: np.ndarray,
        rgb: np.ndarray,
        depth_u16: np.ndarray,
        payload_bytes: int,
    ) -> bool:
        source_episode_id = int(source_episode_id)
        source_frame_id = int(source_frame_id)
        if source_episode_id < 0 or source_frame_id < 0:
            raise ValueError("source episode and frame IDs must be non-negative")
        if self.source_episode_id is None:
            self.source_episode_id = source_episode_id
        elif source_episode_id != self.source_episode_id:
            raise ValueError("source episode changed; reset the stateful session")
        if source_frame_id <= self.last_source_frame_id:
            return False
        previous_buffer_episode = self.buffer.episode_id
        internal_frame_id = self.buffer.append(
            timestamp=float(timestamp),
            clock_domain=str(clock_domain),
            rgb=rgb,
            depth_u16=depth_u16,
            odom=np.asarray(odom, dtype=np.float64),
        )
        if self.buffer.episode_id != previous_buffer_episode:
            self._clear_snapshot_cache_locked()
            self._source_ids.clear()
        self._source_ids[internal_frame_id] = source_frame_id
        minimum_kept = internal_frame_id - self.buffer.capacity - 4
        self._source_ids = {
            key: value
            for key, value in self._source_ids.items()
            if key >= minimum_kept
        }
        self.last_source_frame_id = source_frame_id
        self.accepted_frames += 1
        self.accepted_payload_bytes += int(payload_bytes)
        return True

    def append(
        self,
        *,
        source_episode_id: int,
        source_frame_id: int,
        timestamp: float,
        clock_domain: str,
        odom: np.ndarray,
        rgb: np.ndarray,
        depth_u16: np.ndarray,
        payload_bytes: int,
    ) -> tuple[bool, Optional[RealSensorSnapshot], Optional[int]]:
        with self._lock:
            accepted = self._append_locked(
                source_episode_id=source_episode_id,
                source_frame_id=source_frame_id,
                timestamp=timestamp,
                clock_domain=clock_domain,
                odom=odom,
                rgb=rgb,
                depth_u16=depth_u16,
                payload_bytes=payload_bytes,
            )
            if accepted:
                self._cache_current_snapshot_locked()
            snapshot, source_anchor = self.snapshot()
            return accepted, snapshot, source_anchor

    def append_batch(
        self,
        frames: list[dict[str, object]],
    ) -> tuple[int, Optional[RealSensorSnapshot], Optional[int]]:
        """Append one ordered micro-batch and publish every nominal snapshot.

        Frames still arrive over one HTTP request, but temporal validity depends
        on the exact anchor timestamp.  Inspecting only the final frame can miss
        a valid anchor earlier in the same 3--4 frame batch and unnecessarily
        retain an old snapshot.
        """

        if not frames:
            raise ValueError("stateful frame batch must not be empty")
        if len(frames) > STATEFUL_MAX_FRAME_BATCH_SIZE:
            raise ValueError(
                "stateful frame batch exceeds maximum size "
                f"{STATEFUL_MAX_FRAME_BATCH_SIZE}"
            )
        with self._lock:
            frame_ids = [int(frame["source_frame_id"]) for frame in frames]
            if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
                raise ValueError("stateful batch frame IDs must be strictly increasing")
            accepted_count = 0
            for frame in frames:
                accepted = self._append_locked(
                    source_episode_id=int(frame["source_episode_id"]),
                    source_frame_id=int(frame["source_frame_id"]),
                    timestamp=float(frame["timestamp"]),
                    clock_domain=str(frame["clock_domain"]),
                    odom=np.asarray(frame["odom"], dtype=np.float64),
                    rgb=np.asarray(frame["rgb"]),
                    depth_u16=np.asarray(frame["depth_u16"]),
                    payload_bytes=int(frame["payload_bytes"]),
                )
                accepted_count += int(accepted)
                if accepted:
                    self._cache_current_snapshot_locked()
            snapshot, source_anchor = self.snapshot()
            return accepted_count, snapshot, source_anchor

    def snapshot_record(
        self,
        version: Optional[int] = None,
    ) -> Optional[StatefulSnapshotRecord]:
        with self._lock:
            now = time.monotonic()
            while (
                self._snapshot_records
                and now - self._snapshot_records[0].created_monotonic
                > self.snapshot_ttl_s
            ):
                self._snapshot_records.popleft()
            if not self._snapshot_records:
                return None
            if version is None:
                return self._snapshot_records[-1]
            requested = int(version)
            return next(
                (
                    record
                    for record in self._snapshot_records
                    if record.version == requested
                ),
                None,
            )

    def snapshot(
        self,
        version: Optional[int] = None,
    ) -> tuple[Optional[RealSensorSnapshot], Optional[int]]:
        record = self.snapshot_record(version)
        if record is None:
            return None, None
        return record.snapshot, record.source_anchor_frame_id

    def status(self) -> dict[str, object]:
        record = self.snapshot_record()
        now = time.monotonic()
        return {
            "session_id": self.session_id,
            "source_episode_id": self.source_episode_id,
            "last_source_frame_id": self.last_source_frame_id,
            "accepted_frames": self.accepted_frames,
            "accepted_payload_bytes": self.accepted_payload_bytes,
            "ready": record is not None,
            "source_anchor_frame_id": (
                None if record is None else record.source_anchor_frame_id
            ),
            "snapshot_version": None if record is None else record.version,
            "snapshot_age_s": (
                None
                if record is None
                else max(0.0, now - record.created_monotonic)
            ),
            "snapshot_ttl_s": self.snapshot_ttl_s,
            "degraded_snapshot_rejections": self._degraded_snapshot_rejections,
        }
