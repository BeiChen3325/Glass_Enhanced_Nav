#!/usr/bin/env python3
"""ROS 2 NavDP HTTP client that publishes velocity commands for the Go2 bridge."""

from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image as PIL_Image


def configure_unitree_dds_defaults() -> dict[str, str]:
    """Use the deployed Go2 CycloneDDS profile when the shell omitted it."""
    applied: dict[str, str] = {}
    home = Path.home()
    cyclone_config = home / "cyclonedds_ws" / "cyclonedds.xml"
    cyclone_home = home / "cyclonedds" / "install"
    defaults = {
        "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
        "CYCLONEDDS_URI": str(cyclone_config),
        "CYCLONEDDS_HOME": str(cyclone_home),
    }
    if not cyclone_config.is_file():
        return applied
    for key, value in defaults.items():
        if key == "CYCLONEDDS_HOME" and not cyclone_home.is_dir():
            continue
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


# The RMW implementation is selected when ROS is initialized/imported.  Apply
# only deployment-local defaults and never override an explicit operator value.
UNITREE_DDS_DEFAULTS_APPLIED = configure_unitree_dds_defaults()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.camera.camera_config import load_realsense_config  # noqa: E402
from go2_internnav.config.workspace_paths import (  # noqa: E402
    load_workspace_paths,
    require_path,
)
from go2_internnav.golden_run import GoldenRunResponseRecorder  # noqa: E402
from go2_internnav.ros2.navigation_status import encode_navigation_status  # noqa: E402
from go2_internnav.navdp import (  # noqa: E402
    BoundedOdomHistory,
    DEFAULT_TRAJECTORY_SMOOTHING_CONFIG,
    EncodedStatefulFrame,
    RealSensorRingBuffer,
    RealSensorSnapshot,
    STATEFUL_TRANSPORT_NAME,
    SUPPORTED_GOAL_MODES,
    SUPPORTED_MODEL_MODES,
    SensorClockResolver,
    encode_stateful_frame,
    local_pointgoal_to_world,
    normalize_goal_mode,
    normalize_model_mode,
    pointgoal_distance_to_odom,
    requires_alignment_depth,
    smooth_trajectory_xy,
    ros_stamp_seconds,
    world_pointgoal_to_local,
    validate_model_goal_mode,
)
from go2_internnav.ros2.topic_names import (  # noqa: E402
    CMD_VEL_TOPIC,
    DEPTH_TOPIC,
    NAVIGATION_STATUS_TOPIC,
    ODOM_TOPIC,
    RGB_CAMERA_INFO_TOPIC,
    RGB_TOPIC,
)
from go2_internnav.safety.depth_stop import (  # noqa: E402
    DEFAULT_DEPTH_RECOVERY_SCAN_REVERSE_INTERVAL_S,
    DEFAULT_DEPTH_RECOVERY_SCAN_YAW_RATE_RADPS,
    DEFAULT_SAFE_CURVE_ESCAPE_MAX_SPEED_MPS,
    DEFAULT_DEPTH_SAFETY_CONFIG,
    DepthBlockLatchState,
    DepthSafetyStatus,
    assess_depth_safety_stop,
    choose_depth_recovery_scan_direction,
    limit_forward_velocity_for_safe_curve,
    should_run_depth_recovery_scan,
    update_depth_block_latch,
)

frame_data: dict[int, dict[str, Any]] = {}
WORKSPACE_PATHS = load_workspace_paths()

ROS_RUNTIME_IMPORT_ERRORS: list[str] = []
CONTROLLER_RUNTIME_IMPORT_ERRORS: list[str] = []

try:
    import rclpy  # noqa: E402
    from geometry_msgs.msg import PointStamped, Twist  # noqa: E402
    from nav_msgs.msg import Odometry  # noqa: E402
    from sensor_msgs.msg import CameraInfo, Image, Joy  # noqa: E402
    from std_msgs.msg import String  # noqa: E402
    from cv_bridge import CvBridge  # noqa: E402
    from message_filters import ApproximateTimeSynchronizer, Subscriber  # noqa: E402
    from rclpy.node import Node  # noqa: E402
    from rclpy.qos import (  # noqa: E402
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
except Exception as exc:  # pragma: no cover - exercised by --help without ROS sourced.
    ROS_RUNTIME_IMPORT_ERRORS.append(
        f"ROS 2 runtime imports failed: {type(exc).__name__}: {exc}"
    )
    rclpy = None
    PointStamped = None
    Twist = None
    Odometry = None
    Image = None
    CameraInfo = None
    Joy = None
    String = None
    CvBridge = None
    ApproximateTimeSynchronizer = None
    Subscriber = None
    Node = object
    QoSDurabilityPolicy = None
    QoSHistoryPolicy = None
    QoSProfile = None
    QoSReliabilityPolicy = None

try:
    from go2_internnav.control import Mpc_controller, PID_controller  # noqa: E402
except Exception as exc:  # pragma: no cover - exercised by --help in light envs.
    CONTROLLER_RUNTIME_IMPORT_ERRORS.append(
        f"deployment controller imports failed: {type(exc).__name__}: {exc}"
    )
    Mpc_controller = None
    PID_controller = None

try:
    from thread_utils import ReadWriteLock  # noqa: E402
except Exception:  # pragma: no cover - only used to keep --help import-light.
    class ReadWriteLock:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._lock = threading.RLock()

        def acquire_read(self) -> None:
            self._lock.acquire()

        def release_read(self) -> None:
            self._lock.release()

        def acquire_write(self) -> None:
            self._lock.acquire()

        def release_write(self) -> None:
            self._lock.release()


DEFAULT_SERVER_HOST = os.environ.get("NAVDP_SERVER_HOST", WORKSPACE_PATHS.server_host)
DEFAULT_SERVER_PORT = int(os.environ.get("NAVDP_SERVER_PORT", WORKSPACE_PATHS.server_port))
DEFAULT_SERVER_ENDPOINT = "/eval_navdp"
DEFAULT_SERVER_URL = f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}{DEFAULT_SERVER_ENDPOINT}"
DEFAULT_CAMERA_CONFIG = WORKSPACE_PATHS.realsense_config
HTTP_DEPTH_UNITS_PER_METER = 10000.0
NAVDP_GOAL_POINT_TOPIC = "/navdp/goal_point"
NAVDP_GOAL_IMAGE_TOPIC = "/navdp/goal_image"
NAVDP_GOAL_MODE_TOPIC = "/navdp/goal_mode"
DEFAULT_MAX_LINEAR_SPEED_MPS = 0.30
DEFAULT_MAX_ANGULAR_SPEED_RADPS = 0.50
MAX_LINEAR_ACCEL_MPS2 = 0.8
MAX_ANGULAR_ACCEL_RADPS2 = 1.2
MPC_TRACKING_PROFILES = ("legacy", "smooth_continuous")
SMOOTH_CONTINUOUS_PROFILE = "smooth_continuous"
DEFAULT_MPC_TRACKING_PROFILE = SMOOTH_CONTINUOUS_PROFILE
SMOOTH_MPC_ANGULAR_DELTA_WEIGHT = 1.0
SMOOTH_MPC_MAX_ANGULAR_ACCEL_RADPS2 = 0.6
DEFAULT_STATEFUL_SNAPSHOT_WAIT_TIMEOUT_S = 0.20
DEFAULT_MAX_RETAINED_TRAJECTORY_AGE_S = 0.60
SUPPORTED_MANUAL_OVERRIDE_MSG_TYPES = (
    "sensor_msgs/msg/Joy",
    "unitree_go/msg/WirelessController",
)
SPECIAL_HOLD_ACTIONS = {(5,), (9,)}
V4_ENDPOINTS = {
    "pointgoal": "/v4/pointgoal_step",
    "imagegoal": "/v4/imagegoal_step",
    "nogoal": "/v4/nogoal_step",
}


class ControlMode(Enum):
    Stop_Mode = 0
    PID_Mode = 1
    MPC_Mode = 2


policy_init = True
mpc = None
pid = None
http_idx = -1
first_running_time = 0.0
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None
navigation_stopped = False
shutdown_event = threading.Event()

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()


def ensure_runtime_dependencies(*, require_controllers: bool = True) -> None:
    global pid

    errors = list(ROS_RUNTIME_IMPORT_ERRORS)
    if require_controllers:
        errors.extend(CONTROLLER_RUNTIME_IMPORT_ERRORS)
    if errors:
        joined_errors = "\n  - ".join(errors)
        raise SystemExit(
            "[FAIL] missing runtime dependencies. Source the ROS 2 and deployment "
            f"environments before running the client:\n  - {joined_errors}"
        )
    if require_controllers and pid is None:
        pid = PID_controller(
            Kp_trans=2.0,
            Kd_trans=0.0,
            Kp_yaw=1.5,
            Kd_yaw=0.0,
            max_v=0.6,
            max_w=DEFAULT_MAX_ANGULAR_SPEED_RADPS,
        )


def auto_visualize_recorded_responses(
    response_path: Path,
    *,
    disabled: bool = False,
) -> bool:
    """Render a closed response JSONL after sensor/control cleanup is complete."""

    resolved_path = Path(response_path).expanduser().resolve()
    if disabled:
        print(
            f"[AUTO VISUALIZATION] disabled for {resolved_path}",
            flush=True,
        )
        return False
    if not resolved_path.is_file() or resolved_path.stat().st_size == 0:
        print(
            f"[AUTO VISUALIZATION] skipped: response log is missing or empty: "
            f"{resolved_path}",
            flush=True,
        )
        return False

    visualizer = PROJECT_ROOT / "scripts" / "visualize_navdp_response_sequence.py"
    if not visualizer.is_file():
        print(
            f"[AUTO VISUALIZATION ERROR] visualizer not found: {visualizer}",
            flush=True,
        )
        return False

    command = [
        sys.executable,
        str(visualizer),
        "--responses",
        str(resolved_path),
    ]
    print(
        "[AUTO VISUALIZATION] starting after client shutdown: "
        f"responses={resolved_path}",
        flush=True,
    )
    try:
        completed = subprocess.run(command, check=False, timeout=1800.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(
            f"[AUTO VISUALIZATION ERROR] {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    if completed.returncode != 0:
        print(
            f"[AUTO VISUALIZATION ERROR] renderer exited with "
            f"status={completed.returncode}",
            flush=True,
        )
        return False
    print("[AUTO VISUALIZATION] completed", flush=True)
    return True


def rewind_bytes(value: io.BytesIO) -> io.BytesIO:
    value.seek(0)
    return value


def navdp_eval(
    image_bytes: io.BytesIO,
    depth_bytes: io.BytesIO,
    alignment_depth_bytes: io.BytesIO | None,
    goal_request: dict[str, Any],
    *,
    inference_odom: list[float],
    model_mode: str,
    url: str = DEFAULT_SERVER_URL,
    recorder: GoldenRunResponseRecorder | None = None,
    record_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global policy_init, http_idx, first_running_time
    request_reset = policy_init
    request_idx = http_idx
    goal_mode = goal_request["goal_mode"]
    goal_payload = goal_request.get("goal")
    data: dict[str, Any] = {
        "reset": request_reset,
        "idx": request_idx,
        "goal_mode": goal_mode,
        "model_mode": model_mode,
        "inference_odom": [float(value) for value in inference_odom[:3]],
    }
    if goal_payload is not None:
        data["goal"] = goal_payload
    if goal_request.get("seed") is not None:
        data["seed"] = int(goal_request["seed"])

    files: dict[str, tuple[str, io.BytesIO, str]] = {
        "image": ("rgb_image.jpg", rewind_bytes(image_bytes), "image/jpeg"),
        "depth": ("depth_image.png", rewind_bytes(depth_bytes), "image/png"),
    }
    goal_image_bytes = goal_request.get("goal_image_bytes")
    if goal_image_bytes is not None:
        files["goal_image"] = (
            "goal_image.jpg",
            rewind_bytes(goal_image_bytes),
            "image/jpeg",
        )
    if requires_alignment_depth(model_mode):
        if alignment_depth_bytes is None:
            raise ValueError(
                "x-navdp-glass requires the raw float alignment-depth transport"
            )
        files["alignment_depth"] = (
            "alignment_depth.npy",
            rewind_bytes(alignment_depth_bytes),
            "application/octet-stream",
        )

    start = time.time()
    response = requests.post(
        url,
        files=files,
        data={"json": json.dumps(data)},
        timeout=100,
    )
    elapsed_s = time.time() - start
    print(
        f"response status={response.status_code} bytes={len(response.content)}",
        flush=True,
    )
    response.raise_for_status()
    parsed_response = response.json()
    response_model_mode = normalize_model_mode(parsed_response.get("model_mode", ""))
    if response_model_mode != model_mode:
        raise ValueError(
            f"server model mismatch: requested {model_mode}, received {response_model_mode}"
        )
    if requires_alignment_depth(model_mode):
        diagnostics = parsed_response.get("ransac") or []
        success_count = sum(bool(item.get("success")) for item in diagnostics)
        print(
            f"[GLASSRECON] aligned={success_count}/{len(diagnostics)} "
            f"fallback={len(diagnostics) - success_count}",
            flush=True,
        )
    # Consume the reset only after a complete, mode-matched response.  A
    # network/model failure must retry with reset=true rather than silently
    # advancing partially initialized agent state.
    policy_init = False
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {elapsed_s:.3f}s", flush=True)
    if recorder is not None:
        recorder.write(
            reset=request_reset,
            idx=request_idx,
            response=parsed_response,
            elapsed_s=elapsed_s,
            server_url=url,
            goal_mode=goal_mode,
            goal=goal_payload,
            metadata=record_metadata,
        )
    return parsed_response


def server_origin(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"server URL must be absolute: {url}")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def valid_v4_checkpoint_load_report(report: dict[str, Any]) -> bool:
    """Accept complete V4 or the exact pre-BEV 1048-tensor legacy contract."""
    unexpected = report.get("unexpected_keys") or []
    missing = report.get("missing_keys") or []
    loaded = report.get("loaded_key_count")
    if unexpected:
        return False
    if loaded == 1131 and not missing:
        return True
    return (
        loaded == 1048
        and len(missing) == 83
        and all(str(key).startswith("bev_encoder.") for key in missing)
    )


@dataclass
class _PendingStatefulFrame:
    item: dict[str, Any]
    future: Future[tuple[EncodedStatefulFrame, float]]
    enqueued_monotonic: float
    generation: int


@dataclass(frozen=True)
class StatefulInferenceTicket:
    """One immutable client/server snapshot pair leased for inference."""

    session_id: str
    snapshot_version: int
    source_anchor_frame_id: int
    client_frame_watermark: int
    source_lag_frames: int
    wait_s: float


class StatefulSnapshotUnavailable(ValueError):
    """A fresh stateful snapshot was not available before the short deadline."""


class StatefulInferenceRetryableError(ValueError):
    """The server rejected a snapshot transiently after bounded retry."""


@dataclass
class _PreparedRgbDepthFrame:
    receipt_timestamp: float
    rgb_image: np.ndarray
    depth_image: np.ndarray
    depth_u16: np.ndarray
    image_bytes: io.BytesIO | None
    depth_bytes: io.BytesIO | None
    alignment_depth_bytes: io.BytesIO | None
    rgb_time: float
    depth_time: float
    rgb_header: float
    depth_header: float


class StatefulV4ClientTransport:
    """Parallel-encode and micro-batch stateful RGB-D frames in source order."""

    def __init__(
        self,
        server_url: str,
        *,
        encode_workers: int = 2,
        upload_queue_size: int = 12,
        upload_batch_size: int = 4,
        batch_max_wait_s: float = 0.10,
    ) -> None:
        if int(encode_workers) <= 0:
            raise ValueError("stateful encode worker count must be positive")
        if int(upload_queue_size) <= 0:
            raise ValueError("stateful upload queue size must be positive")
        if int(upload_batch_size) <= 0:
            raise ValueError("stateful upload batch size must be positive")
        if float(batch_max_wait_s) <= 0.0:
            raise ValueError("stateful batch maximum wait must be positive")
        self.origin = server_origin(server_url)
        self.encode_workers = int(encode_workers)
        self.upload_queue_size = int(upload_queue_size)
        self.upload_batch_size = int(upload_batch_size)
        self.batch_max_wait_s = float(batch_max_wait_s)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pending: deque[_PendingStatefulFrame] = deque()
        self._generation = 0
        self._stop = threading.Event()
        self._paused = False
        self._pause_reason: str | None = None
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=self.encode_workers,
            thread_name_prefix="v4-stateful-encode",
        )
        self._session_id: str | None = None
        self._session_episode_id: int | None = None
        self._batch_endpoint: str | None = None
        self._negotiated = False
        self._uses_stateful = False
        self._server_ready = False
        self._max_source_lag_frames: int | None = None
        self._snapshot_version: int | None = None
        self._source_anchor_frame_id: int | None = None
        self._last_leased_snapshot_version: int | None = None
        self._last_inference_finished_monotonic: float | None = None
        self._last_inference_interval_s: float | None = None
        self._last_snapshot_wait_s = 0.0
        self._inference_retry_count = 0
        self._latest_enqueued_frame_id = -1
        self._enqueued_frames = 0
        self._uploaded_frames = 0
        self._uploaded_bytes = 0
        self._dropped_frames = 0
        self._http_409_count = 0
        self._upload_started_monotonic: float | None = None
        self._last_queue_age_s = 0.0
        self._last_encode_s = 0.0
        self._upload_session = requests.Session()
        self._upload_session.trust_env = False
        self._inference_session = requests.Session()
        self._inference_session.trust_env = False

    @property
    def negotiated(self) -> bool:
        with self._lock:
            return self._negotiated

    @property
    def uses_stateful(self) -> bool:
        with self._lock:
            return self._uses_stateful

    @property
    def server_ready(self) -> bool:
        with self._lock:
            return self._server_ready

    @property
    def server_snapshot_fresh(self) -> bool:
        """Whether a request can be sent without tripping server lag limits."""

        with self._lock:
            if not self._server_ready or self._source_anchor_frame_id is None:
                return False
            if self._max_source_lag_frames is None:
                return True
            return (
                self._latest_enqueued_frame_id - self._source_anchor_frame_id
                <= self._max_source_lag_frames
            )

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            duration = (
                0.0
                if self._upload_started_monotonic is None
                else max(0.0, time.monotonic() - self._upload_started_monotonic)
            )
            return {
                "enqueued_frames": self._enqueued_frames,
                "uploaded_frames": self._uploaded_frames,
                "uploaded_fps": (
                    0.0 if duration <= 0.0 else self._uploaded_frames / duration
                ),
                "dropped_oldest_frames": self._dropped_frames,
                "pending_frames": len(self._pending),
                "latest_enqueued_frame_id": self._latest_enqueued_frame_id,
                "source_anchor_frame_id": self._source_anchor_frame_id,
                "source_lag_frames": (
                    None
                    if self._source_anchor_frame_id is None
                    else max(
                        0,
                        self._latest_enqueued_frame_id
                        - self._source_anchor_frame_id,
                    )
                ),
                "snapshot_version": self._snapshot_version,
                "last_leased_snapshot_version": self._last_leased_snapshot_version,
                "http_409_count": self._http_409_count,
                "inference_retry_count": self._inference_retry_count,
                "last_snapshot_wait_s": self._last_snapshot_wait_s,
                "last_inference_interval_s": self._last_inference_interval_s,
                "last_queue_age_s": self._last_queue_age_s,
                "last_encode_s": self._last_encode_s,
                "paused": self._paused,
                "pause_reason": self._pause_reason,
            }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._upload_worker,
            name="v4-stateful-upload",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._condition:
            for pending in self._pending:
                pending.future.cancel()
            self._pending.clear()
        # Python 3.8 (used on the Go2) has no ``cancel_futures`` keyword.
        # Pending futures were cancelled explicitly above, so the compatible
        # shutdown form preserves the same bounded, non-blocking behaviour.
        self._executor.shutdown(wait=False)
        print(
            "[V4 STATEFUL SUMMARY] "
            + json.dumps(self.metrics(), sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        self._upload_session.close()
        self._inference_session.close()

    def invalidate(self, reason: str) -> None:
        with self._condition:
            self._session_id = None
            self._session_episode_id = None
            self._batch_endpoint = None
            self._negotiated = False
            self._uses_stateful = False
            self._server_ready = False
            self._max_source_lag_frames = None
            self._snapshot_version = None
            self._source_anchor_frame_id = None
            self._last_leased_snapshot_version = None
            self._generation += 1
            for pending in self._pending:
                pending.future.cancel()
            self._pending.clear()
            self._condition.notify_all()
        print(f"[V4 STATEFUL RESET] reason={reason}", flush=True)

    def pause(self, reason: str) -> None:
        """Fence all queued uploads after a terminal navigation stop."""

        with self._condition:
            if self._paused:
                return
            self._paused = True
            self._pause_reason = str(reason)
            self._session_id = None
            self._session_episode_id = None
            self._batch_endpoint = None
            self._negotiated = False
            self._uses_stateful = False
            self._server_ready = False
            self._max_source_lag_frames = None
            self._snapshot_version = None
            self._source_anchor_frame_id = None
            self._last_leased_snapshot_version = None
            self._generation += 1
            discarded = len(self._pending)
            for pending in self._pending:
                pending.future.cancel()
            self._pending.clear()
            self._condition.notify_all()
        print(
            f"[V4 STATEFUL PAUSE] reason={reason} discarded_pending={discarded}",
            flush=True,
        )

    def resume(self, reason: str) -> None:
        """Start a fresh upload session after an explicit new goal."""

        with self._condition:
            self._paused = False
            self._pause_reason = None
            self._session_id = None
            self._session_episode_id = None
            self._batch_endpoint = None
            self._negotiated = False
            self._uses_stateful = False
            self._server_ready = False
            self._max_source_lag_frames = None
            self._snapshot_version = None
            self._source_anchor_frame_id = None
            self._last_leased_snapshot_version = None
            self._generation += 1
            for pending in self._pending:
                pending.future.cancel()
            self._pending.clear()
            self._condition.notify_all()
        print(f"[V4 STATEFUL RESUME] reason={reason}", flush=True)

    @staticmethod
    def _encode_item(item: dict[str, Any]) -> tuple[EncodedStatefulFrame, float]:
        started = time.monotonic()
        encoded = encode_stateful_frame(
            item["rgb"],
            item["depth_u16"],
            item["intrinsic"],
        )
        return encoded, time.monotonic() - started

    def enqueue(
        self,
        rgb: np.ndarray,
        depth_u16: np.ndarray,
        intrinsic: np.ndarray,
        *,
        episode_id: int,
        frame_id: int,
        timestamp: float,
        clock_domain: str,
        odom: np.ndarray,
    ) -> None:
        with self._condition:
            if self._stop.is_set() or self._paused:
                return
            if self._negotiated and not self._uses_stateful:
                return
            generation = self._generation
        item = {
            "rgb": np.asarray(rgb, dtype=np.uint8).copy(),
            "depth_u16": np.asarray(depth_u16, dtype=np.uint16).copy(),
            "intrinsic": np.asarray(intrinsic, dtype=np.float64).copy(),
            "episode_id": int(episode_id),
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "clock_domain": str(clock_domain),
            "odom": np.asarray(odom, dtype=np.float64).tolist(),
        }
        enqueued = time.monotonic()
        future = self._executor.submit(self._encode_item, item)
        with self._condition:
            if self._paused or generation != self._generation:
                future.cancel()
                return
            if len(self._pending) >= self.upload_queue_size:
                dropped = self._pending.popleft()
                dropped.future.cancel()
                self._dropped_frames += 1
                if self._dropped_frames == 1 or self._dropped_frames % 30 == 0:
                    print(
                        "[V4 STATEFUL BACKPRESSURE] "
                        f"dropped_oldest_frames={self._dropped_frames}",
                        flush=True,
                    )
            self._pending.append(
                _PendingStatefulFrame(item, future, enqueued, generation)
            )
            if self._upload_started_monotonic is None:
                self._upload_started_monotonic = enqueued
            self._enqueued_frames += 1
            self._latest_enqueued_frame_id = max(
                self._latest_enqueued_frame_id, int(frame_id)
            )
            self._condition.notify_all()

    def _reset_session(
        self,
        item: dict[str, Any],
        encoded: EncodedStatefulFrame,
    ) -> bool:
        started = time.monotonic()
        response = self._upload_session.post(
            self.origin + "/v4/navigator_reset",
            json={
                "intrinsic": encoded.camera_intrinsic.tolist(),
                "batch_size": 1,
                "stop_threshold": -3.0,
                "transport": STATEFUL_TRANSPORT_NAME,
            },
            timeout=100,
        )
        response.raise_for_status()
        payload = response.json()
        report = payload.get("checkpoint_load_report", {})
        if not valid_v4_checkpoint_load_report(report):
            raise ValueError(f"server did not strictly load V4: {report}")
        negotiated = payload.get("transport")
        transport_config = payload.get("transport_config") or {}
        server_batch_size = int(transport_config.get("max_frame_batch_size", 1))
        max_source_lag = transport_config.get("max_source_lag_frames")
        with self._condition:
            self._negotiated = True
            self._uses_stateful = negotiated == STATEFUL_TRANSPORT_NAME
            self._server_ready = False
            self._max_source_lag_frames = (
                None if max_source_lag is None else int(max_source_lag)
            )
            self._snapshot_version = None
            self._source_anchor_frame_id = None
            self._last_leased_snapshot_version = None
            self._session_id = payload.get("session_id")
            self._session_episode_id = int(item["episode_id"])
            self._batch_endpoint = payload.get("frame_batch_endpoint")
            self.upload_batch_size = max(
                1, min(self.upload_batch_size, server_batch_size)
            )
        print(
            "[V4 STATEFUL NEGOTIATED] "
            f"transport={negotiated} batch_size={self.upload_batch_size} "
            f"batch_endpoint={self._batch_endpoint} "
            f"reset_s={time.monotonic() - started:.3f} "
            f"session={self._session_id}",
            flush=True,
        )
        return self._uses_stateful

    @staticmethod
    def _frame_metadata(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "episode_id": item["episode_id"],
            "frame_id": item["frame_id"],
            "timestamp": item["timestamp"],
            "clock_domain": item["clock_domain"],
            "odom": item["odom"],
        }

    def _post_encoded_batch(
        self,
        records: list[tuple[dict[str, Any], EncodedStatefulFrame]],
    ) -> tuple[dict[str, Any], float]:
        with self._lock:
            session_id = self._session_id
            batch_endpoint = self._batch_endpoint
        if not session_id:
            raise ValueError("stateful upload session is missing")
        started = time.monotonic()
        if batch_endpoint:
            metadata = {
                "session_id": session_id,
                "frames": [self._frame_metadata(item) for item, _ in records],
            }
            files: dict[str, tuple[str, bytes, str]] = {}
            for index, (_, encoded) in enumerate(records):
                files[f"rgb_{index}"] = (
                    f"rgb_{index}.jpg",
                    encoded.rgb_jpeg,
                    "image/jpeg",
                )
                files[f"depth_{index}"] = (
                    f"depth_{index}.png",
                    encoded.depth_png,
                    "image/png",
                )
            response = self._upload_session.post(
                self.origin + str(batch_endpoint),
                data={"metadata": json.dumps(metadata, separators=(",", ":"))},
                files=files,
                timeout=30,
            )
            response.raise_for_status()
            return response.json(), time.monotonic() - started

        payload: dict[str, Any] = {}
        for item, encoded in records:
            metadata = {"session_id": session_id, **self._frame_metadata(item)}
            response = self._upload_session.post(
                self.origin + "/v4/stateful/frame",
                data={"metadata": json.dumps(metadata, separators=(",", ":"))},
                files={
                    "rgb": ("rgb.jpg", encoded.rgb_jpeg, "image/jpeg"),
                    "depth": ("depth.png", encoded.depth_png, "image/png"),
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        return payload, time.monotonic() - started

    def _take_pending_batch(self) -> list[_PendingStatefulFrame]:
        with self._condition:
            while (
                (not self._pending or self._paused)
                and not self._stop.is_set()
                and not shutdown_event.is_set()
            ):
                self._condition.wait(timeout=0.1)
            if self._paused or not self._pending:
                return []
            oldest = self._pending[0]
            remaining = self.batch_max_wait_s - (
                time.monotonic() - oldest.enqueued_monotonic
            )
            if len(self._pending) < self.upload_batch_size and remaining > 0.0:
                self._condition.wait(timeout=min(remaining, 0.02))
                return []
            episode_id = int(oldest.item["episode_id"])
            count = 0
            for pending in self._pending:
                if (
                    count >= self.upload_batch_size
                    or int(pending.item["episode_id"]) != episode_id
                ):
                    break
                count += 1
            return [self._pending.popleft() for _ in range(count)]

    def _upload_worker(self) -> None:
        while not self._stop.is_set() and not shutdown_event.is_set():
            pending_batch = self._take_pending_batch()
            if not pending_batch:
                continue
            try:
                encoded_batch: list[tuple[dict[str, Any], EncodedStatefulFrame]] = []
                encode_durations: list[float] = []
                for pending in pending_batch:
                    encoded, encode_s = pending.future.result(timeout=5.0)
                    if pending.generation != self._generation:
                        continue
                    encoded_batch.append((pending.item, encoded))
                    encode_durations.append(encode_s)
                if not encoded_batch:
                    continue
                batch_generation = pending_batch[0].generation
                first_item, first_encoded = encoded_batch[0]
                with self._lock:
                    if self._paused or batch_generation != self._generation:
                        continue
                    reset_needed = (
                        self._session_id is None
                        or self._session_episode_id
                        != int(first_item["episode_id"])
                    )
                if reset_needed and not self._reset_session(first_item, first_encoded):
                    continue
                with self._lock:
                    if self._paused or batch_generation != self._generation:
                        continue
                payload, http_s = self._post_encoded_batch(encoded_batch)
                now = time.monotonic()
                accepted_count = int(
                    payload.get(
                        "accepted_count",
                        int(bool(payload.get("accepted"))),
                    )
                )
                queue_age_s = max(
                    0.0,
                    now - pending_batch[-1].enqueued_monotonic,
                )
                batch_bytes = sum(encoded.payload_bytes for _, encoded in encoded_batch)
                with self._condition:
                    self._server_ready = bool(payload.get("ready"))
                    self._snapshot_version = payload.get("snapshot_version")
                    source_anchor = payload.get("source_anchor_frame_id")
                    self._source_anchor_frame_id = (
                        None if source_anchor is None else int(source_anchor)
                    )
                    self._uploaded_frames += accepted_count
                    self._uploaded_bytes += batch_bytes
                    self._last_queue_age_s = queue_age_s
                    self._last_encode_s = max(encode_durations, default=0.0)
                    uploaded_frames = self._uploaded_frames
                    metrics = self.metrics()
                    self._condition.notify_all()
                if uploaded_frames and (
                    uploaded_frames % 30 < accepted_count
                    or self._dropped_frames
                    or not payload.get("ready")
                ):
                    print(
                        "[V4 STATEFUL UPLOAD] "
                        f"frames={uploaded_frames} batch={len(encoded_batch)} "
                        f"accepted_fps={metrics['uploaded_fps']:.2f} "
                        f"queue_age_ms={queue_age_s * 1000.0:.1f} "
                        f"encode_max_ms={max(encode_durations, default=0.0) * 1000.0:.1f} "
                        f"http_ms={http_s * 1000.0:.1f} "
                        f"source_lag={metrics['source_lag_frames']} "
                        f"dropped={self._dropped_frames} "
                        f"ready={payload.get('ready')}",
                        flush=True,
                    )
            except (
                requests.RequestException,
                ValueError,
                json.JSONDecodeError,
                TimeoutError,
            ) as exc:
                with self._condition:
                    self._session_id = None
                    self._server_ready = False
                    self._snapshot_version = None
                    self._condition.notify_all()
                print(
                    f"[V4 STATEFUL UPLOAD ERROR] {type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(0.2)

    def acquire_inference_ticket(
        self,
        *,
        timeout_s: float = 0.20,
    ) -> StatefulInferenceTicket:
        """Wait briefly for and atomically freeze one fresh server snapshot."""

        if timeout_s <= 0.0:
            raise ValueError("stateful snapshot wait timeout must be positive")
        started = time.monotonic()
        deadline = started + float(timeout_s)
        with self._condition:
            while True:
                session_id = self._session_id
                version = self._snapshot_version
                source_anchor = self._source_anchor_frame_id
                watermark = self._latest_enqueued_frame_id
                max_lag = self._max_source_lag_frames
                lag = (
                    None
                    if source_anchor is None
                    else max(0, int(watermark) - int(source_anchor))
                )
                is_new_version = (
                    version is not None
                    and version != self._last_leased_snapshot_version
                )
                is_fresh = lag is not None and (max_lag is None or lag <= max_lag)
                if (
                    session_id
                    and self._server_ready
                    and version is not None
                    and is_new_version
                    and is_fresh
                ):
                    waited = time.monotonic() - started
                    self._last_leased_snapshot_version = int(version)
                    self._last_snapshot_wait_s = waited
                    return StatefulInferenceTicket(
                        session_id=str(session_id),
                        snapshot_version=int(version),
                        source_anchor_frame_id=int(source_anchor),
                        client_frame_watermark=int(watermark),
                        source_lag_frames=int(lag),
                        wait_s=waited,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._last_snapshot_wait_s = time.monotonic() - started
                    raise StatefulSnapshotUnavailable(
                        "fresh stateful snapshot wait timed out:"
                        f"ready={self._server_ready}:version={version}:"
                        f"last_leased={self._last_leased_snapshot_version}:"
                        f"source_anchor={source_anchor}:watermark={watermark}:"
                        f"source_lag={lag}:limit={max_lag}"
                    )
                self._condition.wait(timeout=min(remaining, 0.05))

    def infer(
        self,
        goal_request: dict[str, Any],
        *,
        recorder: GoldenRunResponseRecorder | None,
        record_metadata: dict[str, Any] | None,
        snapshot_wait_timeout_s: float = 0.20,
    ) -> dict[str, Any]:
        global first_running_time, http_idx
        goal_mode = str(goal_request["goal_mode"])
        if goal_mode not in ("pointgoal", "nogoal"):
            raise ValueError(f"stateful transport does not support {goal_mode}")
        endpoint = f"/v4/stateful/{goal_mode}_step"
        ticket = self.acquire_inference_ticket(timeout_s=snapshot_wait_timeout_s)
        snapshot_wait_s = ticket.wait_s
        started = time.monotonic()
        retry_count = 0
        while True:
            payload: dict[str, Any] = {
                "session_id": ticket.session_id,
                "snapshot_version": ticket.snapshot_version,
                "latest_client_frame_id": ticket.client_frame_watermark,
            }
            if goal_mode == "pointgoal":
                payload["world_goal"] = goal_request.get("world_goal")
            response = self._inference_session.post(
                self.origin + endpoint,
                json=payload,
                timeout=100,
            )
            if response.status_code != 409:
                break
            try:
                detail = response.json()
            except json.JSONDecodeError:
                detail = {"error": response.text[:200]}
            with self._lock:
                self._http_409_count += 1
                observed_retry_version = self._snapshot_version
            print(
                "[V4 STATEFUL 409] "
                f"detail={json.dumps(detail, sort_keys=True)} "
                f"requested_version={ticket.snapshot_version} "
                f"observed_retry_version={observed_retry_version}",
                flush=True,
            )
            if not detail.get("retryable"):
                response.raise_for_status()
            if retry_count >= 1:
                raise StatefulInferenceRetryableError(
                    f"stateful inference HTTP 409 after {retry_count} retries: {detail}"
                )
            retry_count += 1
            with self._lock:
                self._inference_retry_count += 1
            ticket = self.acquire_inference_ticket(
                timeout_s=snapshot_wait_timeout_s
            )
            snapshot_wait_s += ticket.wait_s
        elapsed_s = time.monotonic() - started
        finished = time.monotonic()
        with self._lock:
            interval_s = (
                None
                if self._last_inference_finished_monotonic is None
                else finished - self._last_inference_finished_monotonic
            )
            self._last_inference_finished_monotonic = finished
            self._last_inference_interval_s = interval_s
        print(
            "[V4 STATEFUL INFERENCE] "
            f"status={response.status_code} elapsed_s={elapsed_s:.3f} "
            f"snapshot_wait_s={snapshot_wait_s:.3f} "
            f"interval_s={'first' if interval_s is None else f'{interval_s:.3f}'} "
            f"version={ticket.snapshot_version} "
            f"source_anchor={ticket.source_anchor_frame_id} "
            f"client_watermark={ticket.client_frame_watermark} "
            f"source_lag={ticket.source_lag_frames} retries={retry_count} "
            f"bytes={len(response.content)}",
            flush=True,
        )
        response.raise_for_status()
        parsed = response.json()
        if parsed.get("session_id") != ticket.session_id:
            raise ValueError("stateful response session mismatch")
        if int(parsed.get("snapshot_version", -1)) != ticket.snapshot_version:
            raise ValueError("stateful response snapshot version mismatch")
        source_anchor = parsed.get("source_anchor_frame_id")
        if source_anchor is not None:
            parsed["client_source_lag_frames"] = max(
                0, ticket.client_frame_watermark - int(source_anchor)
            )
        parsed["client_frame_watermark"] = ticket.client_frame_watermark
        parsed["snapshot_wait_s"] = snapshot_wait_s
        parsed["trajectory_interval_s"] = interval_s
        parsed["stateful_retry_count"] = retry_count
        http_idx += 1
        if http_idx == 0:
            first_running_time = time.time()
        if recorder is not None:
            recorder.write(
                reset=http_idx == 0,
                idx=http_idx,
                response=parsed,
                elapsed_s=elapsed_s,
                server_url=self.origin + endpoint,
                goal_mode=goal_mode,
                goal=goal_request.get("goal"),
                metadata={
                    **(record_metadata or {}),
                    "snapshot_wait_s": snapshot_wait_s,
                    "client_frame_watermark": ticket.client_frame_watermark,
                    "request_source_anchor_frame_id": ticket.source_anchor_frame_id,
                    "request_source_lag_frames": ticket.source_lag_frames,
                    "stateful_retry_count": retry_count,
                    "trajectory_interval_s": interval_s,
                },
            )
        return parsed


def navdp_eval_v4(
    snapshot: RealSensorSnapshot,
    goal_request: dict[str, Any],
    *,
    intrinsic: np.ndarray,
    url: str = DEFAULT_SERVER_URL,
    recorder: GoldenRunResponseRecorder | None = None,
    record_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a complete immutable sensor history to the stateless V4 API."""

    global policy_init, http_idx, first_running_time
    origin = server_origin(url)
    request_reset = policy_init
    if request_reset:
        reset_response = requests.post(
            origin + "/v4/navigator_reset",
            json={
                "intrinsic": np.asarray(intrinsic, dtype=float).tolist(),
                "batch_size": 1,
                "stop_threshold": -3.0,
            },
            timeout=100,
        )
        reset_response.raise_for_status()
        report = reset_response.json().get("checkpoint_load_report", {})
        if not valid_v4_checkpoint_load_report(report):
            raise ValueError(f"server did not strictly load the V4 checkpoint: {report}")
        policy_init = False

    goal_mode = goal_request["goal_mode"]
    point_goal = None
    image_goal = None
    if goal_mode == "pointgoal":
        goal = goal_request.get("goal")
        if not isinstance(goal, dict):
            raise ValueError("pointgoal request is missing goal")
        point_goal = np.asarray([goal["x"], goal["y"]], dtype=np.float32)
    elif goal_mode == "imagegoal":
        image_bytes = goal_request.get("goal_image_bytes")
        if image_bytes is None:
            raise ValueError("imagegoal request is missing goal image")
        image_bytes.seek(0)
        image_goal = np.asarray(PIL_Image.open(image_bytes).convert("RGB"))

    history_bytes = snapshot.to_npz_bytes(
        point_goal=point_goal,
        image_goal=image_goal,
        camera_intrinsic=intrinsic,
    )
    start = time.time()
    response = requests.post(
        origin + V4_ENDPOINTS[goal_mode],
        files={
            "history": (
                "navdp_v4_history.npz",
                history_bytes,
                "application/octet-stream",
            )
        },
        timeout=100,
    )
    elapsed_s = time.time() - start
    print(
        f"V4 response status={response.status_code} bytes={len(response.content)} "
        f"anchor={snapshot.anchor_frame_id} elapsed={elapsed_s:.3f}s",
        flush=True,
    )
    response.raise_for_status()
    parsed_response = response.json()
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    if recorder is not None:
        recorder.write(
            reset=request_reset,
            idx=http_idx,
            response=parsed_response,
            elapsed_s=elapsed_s,
            server_url=origin + V4_ENDPOINTS[goal_mode],
            goal_mode=goal_mode,
            goal=goal_request.get("goal"),
            metadata=record_metadata,
        )
    return parsed_response


def build_server_url(
    *,
    server_url: str | None,
    server_host: str,
    server_port: int,
    server_endpoint: str,
) -> str:
    if server_url:
        return server_url
    endpoint = server_endpoint if server_endpoint.startswith("/") else f"/{server_endpoint}"
    return f"http://{server_host}:{server_port}{endpoint}"


def normalize_discrete_actions(actions: Any) -> list[int]:
    if isinstance(actions, np.ndarray):
        actions = actions.tolist()
    if isinstance(actions, (int, np.integer)):
        return [int(actions)]

    normalized = []
    for action in actions:
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)):
            normalized.extend(normalize_discrete_actions(action))
        else:
            normalized.append(int(action))
    return normalized


def clear_motion_state(*, publish_stop: bool = True, force_publish: bool = False) -> None:
    global mpc, trajs_in_world, desired_v, desired_w

    mpc_rw_lock.acquire_write()
    try:
        mpc = None
        trajs_in_world = None
    finally:
        mpc_rw_lock.release_write()

    desired_v, desired_w = 0.0, 0.0
    if manager is not None:
        manager.last_trajs_in_world = None
        manager.last_raw_trajs_in_world = None
        manager.last_valid_trajectory_monotonic = None
        if publish_stop:
            manager.move(0.0, 0.0, 0.0, immediate=True, force_publish=force_publish)


def hold_current_motion(reason: str) -> None:
    global current_control_mode

    current_control_mode = ControlMode.Stop_Mode
    clear_motion_state()
    if manager is not None and not manager.depth_forward_blocked:
        manager.publish_navigation_status("holding", reason.split(":", 1)[0])
    print(
        f"[CLIENT HOLD] reason={reason}; cleared MPC and waiting for next planner output",
        flush=True,
    )


def handle_transient_planning_gap(reason: str) -> None:
    """Keep a recent MPC reference briefly while stateful transport catches up."""

    if manager is None:
        return
    age_s = manager.valid_trajectory_age_s()
    if manager.dry_run:
        print(
            f"[CLIENT RETRY] reason={reason}; dry_run=true no motion state",
            flush=True,
        )
        return
    if (
        current_control_mode == ControlMode.MPC_Mode
        and age_s is not None
        and age_s <= DEFAULT_MAX_RETAINED_TRAJECTORY_AGE_S
    ):
        print(
            "[CLIENT RETAIN] "
            f"reason={reason} trajectory_age_s={age_s:.3f} "
            f"limit_s={DEFAULT_MAX_RETAINED_TRAJECTORY_AGE_S:.3f}",
            flush=True,
        )
        return
    hold_current_motion(f"{reason}:no_recent_trajectory")


def request_client_stop(reason: str, *, distance_to_goal_m: float | None = None) -> None:
    global current_control_mode, navigation_stopped

    navigation_stopped = True
    current_control_mode = ControlMode.Stop_Mode
    clear_motion_state()
    if manager is not None:
        state = "goal_reached" if reason == "pointgoal_reached" else "model_stopped"
        manager.publish_navigation_status(
            state,
            reason,
            distance_to_goal_m=distance_to_goal_m,
            force=True,
        )
        manager.pause_sensor_uploads(reason)
        manager.snapshot_and_render_recorded_trajectories(reason)
    print(f"[CLIENT STOP] reason={reason}; cleared MPC and stopped client planning", flush=True)
    if manager is not None and manager.evaluation_mode:
        manager.finish_evaluation(reason)


def resume_navigation(reason: str) -> None:
    global policy_init, navigation_stopped, current_control_mode

    if manager is not None and getattr(manager, "manual_override_latched", False):
        print(
            "[SAFETY] resume ignored; manual override latched until client restart",
            flush=True,
        )
        return

    policy_init = True
    navigation_stopped = False
    current_control_mode = ControlMode.Stop_Mode
    clear_motion_state()
    if manager is not None and getattr(manager, "v4_history", False):
        manager.resume_sensor_uploads(reason)
    if manager is not None:
        if manager.depth_forward_blocked:
            depth_reason = (
                "depth_obstacle"
                if manager.depth_safety_status is None
                else manager.depth_safety_status.reason
            )
            manager.publish_navigation_status(
                "obstacle_blocked",
                depth_reason,
                force=True,
            )
        else:
            manager.publish_navigation_status("waiting_for_plan", reason, force=True)
        manager.begin_trajectory_recording_episode(reason)
    print(f"[CLIENT RESUME] reason={reason}; reset planner state", flush=True)


def format_depth_safety_status(status: DepthSafetyStatus) -> str:
    min_depth = "nan" if status.min_depth_m is None else f"{status.min_depth_m:.3f}"
    p05_depth = "nan" if status.p05_depth_m is None else f"{status.p05_depth_m:.3f}"
    threshold = "nan" if status.threshold_m is None else f"{status.threshold_m:.3f}"
    return (
        f"reason={status.reason} "
        f"threshold_m={threshold} "
        f"min_depth_m={min_depth} "
        f"p05_depth_m={p05_depth} "
        f"close_fraction={status.close_fraction:.3f} "
        f"valid_fraction={status.valid_fraction:.3f} "
        f"roi_shape={status.roi_shape}"
    )


def control_thread() -> None:
    global current_control_mode, desired_v, desired_w
    while not shutdown_event.is_set():
        if navigation_stopped:
            desired_v, desired_w = 0.0, 0.0
            # request_client_stop() already emitted the one terminal zero.
            # Silence here lets the independent sport bridge issue StopMove on
            # its command timeout instead of receiving endless zero Move calls.
            time.sleep(0.1)
            continue
        if manager is not None and manager.manual_override_latched:
            desired_v, desired_w = 0.0, 0.0
            time.sleep(0.1)
            continue

        if current_control_mode == ControlMode.Stop_Mode:
            desired_v, desired_w = 0.0, 0.0
            if manager is not None:
                manager.move(0.0, 0.0, 0.0, immediate=True)
        elif current_control_mode == ControlMode.MPC_Mode:
            trajectory_age_s = (
                None if manager is None else manager.valid_trajectory_age_s()
            )
            if (
                trajectory_age_s is None
                or trajectory_age_s > DEFAULT_MAX_RETAINED_TRAJECTORY_AGE_S
            ):
                age_label = (
                    "none" if trajectory_age_s is None else f"{trajectory_age_s:.3f}"
                )
                hold_current_motion(f"trajectory_stale:age_s={age_label}")
                time.sleep(0.1)
                continue
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            solve_error = None
            opt_u_controls = None
            if manager is not None and odom is not None:
                previous_w = (
                    manager._last_command_vyaw
                    if manager.mpc_tracking_profile == SMOOTH_CONTINUOUS_PROFILE
                    else 0.0
                )
                mpc_rw_lock.acquire_read()
                try:
                    if mpc is not None:
                        opt_u_controls, _ = mpc.solve(
                            np.array(odom),
                            previous_w=previous_w,
                        )
                except Exception as exc:
                    solve_error = exc
                finally:
                    mpc_rw_lock.release_read()
            if solve_error is not None:
                hold_current_motion(f"mpc_solve_error:{type(solve_error).__name__}")
                print(f"[MPC ERROR] {solve_error}", flush=True)
                time.sleep(0.1)
                continue
            if opt_u_controls is not None:
                opt_u_controls = np.asarray(opt_u_controls, dtype=float)
                if (
                    opt_u_controls.ndim != 2
                    or opt_u_controls.shape[0] == 0
                    or opt_u_controls.shape[1] < 2
                    or not np.all(np.isfinite(opt_u_controls[:, :2]))
                ):
                    hold_current_motion("mpc_invalid_control_shape")
                    time.sleep(0.1)
                    continue
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                if current_control_mode == ControlMode.MPC_Mode and not navigation_stopped:
                    desired_v, desired_w = v, w
                    manager.move(
                        v,
                        0.0,
                        w,
                        allow_depth_recovery_scan=True,
                    )
        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if odom is not None and homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, _, _ = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                if current_control_mode == ControlMode.PID_Mode and not navigation_stopped:
                    desired_v, desired_w = v, w
                    manager.move(v, 0.0, w)

        time.sleep(0.1)


def planning_thread() -> None:
    global current_control_mode, frame_data, mpc, trajs_in_world

    while not shutdown_event.is_set():
        start_time = time.time()
        desired_time = 0.3
        time.sleep(0.05)

        if manager.manual_override_latched:
            manager.new_image_arrived = False
            time.sleep(0.1)
            continue

        if navigation_stopped:
            manager.new_image_arrived = False
            time.sleep(0.1)
            continue

        snapshot = None
        if manager.v4_history:
            snapshot = manager.temporal_sensor_buffer.snapshot()
            if snapshot is None:
                time.sleep(0.01)
                continue
            if snapshot.anchor_frame_id == manager.last_v4_anchor_frame_id:
                time.sleep(0.01)
                continue
            if manager.camera_intrinsic is None:
                print("[V4 CACHE] waiting for RGB CameraInfo", flush=True)
                time.sleep(0.1)
                continue
            if manager.stateful_transport is not None:
                if not manager.stateful_transport.negotiated:
                    time.sleep(0.01)
                    continue
            manager.last_v4_anchor_frame_id = snapshot.anchor_frame_id
            odom_infer = snapshot.anchor_odom.tolist()
            infer_rgb = snapshot.rgb_history[-1]
            infer_depth = snapshot.depth_history_u16[-1].astype(np.float32) / 10000.0
            rgb_bytes = None
            depth_bytes = None
            alignment_depth_bytes = None
        else:
            if not manager.new_image_arrived:
                time.sleep(0.01)
                continue
            manager.new_image_arrived = False
            rgb_depth_rw_lock.acquire_read()
            rgb_bytes = copy.deepcopy(manager.rgb_bytes)
            depth_bytes = copy.deepcopy(manager.depth_bytes)
            alignment_depth_bytes = copy.deepcopy(manager.alignment_depth_bytes)
            infer_rgb = copy.deepcopy(manager.rgb_image)
            infer_depth = copy.deepcopy(manager.depth_image)
            rgb_time = manager.rgb_time
            rgb_depth_rw_lock.release_read()
            odom_rw_lock.acquire_read()
            min_diff = 1e10
            odom_infer = None
            for odom in manager.odom_queue:
                diff = abs(odom[0] - rgb_time)
                if diff < min_diff:
                    min_diff = diff
                    odom_infer = copy.deepcopy(odom[1])
            odom_rw_lock.release_read()

            alignment_missing = (
                requires_alignment_depth(manager.model_mode)
                and alignment_depth_bytes is None
            )
            if (
                odom_infer is None
                or rgb_bytes is None
                or depth_bytes is None
                or alignment_missing
            ):
                print(
                    "skip planning. "
                    f"odom_infer: {odom_infer is not None} "
                    f"rgb_bytes: {rgb_bytes is not None} "
                    f"depth_bytes: {depth_bytes is not None}",
                    f"alignment_depth_bytes: {alignment_depth_bytes is not None}",
                    flush=True,
                )
                time.sleep(0.1)
                continue

        goal_request = manager.build_goal_request(odom_infer)
        if goal_request["status"] == "missing":
            hold_current_motion(goal_request["reason"])
            time.sleep(max(0, desired_time - (time.time() - start_time)))
            continue
        if goal_request["status"] == "reached":
            request_client_stop(
                goal_request["reason"],
                distance_to_goal_m=goal_request.get("distance_to_goal_m"),
            )
            time.sleep(max(0, desired_time - (time.time() - start_time)))
            continue
        if snapshot is not None:
            manager.save_v4_snapshot(snapshot, goal_request)

        frame_data[http_idx] = {
            "infer_rgb": copy.deepcopy(infer_rgb),
            "infer_depth": copy.deepcopy(infer_depth),
            "infer_odom": copy.deepcopy(odom_infer),
        }
        if len(frame_data) > 100:
            del frame_data[min(frame_data.keys())]

        frame_metadata = manager.prepare_inference_record(
            odom_infer, infer_rgb, infer_depth, goal_request
        )
        if snapshot is not None:
            frame_metadata.update(
                {
                    "episode_id": snapshot.episode_id,
                    "anchor_frame_id": snapshot.anchor_frame_id,
                    "anchor_timestamp": snapshot.anchor_timestamp,
                    "clock_domain": snapshot.clock_domain,
                    "source_fps_hz": snapshot.source_fps_hz,
                    "max_selection_error_s": snapshot.max_selection_error_s,
                    "rgb_duplicate_ratio": snapshot.rgb_duplicate_ratio,
                    "depth_duplicate_ratio": snapshot.depth_duplicate_ratio,
                }
            )
        try:
            if snapshot is not None:
                if (
                    manager.stateful_transport is not None
                    and manager.stateful_transport.uses_stateful
                ):
                    response = manager.stateful_transport.infer(
                        goal_request,
                        recorder=manager.response_recorder,
                        record_metadata=frame_metadata,
                        snapshot_wait_timeout_s=(
                            DEFAULT_STATEFUL_SNAPSHOT_WAIT_TIMEOUT_S
                        ),
                    )
                else:
                    response = navdp_eval_v4(
                        snapshot,
                        goal_request,
                        intrinsic=manager.camera_intrinsic,
                        url=manager.server_url,
                        recorder=manager.response_recorder,
                        record_metadata=frame_metadata,
                    )
                if (
                    snapshot.episode_id
                    != manager.temporal_sensor_buffer.episode_id
                ):
                    print(
                        "[V4 CACHE] discarded response from an expired episode",
                        flush=True,
                    )
                    continue
                if (
                    response.get("transport") != STATEFUL_TRANSPORT_NAME
                    and (
                    response.get("episode_ids") != [snapshot.episode_id]
                    or response.get("anchor_frame_id") != snapshot.anchor_frame_id
                    )
                ):
                    raise ValueError("V4 response anchor metadata mismatch")
            else:
                response = navdp_eval(
                    rgb_bytes,
                    depth_bytes,
                    alignment_depth_bytes,
                    goal_request,
                    inference_odom=odom_infer,
                    model_mode=manager.model_mode,
                    url=manager.server_url,
                    recorder=manager.response_recorder,
                    record_metadata=frame_metadata,
                )
        except (
            StatefulSnapshotUnavailable,
            StatefulInferenceRetryableError,
        ) as exc:
            handle_transient_planning_gap(type(exc).__name__)
            print(f"[CLIENT TRANSIENT] {exc}", flush=True)
            continue
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            hold_current_motion(f"http_error:{type(exc).__name__}")
            print(f"[CLIENT HTTP ERROR] {exc}", flush=True)
            continue

        if "trajectory" in response:
            try:
                trajectory = np.asarray(response["trajectory"], dtype=float)
                if trajectory.ndim == 3 and trajectory.shape[0] == 1:
                    trajectory = trajectory[0]
            except (TypeError, ValueError):
                hold_current_motion("invalid_trajectory")
                continue
            if (
                trajectory.ndim != 2
                or trajectory.shape[0] == 0
                or trajectory.shape[1] < 2
                or not np.all(np.isfinite(trajectory[:, :2]))
            ):
                hold_current_motion("invalid_trajectory")
                continue

            manager.update_selected_trajectory_safety(response, trajectory)

            response_anchor_odom = response.get("anchor_odom")
            odom = (
                odom_infer
                if response_anchor_odom is None
                else [float(value) for value in response_anchor_odom]
            )
            traj_len = float(np.linalg.norm(trajectory[-1][:2]))
            if manager.dry_run:
                manager.mark_valid_trajectory()
                print(
                    f"[DRY RUN] trajectory_points={trajectory.shape[0]} traj_len={traj_len:.3f}",
                    flush=True,
                )
                if manager.v4_history:
                    manager.completed_v4_requests += 1
                    if (
                        manager.max_v4_requests > 0
                        and manager.completed_v4_requests >= manager.max_v4_requests
                    ):
                        print(
                            "[SAFETY] completed requested V4 dry-run inferences; "
                            "shutting down without constructing a controller",
                            flush=True,
                        )
                        shutdown_event.set()
                        if rclpy.ok():
                            rclpy.shutdown()
                # A dry run validates only the sensor/cache/model path.  Do not
                # construct MPC/PID references or update any motion state.
                continue
            else:
                print(f"traj len {traj_len}", flush=True)

            raw_local_reference = trajectory[3:, :2].copy()
            if len(raw_local_reference) == 0:
                hold_current_motion("empty_mpc_reference")
                continue

            if manager.mpc_tracking_profile == SMOOTH_CONTINUOUS_PROFILE:
                smoothing_result = smooth_trajectory_xy(
                    raw_local_reference,
                    DEFAULT_TRAJECTORY_SMOOTHING_CONFIG,
                )
                local_reference = smoothing_result.trajectory_xy
                print(
                    "[MPC REFERENCE] "
                    f"profile={manager.mpc_tracking_profile} "
                    f"applied={smoothing_result.applied} "
                    f"reason={smoothing_result.reason} "
                    f"max_deviation_m={smoothing_result.max_deviation_m:.4f} "
                    f"mean_corner_rad="
                    f"{smoothing_result.mean_abs_corner_before_rad:.4f}->"
                    f"{smoothing_result.mean_abs_corner_after_rad:.4f}",
                    flush=True,
                )
            else:
                local_reference = raw_local_reference

            yaw = float(odom[2])
            world_rotation = np.array(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
                dtype=float,
            )
            world_origin = np.asarray(odom[:2], dtype=float)
            raw_trajs_in_world = raw_local_reference @ world_rotation.T + world_origin
            trajs_in_world = local_reference @ world_rotation.T + world_origin

            print(f"{time.time()} update traj", flush=True)

            manager.last_raw_trajs_in_world = raw_trajs_in_world
            manager.last_trajs_in_world = trajs_in_world
            controller_error = None
            mpc_rw_lock.acquire_write()
            try:
                if mpc is None:
                    controller_kwargs = {}
                    if manager.mpc_tracking_profile == SMOOTH_CONTINUOUS_PROFILE:
                        controller_kwargs = {
                            "angular_delta_weight": SMOOTH_MPC_ANGULAR_DELTA_WEIGHT,
                            "max_angular_accel_radps2": SMOOTH_MPC_MAX_ANGULAR_ACCEL_RADPS2,
                        }
                    mpc = Mpc_controller(
                        np.array(trajs_in_world),
                        **controller_kwargs,
                    )
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
            except Exception as exc:
                mpc = None
                controller_error = exc
            finally:
                mpc_rw_lock.release_write()
            if controller_error is not None:
                hold_current_motion(
                    f"mpc_reference_error:{type(controller_error).__name__}"
                )
                print(f"[MPC REFERENCE ERROR] {controller_error}", flush=True)
                continue
            manager.mark_valid_trajectory()
            current_control_mode = ControlMode.MPC_Mode
            if not manager.depth_forward_blocked:
                manager.publish_navigation_status("running", "valid_trajectory")
            manager.mark_first_trajectory()
        elif "discrete_action" in response:
            actions = normalize_discrete_actions(response["discrete_action"])
            print(f"[CLIENT ACTION] discrete_action={actions}", flush=True)
            if 0 in actions:
                request_client_stop("model_stop_action")
            elif tuple(actions) in SPECIAL_HOLD_ACTIONS:
                hold_current_motion(f"special_action_{actions}")
            else:
                current_control_mode = ControlMode.Stop_Mode
                clear_motion_state()
                manager.incremental_change_goal(actions)
                current_control_mode = ControlMode.PID_Mode

        time.sleep(max(0, desired_time - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(
        self,
        *,
        model_mode: str,
        goal_mode: str,
        point_goal: tuple[float, float] | None,
        image_goal_path: Path | None,
        server_url: str,
        depth_scale_m_per_unit: float,
        pointgoal_stop_radius: float,
        dry_run: bool,
        v4_history: bool,
        rgb_topic: str,
        depth_topic: str,
        rgb_camera_info_topic: str,
        odom_topic: str,
        stateful_encode_workers: int,
        stateful_upload_queue_size: int,
        stateful_upload_batch_size: int,
        stateful_batch_max_wait_s: float,
        depth_trajectory_aware_escape: bool,
        depth_safe_curve_max_speed_mps: float,
        depth_recovery_scan_enabled: bool,
        depth_recovery_scan_yaw_rate_radps: float,
        depth_recovery_scan_reverse_interval_s: float,
        temporal_buffer_capacity: int,
        max_frame_selection_error_s: float,
        v4_snapshot_dir: Path | None,
        max_saved_v4_snapshots: int,
        max_v4_requests: int,
        max_linear_speed: float,
        max_angular_speed: float,
        mpc_tracking_profile: str,
        manual_override_topic: str | None,
        manual_override_deadzone: float,
        manual_override_latch: bool,
        manual_override_msg_type: str,
        response_recorder: GoldenRunResponseRecorder | None = None,
        evaluation_scenario_id: str | None = None,
        evaluation_trajectory_type: str | None = None,
        evaluation_world_goal: tuple[float, float] | None = None,
        expected_start: tuple[float, float, float] | None = None,
        adapter_control_url: str | None = None,
        result_json: Path | None = None,
        odom_csv: Path | None = None,
        sensor_dir: Path | None = None,
        motion_timeout_s: float = 120.0,
        stall_timeout_s: float = 15.0,
        stall_progress_m: float = 0.10,
        evaluation_seed: int | None = None,
    ) -> None:
        super().__init__("go2_navdp_manager")
        self.model_mode = normalize_model_mode(model_mode)
        self.goal_mode = normalize_goal_mode(goal_mode)
        validate_model_goal_mode(self.model_mode, self.goal_mode)
        self.server_url = server_url
        self.depth_scale_m_per_unit = depth_scale_m_per_unit
        self.pointgoal_stop_radius = pointgoal_stop_radius
        self.dry_run = dry_run
        self.v4_history = bool(v4_history)
        self.rgb_topic = str(rgb_topic)
        self.depth_topic = str(depth_topic)
        self.rgb_camera_info_topic = str(rgb_camera_info_topic)
        self.odom_topic = str(odom_topic)
        self.depth_trajectory_aware_escape = bool(depth_trajectory_aware_escape)
        self.depth_safe_curve_max_speed_mps = float(
            depth_safe_curve_max_speed_mps
        )
        self.depth_recovery_scan_enabled = bool(depth_recovery_scan_enabled)
        self.depth_recovery_scan_yaw_rate_radps = float(
            depth_recovery_scan_yaw_rate_radps
        )
        self.depth_recovery_scan_reverse_interval_s = float(
            depth_recovery_scan_reverse_interval_s
        )
        self.stateful_transport = (
            StatefulV4ClientTransport(
                server_url,
                encode_workers=stateful_encode_workers,
                upload_queue_size=stateful_upload_queue_size,
                upload_batch_size=stateful_upload_batch_size,
                batch_max_wait_s=stateful_batch_max_wait_s,
            )
            if self.v4_history
            else None
        )
        self.temporal_odom_history = BoundedOdomHistory()
        self.sensor_clock_resolver = SensorClockResolver()
        self.temporal_sensor_buffer = RealSensorRingBuffer(
            capacity=temporal_buffer_capacity,
            max_frame_selection_error_s=max_frame_selection_error_s,
        )
        self.last_v4_anchor_frame_id = -1
        self.camera_intrinsic: np.ndarray | None = None
        self.camera_info_size: tuple[int, int] | None = None
        self.v4_snapshot_dir = (
            None
            if v4_snapshot_dir is None
            else v4_snapshot_dir.expanduser().resolve()
        )
        self.max_saved_v4_snapshots = int(max_saved_v4_snapshots)
        self.saved_v4_snapshots = 0
        self.max_v4_requests = int(max_v4_requests)
        self.completed_v4_requests = 0
        self.last_valid_trajectory_monotonic: float | None = None
        if self.v4_snapshot_dir is not None:
            self.v4_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        if mpc_tracking_profile not in MPC_TRACKING_PROFILES:
            raise ValueError(
                f"unsupported MPC tracking profile: {mpc_tracking_profile!r}"
            )
        self.mpc_tracking_profile = mpc_tracking_profile
        self.manual_override_topic = manual_override_topic
        self.manual_override_deadzone = manual_override_deadzone
        self.manual_override_latch_enabled = manual_override_latch
        self.manual_override_msg_type = manual_override_msg_type
        self.manual_override_latched = False
        self.manual_override_sub = None
        self.navigation_status_state = "starting"
        self.navigation_status_reason = "client_initializing"
        self.navigation_status_pub = None
        self.response_recorder = response_recorder
        self.auto_visualize_responses = True
        self._trajectory_render_lock = threading.Lock()
        self._trajectory_render_started = False
        self._trajectory_episode_start_offset = 0
        self.goal_lock = threading.Lock()
        self.evaluation_mode = evaluation_scenario_id is not None
        self.evaluation_scenario_id = evaluation_scenario_id
        self.evaluation_trajectory_type = evaluation_trajectory_type
        self.expected_start = expected_start
        self.adapter_control_url = adapter_control_url.rstrip("/") if adapter_control_url else None
        self.result_json = result_json.expanduser().resolve() if result_json else None
        self.odom_csv_path = odom_csv.expanduser().resolve() if odom_csv else None
        self.sensor_dir = sensor_dir.expanduser().resolve() if sensor_dir else None
        self.motion_timeout_s = motion_timeout_s
        self.stall_timeout_s = stall_timeout_s
        self.stall_progress_m = stall_progress_m
        self.evaluation_seed = evaluation_seed
        self.pending_local_pointgoal = None if self.evaluation_mode else point_goal
        self.point_goal_world: tuple[float, float] | None = (
            evaluation_world_goal if self.evaluation_mode else None
        )
        self.goal_image_bytes: io.BytesIO | None = None
        self.evaluation_start_ready = not self.evaluation_mode
        self.evaluation_actual_start: list[float] | None = None
        self.evaluation_started_wall_time = time.time()
        self.motion_started_monotonic: float | None = None
        self.best_goal_distance = math.inf
        self.last_progress_monotonic: float | None = None
        self.path_length_m = 0.0
        self._last_path_xy: tuple[float, float] | None = None
        self.final_reason: str | None = None
        self._evaluation_finish_lock = threading.Lock()
        self._evaluation_finished = False
        self.depth_block_events: list[dict[str, Any]] = []
        self._active_depth_block_started: float | None = None
        self.selected_trajectory_safe = False
        self.selected_trajectory_safety_monotonic: float | None = None
        self.selected_trajectory_end_y_m: float | None = None
        self.depth_safe_curve_escape_events: list[dict[str, Any]] = []
        self._active_depth_safe_curve_escape_started: float | None = None
        self.depth_recovery_preferred_direction = 1
        self.depth_recovery_active_direction = 1
        self.depth_recovery_left_clearance_m: float | None = None
        self.depth_recovery_right_clearance_m: float | None = None
        self.depth_recovery_scan_events: list[dict[str, Any]] = []
        self._active_depth_recovery_scan_started: float | None = None
        self._depth_recovery_scan_leg_started: float | None = None
        self._odom_file = None
        self._odom_writer = None
        if self.evaluation_mode:
            if self.goal_mode != "pointgoal" or self.point_goal_world is None:
                raise ValueError("evaluation mode requires an explicit world pointgoal")
            if self.result_json is None or self.odom_csv_path is None or self.sensor_dir is None:
                raise ValueError("evaluation mode requires result JSON, odom CSV, and sensor dir")
            self.result_json.parent.mkdir(parents=True, exist_ok=True)
            self.odom_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.sensor_dir.mkdir(parents=True, exist_ok=True)
            self._odom_file = self.odom_csv_path.open("w", encoding="utf-8", newline="")
            self._odom_writer = csv.DictWriter(
                self._odom_file,
                fieldnames=("timestamp_unix_s", "x", "y", "yaw", "linear_x", "angular_z"),
            )
            self._odom_writer.writeheader()

        image_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.navigation_status_pub = self.create_publisher(
            String,
            NAVIGATION_STATUS_TOPIC,
            status_qos,
        )
        print("[NAVDP_CLIENT] using BEST_EFFORT QoS for RGB/depth subscriptions", flush=True)
        print(f"[NAVDP_CLIENT] server url: {self.server_url}", flush=True)
        print(f"[NAVDP_CLIENT] model_mode: {self.model_mode}", flush=True)
        print(
            f"[NAVDP_CLIENT] temporal_mode={'v4_snapshot' if self.v4_history else 'legacy_request_history'}",
            flush=True,
        )
        if self.dry_run:
            print("[SAFETY] dry-run: not publishing cmd_vel", flush=True)
            print(f"[NAVDP_CLIENT] dry-run command topic: {CMD_VEL_TOPIC}", flush=True)
        else:
            print(f"[NAVDP_CLIENT] publishing {CMD_VEL_TOPIC}", flush=True)
        print(
            f"[SAFETY] max_linear_speed={self.max_linear_speed:.3f} m/s "
            f"max_angular_speed={self.max_angular_speed:.3f} rad/s",
            flush=True,
        )
        print(
            "[MPC CONFIG] "
            f"tracking_profile={self.mpc_tracking_profile} "
            f"savgol_window={DEFAULT_TRAJECTORY_SMOOTHING_CONFIG.window_length} "
            f"savgol_poly={DEFAULT_TRAJECTORY_SMOOTHING_CONFIG.polynomial_order} "
            f"max_deviation_m={DEFAULT_TRAJECTORY_SMOOTHING_CONFIG.max_deviation_m:.3f} "
            f"angular_delta_weight={SMOOTH_MPC_ANGULAR_DELTA_WEIGHT:.2f} "
            f"max_angular_accel_radps2={SMOOTH_MPC_MAX_ANGULAR_ACCEL_RADPS2:.2f}",
            flush=True,
        )
        print(f"[NAVDP_CLIENT] goal_mode: {self.goal_mode}", flush=True)
        print(
            "[NAVDP_CLIENT] goal topics: "
            f"{NAVDP_GOAL_POINT_TOPIC}, {NAVDP_GOAL_IMAGE_TOPIC}, {NAVDP_GOAL_MODE_TOPIC}",
            flush=True,
        )
        if self.response_recorder is not None:
            print(
                f"[GOLDEN RECORD] writing model responses to {self.response_recorder.path}",
                flush=True,
            )
        print(
            f"[NAVDP_CLIENT] depth scale: {self.depth_scale_m_per_unit:.12g} m/raw_unit",
            flush=True,
        )

        self.depth_safety_config = DEFAULT_DEPTH_SAFETY_CONFIG
        # Evaluation starts fail-closed until the first synchronized depth frame
        # has actually run through the safety filter.  This prevents a valid
        # trajectory arriving a few frames before depth from briefly commanding
        # forward motion into a close obstacle.
        self.depth_forward_blocked = self.evaluation_mode
        self.depth_safety_blocked_since: float | None = (
            time.monotonic() if self.evaluation_mode else None
        )
        if self.evaluation_mode:
            self._active_depth_block_started = self.depth_safety_blocked_since
        self.depth_safety_status = None
        print(
            "[DEPTH SAFETY CONFIG] "
            f"block_distance_m={self.depth_safety_config.block_distance_m:.2f} "
            f"clear_distance_m={self.depth_safety_config.clear_distance_m:.2f} "
            f"roi=({self.depth_safety_config.roi_x_min:.2f},"
            f"{self.depth_safety_config.roi_y_min:.2f})-"
            f"({self.depth_safety_config.roi_x_max:.2f},"
            f"{self.depth_safety_config.roi_y_max:.2f}) "
            f"min_close_fraction={self.depth_safety_config.min_close_fraction:.3f} "
            f"min_block_duration_s={self.depth_safety_config.min_block_duration_s:.2f}",
            flush=True,
        )
        if self.evaluation_mode:
            print(
                "[DEPTH SAFETY STARTUP] forward motion blocked until first "
                "synchronized depth assessment",
                flush=True,
            )
        print(
            "[DEPTH SAFETY SAFE CURVE] "
            f"enabled={self.depth_trajectory_aware_escape} "
            f"max_forward_speed_mps={self.depth_safe_curve_max_speed_mps:.3f}",
            flush=True,
        )
        print(
            "[DEPTH SAFETY RECOVERY SCAN] "
            f"enabled={self.depth_recovery_scan_enabled} "
            f"yaw_rate_radps={self.depth_recovery_scan_yaw_rate_radps:.3f} "
            f"reverse_interval_s={self.depth_recovery_scan_reverse_interval_s:.2f}",
            flush=True,
        )
        rgb_down_sub = Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=image_qos,
        )
        depth_down_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=image_qos,
        )

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.syncronizer = ApproximateTimeSynchronizer(
            [rgb_down_sub, depth_down_sub], 32, 0.03
        )
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, sensor_qos
        )
        self.rgb_info_sub = self.create_subscription(
            CameraInfo,
            self.rgb_camera_info_topic,
            self.rgb_camera_info_callback,
            sensor_qos,
        )
        self.goal_point_sub = self.create_subscription(
            PointStamped,
            NAVDP_GOAL_POINT_TOPIC,
            self.goal_point_callback,
            command_qos,
        )
        self.goal_image_sub = self.create_subscription(
            Image,
            NAVDP_GOAL_IMAGE_TOPIC,
            self.goal_image_callback,
            image_qos,
        )
        self.goal_mode_sub = self.create_subscription(
            String,
            NAVDP_GOAL_MODE_TOPIC,
            self.goal_mode_callback,
            command_qos,
        )
        print(
            "[NAVDP_CLIENT] sensor topics: "
            f"rgb={self.rgb_topic} depth={self.depth_topic} "
            f"camera_info={self.rgb_camera_info_topic} odom={self.odom_topic}",
            flush=True,
        )

        self._configure_manual_override(sensor_qos)

        self.control_pub = None
        if not self.dry_run:
            self.control_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 5)

        self.cv_bridge = CvBridge()
        self._sensor_prepare_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="v4-sensor-prepare",
        )
        self._sensor_prepare_condition = threading.Condition()
        self._sensor_prepare_pending: deque[Future[_PreparedRgbDepthFrame]] = deque()
        self._sensor_prepare_stop = threading.Event()
        self._sensor_upload_paused = threading.Event()
        self._sensor_prepare_enqueued = 0
        self._sensor_prepare_processed = 0
        self._sensor_prepare_dropped = 0
        self._sensor_prepare_thread = threading.Thread(
            target=self._sensor_prepare_worker,
            name="v4-sensor-commit",
            daemon=True,
        )
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.alignment_depth_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_trajs_in_world = None
        self.last_raw_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None
        self._last_command_time = None
        self._last_command_vx = 0.0
        self._last_command_vyaw = 0.0
        self._last_dry_run_command_log_time = 0.0
        self._last_dry_run_command: tuple[float, float] | None = None
        self._last_missing_goal_reason = None

        if image_goal_path is not None:
            self.set_image_goal_from_path(image_goal_path)
        if self.evaluation_mode:
            self.evaluation_timer = self.create_timer(0.2, self.evaluation_timer_callback)
            print(
                "[EVALUATION] "
                f"scenario={self.evaluation_scenario_id} type={self.evaluation_trajectory_type} "
                f"world_goal={self.point_goal_world} expected_start={self.expected_start}",
                flush=True,
            )
        initial_state = (
            "obstacle_blocked" if self.depth_forward_blocked else "waiting_for_plan"
        )
        initial_reason = (
            "waiting_for_first_depth_assessment"
            if self.depth_forward_blocked
            else "client_ready"
        )
        self.publish_navigation_status(initial_state, initial_reason, force=True)
        self._sensor_prepare_thread.start()

    def publish_navigation_status(
        self,
        state: str,
        reason: str,
        *,
        distance_to_goal_m: float | None = None,
        force: bool = False,
    ) -> None:
        """Publish planner meaning separately from the bridge's actuator action."""

        state = str(state)
        reason = str(reason)
        changed = (
            state != self.navigation_status_state
            or reason != self.navigation_status_reason
        )
        if not changed and not force:
            return
        self.navigation_status_state = state
        self.navigation_status_reason = reason
        if self.navigation_status_pub is not None:
            msg = String()
            msg.data = encode_navigation_status(
                state,
                reason,
                goal_mode=self.goal_mode,
                distance_to_goal_m=distance_to_goal_m,
            )
            self.navigation_status_pub.publish(msg)
        distance_text = (
            ""
            if distance_to_goal_m is None
            else f" distance_to_goal_m={distance_to_goal_m:.3f}"
        )
        if state == "goal_reached":
            print(
                "[GOAL REACHED] state=goal_reached "
                f"reason={reason}{distance_text}; navigation stopped safely",
                flush=True,
            )
        else:
            print(
                f"[NAVIGATION STATUS] state={state} reason={reason}{distance_text}",
                flush=True,
            )

    def update_selected_trajectory_safety(
        self,
        response: dict[str, Any],
        trajectory: np.ndarray,
    ) -> None:
        """Cache selected-path safety and the clearer side for recovery scanning."""

        selected_safe = False
        safety = response.get("candidate_safety")
        if isinstance(safety, dict):
            candidate_safe = safety.get("candidate_safe")
            try:
                selected_index = int(safety.get("final_selection_index", -1))
            except (TypeError, ValueError):
                selected_index = -1
            if (
                isinstance(candidate_safe, list)
                and 0 <= selected_index < len(candidate_safe)
            ):
                selected_safe = candidate_safe[selected_index] is True
        self.selected_trajectory_safe = selected_safe
        self.selected_trajectory_safety_monotonic = time.monotonic()
        self.selected_trajectory_end_y_m = float(trajectory[-1, 1])

        obstacle = response.get("obstacle")
        obstacle_row = (
            obstacle[0]
            if isinstance(obstacle, list)
            and obstacle
            and isinstance(obstacle[0], dict)
            else None
        )

        def clearance(side: str) -> float | None:
            if obstacle_row is None or not isinstance(obstacle_row.get(side), dict):
                return None
            try:
                value = float(obstacle_row[side].get("nearest_distance_m"))
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        left_clearance = clearance("left")
        right_clearance = clearance("right")
        previous_direction = self.depth_recovery_preferred_direction
        preferred_direction = choose_depth_recovery_scan_direction(
            left_clearance_m=left_clearance,
            right_clearance_m=right_clearance,
            trajectory_end_y_m=self.selected_trajectory_end_y_m,
            previous_direction=previous_direction,
        )
        self.depth_recovery_left_clearance_m = left_clearance
        self.depth_recovery_right_clearance_m = right_clearance
        self.depth_recovery_preferred_direction = preferred_direction
        if self._active_depth_recovery_scan_started is None:
            self.depth_recovery_active_direction = preferred_direction
        if preferred_direction != previous_direction:
            print(
                "[DEPTH SAFETY RECOVERY DIRECTION] "
                f"preferred={'left' if preferred_direction > 0 else 'right'} "
                f"left_clearance_m={left_clearance} "
                f"right_clearance_m={right_clearance} "
                f"trajectory_end_y_m={self.selected_trajectory_end_y_m:.3f}",
                flush=True,
            )

    def prepare_inference_record(
        self,
        odom: list[float],
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        goal_request: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.evaluation_mode:
            return {"inference_odom": [float(value) for value in odom[:3]]}
        sequence = (
            self.response_recorder.request_sequence if self.response_recorder is not None else self.request_cnt
        )
        rgb_name = f"rgb_{sequence:06d}.jpg"
        depth_name = f"depth_{sequence:06d}.png"
        PIL_Image.fromarray(np.asarray(rgb_image, dtype=np.uint8)).save(
            self.sensor_dir / rgb_name, format="JPEG", quality=90
        )
        depth_raw = np.clip(
            np.asarray(depth_image, dtype=np.float32) * HTTP_DEPTH_UNITS_PER_METER,
            0,
            65535,
        ).astype(np.uint16)
        PIL_Image.fromarray(depth_raw).save(self.sensor_dir / depth_name, format="PNG")
        return {
            "scenario_id": self.evaluation_scenario_id,
            "trajectory_type": self.evaluation_trajectory_type,
            "inference_odom": [float(value) for value in odom[:3]],
            "sensor_rgb": f"sensor_frames/{rgb_name}",
            "sensor_depth": f"sensor_frames/{depth_name}",
            "local_pointgoal": copy.deepcopy(goal_request.get("goal")),
        }

    def save_v4_snapshot(
        self,
        snapshot: RealSensorSnapshot,
        goal_request: dict[str, Any],
    ) -> None:
        if (
            self.v4_snapshot_dir is None
            or self.saved_v4_snapshots >= self.max_saved_v4_snapshots
            or self.camera_intrinsic is None
        ):
            return
        point_goal = None
        image_goal = None
        if goal_request["goal_mode"] == "pointgoal":
            goal = goal_request["goal"]
            point_goal = np.asarray([goal["x"], goal["y"]], dtype=np.float32)
        elif goal_request["goal_mode"] == "imagegoal":
            image_bytes = goal_request["goal_image_bytes"]
            image_bytes.seek(0)
            image_goal = np.asarray(PIL_Image.open(image_bytes).convert("RGB"))
        archive = snapshot.to_npz_bytes(
            point_goal=point_goal,
            image_goal=image_goal,
            camera_intrinsic=self.camera_intrinsic,
        )
        destination = self.v4_snapshot_dir / (
            f"episode_{snapshot.episode_id:04d}_anchor_{snapshot.anchor_frame_id:08d}.npz"
        )
        destination.write_bytes(archive.getvalue())
        self.saved_v4_snapshots += 1
        print(f"[V4 SNAPSHOT] saved {destination}", flush=True)

    def mark_first_trajectory(self) -> None:
        if not self.evaluation_mode or self.motion_started_monotonic is not None:
            return
        now = time.monotonic()
        self.motion_started_monotonic = now
        self.last_progress_monotonic = now
        if self.odom is not None and self.point_goal_world is not None:
            self.best_goal_distance = pointgoal_distance_to_odom(
                self.point_goal_world, self.odom
            )
        print("[EVALUATION] motion timer started after first valid trajectory", flush=True)

    def mark_valid_trajectory(self) -> None:
        self.last_valid_trajectory_monotonic = time.monotonic()

    def valid_trajectory_age_s(self) -> float | None:
        if self.last_valid_trajectory_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self.last_valid_trajectory_monotonic)

    def _adapter_status(self) -> dict[str, Any]:
        if self.adapter_control_url is None:
            return {}
        response = requests.get(
            f"{self.adapter_control_url}/trial/status", timeout=0.4
        )
        response.raise_for_status()
        value = response.json()
        return value if isinstance(value, dict) else {}

    def evaluation_timer_callback(self) -> None:
        if not self.evaluation_mode or self._evaluation_finished:
            return
        if not self.evaluation_start_ready:
            if time.time() - self.evaluation_started_wall_time > 30.0:
                self.finish_evaluation("start_gate_timeout")
            return
        try:
            adapter_status = self._adapter_status()
        except requests.RequestException as exc:
            print(f"[EVALUATION] adapter status unavailable: {exc}", flush=True)
            adapter_status = {}
        if adapter_status.get("collision"):
            self.finish_evaluation(
                f"collision:{adapter_status.get('collision_reason', 'unknown')}"
            )
            return
        if self.odom is not None and self.point_goal_world is not None:
            distance = pointgoal_distance_to_odom(self.point_goal_world, self.odom)
            if distance <= self.pointgoal_stop_radius:
                self.finish_evaluation("pointgoal_reached")
                return
        if self.motion_started_monotonic is None:
            return
        now = time.monotonic()
        if now - self.motion_started_monotonic >= self.motion_timeout_s:
            self.finish_evaluation("motion_timeout")
            return
        if (
            self.last_progress_monotonic is not None
            and now - self.last_progress_monotonic >= self.stall_timeout_s
        ):
            self.finish_evaluation("stall")

    def finish_evaluation(self, reason: str) -> None:
        global current_control_mode, navigation_stopped
        if not self.evaluation_mode:
            return
        with self._evaluation_finish_lock:
            if self._evaluation_finished:
                return
            self._evaluation_finished = True
            self.final_reason = reason
        navigation_stopped = True
        current_control_mode = ControlMode.Stop_Mode
        shutdown_event.set()
        try:
            self.publish_zero_command(f"evaluation_terminal:{reason}", force_publish=True)
        except Exception as exc:
            print(f"[EVALUATION] failed to publish terminal zero: {exc}", flush=True)
        self.pause_sensor_uploads(f"evaluation_terminal:{reason}")
        self.snapshot_and_render_recorded_trajectories(
            f"evaluation_terminal:{reason}"
        )
        try:
            adapter_status = self._adapter_status()
        except requests.RequestException:
            adapter_status = {}
        final_distance = None
        final_odom = None
        if self.odom is not None:
            final_odom = [float(value) for value in self.odom[:3]]
            if self.point_goal_world is not None:
                final_distance = pointgoal_distance_to_odom(
                    self.point_goal_world, self.odom
                )
        active_depth_duration = None
        if self._active_depth_block_started is not None:
            active_depth_duration = time.monotonic() - self._active_depth_block_started
        active_safe_curve_duration = None
        if self._active_depth_safe_curve_escape_started is not None:
            active_safe_curve_duration = (
                time.monotonic() - self._active_depth_safe_curve_escape_started
            )
        active_recovery_scan_duration = None
        if self._active_depth_recovery_scan_started is not None:
            active_recovery_scan_duration = (
                time.monotonic() - self._active_depth_recovery_scan_started
            )
        recovered = all(
            event.get("recovered", False) and event.get("duration_s", math.inf) <= 10.0
            for event in self.depth_block_events
        ) and active_depth_duration is None
        result = {
            "schema_version": 1,
            "scenario_id": self.evaluation_scenario_id,
            "trajectory_type": self.evaluation_trajectory_type,
            "goal_mode": "pointgoal",
            "world_goal_xy": list(self.point_goal_world),
            "expected_start": list(self.expected_start) if self.expected_start else None,
            "actual_start": self.evaluation_actual_start,
            "terminal_reason": reason,
            "reached_goal": reason == "pointgoal_reached",
            "collision": reason.startswith("collision:"),
            "stall": reason == "stall",
            "motion_timeout": reason == "motion_timeout",
            "final_odom": final_odom,
            "final_goal_distance_m": final_distance,
            "path_length_m": self.path_length_m,
            "motion_duration_s": (
                None
                if self.motion_started_monotonic is None
                else time.monotonic() - self.motion_started_monotonic
            ),
            "minimum_clearance_m": adapter_status.get("minimum_clearance_m"),
            "start_target_center_distance_m": adapter_status.get(
                "start_target_center_distance_m"
            ),
            "depth_block_events": self.depth_block_events,
            "depth_active_block_duration_s": active_depth_duration,
            "depth_blocks_recovered": recovered,
            "depth_trajectory_aware_escape_enabled": self.depth_trajectory_aware_escape,
            "depth_safe_curve_escape_max_speed_mps": (
                self.depth_safe_curve_max_speed_mps
            ),
            "depth_safe_curve_escape_events": self.depth_safe_curve_escape_events,
            "depth_safe_curve_active_duration_s": active_safe_curve_duration,
            "depth_recovery_scan_enabled": self.depth_recovery_scan_enabled,
            "depth_recovery_scan_yaw_rate_radps": (
                self.depth_recovery_scan_yaw_rate_radps
            ),
            "depth_recovery_scan_reverse_interval_s": (
                self.depth_recovery_scan_reverse_interval_s
            ),
            "depth_recovery_scan_events": self.depth_recovery_scan_events,
            "depth_recovery_scan_active_duration_s": active_recovery_scan_duration,
            "server_url": self.server_url,
            "evaluation_seed": self.evaluation_seed,
            "timestamp_unix_s": time.time(),
        }
        temporary = self.result_json.with_suffix(self.result_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.result_json)
        if self._odom_file is not None and not self._odom_file.closed:
            self._odom_file.flush()
        print(f"[EVALUATION TERMINAL] reason={reason} result={self.result_json}", flush=True)

        def stop_ros() -> None:
            time.sleep(0.15)
            if rclpy is not None and rclpy.ok():
                rclpy.shutdown()

        threading.Thread(target=stop_ros, name="evaluation-shutdown", daemon=True).start()

    def close_evaluation_files(self) -> None:
        if self._odom_file is not None and not self._odom_file.closed:
            self._odom_file.close()

    def _configure_manual_override(self, sensor_qos: Any) -> None:
        if not self.manual_override_topic:
            print(
                "[SAFETY] manual override topic not configured; remote latch disabled",
                flush=True,
            )
            return
        if not self.manual_override_latch_enabled:
            print("[SAFETY] manual override latch disabled by configuration", flush=True)
            return

        msg_cls = self._resolve_manual_override_msg_class(self.manual_override_msg_type)
        self.manual_override_sub = self.create_subscription(
            msg_cls,
            self.manual_override_topic,
            self.manual_override_callback,
            sensor_qos,
        )
        print(
            "[SAFETY] manual override latch enabled: "
            f"topic={self.manual_override_topic} "
            f"msg_type={self.manual_override_msg_type} "
            f"deadzone={self.manual_override_deadzone:.3f}",
            flush=True,
        )

    def _resolve_manual_override_msg_class(self, msg_type: str) -> Any:
        if msg_type == "sensor_msgs/msg/Joy":
            return Joy
        if msg_type == "unitree_go/msg/WirelessController":
            try:
                from unitree_go.msg import WirelessController  # noqa: PLC0415
            except Exception as exc:
                raise RuntimeError(
                    "manual override msg type unitree_go/msg/WirelessController is not "
                    "available; source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash"
                ) from exc
            return WirelessController
        raise ValueError(f"unsupported manual override msg type: {msg_type}")

    def _manual_override_input_active(self, msg: Any) -> bool:
        deadzone = self.manual_override_deadzone
        if self.manual_override_msg_type == "sensor_msgs/msg/Joy":
            axes = getattr(msg, "axes", []) or []
            buttons = getattr(msg, "buttons", []) or []
            axis_active = any(
                math.isfinite(float(value)) and abs(float(value)) > deadzone
                for value in axes
            )
            button_active = any(int(value) != 0 for value in buttons)
            return axis_active or button_active

        if self.manual_override_msg_type == "unitree_go/msg/WirelessController":
            sticks = (
                getattr(msg, "lx", 0.0),
                getattr(msg, "ly", 0.0),
                getattr(msg, "rx", 0.0),
                getattr(msg, "ry", 0.0),
            )
            stick_active = any(
                math.isfinite(float(value)) and abs(float(value)) > deadzone
                for value in sticks
            )
            key_active = int(getattr(msg, "keys", 0)) != 0
            return stick_active or key_active

        return False

    def manual_override_callback(self, msg: Any) -> None:
        if self.manual_override_latched or not self.manual_override_latch_enabled:
            return
        if self._manual_override_input_active(msg):
            self.latch_manual_override("manual_override_input")

    def latch_manual_override(self, reason: str) -> None:
        global current_control_mode, navigation_stopped

        if self.manual_override_latched:
            return

        self.manual_override_latched = True
        navigation_stopped = True
        current_control_mode = ControlMode.Stop_Mode
        clear_motion_state(publish_stop=False)
        self.publish_navigation_status("manual_override", reason, force=True)
        print(
            f"[SAFETY] manual override latched: reason={reason}; "
            "automatic control disabled until client restart",
            flush=True,
        )
        self.publish_zero_command(f"manual_override_latched:{reason}", force_publish=True)
        self.pause_sensor_uploads(f"manual_override_latched:{reason}")
        self.snapshot_and_render_recorded_trajectories(
            f"manual_override_latched:{reason}"
        )

    def publish_zero_command(self, reason: str, *, force_publish: bool = False) -> None:
        self._last_command_time = time.monotonic()
        self._last_command_vx = 0.0
        self._last_command_vyaw = 0.0
        request = Twist()
        request.linear.x = 0.0
        request.linear.y = 0.0
        request.angular.z = 0.0
        self._emit_cmd_vel(request, reason=reason, force_publish=force_publish)

    def _encode_goal_image(self, rgb_image: np.ndarray) -> io.BytesIO:
        goal = PIL_Image.fromarray(rgb_image)
        image_bytes = io.BytesIO()
        goal.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        return image_bytes

    def set_image_goal_from_path(self, path: Path) -> None:
        resolved = require_path(path, "image goal")
        goal = PIL_Image.open(resolved).convert("RGB")
        with self.goal_lock:
            self.goal_image_bytes = self._encode_goal_image(np.asarray(goal))
        print(f"[NAVDP_CLIENT] loaded image goal: {resolved}", flush=True)

    def _anchor_pending_pointgoal_locked(self, odom: list[float] | None) -> None:
        if self.pending_local_pointgoal is not None and odom is not None:
            self.point_goal_world = local_pointgoal_to_world(self.pending_local_pointgoal, odom)
            print(
                "[NAVDP_CLIENT] anchored pointgoal "
                f"local={self.pending_local_pointgoal} world={self.point_goal_world}",
                flush=True,
            )
            self.pending_local_pointgoal = None

    def goal_point_callback(self, msg: PointStamped) -> None:
        if self.evaluation_mode:
            print("[EVALUATION] ignored runtime goal topic; world pointgoal is immutable", flush=True)
            return
        local_goal = (float(msg.point.x), float(msg.point.y))
        with self.goal_lock:
            self.goal_mode = "pointgoal"
            self.pending_local_pointgoal = local_goal
            self.point_goal_world = None
            odom = self.odom.copy() if self.odom is not None else None
            self._anchor_pending_pointgoal_locked(odom)
        resume_navigation("goal_point_topic")

    def goal_image_callback(self, msg: Image) -> None:
        if self.evaluation_mode:
            print("[EVALUATION] ignored image goal topic", flush=True)
            return
        try:
            validate_model_goal_mode(self.model_mode, "imagegoal")
        except ValueError as exc:
            print(f"[NAVDP_CLIENT] ignored incompatible image goal: {exc}", flush=True)
            return
        rgb = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")[:, :, :]
        with self.goal_lock:
            self.goal_image_bytes = self._encode_goal_image(rgb)
            self.goal_mode = "imagegoal"
        resume_navigation("goal_image_topic")

    def goal_mode_callback(self, msg: String) -> None:
        if self.evaluation_mode:
            print("[EVALUATION] ignored goal mode topic; pointgoal is mandatory", flush=True)
            return
        try:
            goal_mode = normalize_goal_mode(msg.data)
            validate_model_goal_mode(self.model_mode, goal_mode)
        except ValueError as exc:
            print(f"[NAVDP_CLIENT] ignored invalid goal_mode topic value: {exc}", flush=True)
            return
        with self.goal_lock:
            self.goal_mode = goal_mode
        resume_navigation("goal_mode_topic")

    def build_goal_request(self, odom: list[float]) -> dict[str, Any]:
        with self.goal_lock:
            if self.evaluation_mode and not self.evaluation_start_ready:
                return {
                    "status": "missing",
                    "reason": "waiting_for_expected_start",
                    "goal_mode": "pointgoal",
                }
            self._anchor_pending_pointgoal_locked(odom)
            goal_mode = self.goal_mode
            if goal_mode == "pointgoal":
                if self.point_goal_world is None:
                    return {
                        "status": "missing",
                        "reason": "missing_pointgoal",
                        "goal_mode": goal_mode,
                    }
                distance = pointgoal_distance_to_odom(self.point_goal_world, odom)
                if distance <= self.pointgoal_stop_radius:
                    return {
                        "status": "reached",
                        "reason": "pointgoal_reached",
                        "goal_mode": goal_mode,
                        "distance_to_goal_m": float(distance),
                    }
                local_goal = world_pointgoal_to_local(self.point_goal_world, odom)
                return {
                    "status": "ready",
                    "goal_mode": goal_mode,
                    "goal": {"x": local_goal[0], "y": local_goal[1]},
                    "world_goal": [
                        float(self.point_goal_world[0]),
                        float(self.point_goal_world[1]),
                    ],
                    "seed": self.evaluation_seed if self.evaluation_mode else None,
                }
            if goal_mode == "imagegoal":
                if self.goal_image_bytes is None:
                    return {
                        "status": "missing",
                        "reason": "missing_imagegoal",
                        "goal_mode": goal_mode,
                    }
                return {
                    "status": "ready",
                    "goal_mode": goal_mode,
                    "goal_image_bytes": io.BytesIO(self.goal_image_bytes.getvalue()),
                }
            return {"status": "ready", "goal_mode": "nogoal"}

    def rgb_camera_info_callback(self, msg: CameraInfo) -> None:
        try:
            intrinsic = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            width = int(msg.width)
            height = int(msg.height)
        except (TypeError, ValueError) as exc:
            print(f"[V4 CAMERA INFO] rejected malformed CameraInfo: {exc}", flush=True)
            return
        if (
            width <= 0
            or height <= 0
            or not np.all(np.isfinite(intrinsic))
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
        ):
            print("[V4 CAMERA INFO] rejected invalid intrinsic", flush=True)
            return
        self.camera_intrinsic = intrinsic.astype(np.float32)
        self.camera_info_size = (width, height)

    def _prepare_rgb_depth_frame(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        receipt_timestamp: float,
    ) -> _PreparedRgbDepthFrame:
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")[:, :, :]
        rgb_image = np.asarray(raw_image, dtype=np.uint8).copy()
        image_bytes = None
        if not self.v4_history:
            image = PIL_Image.fromarray(rgb_image)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes.seek(0)

        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        depth_image = raw_depth.astype(np.float32)
        depth_image *= self.depth_scale_m_per_unit
        depth_u16 = (
            np.clip(depth_image * HTTP_DEPTH_UNITS_PER_METER, 0, 65535)
        ).astype(np.uint16)
        depth_bytes = None
        if not self.v4_history:
            depth = PIL_Image.fromarray(depth_u16)
            depth_bytes = io.BytesIO()
            depth.save(depth_bytes, format="PNG")
            depth_bytes.seek(0)

        alignment_depth_bytes = None
        if requires_alignment_depth(self.model_mode):
            # GlassRecon aligns the relative DPT output against the original
            # metric samples.  Keep this float transport independent of the
            # baseline policy PNG, whose 0.1 mm encoding saturates at 6.5535 m.
            alignment_depth_bytes = io.BytesIO()
            np.save(
                alignment_depth_bytes,
                depth_image.astype(np.float32, copy=False),
                allow_pickle=False,
            )
            alignment_depth_bytes.seek(0)

        rgb_header = ros_stamp_seconds(rgb_msg)
        depth_header = ros_stamp_seconds(depth_msg)
        return _PreparedRgbDepthFrame(
            receipt_timestamp=float(receipt_timestamp),
            rgb_image=rgb_image,
            depth_image=depth_image,
            depth_u16=depth_u16,
            image_bytes=image_bytes,
            depth_bytes=depth_bytes,
            alignment_depth_bytes=alignment_depth_bytes,
            rgb_time=rgb_header,
            depth_time=depth_header,
            rgb_header=rgb_header,
            depth_header=depth_header,
        )

    def rgb_depth_down_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self._sensor_prepare_stop.is_set() or self._sensor_upload_paused.is_set():
            return
        future = self._sensor_prepare_executor.submit(
            self._prepare_rgb_depth_frame,
            rgb_msg,
            depth_msg,
            time.monotonic(),
        )
        with self._sensor_prepare_condition:
            if self._sensor_upload_paused.is_set():
                future.cancel()
                return
            if len(self._sensor_prepare_pending) >= 4:
                dropped = self._sensor_prepare_pending.popleft()
                dropped.cancel()
                self._sensor_prepare_dropped += 1
            self._sensor_prepare_pending.append(future)
            self._sensor_prepare_enqueued += 1
            self._sensor_prepare_condition.notify()

    def _sensor_prepare_worker(self) -> None:
        while not self._sensor_prepare_stop.is_set():
            with self._sensor_prepare_condition:
                while (
                    not self._sensor_prepare_pending
                    and not self._sensor_prepare_stop.is_set()
                ):
                    self._sensor_prepare_condition.wait(timeout=0.1)
                if self._sensor_prepare_stop.is_set():
                    return
                future = self._sensor_prepare_pending.popleft()
            try:
                prepared = future.result(timeout=2.0)
                if self._sensor_upload_paused.is_set():
                    continue
                self._commit_rgb_depth_frame(prepared)
                self._sensor_prepare_processed += 1
            except Exception as exc:
                print(
                    "[V4 SENSOR PREPARE ERROR] "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    def begin_trajectory_recording_episode(self, reason: str) -> None:
        """Arm a new response-file segment for the next terminal stop."""

        recorder = self.response_recorder
        if recorder is None:
            return
        try:
            start_offset = recorder.path.stat().st_size
        except OSError:
            start_offset = 0
        with self._trajectory_render_lock:
            self._trajectory_episode_start_offset = start_offset
            self._trajectory_render_started = False
        print(
            "[TRAJECTORY RECORD] "
            f"new_episode reason={reason} start_offset={start_offset}",
            flush=True,
        )

    def snapshot_and_render_recorded_trajectories(self, reason: str) -> None:
        """Freeze this episode and render every successful plan in the background."""

        recorder = self.response_recorder
        if recorder is None:
            print(
                "[AUTO VISUALIZATION] skipped: --record-responses was not provided",
                flush=True,
            )
            return
        if not self.auto_visualize_responses:
            print("[AUTO VISUALIZATION] disabled", flush=True)
            return

        with self._trajectory_render_lock:
            if self._trajectory_render_started:
                return
            self._trajectory_render_started = True
            start_offset = self._trajectory_episode_start_offset

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_root = recorder.path.parent / f"trajectory_run_{timestamp}"
        snapshot_path = output_root / f"planning_responses_{timestamp}.jsonl"
        render_log_path = output_root / "trajectory_render.log"
        try:
            with recorder.path.open("rb") as source_file:
                source_file.seek(start_offset)
                snapshot_data = source_file.read()
            if not snapshot_data.strip():
                print(
                    "[AUTO VISUALIZATION] "
                    f"skipped reason={reason}: no successful planning responses",
                    flush=True,
                )
                return
            output_root.mkdir(parents=True, exist_ok=False)
            with snapshot_path.open("xb") as snapshot_file:
                snapshot_file.write(snapshot_data)
        except OSError as exc:
            print(
                "[AUTO VISUALIZATION ERROR] "
                f"failed to freeze responses: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        visualizer = PROJECT_ROOT / "scripts" / "visualize_navdp_response_sequence.py"
        command = [
            sys.executable,
            str(visualizer),
            "--responses",
            str(snapshot_path),
            "--output-root",
            str(output_root),
        ]
        try:
            with render_log_path.open("w", encoding="utf-8") as render_log:
                render_process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=render_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            print(
                "[AUTO VISUALIZATION ERROR] "
                f"failed to start renderer: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        record_count = snapshot_data.count(b"\n")
        print(
            "[AUTO VISUALIZATION] "
            f"started pid={render_process.pid} reason={reason} "
            f"records={record_count} output_root={output_root} "
            f"snapshot={snapshot_path} log={render_log_path}",
            flush=True,
        )

    def pause_sensor_uploads(self, reason: str) -> None:
        """Stop local RGB-D buffering and remote stateful uploads."""

        if self._sensor_upload_paused.is_set():
            return
        self._sensor_upload_paused.set()
        with self._sensor_prepare_condition:
            discarded = len(self._sensor_prepare_pending)
            for future in self._sensor_prepare_pending:
                future.cancel()
            self._sensor_prepare_pending.clear()
            self._sensor_prepare_condition.notify_all()
        self.new_image_arrived = False
        if self.stateful_transport is not None:
            self.stateful_transport.pause(reason)
        print(
            "[V4 SENSOR PAUSE] "
            f"reason={reason} discarded_pending_prepares={discarded}",
            flush=True,
        )

    def resume_sensor_uploads(self, reason: str) -> None:
        """Reset temporal state and resume uploads for an explicit new goal."""

        self.temporal_sensor_buffer.reset(f"navigation_resume:{reason}")
        self.last_v4_anchor_frame_id = -1
        if self.stateful_transport is not None:
            self.stateful_transport.resume(f"navigation_resume:{reason}")
        self._sensor_upload_paused.clear()
        print(f"[V4 SENSOR RESUME] reason={reason}", flush=True)

    def close_sensor_pipeline(self) -> None:
        self._sensor_prepare_stop.set()
        with self._sensor_prepare_condition:
            self._sensor_prepare_condition.notify_all()
        self._sensor_prepare_thread.join(timeout=2.0)
        with self._sensor_prepare_condition:
            for future in self._sensor_prepare_pending:
                future.cancel()
            pending = len(self._sensor_prepare_pending)
            self._sensor_prepare_pending.clear()
        self._sensor_prepare_executor.shutdown(wait=False)
        print(
            "[V4 SENSOR SUMMARY] "
            f"enqueued={self._sensor_prepare_enqueued} "
            f"processed={self._sensor_prepare_processed} "
            f"dropped_oldest={self._sensor_prepare_dropped} pending={pending}",
            flush=True,
        )

    def _commit_rgb_depth_frame(self, prepared: _PreparedRgbDepthFrame) -> None:
        if self._sensor_upload_paused.is_set():
            return
        receipt_timestamp = prepared.receipt_timestamp
        rgb_image = prepared.rgb_image
        depth_image = prepared.depth_image
        depth_u16 = prepared.depth_u16
        image_bytes = prepared.image_bytes
        depth_bytes = prepared.depth_bytes
        alignment_depth_bytes = prepared.alignment_depth_bytes

        rgb_depth_rw_lock.acquire_write()
        self.rgb_image = rgb_image
        self.rgb_bytes = image_bytes
        self.rgb_time = prepared.rgb_time
        self.last_rgb_time = self.rgb_time

        self.depth_image = depth_image
        self.depth_bytes = depth_bytes
        self.alignment_depth_bytes = alignment_depth_bytes
        self.depth_time = prepared.depth_time
        self.last_depth_time = self.depth_time
        rgb_depth_rw_lock.release_write()

        self.new_vis_image_arrived = True
        self.new_image_arrived = True
        self.update_depth_safety_filter(depth_image)

        if self.v4_history:
            rgb_header = prepared.rgb_header
            depth_header = prepared.depth_header
            if (
                math.isfinite(rgb_header)
                and math.isfinite(depth_header)
                and abs(rgb_header - depth_header) <= 0.03
            ):
                camera_header = 0.5 * (rgb_header + depth_header)
            else:
                camera_header = math.nan
            resolution = self.sensor_clock_resolver.resolve(
                camera_header_timestamp=camera_header,
                receipt_timestamp=receipt_timestamp,
                odom_history=self.temporal_odom_history,
            )
            if resolution is None:
                return
            previous_episode = self.temporal_sensor_buffer.episode_id
            frame_id = self.temporal_sensor_buffer.append(
                timestamp=resolution.timestamp,
                clock_domain=resolution.clock_domain,
                rgb=rgb_image,
                depth_u16=depth_u16,
                odom=resolution.odom,
            )
            if self.stateful_transport is not None and self.camera_intrinsic is not None:
                self.stateful_transport.enqueue(
                    rgb_image,
                    depth_u16,
                    self.camera_intrinsic,
                    episode_id=self.temporal_sensor_buffer.episode_id,
                    frame_id=frame_id,
                    timestamp=resolution.timestamp,
                    clock_domain=resolution.clock_domain,
                    odom=resolution.odom,
                )
            if self.temporal_sensor_buffer.episode_id != previous_episode:
                self.last_v4_anchor_frame_id = -1
                print(
                    "[V4 CACHE RESET] "
                    f"reason={self.temporal_sensor_buffer.last_reset_reason} "
                    f"episode={self.temporal_sensor_buffer.episode_id}",
                    flush=True,
                )
            if frame_id % 30 == 0:
                snapshot = self.temporal_sensor_buffer.snapshot()
                print(
                    "[V4 CACHE] "
                    f"frame={frame_id} episode={self.temporal_sensor_buffer.episode_id} "
                    f"clock={resolution.clock_domain} "
                    f"odom_error_ms={resolution.odom_error_s * 1000.0:.1f} "
                    f"ready={snapshot is not None} "
                    f"memory_mib={self.temporal_sensor_buffer.memory_bytes / 2**20:.1f}",
                    flush=True,
                )

    def odom_callback(self, msg: Odometry) -> None:
        receipt_timestamp = time.monotonic()
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        xx = msg.pose.pose.orientation.x
        yy = msg.pose.pose.orientation.y
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * (ww * zz + xx * yy), 1 - 2 * (yy * yy + zz * zz))
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()
        self.temporal_odom_history.append(
            header_timestamp=ros_stamp_seconds(msg),
            receipt_timestamp=receipt_timestamp,
            pose=np.asarray(self.odom, dtype=np.float64),
        )

        now = time.monotonic()
        current_xy = (float(self.odom[0]), float(self.odom[1]))
        if self.evaluation_mode:
            if self._odom_writer is not None:
                self._odom_writer.writerow(
                    {
                        "timestamp_unix_s": f"{time.time():.9f}",
                        "x": f"{self.odom[0]:.9f}",
                        "y": f"{self.odom[1]:.9f}",
                        "yaw": f"{self.odom[2]:.9f}",
                        "linear_x": f"{self.linear_vel:.9f}",
                        "angular_z": f"{self.angular_vel:.9f}",
                    }
                )
                self._odom_file.flush()
            if self._last_path_xy is not None and self.evaluation_start_ready:
                self.path_length_m += math.dist(self._last_path_xy, current_xy)
            self._last_path_xy = current_xy
            if not self.evaluation_start_ready and self.expected_start is not None:
                position_error = math.dist(current_xy, self.expected_start[:2])
                yaw_error = abs(
                    (yaw - self.expected_start[2] + math.pi) % (2.0 * math.pi) - math.pi
                )
                if position_error <= 0.12 and yaw_error <= math.radians(8.0):
                    self.evaluation_start_ready = True
                    self.evaluation_actual_start = [self.odom[0], self.odom[1], self.odom[2]]
                    if self.point_goal_world is not None:
                        self.best_goal_distance = pointgoal_distance_to_odom(
                            self.point_goal_world, self.odom
                        )
                    print(
                        "[EVALUATION] expected start gate passed "
                        f"position_error={position_error:.3f} yaw_error_deg={math.degrees(yaw_error):.2f}",
                        flush=True,
                    )
            if self.evaluation_start_ready and self.motion_started_monotonic is not None:
                if self.point_goal_world is not None:
                    distance = pointgoal_distance_to_odom(
                        self.point_goal_world, self.odom
                    )
                    if distance <= self.best_goal_distance - self.stall_progress_m:
                        self.best_goal_distance = distance
                        self.last_progress_monotonic = now

        with self.goal_lock:
            self._anchor_pending_pointgoal_locked(self.odom)

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions: list[int]) -> None:
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it")
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def _depth_recovery_scan_yaw(self, now: float) -> float:
        if self._active_depth_recovery_scan_started is None:
            self._active_depth_recovery_scan_started = now
            self._depth_recovery_scan_leg_started = now
            self.depth_recovery_active_direction = (
                self.depth_recovery_preferred_direction
            )
            print(
                "[DEPTH SAFETY RECOVERY SCAN START] "
                f"direction={'left' if self.depth_recovery_active_direction > 0 else 'right'} "
                f"yaw_rate_radps={self.depth_recovery_scan_yaw_rate_radps:.3f} "
                f"left_clearance_m={self.depth_recovery_left_clearance_m} "
                f"right_clearance_m={self.depth_recovery_right_clearance_m}",
                flush=True,
            )
        elif (
            self._depth_recovery_scan_leg_started is not None
            and now - self._depth_recovery_scan_leg_started
            >= self.depth_recovery_scan_reverse_interval_s
        ):
            self.depth_recovery_active_direction *= -1
            self._depth_recovery_scan_leg_started = now
            print(
                "[DEPTH SAFETY RECOVERY SCAN REVERSE] "
                f"direction={'left' if self.depth_recovery_active_direction > 0 else 'right'}",
                flush=True,
            )
        yaw_rate = min(
            self.depth_recovery_scan_yaw_rate_radps,
            self.max_angular_speed,
        )
        return float(self.depth_recovery_active_direction) * float(yaw_rate)

    def _finish_depth_recovery_scan(self, now: float, reason: str) -> None:
        if self._active_depth_recovery_scan_started is None:
            return
        duration = max(0.0, now - self._active_depth_recovery_scan_started)
        self.depth_recovery_scan_events.append(
            {"duration_s": duration, "reason": reason}
        )
        print(
            "[DEPTH SAFETY RECOVERY SCAN END] "
            f"duration_s={duration:.3f} reason={reason}",
            flush=True,
        )
        self._active_depth_recovery_scan_started = None
        self._depth_recovery_scan_leg_started = None
        self.depth_recovery_active_direction = self.depth_recovery_preferred_direction

    def move(
        self,
        vx: float,
        vy: float,
        vyaw: float,
        immediate: bool = False,
        force_publish: bool = False,
        allow_depth_recovery_scan: bool = False,
    ) -> None:
        if self.manual_override_latched and not force_publish:
            return

        vx = float(np.clip(vx, 0.0, self.max_linear_speed))
        vyaw = float(np.clip(vyaw, -self.max_angular_speed, self.max_angular_speed))
        requested_vx = vx
        requested_vyaw = vyaw
        now = time.monotonic()
        safety_age_s = (
            None
            if self.selected_trajectory_safety_monotonic is None
            else max(0.0, now - self.selected_trajectory_safety_monotonic)
        )
        min_depth_m = (
            None
            if self.depth_safety_status is None
            else self.depth_safety_status.min_depth_m
        )
        vx, vyaw, safe_curve_escape_active = limit_forward_velocity_for_safe_curve(
            vx,
            vyaw,
            blocked=self.depth_forward_blocked,
            enabled=self.depth_trajectory_aware_escape,
            selected_trajectory_safe=self.selected_trajectory_safe,
            selected_trajectory_safety_age_s=safety_age_s,
            min_depth_m=min_depth_m,
            max_escape_speed_mps=self.depth_safe_curve_max_speed_mps,
        )
        recovery_scan_active = should_run_depth_recovery_scan(
            enabled=self.depth_recovery_scan_enabled,
            allow_control_recovery=allow_depth_recovery_scan,
            blocked=self.depth_forward_blocked,
            safe_curve_escape_active=safe_curve_escape_active,
        )
        if recovery_scan_active:
            vx = 0.0
            vyaw = self._depth_recovery_scan_yaw(now)
        else:
            recovery_end_reason = (
                "depth_clear"
                if not self.depth_forward_blocked
                else "control_inactive_or_safe_trajectory"
            )
            self._finish_depth_recovery_scan(now, recovery_end_reason)
        block_forward = (
            self.depth_forward_blocked
            and requested_vx > 0.0
            and not safe_curve_escape_active
        )

        was_safe_curve_escape_active = (
            self._active_depth_safe_curve_escape_started is not None
        )
        if safe_curve_escape_active and not was_safe_curve_escape_active:
            self._active_depth_safe_curve_escape_started = now
            print(
                "[DEPTH SAFETY SAFE CURVE START] "
                f"vx_cap={vx:.3f} vyaw={vyaw:.3f} "
                f"min_depth_m={min_depth_m:.3f} safety_age_s={safety_age_s:.3f}",
                flush=True,
            )
        elif not safe_curve_escape_active and was_safe_curve_escape_active:
            assert self._active_depth_safe_curve_escape_started is not None
            duration = now - self._active_depth_safe_curve_escape_started
            self.depth_safe_curve_escape_events.append({"duration_s": duration})
            self._active_depth_safe_curve_escape_started = None
            print(
                f"[DEPTH SAFETY SAFE CURVE END] duration_s={duration:.3f}",
                flush=True,
            )

        if immediate or self._last_command_time is None:
            smooth_vx = vx
            smooth_vyaw = vyaw
        else:
            dt = max(0.0, min(now - self._last_command_time, 0.5))
            max_dvx = MAX_LINEAR_ACCEL_MPS2 * dt
            max_dvyaw = MAX_ANGULAR_ACCEL_RADPS2 * dt
            smooth_vx = self._last_command_vx + float(
                np.clip(vx - self._last_command_vx, -max_dvx, max_dvx)
            )
            smooth_vyaw = self._last_command_vyaw + float(
                np.clip(vyaw - self._last_command_vyaw, -max_dvyaw, max_dvyaw)
            )

        if block_forward or recovery_scan_active:
            smooth_vx = 0.0
        elif safe_curve_escape_active:
            smooth_vx = min(smooth_vx, vx)
        smooth_vx = float(np.clip(smooth_vx, 0.0, self.max_linear_speed))
        smooth_vyaw = float(np.clip(smooth_vyaw, -self.max_angular_speed, self.max_angular_speed))

        self._last_command_time = now
        self._last_command_vx = smooth_vx
        self._last_command_vyaw = smooth_vyaw

        request = Twist()
        request.linear.x = smooth_vx
        request.linear.y = 0.0
        request.angular.z = smooth_vyaw
        self._emit_cmd_vel(request, reason="move", force_publish=force_publish)

    def _emit_cmd_vel(self, request: Any, *, reason: str, force_publish: bool = False) -> None:
        if self.dry_run:
            self._log_dry_run_command(request, reason)
            return
        if self.manual_override_latched and not force_publish:
            return
        if self.control_pub is None:
            return
        self.control_pub.publish(request)

    def _log_dry_run_command(self, request: Any, reason: str) -> None:
        now = time.monotonic()
        command = (
            float(request.linear.x),
            float(request.angular.z),
        )
        command_changed = self._last_dry_run_command != command
        if command_changed or now - self._last_dry_run_command_log_time >= 1.0:
            print(
                "[DRY RUN CMD] "
                f"reason={reason} "
                f"cmd_vel.linear.x={command[0]:.3f} "
                f"cmd_vel.angular.z={command[1]:.3f}",
                flush=True,
            )
            self._last_dry_run_command = command
            self._last_dry_run_command_log_time = now

    def update_depth_safety_filter(self, depth_image: np.ndarray) -> None:
        now = time.monotonic()
        status = assess_depth_safety_stop(
            depth_image,
            self.depth_safety_config,
            currently_blocked=self.depth_forward_blocked,
        )
        was_blocked = self.depth_forward_blocked
        latch_state = update_depth_block_latch(
            detection_blocked=status.should_block_forward,
            previous=DepthBlockLatchState(
                blocked=was_blocked,
                blocked_since_s=self.depth_safety_blocked_since,
            ),
            now_s=now,
            min_block_duration_s=self.depth_safety_config.min_block_duration_s,
        )
        self.depth_forward_blocked = latch_state.blocked
        self.depth_safety_blocked_since = latch_state.blocked_since_s
        self.depth_safety_status = status

        if self.depth_forward_blocked and not was_blocked:
            if self.evaluation_mode:
                self._active_depth_block_started = now
            print(f"[DEPTH SAFETY BLOCK] {format_depth_safety_status(status)}", flush=True)
            if not navigation_stopped:
                self.publish_navigation_status("obstacle_blocked", status.reason)
        elif was_blocked and not self.depth_forward_blocked:
            if self.evaluation_mode and self._active_depth_block_started is not None:
                duration = now - self._active_depth_block_started
                self.depth_block_events.append(
                    {"duration_s": duration, "recovered": duration <= 10.0}
                )
                self._active_depth_block_started = None
            print(f"[DEPTH SAFETY CLEAR] {format_depth_safety_status(status)}", flush=True)
            if not navigation_stopped:
                if current_control_mode in (ControlMode.MPC_Mode, ControlMode.PID_Mode):
                    self.publish_navigation_status("running", "depth_obstacle_cleared")
                else:
                    self.publish_navigation_status(
                        "waiting_for_plan",
                        "depth_obstacle_cleared",
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=None,
        help="Full NavDP server endpoint. Overrides --server-host/--server-port/--server-endpoint.",
    )
    parser.add_argument(
        "--server-host",
        default=DEFAULT_SERVER_HOST,
        help="NavDP server host used when --server-url is not provided.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="NavDP server port used when --server-url is not provided.",
    )
    parser.add_argument(
        "--server-endpoint",
        default=DEFAULT_SERVER_ENDPOINT,
        help="NavDP server endpoint path used when --server-url is not provided.",
    )
    parser.add_argument(
        "--model-mode",
        choices=sorted(SUPPORTED_MODEL_MODES),
        default="navdp",
        help="Must match the model mode selected by terminal C.",
    )
    parser.add_argument(
        "--goal-mode",
        choices=sorted(SUPPORTED_GOAL_MODES),
        default="nogoal",
        help="Initial NavDP goal mode.",
    )
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--image-goal-path", type=Path, default=None)
    parser.add_argument("--pointgoal-stop-radius", type=float, default=0.5)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
        help="RealSense calibration JSON used to read depth_scale_m_per_unit.",
    )
    parser.add_argument(
        "--record-responses",
        type=Path,
        default=None,
        help="Write each successful /eval_navdp JSON response to this JSONL file.",
    )
    parser.add_argument(
        "--no-auto-visualize-responses",
        action="store_true",
        help=(
            "Do not generate timestamped per-planning-frame PNG/HTML/GIF output "
            "when a --record-responses run terminates."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute NavDP commands but never publish cmd_vel.",
    )
    parser.add_argument(
        "--v4-history",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--rgb-topic", default=RGB_TOPIC)
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    parser.add_argument("--rgb-camera-info-topic", default=RGB_CAMERA_INFO_TOPIC)
    parser.add_argument("--odom-topic", default=ODOM_TOPIC)
    parser.add_argument(
        "--stateful-encode-workers", type=int, default=2, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--stateful-upload-queue-size",
        type=int,
        default=12,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stateful-upload-batch-size", type=int, default=4, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--stateful-batch-max-wait-s", type=float, default=0.10, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--depth-trajectory-aware-escape",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--depth-safe-curve-max-speed-mps",
        type=float,
        default=DEFAULT_SAFE_CURVE_ESCAPE_MAX_SPEED_MPS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-depth-recovery-scan",
        action="store_true",
        help=(
            "Disable the in-place scan used when the depth stop is active and "
            "the guarded low-speed curve cannot execute."
        ),
    )
    parser.add_argument(
        "--depth-recovery-scan-yaw-rate-radps",
        type=float,
        default=DEFAULT_DEPTH_RECOVERY_SCAN_YAW_RATE_RADPS,
        help="In-place depth-stop recovery scan yaw rate (default: 0.25 rad/s).",
    )
    parser.add_argument(
        "--depth-recovery-scan-reverse-interval-s",
        type=float,
        default=DEFAULT_DEPTH_RECOVERY_SCAN_REVERSE_INTERVAL_S,
        help="Reverse a still-blocked recovery scan after this many seconds.",
    )
    parser.add_argument(
        "--temporal-buffer-capacity", type=int, default=32, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--max-frame-selection-error-s", type=float, default=0.045, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--v4-snapshot-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-saved-v4-snapshots", type=int, default=20, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--max-v4-requests",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=DEFAULT_MAX_LINEAR_SPEED_MPS,
        help="Final forward speed limit in m/s.",
    )
    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=DEFAULT_MAX_ANGULAR_SPEED_RADPS,
        help="Final yaw speed limit in rad/s.",
    )
    parser.add_argument(
        "--mpc-tracking-profile",
        choices=MPC_TRACKING_PROFILES,
        default=DEFAULT_MPC_TRACKING_PROFILE,
        help=(
            "MPC reference tracking mode. smooth_continuous enables bounded "
            "Savitzky-Golay reference smoothing and yaw-rate continuity."
        ),
    )
    parser.add_argument(
        "--manual-override-topic",
        default=None,
        help="Optional remote/manual-control topic that permanently latches automatic control off.",
    )
    parser.add_argument(
        "--manual-override-deadzone",
        type=float,
        default=0.05,
        help="Absolute joystick deadzone used by manual override latch.",
    )
    parser.add_argument(
        "--manual-override-latch",
        action="store_true",
        default=True,
        help="Latch manual override permanently until client restart (default: enabled).",
    )
    parser.add_argument(
        "--manual-override-msg-type",
        choices=SUPPORTED_MANUAL_OVERRIDE_MSG_TYPES,
        default="sensor_msgs/msg/Joy",
        help="Message type for --manual-override-topic.",
    )
    parser.add_argument("--evaluation-scenario-id", default=None)
    parser.add_argument(
        "--evaluation-trajectory-type", choices=("straight", "avoidance"), default=None
    )
    parser.add_argument("--world-goal-x", type=float, default=None)
    parser.add_argument("--world-goal-y", type=float, default=None)
    parser.add_argument("--expected-start-x", type=float, default=None)
    parser.add_argument("--expected-start-y", type=float, default=None)
    parser.add_argument("--expected-start-yaw", type=float, default=None)
    parser.add_argument("--adapter-control-url", default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--odom-csv", type=Path, default=None)
    parser.add_argument("--sensor-dir", type=Path, default=None)
    parser.add_argument("--motion-timeout-s", type=float, default=120.0)
    parser.add_argument("--stall-timeout-s", type=float, default=15.0)
    parser.add_argument("--stall-progress-m", type=float, default=0.10)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parsed = parser.parse_args()
    parsed.model_mode = normalize_model_mode(parsed.model_mode)
    try:
        validate_model_goal_mode(parsed.model_mode, parsed.goal_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if parsed.v4_history:
        parser.error(
            "--v4-history belongs to the retired BEV deployment and is not "
            "compatible with the current NavDP/X-NavDP server"
        )
    if (parsed.goal_x is None) != (parsed.goal_y is None):
        parser.error("--goal-x and --goal-y must be provided together")
    if parsed.pointgoal_stop_radius <= 0.0:
        parser.error("--pointgoal-stop-radius must be positive")
    if parsed.max_linear_speed <= 0.0:
        parser.error("--max-linear-speed must be positive")
    if parsed.max_angular_speed <= 0.0:
        parser.error("--max-angular-speed must be positive")
    if parsed.temporal_buffer_capacity < 29:
        parser.error("--temporal-buffer-capacity must be at least 29")
    if not 0.0 < parsed.max_frame_selection_error_s <= 0.050:
        parser.error("--max-frame-selection-error-s must be in (0, 0.050]")
    if parsed.max_saved_v4_snapshots < 0:
        parser.error("--max-saved-v4-snapshots must be non-negative")
    if parsed.max_v4_requests < 0:
        parser.error("--max-v4-requests must be non-negative")
    if parsed.max_v4_requests and (not parsed.dry_run or not parsed.v4_history):
        parser.error("--max-v4-requests requires --dry-run --v4-history")
    if parsed.stateful_encode_workers <= 0:
        parser.error("--stateful-encode-workers must be positive")
    if parsed.stateful_upload_queue_size <= 0:
        parser.error("--stateful-upload-queue-size must be positive")
    if parsed.stateful_upload_batch_size <= 0:
        parser.error("--stateful-upload-batch-size must be positive")
    if parsed.stateful_batch_max_wait_s <= 0.0:
        parser.error("--stateful-batch-max-wait-s must be positive")
    if parsed.depth_trajectory_aware_escape and not parsed.v4_history:
        parser.error("--depth-trajectory-aware-escape requires --v4-history")
    if not 0.0 < parsed.depth_safe_curve_max_speed_mps <= 0.10:
        parser.error("--depth-safe-curve-max-speed-mps must be in (0, 0.10]")
    if parsed.depth_recovery_scan_yaw_rate_radps <= 0.0:
        parser.error("--depth-recovery-scan-yaw-rate-radps must be positive")
    if parsed.depth_recovery_scan_reverse_interval_s <= 0.0:
        parser.error("--depth-recovery-scan-reverse-interval-s must be positive")
    if parsed.manual_override_deadzone < 0.0:
        parser.error("--manual-override-deadzone must be non-negative")
    evaluation_values = (
        parsed.evaluation_scenario_id,
        parsed.evaluation_trajectory_type,
        parsed.world_goal_x,
        parsed.world_goal_y,
        parsed.expected_start_x,
        parsed.expected_start_y,
        parsed.expected_start_yaw,
        parsed.adapter_control_url,
        parsed.result_json,
        parsed.odom_csv,
        parsed.sensor_dir,
        parsed.evaluation_seed,
    )
    if any(value is not None for value in evaluation_values):
        if not all(value is not None for value in evaluation_values):
            parser.error(
                "evaluation mode requires scenario/type, world goal, expected start, "
                "adapter URL, result JSON, odom CSV, and sensor dir"
            )
        if parsed.goal_mode != "pointgoal":
            parser.error("evaluation mode requires --goal-mode pointgoal")
        if parsed.goal_x is not None or parsed.goal_y is not None:
            parser.error("evaluation mode uses --world-goal-x/y, not --goal-x/y")
        if parsed.record_responses is None:
            parser.error("evaluation mode requires --record-responses")
        if parsed.dry_run:
            parser.error("evaluation mode cannot use --dry-run")
    if parsed.motion_timeout_s <= 0.0 or parsed.stall_timeout_s <= 0.0:
        parser.error("evaluation timeouts must be positive")
    if parsed.stall_progress_m <= 0.0:
        parser.error("--stall-progress-m must be positive")
    if parsed.manual_override_topic is not None:
        parsed.manual_override_topic = parsed.manual_override_topic.strip()
        if not parsed.manual_override_topic:
            parsed.manual_override_topic = None
    if not str(parsed.server_host).strip():
        parser.error("--server-host must not be empty")
    if parsed.server_port <= 0:
        parser.error("--server-port must be positive")
    if not str(parsed.server_endpoint).strip():
        parser.error("--server-endpoint must not be empty")
    parsed.server_url = build_server_url(
        server_url=parsed.server_url.strip() if parsed.server_url else None,
        server_host=parsed.server_host.strip(),
        server_port=parsed.server_port,
        server_endpoint=parsed.server_endpoint.strip(),
    )
    return parsed


if __name__ == "__main__":
    cli_args = parse_args()
    ensure_runtime_dependencies(require_controllers=not cli_args.dry_run)
    try:
        cli_args.camera_config = require_path(cli_args.camera_config, "camera config")
        if cli_args.image_goal_path is not None:
            cli_args.image_goal_path = require_path(cli_args.image_goal_path, "image goal")
    except FileNotFoundError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc

    print(f"[NAVDP_CLIENT] server url: {cli_args.server_url}", flush=True)
    print(f"[NAVDP_CLIENT] camera config: {cli_args.camera_config}", flush=True)
    if UNITREE_DDS_DEFAULTS_APPLIED:
        configured = ", ".join(
            f"{key}={value}" for key, value in UNITREE_DDS_DEFAULTS_APPLIED.items()
        )
        print(f"[ROS TRANSPORT DEFAULTS] {configured}", flush=True)
    camera_calibration = load_realsense_config(cli_args.camera_config)
    rclpy.init()
    manager = None
    response_recorder = None

    termination_signal: str | None = None

    def request_graceful_shutdown(signum: int, _frame: Any) -> None:
        """Turn terminal/session termination into the normal cleanup path."""

        global termination_signal, shutdown_event
        nonlocal_signal_name = signal.Signals(signum).name
        if shutdown_event.is_set():
            return
        termination_signal = nonlocal_signal_name
        print(
            f"[NAVDP_CLIENT] received {nonlocal_signal_name}; "
            "closing logs and generating trajectory visualization",
            flush=True,
        )
        shutdown_event.set()
        # Do not call rclpy.shutdown() from the Python signal handler.  Foxy can
        # deadlock if the main thread is inside spin() while another callback is
        # using the same context.  KeyboardInterrupt unwinds spin into finally,
        # where sensor threads are closed before the ROS context.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_graceful_shutdown)
    signal.signal(signal.SIGHUP, request_graceful_shutdown)

    try:
        if cli_args.record_responses is not None:
            response_recorder = GoldenRunResponseRecorder(cli_args.record_responses)
            print(
                f"[GOLDEN RECORD] opened {response_recorder.path} "
                "(existing file is overwritten)",
                flush=True,
            )

        point_goal = None
        if cli_args.goal_x is not None and cli_args.goal_y is not None:
            point_goal = (float(cli_args.goal_x), float(cli_args.goal_y))

        manager = Go2Manager(
            model_mode=cli_args.model_mode,
            goal_mode=cli_args.goal_mode,
            point_goal=point_goal,
            image_goal_path=cli_args.image_goal_path,
            server_url=cli_args.server_url,
            depth_scale_m_per_unit=camera_calibration.depth_scale_m_per_unit,
            pointgoal_stop_radius=cli_args.pointgoal_stop_radius,
            dry_run=cli_args.dry_run,
            v4_history=cli_args.v4_history,
            rgb_topic=cli_args.rgb_topic,
            depth_topic=cli_args.depth_topic,
            rgb_camera_info_topic=cli_args.rgb_camera_info_topic,
            odom_topic=cli_args.odom_topic,
            stateful_encode_workers=cli_args.stateful_encode_workers,
            stateful_upload_queue_size=cli_args.stateful_upload_queue_size,
            stateful_upload_batch_size=cli_args.stateful_upload_batch_size,
            stateful_batch_max_wait_s=cli_args.stateful_batch_max_wait_s,
            depth_trajectory_aware_escape=cli_args.depth_trajectory_aware_escape,
            depth_safe_curve_max_speed_mps=(
                cli_args.depth_safe_curve_max_speed_mps
            ),
            depth_recovery_scan_enabled=(
                not cli_args.disable_depth_recovery_scan
            ),
            depth_recovery_scan_yaw_rate_radps=(
                cli_args.depth_recovery_scan_yaw_rate_radps
            ),
            depth_recovery_scan_reverse_interval_s=(
                cli_args.depth_recovery_scan_reverse_interval_s
            ),
            temporal_buffer_capacity=cli_args.temporal_buffer_capacity,
            max_frame_selection_error_s=cli_args.max_frame_selection_error_s,
            v4_snapshot_dir=cli_args.v4_snapshot_dir,
            max_saved_v4_snapshots=cli_args.max_saved_v4_snapshots,
            max_v4_requests=cli_args.max_v4_requests,
            max_linear_speed=cli_args.max_linear_speed,
            max_angular_speed=cli_args.max_angular_speed,
            mpc_tracking_profile=cli_args.mpc_tracking_profile,
            manual_override_topic=cli_args.manual_override_topic,
            manual_override_deadzone=cli_args.manual_override_deadzone,
            manual_override_latch=cli_args.manual_override_latch,
            manual_override_msg_type=cli_args.manual_override_msg_type,
            response_recorder=response_recorder,
            evaluation_scenario_id=cli_args.evaluation_scenario_id,
            evaluation_trajectory_type=cli_args.evaluation_trajectory_type,
            evaluation_world_goal=(
                None
                if cli_args.world_goal_x is None
                else (float(cli_args.world_goal_x), float(cli_args.world_goal_y))
            ),
            expected_start=(
                None
                if cli_args.expected_start_x is None
                else (
                    float(cli_args.expected_start_x),
                    float(cli_args.expected_start_y),
                    float(cli_args.expected_start_yaw),
                )
            ),
            adapter_control_url=cli_args.adapter_control_url,
            result_json=cli_args.result_json,
            odom_csv=cli_args.odom_csv,
            sensor_dir=cli_args.sensor_dir,
            motion_timeout_s=cli_args.motion_timeout_s,
            stall_timeout_s=cli_args.stall_timeout_s,
            stall_progress_m=cli_args.stall_progress_m,
            evaluation_seed=cli_args.evaluation_seed,
        )
        manager.auto_visualize_responses = not cli_args.no_auto_visualize_responses

        if manager.stateful_transport is not None:
            manager.stateful_transport.start()

        planning_thread_instance = threading.Thread(target=planning_thread)
        planning_thread_instance.daemon = True

        if not cli_args.dry_run:
            control_thread_instance = threading.Thread(target=control_thread)
            control_thread_instance.daemon = True
            control_thread_instance.start()
        else:
            print(
                "[SAFETY] dry-run: control thread was not started and no cmd_vel publisher exists",
                flush=True,
            )
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        if termination_signal is None:
            print(
                "[NAVDP_CLIENT] received Ctrl+C; closing logs and generating "
                "trajectory visualization",
                flush=True,
            )
    except Exception:
        if termination_signal is None:
            raise
    finally:
        shutdown_event.set()
        recorded_responses_path = None
        if manager is not None:
            manager.close_sensor_pipeline()
            if manager.stateful_transport is not None:
                manager.stateful_transport.close()
            if manager.evaluation_mode and not manager._evaluation_finished:
                manager.finish_evaluation("client_shutdown")
            manager.close_evaluation_files()
            manager.destroy_node()
        if response_recorder is not None:
            recorded_responses_path = response_recorder.path
            response_recorder.close()
            print(f"[GOLDEN RECORD] closed {response_recorder.path}", flush=True)
        if rclpy.ok():
            rclpy.shutdown()
        stop_render_already_started = bool(
            manager is not None and manager._trajectory_render_started
        )
        if recorded_responses_path is not None and not stop_render_already_started:
            auto_visualize_recorded_responses(
                recorded_responses_path,
                disabled=cli_args.no_auto_visualize_responses,
            )
