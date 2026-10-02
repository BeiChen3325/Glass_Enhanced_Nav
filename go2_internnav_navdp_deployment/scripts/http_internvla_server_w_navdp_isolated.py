#!/usr/bin/env python3
"""Unified real-robot HTTP server for NavDP, X-NavDP, and X-NavDP+GlassRecon."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.camera.camera_config import (  # noqa: E402
    CameraCalibration,
    load_realsense_config,
    map_color_intrinsics_to_isaacsim,
)
from go2_internnav.config.workspace_paths import (  # noqa: E402
    load_workspace_paths,
    require_path,
)
from go2_internnav.navdp import (  # noqa: E402
    NAVDP_MODE,
    SUPPORTED_GOAL_MODES,
    SUPPORTED_MODEL_MODES,
    X_NAVDP_GLASS_MODE,
    default_checkpoint,
    default_dav2_checkpoint,
    is_x_navdp_mode,
    normalize_goal_mode,
    normalize_model_mode,
    normalize_navdp_trajectory,
    requires_alignment_depth,
    validate_model_goal_mode,
)


WORKSPACE_PATHS = load_workspace_paths()
DEFAULT_DEVICE = WORKSPACE_PATHS.server_device
DEFAULT_HOST = WORKSPACE_PATHS.server_host
DEFAULT_PORT = WORKSPACE_PATHS.server_port
DEFAULT_NAVDP_ROOT = WORKSPACE_PATHS.navdp_root
DEFAULT_CAMERA_CONFIG = WORKSPACE_PATHS.realsense_config
DEFAULT_CAMERA_INFO_MODE = WORKSPACE_PATHS.camera_info_mode
HTTP_DEPTH_UNITS_PER_METER = 10000.0

app = Flask(__name__)
args: argparse.Namespace
camera_calibration: CameraCalibration
navdp_navigator: Any = None
navdp_agent_class: Any = None
navdp_lock = threading.RLock()
request_idx = 0


def camera_intrinsic_for_size(width: int, height: int) -> np.ndarray:
    """Map the deployed RealSense calibration to the incoming RGB grid."""
    mapping = map_color_intrinsics_to_isaacsim(
        camera_calibration,
        width,
        height,
        camera_info_mode=args.camera_info_mode,
    )
    return np.asarray(mapping.camera_info.matrix(), dtype=np.float32)


def json_error(message: str, status_code: int = 400):
    return jsonify({"error": message, "model_mode": args.model_mode}), status_code


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_navdp_agent_class() -> Any:
    """Load exactly the policy implementation selected at process startup."""
    global navdp_agent_class
    if navdp_agent_class is not None:
        return navdp_agent_class

    if args.model_mode == NAVDP_MODE:
        module_root = args.navdp_root / "baselines" / "navdp"
        if not module_root.is_dir():
            raise FileNotFoundError(f"NavDP baseline directory is missing: {module_root}")
        sys.path.insert(0, str(module_root))
        module = importlib.import_module("policy_agent")
    else:
        module_root = args.navdp_root / "baselines" / "x-navdp"
        source_file = module_root / "eval" / "src" / "policy_agent.py"
        if not source_file.is_file():
            raise FileNotFoundError(f"X-NavDP policy agent is missing: {source_file}")
        sys.path.insert(0, str(module_root))
        module = importlib.import_module("eval.src.policy_agent")

    navdp_agent_class = module.NavDP_Agent
    return navdp_agent_class


def _create_navigator(intrinsic: np.ndarray) -> Any:
    agent_class = load_navdp_agent_class()
    common = dict(
        image_intrinsic=intrinsic,
        image_size=args.image_size,
        memory_size=args.memory_size,
        predict_size=args.predict_size,
        temporal_depth=args.temporal_depth,
        heads=args.heads,
        token_dim=args.token_dim,
        navi_model=str(args.checkpoint),
        device=args.device,
    )
    if args.model_mode == NAVDP_MODE:
        return agent_class(**common)
    return agent_class(
        **common,
        embodiment=2,
        is_real=True,
        enable_dpt_prior=requires_alignment_depth(args.model_mode),
        dav2_checkpoint=(
            str(args.dav2_checkpoint)
            if requires_alignment_depth(args.model_mode)
            else None
        ),
        enable_glassrecon=requires_alignment_depth(args.model_mode),
        dpt_color_mode=args.dpt_color_mode,
        prior_representation=args.prior_representation,
        alignment_min_depth=args.alignment_min_depth,
        alignment_max_depth=args.alignment_max_depth,
        ransac_num_rows=args.ransac_num_rows,
        ransac_num_cols=args.ransac_num_cols,
        ransac_points_per_partition=args.ransac_points_per_partition,
        ransac_iterations=args.ransac_iterations,
        ransac_seed=args.ransac_seed,
        return_glass_debug_depths=args.return_glass_debug_depths,
    )


def reset_or_create_navigator(
    intrinsic: np.ndarray,
    *,
    batch_size: int,
    stop_threshold: float,
) -> Any:
    """Create the selected agent lazily, then reset all policy state."""
    global navdp_navigator
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"intrinsic must be a finite [3,3] matrix, got {intrinsic.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    with navdp_lock:
        if navdp_navigator is None:
            navdp_navigator = _create_navigator(intrinsic)
            print(f"[MODEL] created agent mode={args.model_mode}", flush=True)
        else:
            navdp_navigator.image_intrinsic = intrinsic
        if args.model_mode == NAVDP_MODE:
            navdp_navigator.reset(int(batch_size), float(stop_threshold))
        else:
            navdp_navigator.reset(int(batch_size))
        print(
            f"[MODEL] reset mode={args.model_mode} batch_size={batch_size}",
            flush=True,
        )
        return navdp_navigator


def ensure_navigator_for_request(
    width: int,
    height: int,
    *,
    batch_size: int,
    stop_threshold: float,
    force_reset: bool,
) -> Any:
    with navdp_lock:
        current_batch_size = getattr(navdp_navigator, "batch_size", None)
        if navdp_navigator is None or force_reset or current_batch_size != batch_size:
            return reset_or_create_navigator(
                camera_intrinsic_for_size(width, height),
                batch_size=batch_size,
                stop_threshold=stop_threshold,
            )
        return navdp_navigator


def read_rgb(file_storage: Any) -> np.ndarray:
    """Decode a standards-compliant JPEG into true RGB numeric channels."""
    with Image.open(file_storage.stream) as image:
        return np.asarray(image.convert("RGB"))


def read_depth_m(file_storage: Any) -> np.ndarray:
    with Image.open(file_storage.stream) as depth:
        depth_raw = np.asarray(depth.convert("I"))
    return depth_raw.astype(np.float32) / HTTP_DEPTH_UNITS_PER_METER


def read_alignment_depth(file_storage: Any, *, batch_size: int) -> np.ndarray:
    """Decode the lossless float metric depth required by GlassRecon."""
    raw = np.load(file_storage.stream, allow_pickle=False)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]
    if raw.ndim != 3 or raw.shape[-1] != 1:
        raise ValueError(
            f"alignment_depth must decode to [B*H,W,1], got {raw.shape}"
        )
    if raw.shape[0] % batch_size != 0:
        raise ValueError(
            f"alignment_depth height {raw.shape[0]} is not divisible by batch_size={batch_size}"
        )
    return raw.reshape((batch_size, -1, raw.shape[1], 1))


def batch_rgb(image_rgb: np.ndarray, batch_size: int) -> np.ndarray:
    if image_rgb.shape[0] % batch_size != 0:
        raise ValueError("concatenated RGB height is not divisible by batch size")
    return image_rgb.reshape((batch_size, -1, image_rgb.shape[1], 3))


def batch_depth(depth_m: np.ndarray, batch_size: int) -> np.ndarray:
    if depth_m.shape[0] % batch_size != 0:
        raise ValueError("concatenated depth height is not divisible by batch size")
    return depth_m[:, :, np.newaxis].reshape((batch_size, -1, depth_m.shape[1], 1))


def parse_form_json() -> dict[str, Any]:
    raw = request.form.get("json", request.form.get("data"))
    if raw is None:
        if request.is_json:
            payload = request.get_json()
            if isinstance(payload, dict):
                return payload
        raise ValueError("multipart form must include a 'json' field")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("json field must decode to an object")
    return payload


def pointgoal_from_payload(payload: dict[str, Any]) -> np.ndarray:
    goal = payload.get("goal", payload.get("point_goal", payload.get("pointgoal")))
    if not isinstance(goal, dict):
        raise ValueError("pointgoal mode requires goal={'x': float, 'y': float}")
    xy = np.asarray([float(goal["x"]), float(goal["y"])], dtype=np.float32)
    if not np.isfinite(xy).all():
        raise ValueError("pointgoal x/y must be finite")
    return np.asarray([[xy[0], xy[1], 0.0]], dtype=np.float32)


def odom_to_x_navdp_state(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    odom = np.asarray(payload.get("inference_odom"), dtype=np.float64)
    if odom.shape != (3,) or not np.isfinite(odom).all():
        raise ValueError(
            "X-NavDP requests require finite inference_odom=[world_x, world_y, yaw]"
        )
    half_yaw = 0.5 * float(odom[2])
    robot_pos = np.asarray([[odom[0], odom[1], 0.0]], dtype=np.float64)
    robot_quat = np.asarray(
        [[0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]],
        dtype=np.float64,
    )
    return robot_pos, robot_quat


def run_policy(
    *,
    goal_mode: str,
    payload: dict[str, Any],
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    batch_size: int,
    force_reset: bool,
    stop_threshold: float,
    goal_image_rgb: np.ndarray | None = None,
    alignment_depth: np.ndarray | None = None,
) -> tuple[Any, Any, Any, dict[str, Any] | None]:
    goal_mode = normalize_goal_mode(goal_mode)
    validate_model_goal_mode(args.model_mode, goal_mode)
    if image_rgb.shape[:2] != depth_m.shape[:2]:
        raise ValueError(
            f"image/depth shape mismatch: {image_rgb.shape[:2]} vs {depth_m.shape[:2]}"
        )
    navigator = ensure_navigator_for_request(
        image_rgb.shape[1],
        image_rgb.shape[0] // batch_size,
        batch_size=batch_size,
        stop_threshold=stop_threshold,
        force_reset=force_reset,
    )
    image_input = batch_rgb(image_rgb, batch_size)
    depth_input = batch_depth(depth_m, batch_size)

    with navdp_lock:
        if goal_mode == "pointgoal":
            goal = pointgoal_from_payload(payload)
            if is_x_navdp_mode(args.model_mode):
                robot_pos, robot_quat = odom_to_x_navdp_state(payload)
                prediction = navigator.step_pointgoal_with_guidance(
                    goal,
                    image_input,
                    depth_input,
                    robot_pos,
                    robot_quat,
                    alignment_depths=alignment_depth,
                )
            else:
                prediction = navigator.step_pointgoal(goal, image_input, depth_input)
        elif goal_mode == "imagegoal":
            if goal_image_rgb is None:
                raise ValueError("imagegoal mode requires goal_image file")
            prediction = navigator.step_imagegoal(
                batch_rgb(goal_image_rgb, batch_size), image_input, depth_input
            )
        else:
            prediction = navigator.step_nogoal(image_input, depth_input)

    trajectory, all_trajectory, all_values = prediction[:3]
    depth_debug = prediction[4] if len(prediction) == 5 else None
    return trajectory, all_trajectory, all_values, depth_debug


def _validate_requested_model(payload: dict[str, Any]) -> None:
    requested_mode = payload.get("model_mode")
    if requested_mode is None:
        return
    normalized = normalize_model_mode(requested_mode)
    if normalized != args.model_mode:
        raise ValueError(
            f"client requested model_mode={normalized}, but server runs {args.model_mode}"
        )


def eval_navdp_impl():
    global request_idx
    started = time.time()
    try:
        payload = parse_form_json()
        _validate_requested_model(payload)
        goal_mode = normalize_goal_mode(payload.get("goal_mode"))
        validate_model_goal_mode(args.model_mode, goal_mode)
        force_reset = bool(payload.get("reset", False))
        stop_threshold = float(payload.get("stop_threshold", args.stop_threshold))
        image_rgb = read_rgb(request.files["image"])
        depth_m = read_depth_m(request.files["depth"])
        goal_file = request.files.get("goal_image") or request.files.get("goal")
        goal_image_rgb = None if goal_file is None else read_rgb(goal_file)
        alignment_depth = None
        if requires_alignment_depth(args.model_mode):
            alignment_file = request.files.get("alignment_depth")
            if alignment_file is None:
                raise ValueError(
                    "x-navdp-glass requires alignment_depth.npy with raw float metric depth"
                )
            alignment_depth = read_alignment_depth(alignment_file, batch_size=1)
        trajectory, all_trajectory, all_values, depth_debug = run_policy(
            goal_mode=goal_mode,
            payload=payload,
            image_rgb=image_rgb,
            depth_m=depth_m,
            batch_size=1,
            force_reset=force_reset,
            stop_threshold=stop_threshold,
            goal_image_rgb=goal_image_rgb,
            alignment_depth=alignment_depth,
        )
        normalized_trajectory = normalize_navdp_trajectory(trajectory)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return json_error(str(exc), 400)
    except Exception as exc:
        print(f"[MODEL ERROR] {type(exc).__name__}: {exc}", flush=True)
        return json_error("model inference failed; inspect terminal C", 500)

    response = {
        "model_mode": args.model_mode,
        "goal_mode": goal_mode,
        "idx": int(payload.get("idx", request_idx)),
        "trajectory": normalized_trajectory,
        "all_trajectory": to_jsonable(all_trajectory),
        "all_values": to_jsonable(all_values),
        "elapsed_s": time.time() - started,
    }
    if args.model_mode == X_NAVDP_GLASS_MODE and depth_debug is not None:
        response["ransac"] = depth_debug.get("ransac")
        if getattr(args, "return_glass_debug_depths", False):
            for key in (
                "relative_depth",
                "aligned_depth",
                "clipped_enhanced_depth",
            ):
                if key in depth_debug:
                    response[key] = to_jsonable(depth_debug[key])
    request_idx += 1
    print(
        f"[EVAL] idx={response['idx']} model={args.model_mode} goal={goal_mode} "
        f"reset={force_reset} points={len(normalized_trajectory)} "
        f"elapsed_s={response['elapsed_s']:.3f}",
        flush=True,
    )
    return jsonify(response)


@app.route("/eval_navdp", methods=["POST"])
def eval_navdp():
    return eval_navdp_impl()


@app.route("/eval_dual", methods=["POST"])
def eval_dual():
    return eval_navdp_impl()


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify(
        {
            "status": "ok",
            "model_mode": args.model_mode,
            "checkpoint": str(args.checkpoint),
            "agent_loaded": navdp_navigator is not None,
            "goal_modes": (
                ["pointgoal"]
                if is_x_navdp_mode(args.model_mode)
                else sorted(SUPPORTED_GOAL_MODES)
            ),
        }
    )


@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    try:
        payload = request.get_json(force=True)
        intrinsic = np.asarray(payload.get("intrinsic"), dtype=np.float32)
        reset_or_create_navigator(
            intrinsic,
            batch_size=int(payload.get("batch_size", args.batch_size)),
            stop_threshold=float(payload.get("stop_threshold", args.stop_threshold)),
        )
    except Exception as exc:
        return json_error(str(exc), 400)
    return jsonify({"algo": args.model_mode, "model_mode": args.model_mode})


def _native_step(goal_mode: str):
    try:
        validate_model_goal_mode(args.model_mode, goal_mode)
        image_rgb = read_rgb(request.files["image"])
        depth_m = read_depth_m(request.files["depth"])
        if goal_mode == "pointgoal":
            goal_data = json.loads(request.form.get("goal_data"))
            goal_x = np.asarray(goal_data["goal_x"], dtype=np.float32)
            goal_y = np.asarray(goal_data["goal_y"], dtype=np.float32)
            batch_size = int(goal_x.shape[0])
            if batch_size != 1 or goal_y.shape != (1,):
                raise ValueError("real-robot deployment supports native batch_size=1")
            payload: dict[str, Any] = {
                "goal": {"x": float(goal_x[0]), "y": float(goal_y[0])}
            }
            state_data = json.loads(request.form.get("state_data", "{}"))
            if is_x_navdp_mode(args.model_mode):
                robot_pos = np.asarray(state_data.get("robot_pos"), dtype=np.float64)
                robot_quat = np.asarray(state_data.get("robot_quat"), dtype=np.float64)
                if robot_pos.shape != (batch_size, 3) or robot_quat.shape != (batch_size, 4):
                    raise ValueError("state_data must contain robot_pos [B,3] and robot_quat [B,4]")
                yaw = math.atan2(
                    2.0
                    * (
                        robot_quat[0, 3] * robot_quat[0, 2]
                        + robot_quat[0, 0] * robot_quat[0, 1]
                    ),
                    1.0
                    - 2.0
                    * (robot_quat[0, 1] ** 2 + robot_quat[0, 2] ** 2),
                )
                payload["inference_odom"] = [robot_pos[0, 0], robot_pos[0, 1], yaw]
        else:
            batch_size = int(getattr(navdp_navigator, "batch_size", args.batch_size))
            payload = {}
        goal_file = request.files.get("goal")
        goal_image_rgb = None if goal_file is None else read_rgb(goal_file)
        alignment_depth = None
        if requires_alignment_depth(args.model_mode):
            alignment_file = request.files.get("alignment_depth")
            if alignment_file is None:
                raise ValueError("x-navdp-glass requires alignment_depth.npy")
            alignment_depth = read_alignment_depth(alignment_file, batch_size=batch_size)
        trajectory, all_trajectory, all_values, depth_debug = run_policy(
            goal_mode=goal_mode,
            payload=payload,
            image_rgb=image_rgb,
            depth_m=depth_m,
            batch_size=batch_size,
            force_reset=False,
            stop_threshold=args.stop_threshold,
            goal_image_rgb=goal_image_rgb,
            alignment_depth=alignment_depth,
        )
    except Exception as exc:
        return json_error(str(exc), 400)
    response = {
        "model_mode": args.model_mode,
        "trajectory": to_jsonable(trajectory),
        "all_trajectory": to_jsonable(all_trajectory),
        "all_values": to_jsonable(all_values),
    }
    if args.model_mode == X_NAVDP_GLASS_MODE and depth_debug is not None:
        response["ransac"] = depth_debug.get("ransac")
        if getattr(args, "return_glass_debug_depths", False):
            for key in (
                "relative_depth",
                "aligned_depth",
                "clipped_enhanced_depth",
            ):
                if key in depth_debug:
                    response[key] = to_jsonable(depth_debug[key])
    return jsonify(response)


@app.route("/pointgoal_step", methods=["POST"])
def pointgoal_step():
    return _native_step("pointgoal")


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    return _native_step("imagegoal")


@app.route("/nogoal_step", methods=["POST"])
def nogoal_step():
    return _native_step("nogoal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-mode",
        choices=sorted(SUPPORTED_MODEL_MODES),
        default=NAVDP_MODE,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dav2-checkpoint", type=Path)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--navdp-root", type=Path, default=DEFAULT_NAVDP_ROOT)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument(
        "--camera-info-mode",
        choices=("effective", "real"),
        default=DEFAULT_CAMERA_INFO_MODE,
    )
    parser.add_argument("--stop-threshold", type=float, default=-3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--memory-size", type=int, default=8)
    parser.add_argument("--predict-size", type=int, default=24)
    parser.add_argument("--temporal-depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=384)
    parser.add_argument(
        "--dpt-color-mode",
        choices=("shared_xnavdp", "official_rgb"),
        default="shared_xnavdp",
    )
    parser.add_argument(
        "--prior-representation",
        choices=("inverse_depth", "depth"),
        default="inverse_depth",
    )
    parser.add_argument("--alignment-min-depth", type=float, default=0.01)
    parser.add_argument("--alignment-max-depth", type=float, default=100.0)
    parser.add_argument("--ransac-num-rows", type=int, default=4)
    parser.add_argument("--ransac-num-cols", type=int, default=4)
    parser.add_argument("--ransac-points-per-partition", type=int, default=10)
    parser.add_argument("--ransac-iterations", type=int, default=100)
    parser.add_argument("--ransac-seed", type=int, default=7)
    parser.add_argument(
        "--return-glass-debug-depths",
        action="store_true",
        help=(
            "Return relative/aligned/enhanced 224x224 depth maps; disabled by "
            "default to avoid GPU-to-CPU copies in the navigation path."
        ),
    )
    parsed = parser.parse_args()
    parsed.model_mode = normalize_model_mode(parsed.model_mode)
    if is_x_navdp_mode(parsed.model_mode) and sys.version_info < (3, 9):
        parser.error(
            f"{parsed.model_mode} requires Python >= 3.9 because the current "
            "NavDP X-NavDP/DPT source uses modern string APIs; got "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    if parsed.batch_size != 1:
        parser.error("real-robot deployment requires --batch-size 1")
    for name in (
        "image_size",
        "memory_size",
        "predict_size",
        "temporal_depth",
        "heads",
        "token_dim",
        "ransac_num_rows",
        "ransac_num_cols",
        "ransac_points_per_partition",
        "ransac_iterations",
    ):
        if getattr(parsed, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not parsed.alignment_min_depth < parsed.alignment_max_depth:
        parser.error("--alignment-min-depth must be less than --alignment-max-depth")
    if parsed.return_glass_debug_depths and parsed.model_mode != X_NAVDP_GLASS_MODE:
        parser.error("--return-glass-debug-depths requires --model-mode x-navdp-glass")
    return parsed


def print_startup_configuration(parsed: argparse.Namespace) -> None:
    display_host = "<SERVER_IP>" if parsed.host == "0.0.0.0" else parsed.host
    print("=== NAVDP REAL-ROBOT SERVER ===")
    print(f"deployment_root: {WORKSPACE_PATHS.deployment_root}")
    print(f"navdp_root: {parsed.navdp_root}")
    print(f"model_mode: {parsed.model_mode}")
    print(f"checkpoint: {parsed.checkpoint}")
    print(f"dav2_checkpoint: {parsed.dav2_checkpoint}")
    print(f"device: {parsed.device}")
    print(f"camera_config: {parsed.camera_config}")
    print(f"camera_info_mode: {parsed.camera_info_mode}")
    print(f"server_url: http://{display_host}:{parsed.port}/eval_navdp")
    print(f"health_url: http://{display_host}:{parsed.port}/healthz")
    print(
        "goal_modes: "
        + (
            "pointgoal"
            if is_x_navdp_mode(parsed.model_mode)
            else ",".join(sorted(SUPPORTED_GOAL_MODES))
        )
    )
    if parsed.model_mode == X_NAVDP_GLASS_MODE:
        print(
            "glassrecon: enabled "
            f"dpt_color={parsed.dpt_color_mode} prior={parsed.prior_representation} "
            f"ransac={parsed.ransac_num_rows}x{parsed.ransac_num_cols}/"
            f"{parsed.ransac_points_per_partition}pts/{parsed.ransac_iterations}iters/"
            f"seed={parsed.ransac_seed}"
        )
        print(
            "glass_debug_depths: "
            + ("enabled" if parsed.return_glass_debug_depths else "disabled")
        )
    print("agent_loading: lazy_on_first_request")
    print("=== END NAVDP REAL-ROBOT SERVER ===", flush=True)


def main() -> int:
    global args, camera_calibration
    args = parse_args()
    try:
        args.navdp_root = require_path(args.navdp_root, "NavDP root")
        if args.checkpoint is None:
            args.checkpoint = default_checkpoint(args.navdp_root, args.model_mode)
        args.checkpoint = require_path(args.checkpoint, f"{args.model_mode} checkpoint")
        if requires_alignment_depth(args.model_mode):
            if args.dav2_checkpoint is None:
                args.dav2_checkpoint = default_dav2_checkpoint(args.navdp_root)
            args.dav2_checkpoint = require_path(
                args.dav2_checkpoint, "Depth Anything V2 checkpoint"
            )
        elif args.dav2_checkpoint is not None:
            raise ValueError(
                "--dav2-checkpoint is valid only with --model-mode x-navdp-glass"
            )
        args.camera_config = require_path(args.camera_config, "camera config")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1

    camera_calibration = load_realsense_config(args.camera_config)
    print_startup_configuration(args)
    app.run(host=args.host, port=args.port, threaded=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
