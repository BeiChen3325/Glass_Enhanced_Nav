#!/usr/bin/env python3
"""Supervise the Go2-side NavDP processes from one operator terminal."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = DEPLOYMENT_ROOT / "logs"
ROS_SETUP = Path("/opt/ros/foxy/setup.bash")
REALSENSE_SETUP = Path("/home/unitree/realsense_foxy_ws/install/setup.bash")
UNITREE_SETUP = Path("/home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash")

MODEL_MODES = ("navdp", "x-navdp", "x-navdp-glass")
GOAL_MODES = ("pointgoal", "imagegoal", "nogoal")
COMPONENT_GROUPS = ("all", "sensors", "navigation")
RUN_MODES = ("dry-run", "real")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def shell_with_setups(setups: Sequence[Path], command: Sequence[str], *, no_proxy: str | None = None) -> list[str]:
    lines = ["set -e"]
    lines.extend(f"source {shlex.quote(str(path))}" for path in setups)
    if no_proxy is not None:
        lines.extend(
            (
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
                f"export NO_PROXY={shlex.quote(no_proxy)}",
                "export no_proxy=\"${NO_PROXY}\"",
            )
        )
    lines.append("exec " + shlex.join([str(value) for value in command]))
    return ["/bin/bash", "-c", "\n".join(lines)]


def realsense_command() -> list[str]:
    return shell_with_setups(
        (ROS_SETUP, REALSENSE_SETUP),
        (
            "ros2",
            "launch",
            "realsense2_camera",
            "rs_launch.py",
            "enable_depth:=true",
            "enable_color:=true",
            "align_depth:=true",
            "depth_width:=640",
            "depth_height:=480",
            "depth_fps:=15.0",
            "color_width:=640",
            "color_height:=480",
            "color_fps:=15.0",
        ),
    )


def input_bridge_command() -> list[str]:
    return shell_with_setups(
        (ROS_SETUP, UNITREE_SETUP),
        ("/usr/bin/python3", str(DEPLOYMENT_ROOT / "scripts/ros2_navdp_input_bridge.py")),
    )


def safety_bridge_command(args: argparse.Namespace) -> list[str]:
    return shell_with_setups(
        (ROS_SETUP, UNITREE_SETUP),
        (
            "/usr/bin/python3",
            str(DEPLOYMENT_ROOT / "scripts/ros2_cmd_vel_sport_safe_bridge.py"),
            "--no-dry-run",
            "--enable-sport-publish",
            "--i-understand-this-can-move-the-robot",
            "--max-vx",
            str(args.max_linear_speed),
            "--max-wz",
            str(args.max_angular_speed),
            "--cmd-timeout-sec",
            str(args.cmd_timeout_sec),
        ),
    )


def client_command(args: argparse.Namespace, response_path: Path) -> list[str]:
    command = [
        "/usr/bin/python3",
        str(DEPLOYMENT_ROOT / "scripts/http_internvla_client_qos_isolated.py"),
        "--server-url",
        args.server_url,
        "--model-mode",
        args.model_mode,
        "--goal-mode",
        args.goal_mode,
        "--pointgoal-stop-radius",
        str(args.pointgoal_stop_radius),
        "--max-linear-speed",
        str(args.max_linear_speed),
        "--max-angular-speed",
        str(args.max_angular_speed),
        "--mpc-tracking-profile",
        args.mpc_tracking_profile,
        "--record-responses",
        str(response_path),
    ]
    if args.goal_mode == "pointgoal":
        command.extend(("--goal-x", str(args.goal_x), "--goal-y", str(args.goal_y)))
    elif args.goal_mode == "imagegoal":
        command.extend(("--image-goal-path", str(args.image_goal_path)))
    if args.run_mode == "dry-run":
        command.append("--dry-run")
    if args.disable_depth_recovery_scan:
        command.append("--disable-depth-recovery-scan")
    return shell_with_setups(
        (ROS_SETUP, UNITREE_SETUP),
        command,
        no_proxy=f"{args.gpu_ip},127.0.0.1,localhost",
    )


def preflight_command(timeout_s: float) -> list[str]:
    return shell_with_setups(
        (ROS_SETUP, UNITREE_SETUP),
        (
            "/usr/bin/python3",
            str(DEPLOYMENT_ROOT / "scripts/ros2_navdp_preflight.py"),
            "--timeout-s",
            str(timeout_s),
        ),
    )


class ManagedProcess:
    def __init__(self, name: str, command: Sequence[str], log_path: Path) -> None:
        self.name = name
        self.command = list(command)
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log_file = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            self.command,
            cwd=str(DEPLOYMENT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._pump_output, daemon=True)
        self._reader.start()
        print(f"[STACK] started {self.name} pid={self.process.pid} log={self.log_path}", flush=True)

    def _pump_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        assert self._log_file is not None
        for line in self.process.stdout:
            self._log_file.write(line)
            print(f"[{self.name}] {line}", end="", flush=True)

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def stop(self, timeout_s: float = 8.0) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            print(f"[STACK] stopping {self.name} with SIGINT", flush=True)
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                print(f"[STACK] {self.name} did not stop; sending SIGTERM", flush=True)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    print(f"[STACK] {self.name} did not terminate; sending SIGKILL", flush=True)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2.0)
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        if self._log_file is not None:
            self._log_file.close()
        print(f"[STACK] stopped {self.name} exit={process.returncode}", flush=True)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.model_mode in ("x-navdp", "x-navdp-glass") and args.goal_mode != "pointgoal":
        parser.error(f"{args.model_mode} supports only --goal-mode pointgoal")
    if args.goal_mode == "pointgoal":
        if args.goal_x is None or args.goal_y is None:
            parser.error("pointgoal requires --goal-x and --goal-y")
    if args.goal_mode == "imagegoal":
        if args.image_goal_path is None:
            parser.error("imagegoal requires --image-goal-path")
        args.image_goal_path = args.image_goal_path.expanduser().resolve()
        if not args.image_goal_path.is_file():
            parser.error(f"image goal does not exist: {args.image_goal_path}")
    if args.run_mode == "real" and args.components != "sensors" and not args.confirm_real_motion:
        parser.error("real mode requires --confirm-real-motion")
    for name in (
        "max_linear_speed",
        "max_angular_speed",
        "pointgoal_stop_radius",
        "cmd_timeout_sec",
        "sensor_timeout_s",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.server_port <= 65535:
        parser.error("--server-port must be in [1, 65535]")
    if args.run_id is not None and not RUN_ID_PATTERN.fullmatch(args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore, and hyphen")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=Path(__file__).name, description=__doc__)
    parser.add_argument("--components", choices=COMPONENT_GROUPS, default="all")
    parser.add_argument("--run-mode", choices=RUN_MODES, default="dry-run")
    parser.add_argument("--confirm-real-motion", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu-ip", default="192.168.8.251")
    parser.add_argument("--server-port", type=int, default=5802)
    parser.add_argument("--model-mode", choices=MODEL_MODES, default="navdp")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="pointgoal")
    parser.add_argument("--goal-x", type=float, default=3.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--image-goal-path", type=Path)
    parser.add_argument("--pointgoal-stop-radius", type=float, default=0.5)
    parser.add_argument("--max-linear-speed", type=float, default=0.20)
    parser.add_argument("--max-angular-speed", type=float, default=0.50)
    parser.add_argument(
        "--mpc-tracking-profile",
        choices=("legacy", "smooth_continuous"),
        default="smooth_continuous",
    )
    parser.add_argument("--cmd-timeout-sec", type=float, default=0.50)
    parser.add_argument("--sensor-timeout-s", type=float, default=30.0)
    parser.add_argument("--disable-depth-recovery-scan", action="store_true")
    args = parser.parse_args(argv)
    args.server_url = f"http://{args.gpu_ip}:{args.server_port}/eval_navdp"
    validate_args(parser, args)
    return args


def resolve_run_id(args: argparse.Namespace) -> str:
    latest = LOG_ROOT / ".latest_run_id"
    if args.run_id:
        run_id = args.run_id
    elif args.components == "navigation" and latest.is_file():
        run_id = latest.read_text(encoding="utf-8").strip()
    else:
        run_id = f"{args.model_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not run_id or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    latest.write_text(run_id + "\n", encoding="utf-8")
    return run_id


def require_runtime_files(include_sensors: bool, include_navigation: bool) -> None:
    required = [ROS_SETUP, UNITREE_SETUP]
    if include_sensors:
        required.append(REALSENSE_SETUP)
    if include_navigation:
        required.extend(
            (
                DEPLOYMENT_ROOT / "scripts/http_internvla_client_qos_isolated.py",
                DEPLOYMENT_ROOT / "scripts/ros2_cmd_vel_sport_safe_bridge.py",
            )
        )
    required.extend(
        (
            DEPLOYMENT_ROOT / "scripts/ros2_navdp_input_bridge.py",
            DEPLOYMENT_ROOT / "scripts/ros2_navdp_preflight.py",
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing runtime files: " + ", ".join(missing))


def acquire_component_locks(names: Sequence[str]) -> list[Any]:
    handles = []
    for name in sorted(names):
        path = LOG_ROOT / f".go2_navdp_{name}.lock"
        handle = path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            for acquired in handles:
                acquired.close()
            raise RuntimeError(f"another {name} stack is already running ({path})") from exc
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        handles.append(handle)
    return handles


def check_server_health(args: argparse.Namespace) -> dict[str, Any]:
    url = f"http://{args.gpu_ip}:{args.server_port}/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"model server health check failed: {url}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(f"model server is not healthy: {payload!r}")
    if payload.get("model_mode") != args.model_mode:
        raise RuntimeError(
            f"model mismatch: terminal C={payload.get('model_mode')!r}, "
            f"Go2 launcher={args.model_mode!r}"
        )
    print(f"[STACK] server health ok: {json.dumps(payload, ensure_ascii=False)}", flush=True)
    return payload


def run_sensor_preflight(timeout_s: float, log_path: Path) -> None:
    print(f"[STACK] waiting for live RGB/depth/CameraInfo/odom (timeout={timeout_s:.1f}s)", flush=True)
    completed = subprocess.run(
        preflight_command(timeout_s),
        cwd=str(DEPLOYMENT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout_s + 10.0,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"sensor preflight failed with exit={completed.returncode}")


def write_stack_config(path: Path, args: argparse.Namespace, run_id: str) -> None:
    payload = {
        "run_id": run_id,
        "components": args.components,
        "run_mode": args.run_mode,
        "model_mode": args.model_mode,
        "goal_mode": args.goal_mode,
        "goal_x": args.goal_x,
        "goal_y": args.goal_y,
        "image_goal_path": None if args.image_goal_path is None else str(args.image_goal_path),
        "gpu_ip": args.gpu_ip,
        "server_port": args.server_port,
        "max_linear_speed": args.max_linear_speed,
        "max_angular_speed": args.max_angular_speed,
        "pointgoal_stop_radius": args.pointgoal_stop_radius,
        "cmd_timeout_sec": args.cmd_timeout_sec,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    include_sensors = args.components in ("all", "sensors")
    include_navigation = args.components in ("all", "navigation")
    locks: list[Any] = []
    try:
        require_runtime_files(include_sensors, include_navigation)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        locks = acquire_component_locks(
            (["sensors"] if include_sensors else [])
            + (["navigation"] if include_navigation else [])
        )
        run_id = resolve_run_id(args)
        run_dir = LOG_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ros_log_dir = run_dir / "ros"
        ros_log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ROS_LOG_DIR"] = str(ros_log_dir)
        write_stack_config(
            run_dir / f"stack_{args.components}_{args.run_mode}.json",
            args,
            run_id,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        for handle in locks:
            handle.close()
        print(f"[STACK ERROR] {exc}", flush=True)
        return 1

    stop_event = threading.Event()
    processes: dict[str, ManagedProcess] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        print(f"[STACK] received {signal.Signals(signum).name}; beginning safe shutdown", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        print(
            f"[STACK] run_id={run_id} components={args.components} mode={args.run_mode} "
            f"model={args.model_mode} goal={args.goal_mode}",
            flush=True,
        )
        if include_navigation:
            check_server_health(args)

        if include_sensors:
            processes["A"] = ManagedProcess(
                "A", realsense_command(), run_dir / "terminal_A_realsense.log"
            )
            processes["A"].start()
            time.sleep(2.0)
            if processes["A"].poll() is not None:
                raise RuntimeError("RealSense process exited during startup")
            processes["B"] = ManagedProcess(
                "B", input_bridge_command(), run_dir / "terminal_B_input_bridge.log"
            )
            processes["B"].start()

        run_sensor_preflight(args.sensor_timeout_s, run_dir / "sensor_preflight.log")

        if include_navigation and args.run_mode == "real":
            processes["D"] = ManagedProcess(
                "D", safety_bridge_command(args), run_dir / "terminal_D_real_safety_bridge.log"
            )
            processes["D"].start()
            time.sleep(1.0)
            if processes["D"].poll() is not None:
                raise RuntimeError("sport safety bridge exited during startup")

        if include_navigation:
            response_path = run_dir / f"responses_{args.run_mode.replace('-', '_')}.jsonl"
            processes["E"] = ManagedProcess(
                "E", client_command(args, response_path), run_dir / f"terminal_E_{args.run_mode}.log"
            )
            processes["E"].start()

        print("[STACK] startup complete; press Ctrl+C once for ordered safe shutdown", flush=True)
        while not stop_event.wait(0.25):
            for name, process in processes.items():
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"component {name} exited unexpectedly: {exit_code}"
                    )
    except (FileNotFoundError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"[STACK ERROR] {exc}", flush=True)
        return_code = 1
    else:
        return_code = 0
    finally:
        # E stops first. D remains alive long enough to observe zero/stale cmd,
        # then its own finally block publishes the independent StopMove request.
        if "E" in processes:
            processes["E"].stop()
        if "D" in processes:
            time.sleep(args.cmd_timeout_sec + 0.25)
            processes["D"].stop()
        if "B" in processes:
            processes["B"].stop()
        if "A" in processes:
            processes["A"].stop()
        for handle in locks:
            handle.close()
        print("[STACK] shutdown complete", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
