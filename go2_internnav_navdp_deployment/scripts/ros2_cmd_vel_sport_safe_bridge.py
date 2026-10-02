#!/usr/bin/python3
"""Dry-run-first /cmd_vel_bridge to Unitree Sport request safety bridge."""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from go2_internnav.ros2.navigation_status import (  # noqa: E402
    NAVIGATION_STATUS_TOPIC,
    decode_navigation_status,
    effective_stop_cause,
)

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

try:
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
except ImportError:  # ROS 2 Foxy compatibility.
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
    from rclpy.qos import QoSHistoryPolicy as HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import QoSReliabilityPolicy as ReliabilityPolicy

from unitree_api.msg import Request
from unitree_go.msg import WirelessController


ROBOT_SPORT_API_ID_STOPMOVE = 1003
ROBOT_SPORT_API_ID_MOVE = 1008
STICK_CHANGE_THRESHOLD = 0.05


@dataclass(frozen=True)
class WirelessState:
    lx: float
    ly: float
    rx: float
    ry: float
    keys: int


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def wireless_state_from_msg(msg: WirelessController) -> WirelessState:
    return WirelessState(
        lx=float(msg.lx),
        ly=float(msg.ly),
        rx=float(msg.rx),
        ry=float(msg.ry),
        keys=int(msg.keys),
    )


def wireless_state_changed(initial: WirelessState, current: WirelessState) -> bool:
    if current.keys != initial.keys:
        return True

    return (
        abs(current.lx - initial.lx) > STICK_CHANGE_THRESHOLD
        or abs(current.ly - initial.ly) > STICK_CHANGE_THRESHOLD
        or abs(current.rx - initial.rx) > STICK_CHANGE_THRESHOLD
        or abs(current.ry - initial.ry) > STICK_CHANGE_THRESHOLD
    )


def build_move_request(vx: float, wz: float) -> Request:
    req = Request()
    req.header.identity.api_id = ROBOT_SPORT_API_ID_MOVE
    req.parameter = json.dumps(
        {"x": float(vx), "y": 0.0, "z": float(wz)},
        separators=(",", ":"),
    )
    return req


def build_stopmove_request() -> Request:
    req = Request()
    req.header.identity.api_id = ROBOT_SPORT_API_ID_STOPMOVE
    return req


class CmdVelSportSafeBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("cmd_vel_sport_safe_bridge")
        self.cmd_topic = args.cmd_topic
        self.wireless_topic = args.wireless_topic
        self.sport_request_topic = args.sport_request_topic
        self.navigation_status_topic = args.navigation_status_topic
        self.max_vx = args.max_vx
        self.max_wz = args.max_wz
        self.cmd_timeout_sec = args.cmd_timeout_sec
        self.dry_run = args.dry_run
        self.enable_sport_publish = args.enable_sport_publish
        self.understand_motion_risk = args.i_understand_this_can_move_the_robot
        self.diagnostic_no_sport_publish = args.diagnostic_no_sport_publish

        self.started_at = time.monotonic()
        self.cmd_count = 0
        self.move_publish_count = 0
        self.stopmove_triggers_handled = set()
        self.last_cmd_at: Optional[float] = None
        self.raw_vx = 0.0
        self.raw_wz = 0.0
        self.limited_vx = 0.0
        self.limited_wz = 0.0
        self.manual_override_latched = False
        self.initial_wireless_state: Optional[WirelessState] = None
        self.last_action = "StopMove"
        self.navigation_state = "unknown"
        self.navigation_reason = "no_status_received"
        self.navigation_distance_to_goal_m: Optional[float] = None
        self.navigation_status_received_at: Optional[float] = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.cmd_subscription = self.create_subscription(
            Twist,
            self.cmd_topic,
            self._on_cmd_vel,
            qos,
        )
        self.wireless_subscription = self.create_subscription(
            WirelessController,
            self.wireless_topic,
            self._on_wireless_controller,
            qos,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.navigation_status_subscription = self.create_subscription(
            String,
            self.navigation_status_topic,
            self._on_navigation_status,
            status_qos,
        )

        self.sport_publisher = None
        if self._should_create_sport_publisher():
            self.sport_publisher = self.create_publisher(
                Request,
                self.sport_request_topic,
                10,
            )
            self.get_logger().warning(
                "real sport publishing ENABLED on %s; manual latch and timeout "
                "gates remain active"
                % self.sport_request_topic
            )
        elif self.diagnostic_no_sport_publish:
            self.get_logger().warning(
                "DIAGNOSTIC NO SPORT PUBLISH: no publisher will be created for %s"
                % self.sport_request_topic
            )
        else:
            self.get_logger().warning(
                "DRY-RUN SAFE MODE: no publisher will be created for %s"
                % self.sport_request_topic
            )

        self.get_logger().info(
            "subscribed cmd=%s wireless=%s navigation_status=%s "
            "max_vx=%.3f max_wz=%.3f "
            "cmd_timeout_sec=%.3f diagnostic_no_sport_publish=%s"
            % (
                self.cmd_topic,
                self.wireless_topic,
                self.navigation_status_topic,
                self.max_vx,
                self.max_wz,
                self.cmd_timeout_sec,
                self.diagnostic_no_sport_publish,
            )
        )
        self.create_timer(1.0, self._print_status)

    def _dangerous_publish_mode_requested(self) -> bool:
        return (
            self.enable_sport_publish
            and self.understand_motion_risk
            and not self.dry_run
        )

    def _should_create_sport_publisher(self) -> bool:
        return (
            self._dangerous_publish_mode_requested()
            and not self.diagnostic_no_sport_publish
        )

    def _cmd_timed_out(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.monotonic()
        if self.last_cmd_at is None:
            return now - self.started_at > self.cmd_timeout_sec
        return now - self.last_cmd_at > self.cmd_timeout_sec

    def _sport_publish_block_reasons(
        self,
        now: Optional[float] = None,
        include_motion_safety_gates: bool = True,
    ) -> List[str]:
        reasons: List[str] = []
        if self.sport_publisher is None:
            reasons.append("publisher_missing")
        if not self._dangerous_publish_mode_requested():
            reasons.append("dangerous_mode_false")
        if self.dry_run:
            reasons.append("dry_run_true")
        if not self.enable_sport_publish:
            reasons.append("enable_sport_publish_false")
        if not self.understand_motion_risk:
            reasons.append("understand_motion_risk_false")
        if self.diagnostic_no_sport_publish:
            reasons.append("diagnostic_no_sport_publish_true")

        if include_motion_safety_gates:
            if self.manual_override_latched:
                reasons.append("manual_latch_true")
            if self._cmd_timed_out(now):
                reasons.append("cmd_timeout_true")
        return reasons

    def _can_publish_sport_request(self, now: Optional[float] = None) -> bool:
        return not self._sport_publish_block_reasons(
            now,
            include_motion_safety_gates=True,
        )

    def _can_publish_stopmove_request(self, now: Optional[float] = None) -> bool:
        return not self._sport_publish_block_reasons(
            now,
            include_motion_safety_gates=False,
        )

    def _request_details(self, req: Request) -> str:
        return "Request(api_id=%d, parameter=%r)" % (
            int(req.header.identity.api_id),
            req.parameter,
        )

    def _log_publish_blocked(
        self,
        action: str,
        trigger: str,
        req: Request,
        reasons: List[str],
    ) -> None:
        self.get_logger().warning(
            "PUBLISH_BLOCKED action=%s trigger=%s reasons=%s %s; no publish"
            % (
                action,
                trigger,
                ",".join(reasons) if reasons else "none",
                self._request_details(req),
            )
        )

    def _publish_sport_request(self, action: str, trigger: str, req: Request) -> None:
        details = self._request_details(req)
        self.get_logger().info(
            "PUBLISH_ATTEMPT action=%s trigger=%s %s"
            % (action, trigger, details)
        )
        self.sport_publisher.publish(req)
        self.get_logger().info(
            "PUBLISH_DONE action=%s trigger=%s %s"
            % (action, trigger, details)
        )
        if action == "Move":
            self.move_publish_count += 1

    def _publish_stopmove_if_needed(
        self,
        trigger: str,
        now: Optional[float] = None,
    ) -> None:
        if trigger in self.stopmove_triggers_handled:
            return
        if self.cmd_count == 0:
            return

        req = build_stopmove_request()
        block_reasons = self._sport_publish_block_reasons(
            now,
            include_motion_safety_gates=False,
        )
        log_reasons = list(block_reasons)
        if trigger == "manual_latch" and self.manual_override_latched:
            log_reasons.append("manual_latch_true")
        if trigger == "cmd_timeout" and self._cmd_timed_out(now):
            log_reasons.append("cmd_timeout_true")
        if self._should_create_sport_publisher() and self.move_publish_count == 0:
            block_reasons.append("no_prior_move_publish")
            log_reasons.append("no_prior_move_publish")

        if block_reasons:
            self._log_publish_blocked("StopMove", trigger, req, log_reasons)
        else:
            self._publish_sport_request("StopMove", trigger, req)
        self.stopmove_triggers_handled.add(trigger)

    def _on_cmd_vel(self, msg: Twist) -> None:
        now = time.monotonic()
        self.cmd_count += 1
        self.stopmove_triggers_handled.discard("cmd_timeout")
        self.last_cmd_at = now
        self.raw_vx = float(msg.linear.x)
        self.raw_wz = float(msg.angular.z)
        self.limited_vx = clamp(self.raw_vx, self.max_vx)
        self.limited_wz = clamp(self.raw_wz, self.max_wz)
        self.last_action = "Move"

        req = build_move_request(self.limited_vx, self.limited_wz)
        if self._can_publish_sport_request(now):
            self._publish_sport_request("Move", "cmd_vel", req)
        else:
            self._log_publish_blocked(
                "Move",
                "cmd_vel",
                req,
                self._sport_publish_block_reasons(
                    now,
                    include_motion_safety_gates=True,
                ),
            )

    def _on_wireless_controller(self, msg: WirelessController) -> None:
        now = time.monotonic()
        current = wireless_state_from_msg(msg)
        if self.initial_wireless_state is None:
            self.initial_wireless_state = current
            self.get_logger().info(
                "wireless controller baseline captured: lx=%.3f ly=%.3f "
                "rx=%.3f ry=%.3f keys=%d"
                % (current.lx, current.ly, current.rx, current.ry, current.keys)
            )
            return

        if self.manual_override_latched:
            return

        if wireless_state_changed(self.initial_wireless_state, current):
            self.manual_override_latched = True
            self.last_action = "StopMove"
            self.get_logger().warning("[SAFETY LATCH] manual override detected")
            self._publish_stopmove_if_needed("manual_latch", now)

    def _on_navigation_status(self, msg: String) -> None:
        status = decode_navigation_status(msg.data)
        previous_state = self.navigation_state
        self.navigation_state = str(status["state"])
        self.navigation_reason = str(status["reason"])
        self.navigation_status_received_at = time.monotonic()
        try:
            distance = status.get("distance_to_goal_m")
            self.navigation_distance_to_goal_m = (
                None if distance is None else float(distance)
            )
        except (TypeError, ValueError):
            self.navigation_distance_to_goal_m = None

        if self.navigation_state == "goal_reached" and previous_state != "goal_reached":
            distance_text = (
                "unknown"
                if self.navigation_distance_to_goal_m is None
                else "%.3f" % self.navigation_distance_to_goal_m
            )
            self.get_logger().warning(
                "[GOAL REACHED] navigation_state=goal_reached "
                "stop_reason=%s distance_to_goal_m=%s; StopMove is expected"
                % (self.navigation_reason, distance_text)
            )
        elif self.navigation_state != previous_state:
            self.get_logger().info(
                "[NAVIGATION STATE] navigation_state=%s reason=%s"
                % (self.navigation_state, self.navigation_reason)
            )

    def _print_status(self) -> None:
        now = time.monotonic()
        timeout = self._cmd_timed_out(now)
        action = "StopMove" if timeout or self.manual_override_latched else "Move"
        self.last_action = action
        mode = "DRY RUN" if self.dry_run else "REAL MODE"
        if self.diagnostic_no_sport_publish:
            mode = "DIAGNOSTIC NO SPORT PUBLISH"
        wireless_baseline = self.initial_wireless_state is not None
        stop_cause = effective_stop_cause(
            manual_override_latched=self.manual_override_latched,
            cmd_timed_out=timeout,
            navigation_state=self.navigation_state,
            navigation_reason=self.navigation_reason,
        )

        if self.last_cmd_at is None:
            cmd_age = "never"
        else:
            cmd_age = "%.2fs" % (now - self.last_cmd_at)
        if self.navigation_status_received_at is None:
            navigation_status_age = "never"
        else:
            navigation_status_age = "%.2fs" % (
                now - self.navigation_status_received_at
            )

        self.get_logger().info(
            "[%s STATUS] cmd_count=%d cmd_age=%s raw_vx=%.3f raw_wz=%.3f "
            "limited_vx=%.3f limited_wz=%.3f dry_run=%s "
            "enable_sport_publish=%s manual_latch=%s timeout=%s "
            "wireless_baseline=%s action=%s navigation_state=%s "
            "navigation_reason=%s stop_cause=%s navigation_status_age=%s%s"
            % (
                mode,
                self.cmd_count,
                cmd_age,
                self.raw_vx,
                self.raw_wz,
                self.limited_vx,
                self.limited_wz,
                self.dry_run,
                self.enable_sport_publish,
                self.manual_override_latched,
                timeout,
                wireless_baseline,
                action,
                self.navigation_state,
                self.navigation_reason,
                stop_cause if action == "StopMove" else "none",
                navigation_status_age,
                "(vx=%.3f, y=0.000, wz=%.3f)"
                % (self.limited_vx, self.limited_wz)
                if action == "Move"
                else "",
            )
        )

        if timeout:
            self._publish_stopmove_if_needed("cmd_timeout", now)

    def shutdown_safely(self) -> None:
        self._publish_stopmove_if_needed("shutdown", time.monotonic())


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Safe dry-run-first bridge from geometry_msgs/Twist on "
            "/cmd_vel_bridge to Unitree Sport Request format."
        )
    )
    parser.add_argument("--cmd-topic", default="/cmd_vel_bridge")
    parser.add_argument("--wireless-topic", default="/wirelesscontroller")
    parser.add_argument("--sport-request-topic", default="/api/sport/request")
    parser.add_argument(
        "--navigation-status-topic",
        default=NAVIGATION_STATUS_TOPIC,
        help="Planner semantic-state topic used to explain StopMove causes.",
    )
    parser.add_argument("--max-vx", type=nonnegative_float, default=0.50)
    parser.add_argument("--max-wz", type=nonnegative_float, default=0.10)
    parser.add_argument("--cmd-timeout-sec", type=positive_float, default=0.50)
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument("--dry-run", dest="dry_run", action="store_true")
    dry_run_group.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--enable-sport-publish", action="store_true", default=False)
    parser.add_argument(
        "--diagnostic-no-sport-publish",
        action="store_true",
        default=False,
        help=(
            "Construct and log exact Unitree Sport Request messages, but never "
            "publish /api/sport/request."
        ),
    )
    parser.add_argument(
        "--i-understand-this-can-move-the-robot",
        action="store_true",
        default=False,
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args, ros_args = parse_args(argv)

    rclpy.init(args=ros_args)
    node = CmdVelSportSafeBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_safely()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
