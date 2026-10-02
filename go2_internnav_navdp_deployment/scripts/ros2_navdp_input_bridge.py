#!/usr/bin/python3
"""Input-only topic relay for NavDP dry-run sensor inputs."""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Type

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

try:
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
except ImportError:  # ROS 2 Foxy exposes the same enum values with QoS* names.
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
    from rclpy.qos import QoSHistoryPolicy as HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import QoSReliabilityPolicy as ReliabilityPolicy

from sensor_msgs.msg import CameraInfo, Image


@dataclass(frozen=True)
class RelaySpec:
    name: str
    input_topic: str
    output_topic: str
    msg_type: Type


class NavDPInputBridge(Node):
    def __init__(self, specs: Sequence[RelaySpec]) -> None:
        super().__init__("navdp_input_bridge")
        self.specs = list(specs)
        self.started_at = time.monotonic()
        self.input_counts: Dict[str, int] = {}
        self.output_counts: Dict[str, int] = {}
        self.last_received_at: Dict[str, float] = {}
        self._publishers = {}
        self._subscriptions = []

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        for spec in self.specs:
            self.input_counts[spec.name] = 0
            self.output_counts[spec.name] = 0
            publisher = self.create_publisher(spec.msg_type, spec.output_topic, sensor_qos)
            self._publishers[spec.name] = publisher
            subscription = self.create_subscription(
                spec.msg_type,
                spec.input_topic,
                self._make_relay_callback(spec),
                sensor_qos,
            )
            self._subscriptions.append(subscription)
            self.get_logger().info(
                "relay %s: %s -> %s"
                % (spec.name, spec.input_topic, spec.output_topic)
            )

        self.get_logger().info(
            "QoS: KEEP_LAST depth=10 BEST_EFFORT VOLATILE for all input and output topics"
        )
        self.create_timer(5.0, self._print_stats)

    def _make_relay_callback(self, spec: RelaySpec):
        def relay_callback(msg) -> None:
            self.input_counts[spec.name] += 1
            self.last_received_at[spec.name] = time.monotonic()
            self._publishers[spec.name].publish(msg)
            self.output_counts[spec.name] += 1

        return relay_callback

    def _print_stats(self) -> None:
        now = time.monotonic()
        lines = ["NavDP input bridge stats:"]
        stale_specs = []

        for spec in self.specs:
            last_at = self.last_received_at.get(spec.name)
            if last_at is None:
                age_text = "never"
                is_stale = now - self.started_at > 3.0
            else:
                age = now - last_at
                age_text = "%.2fs ago" % age
                is_stale = age > 3.0

            lines.append(
                "- %s: input=%d output=%d last=%s"
                % (
                    spec.input_topic,
                    self.input_counts[spec.name],
                    self.output_counts[spec.name],
                    age_text,
                )
            )
            lines.append("  output_topic=%s" % spec.output_topic)

            if is_stale:
                stale_specs.append((spec, age_text))

        self.get_logger().info("\n".join(lines))

        for spec, age_text in stale_specs:
            self.get_logger().warning(
                "no input from %s for more than 3 seconds; last=%s"
                % (spec.input_topic, age_text)
            )


def parse_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Relay RealSense and odom input topics to NavDP default topics."
    )
    parser.add_argument("--rgb-in", default="/camera/color/image_raw")
    parser.add_argument("--depth-in", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--rgb-info-in", default="/camera/color/camera_info")
    parser.add_argument(
        "--depth-info-in", default="/camera/aligned_depth_to_color/camera_info"
    )
    parser.add_argument("--odom-in", default="/utlidar/robot_odom")
    parser.add_argument("--rgb-out", default="/camera/camera/color/image_raw")
    parser.add_argument(
        "--depth-out", default="/camera/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--rgb-info-out", default="/camera/camera/color/camera_info")
    parser.add_argument(
        "--depth-info-out", default="/camera/camera/aligned_depth_to_color/camera_info"
    )
    parser.add_argument("--odom-out", default="/odom_bridge")
    return parser.parse_known_args(argv)


def build_specs(args: argparse.Namespace) -> List[RelaySpec]:
    return [
        RelaySpec("rgb", args.rgb_in, args.rgb_out, Image),
        RelaySpec("depth", args.depth_in, args.depth_out, Image),
        RelaySpec("rgb_info", args.rgb_info_in, args.rgb_info_out, CameraInfo),
        RelaySpec("depth_info", args.depth_info_in, args.depth_info_out, CameraInfo),
        RelaySpec("odom", args.odom_in, args.odom_out, Odometry),
    ]


def main(argv: Sequence[str] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args, ros_args = parse_args(argv)

    rclpy.init(args=ros_args)
    node = NavDPInputBridge(build_specs(args))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
