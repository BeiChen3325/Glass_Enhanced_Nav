#!/usr/bin/env python3
"""Wait for one valid RGB, aligned-depth, CameraInfo, and odom sample."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

try:
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
except ImportError:
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
    from rclpy.qos import QoSHistoryPolicy as HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import QoSReliabilityPolicy as ReliabilityPolicy


class NavDPPreflight(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("navdp_stack_preflight")
        self.messages: dict[str, Any] = {}
        self.counts = {"rgb": 0, "depth": 0, "camera_info": 0, "odom": 0}
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Keep explicit references without shadowing Foxy's read-only
        # ``Node.subscriptions`` property.
        self._subscription_handles = (
            self.create_subscription(Image, args.rgb_topic, self._callback("rgb"), qos),
            self.create_subscription(Image, args.depth_topic, self._callback("depth"), qos),
            self.create_subscription(
                CameraInfo,
                args.camera_info_topic,
                self._callback("camera_info"),
                qos,
            ),
            self.create_subscription(Odometry, args.odom_topic, self._callback("odom"), qos),
        )

    def _callback(self, name: str):
        def store(message: Any) -> None:
            self.messages[name] = message
            self.counts[name] += 1

        return store

    @property
    def ready(self) -> bool:
        return all(name in self.messages for name in self.counts)

    def validate(self) -> dict[str, Any]:
        rgb = self.messages["rgb"]
        depth = self.messages["depth"]
        camera_info = self.messages["camera_info"]
        odom = self.messages["odom"]
        if rgb.width <= 0 or rgb.height <= 0 or depth.width <= 0 or depth.height <= 0:
            raise ValueError("RGB/depth dimensions must be positive")
        if (rgb.width, rgb.height) != (depth.width, depth.height):
            raise ValueError(
                f"RGB/depth dimensions differ: RGB={rgb.width}x{rgb.height}, "
                f"depth={depth.width}x{depth.height}"
            )
        if len(camera_info.k) != 9 or not all(math.isfinite(float(v)) for v in camera_info.k):
            raise ValueError("CameraInfo.k must contain nine finite values")
        if float(camera_info.k[0]) <= 0.0 or float(camera_info.k[4]) <= 0.0:
            raise ValueError("CameraInfo focal lengths must be positive")
        pose = odom.pose.pose
        odom_values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(float(value)) for value in odom_values):
            raise ValueError("odom pose contains non-finite values")
        return {
            "status": "ok",
            "counts": self.counts,
            "rgb": {"width": rgb.width, "height": rgb.height, "encoding": rgb.encoding},
            "depth": {
                "width": depth.width,
                "height": depth.height,
                "encoding": depth.encoding,
            },
            "camera_info": {"fx": camera_info.k[0], "fy": camera_info.k[4]},
            "odom": {"x": pose.position.x, "y": pose.position.y},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument(
        "--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument(
        "--camera-info-topic", default="/camera/camera/color/camera_info"
    )
    parser.add_argument("--odom-topic", default="/odom_bridge")
    args = parser.parse_args()
    if args.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")
    return args


def main() -> int:
    args = parse_args()
    node = None
    try:
        rclpy.init()
        node = NavDPPreflight(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"ROS preflight initialization failed: {type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if rclpy.ok():
            rclpy.shutdown()
        return 2
    deadline = time.monotonic() + args.timeout_s
    try:
        while rclpy.ok() and time.monotonic() < deadline and not node.ready:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.ready:
            missing = [name for name in node.counts if name not in node.messages]
            print(
                json.dumps(
                    {"status": "error", "missing": missing, "counts": node.counts},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 1
        try:
            summary = node.validate()
        except ValueError as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
