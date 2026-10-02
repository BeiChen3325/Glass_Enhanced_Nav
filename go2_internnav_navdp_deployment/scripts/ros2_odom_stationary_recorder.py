#!/usr/bin/python3
"""Record and summarize odometry drift without publishing any ROS messages."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

try:
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
except ImportError:  # ROS 2 Foxy compatibility.
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
    from rclpy.qos import QoSHistoryPolicy as HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import QoSReliabilityPolicy as ReliabilityPolicy


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_delta(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


class OdomStationaryRecorder(Node):
    def __init__(self, topics: Sequence[str], duration_s: float) -> None:
        super().__init__("odom_stationary_recorder")
        self.topics = list(topics)
        self.duration_s = duration_s
        self.started_monotonic = time.monotonic()
        self.started_unix = time.time()
        self.rows: List[Dict[str, object]] = []
        self.counts = {topic: 0 for topic in self.topics}

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Keep explicit references so rclpy does not garbage-collect subscriptions.
        # Do not use ``subscriptions``: Node exposes that name as a read-only
        # property in ROS 2 Foxy.
        self._odom_subscriptions = [
            self.create_subscription(Odometry, topic, self._callback(topic), qos)
            for topic in self.topics
        ]
        self.get_logger().warning(
            "READ-ONLY RECORDER: no publishers are created and no motion commands are sent"
        )
        self.get_logger().info(
            "recording topics=%s duration_s=%.1f" % (",".join(self.topics), duration_s)
        )

    def _callback(self, topic: str):
        def callback(msg: Odometry) -> None:
            pose = msg.pose.pose
            orientation = pose.orientation
            now_unix = time.time()
            self.counts[topic] += 1
            self.rows.append(
                {
                    "topic": topic,
                    "sample_index": self.counts[topic] - 1,
                    "received_unix_s": now_unix,
                    "elapsed_s": now_unix - self.started_unix,
                    "header_sec": int(msg.header.stamp.sec),
                    "header_nanosec": int(msg.header.stamp.nanosec),
                    "frame_id": msg.header.frame_id,
                    "child_frame_id": msg.child_frame_id,
                    "x_m": float(pose.position.x),
                    "y_m": float(pose.position.y),
                    "z_m": float(pose.position.z),
                    "qx": float(orientation.x),
                    "qy": float(orientation.y),
                    "qz": float(orientation.z),
                    "qw": float(orientation.w),
                    "yaw_rad": quaternion_yaw(
                        float(orientation.x),
                        float(orientation.y),
                        float(orientation.z),
                        float(orientation.w),
                    ),
                    "linear_x_mps": float(msg.twist.twist.linear.x),
                    "linear_y_mps": float(msg.twist.twist.linear.y),
                    "angular_z_radps": float(msg.twist.twist.angular.z),
                }
            )

        return callback

def summarize(rows: Sequence[Dict[str, object]], topics: Sequence[str]) -> Dict[str, object]:
    result: Dict[str, object] = {"topics": {}}
    for topic in topics:
        samples = [row for row in rows if row["topic"] == topic]
        if not samples:
            result["topics"][topic] = {"sample_count": 0, "status": "no_data"}
            continue

        first = samples[0]
        last = samples[-1]
        planar_from_start = [
            math.hypot(float(row["x_m"]) - float(first["x_m"]), float(row["y_m"]) - float(first["y_m"]))
            for row in samples
        ]
        yaw_from_start = [
            abs(angle_delta(float(row["yaw_rad"]), float(first["yaw_rad"])))
            for row in samples
        ]
        position_steps = [
            math.hypot(
                float(current["x_m"]) - float(previous["x_m"]),
                float(current["y_m"]) - float(previous["y_m"]),
            )
            for previous, current in zip(samples, samples[1:])
        ]
        yaw_steps = [
            abs(angle_delta(float(current["yaw_rad"]), float(previous["yaw_rad"])))
            for previous, current in zip(samples, samples[1:])
        ]
        elapsed = max(float(last["received_unix_s"]) - float(first["received_unix_s"]), 0.0)
        result["topics"][topic] = {
            "sample_count": len(samples),
            "observed_duration_s": elapsed,
            "mean_rate_hz": (len(samples) - 1) / elapsed if elapsed > 0.0 else None,
            "start": {"x_m": first["x_m"], "y_m": first["y_m"], "yaw_rad": first["yaw_rad"]},
            "end": {"x_m": last["x_m"], "y_m": last["y_m"], "yaw_rad": last["yaw_rad"]},
            "end_position_delta_m": planar_from_start[-1],
            "max_position_drift_m": max(planar_from_start),
            "end_yaw_delta_deg": math.degrees(yaw_from_start[-1]),
            "max_yaw_drift_deg": math.degrees(max(yaw_from_start)),
            "max_position_step_m": max(position_steps, default=0.0),
            "max_yaw_step_deg": math.degrees(max(yaw_steps, default=0.0)),
            "frame_ids": sorted({str(row["frame_id"]) for row in samples}),
            "child_frame_ids": sorted({str(row["child_frame_id"]) for row in samples}),
            "status": "ok",
        }
    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["/utlidar/robot_odom", "/odom_bridge"],
    )
    args = parser.parse_args(argv)
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    return args


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = OdomStationaryRecorder(args.topics, args.duration)
    try:
        deadline = time.monotonic() + args.duration
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        rows = list(node.rows)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    csv_path = args.output_dir / "odom_samples.csv"
    summary_path = args.output_dir / "odom_summary.json"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows, args.topics)
    summary.update({"duration_requested_s": args.duration, "csv_path": str(csv_path)})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("wrote %s and %s" % (csv_path, summary_path), flush=True)
    return 0 if all(summary["topics"][topic]["status"] == "ok" for topic in args.topics) else 2


if __name__ == "__main__":
    raise SystemExit(main())
