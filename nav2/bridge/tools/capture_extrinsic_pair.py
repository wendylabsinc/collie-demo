#!/usr/bin/env python3
"""Capture stationary Hesai and body-frame Go2 point clouds.

This utility is deliberately read-only. It subscribes to the raw ROS graph,
does not publish any topic, and has no Unitree command dependency. Woof must
remain stationary for the complete capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def cloud_xyz(message: PointCloud2) -> np.ndarray:
    points = point_cloud2.read_points_numpy(
        message,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    if points.dtype.names:
        points = np.column_stack(
            [points[name].reshape(-1) for name in ("x", "y", "z")]
        )
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def filter_cloud(
    points: np.ndarray,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    maximum_points: int,
) -> np.ndarray:
    radius = np.linalg.norm(points, axis=1)
    points = points[
        (radius >= minimum_range_m)
        & (radius <= maximum_range_m)
        & (np.abs(points[:, 2]) <= maximum_range_m)
    ]
    if points.shape[0] > maximum_points:
        # A fixed stride preserves the complete azimuth better than truncation
        # and makes repeated captures reproducible.
        indices = np.linspace(
            0,
            points.shape[0] - 1,
            maximum_points,
            dtype=np.int64,
        )
        points = points[indices]
    return points


class StationaryPairCapture(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("collie_stationary_extrinsic_capture")
        self.args = args
        self.hesai_frames: list[np.ndarray] = []
        self.base_frames: list[np.ndarray] = []
        self.hesai_frame_id = ""
        self.base_frame_id = ""
        self.hesai_stamps: list[float] = []
        self.base_stamps: list[float] = []
        self.odom_samples: list[tuple[float, float, float]] = []
        self.capture_started = time.monotonic()
        self.create_subscription(
            PointCloud2,
            args.hesai_topic,
            self._on_hesai,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            args.base_topic,
            self._on_base,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            args.odom_topic,
            self._on_odom,
            qos_profile_sensor_data,
        )

    @staticmethod
    def _stamp_seconds(message: PointCloud2) -> float:
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )

    def _accept(
        self,
        message: PointCloud2,
        frames: list[np.ndarray],
        stamps: list[float],
    ) -> None:
        if len(frames) >= self.args.frames:
            return
        points = filter_cloud(
            cloud_xyz(message),
            minimum_range_m=self.args.minimum_range,
            maximum_range_m=self.args.maximum_range,
            maximum_points=self.args.points_per_frame,
        )
        if points.shape[0] < self.args.minimum_points:
            self.get_logger().warning(
                f"Ignoring sparse {message.header.frame_id!r} cloud with "
                f"{points.shape[0]} usable points"
            )
            return
        frames.append(points)
        stamps.append(self._stamp_seconds(message))

    def _on_hesai(self, message: PointCloud2) -> None:
        self.hesai_frame_id = message.header.frame_id
        self._accept(message, self.hesai_frames, self.hesai_stamps)

    def _on_base(self, message: PointCloud2) -> None:
        self.base_frame_id = message.header.frame_id
        self._accept(message, self.base_frames, self.base_stamps)

    def _on_odom(self, message: Odometry) -> None:
        orientation = message.pose.pose.orientation
        sin_yaw = 2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        )
        cos_yaw = 1.0 - 2.0 * (
            orientation.y * orientation.y
            + orientation.z * orientation.z
        )
        yaw = float(np.arctan2(sin_yaw, cos_yaw))
        position = message.pose.pose.position
        self.odom_samples.append((position.x, position.y, yaw))

    @property
    def complete(self) -> bool:
        return (
            len(self.hesai_frames) >= self.args.frames
            and len(self.base_frames) >= self.args.frames
            and time.monotonic() - self.capture_started
            >= self.args.minimum_duration
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hesai-topic", default="/hesai/points")
    parser.add_argument("--base-topic", default="/utlidar/cloud_base")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--minimum-duration", type=float, default=3.0)
    parser.add_argument("--maximum-motion", type=float, default=0.015)
    parser.add_argument(
        "--maximum-yaw-motion-deg",
        type=float,
        default=2.0,
    )
    parser.add_argument("--minimum-range", type=float, default=0.30)
    parser.add_argument("--maximum-range", type=float, default=8.0)
    parser.add_argument("--minimum-points", type=int, default=250)
    parser.add_argument("--points-per-frame", type=int, default=30_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.timeout <= 0:
        raise SystemExit("--frames and --timeout must be positive")

    rclpy.init()
    node = StationaryPairCapture(args)
    started = time.monotonic()
    try:
        while not node.complete and time.monotonic() - started < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not node.complete:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "capture_timeout",
                    "hesai_frames": len(node.hesai_frames),
                    "base_frames": len(node.base_frames),
                },
                sort_keys=True,
            )
        )
        return 2

    if len(node.odom_samples) < 5:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "insufficient_odometry_for_stationary_proof",
                    "odom_samples": len(node.odom_samples),
                },
                sort_keys=True,
            )
        )
        return 3
    odom = np.asarray(node.odom_samples, dtype=np.float64)
    displacement = np.linalg.norm(odom[:, :2] - odom[0, :2], axis=1)
    yaw_delta = (odom[:, 2] - odom[0, 2] + np.pi) % (2 * np.pi) - np.pi
    maximum_motion = float(np.max(displacement))
    maximum_yaw_motion_deg = float(np.degrees(np.max(np.abs(yaw_delta))))
    stationary = (
        maximum_motion <= args.maximum_motion
        and maximum_yaw_motion_deg <= args.maximum_yaw_motion_deg
    )
    if not stationary:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "robot_moved_during_capture",
                    "maximum_motion_m": maximum_motion,
                    "maximum_yaw_motion_deg": maximum_yaw_motion_deg,
                },
                sort_keys=True,
            )
        )
        return 4

    metadata = {
        "schema": 1,
        "stationary_capture_required": True,
        "hesai_topic": args.hesai_topic,
        "base_topic": args.base_topic,
        "hesai_frame_id": node.hesai_frame_id,
        "base_frame_id": node.base_frame_id,
        "hesai_frame_count": len(node.hesai_frames),
        "base_frame_count": len(node.base_frames),
        "hesai_points_per_frame": [
            int(frame.shape[0]) for frame in node.hesai_frames
        ],
        "base_points_per_frame": [
            int(frame.shape[0]) for frame in node.base_frames
        ],
        "hesai_stamps": node.hesai_stamps,
        "base_stamps": node.base_stamps,
        "odom_sample_count": len(node.odom_samples),
        "maximum_motion_m": maximum_motion,
        "maximum_yaw_motion_deg": maximum_yaw_motion_deg,
        "stationary_verified": True,
        "elapsed_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        hesai_points=np.concatenate(node.hesai_frames, axis=0),
        base_points=np.concatenate(node.base_frames, axis=0),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    result = {
        "ok": True,
        "output": str(args.output),
        **metadata,
        "hesai_total_points": int(sum(metadata["hesai_points_per_frame"])),
        "base_total_points": int(sum(metadata["base_points_per_frame"])),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
