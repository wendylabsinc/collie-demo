#!/usr/bin/env python3
"""Read-only live geometry probe for the deployed Collie Nav2 preview."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import rclpy
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import GetCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class LiveMapProbe(Node):
    def __init__(self) -> None:
        super().__init__("collie_live_map_probe")
        self.map: OccupancyGrid | None = None
        self.local_costmap: OccupancyGrid | None = None
        self.global_costmap: OccupancyGrid | None = None
        self.scan: LaserScan | None = None
        self.odom: list[tuple[float, float, float]] = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.local_costmap_client = self.create_client(
            GetCostmap,
            "/local_costmap/get_costmap",
        )
        self.global_costmap_client = self.create_client(
            GetCostmap,
            "/global_costmap/get_costmap",
        )
        grid_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_map,
            grid_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            "/local_costmap/costmap",
            self._on_local_costmap,
            grid_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            self._on_global_costmap,
            grid_qos,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)

    def _on_map(self, message: OccupancyGrid) -> None:
        self.map = message

    def _on_local_costmap(self, message: OccupancyGrid) -> None:
        self.local_costmap = message

    def _on_global_costmap(self, message: OccupancyGrid) -> None:
        self.global_costmap = message

    def _on_scan(self, message: LaserScan) -> None:
        self.scan = message

    def _on_odom(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self.odom.append(
            (
                float(position.x),
                float(position.y),
                quaternion_yaw(
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ),
            )
        )

    def lookup_pose(self, target: str, source: str) -> dict | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                source,
                Time(),
            )
        except TransformException:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "target_frame": target,
            "source_frame": source,
            "x_m": float(translation.x),
            "y_m": float(translation.y),
            "z_m": float(translation.z),
            "roll_pitch_not_reported": True,
            "yaw_rad": quaternion_yaw(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
        }

    def get_costmap(
        self,
        client,
        *,
        timeout_s: float = 3.0,
    ) -> Costmap | None:
        if not client.wait_for_service(timeout_sec=timeout_s):
            return None
        future = client.call_async(GetCostmap.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.exception() is not None:
            return None
        return future.result().map


def map_metrics(
    message: OccupancyGrid,
    robot_pose: dict | None,
) -> dict:
    width = int(message.info.width)
    height = int(message.info.height)
    resolution = float(message.info.resolution)
    cells = np.asarray(message.data, dtype=np.int16).reshape(height, width)
    occupied = cells >= 65
    free = (cells >= 0) & (cells < 65)
    unknown = cells < 0
    metrics = {
        "width": width,
        "height": height,
        "resolution_m": resolution,
        "occupied_cells": int(occupied.sum()),
        "free_cells": int(free.sum()),
        "unknown_cells": int(unknown.sum()),
        "occupied_fraction_of_known": float(
            occupied.sum() / max(1, occupied.sum() + free.sum())
        ),
        "nearest_occupied_to_robot_m": None,
        "occupied_inside_0_30m": None,
    }
    if robot_pose is None or not occupied.any():
        return metrics
    origin = message.info.origin.position
    rows, columns = np.nonzero(occupied)
    x = float(origin.x) + (columns + 0.5) * resolution
    y = float(origin.y) + (rows + 0.5) * resolution
    distance = np.hypot(x - robot_pose["x_m"], y - robot_pose["y_m"])
    metrics["nearest_occupied_to_robot_m"] = float(distance.min())
    metrics["occupied_inside_0_30m"] = int((distance < 0.30).sum())
    return metrics


def grid_cost_at_pose(
    message: OccupancyGrid | None,
    robot_pose: dict | None,
) -> dict | None:
    if message is None or robot_pose is None:
        return None
    width = int(message.info.width)
    height = int(message.info.height)
    resolution = float(message.info.resolution)
    origin = message.info.origin.position
    column = int((robot_pose["x_m"] - float(origin.x)) / resolution)
    row = int((robot_pose["y_m"] - float(origin.y)) / resolution)
    if not (0 <= column < width and 0 <= row < height):
        return {
            "frame_id": message.header.frame_id,
            "robot_cell_in_bounds": False,
        }
    cells = np.asarray(message.data, dtype=np.int16).reshape(height, width)
    radius_cells = max(1, int(math.ceil(0.20 / resolution)))
    row_start = max(0, row - radius_cells)
    row_end = min(height, row + radius_cells + 1)
    column_start = max(0, column - radius_cells)
    column_end = min(width, column + radius_cells + 1)
    nearby = cells[row_start:row_end, column_start:column_end]
    return {
        "frame_id": message.header.frame_id,
        "robot_cell_in_bounds": True,
        "robot_cell_cost": int(cells[row, column]),
        "maximum_cost_inside_0_20m_square": int(nearby.max()),
    }


def costmap_cost_at_pose(
    message: Costmap | None,
    robot_pose: dict | None,
) -> dict | None:
    if message is None or robot_pose is None:
        return None
    width = int(message.metadata.size_x)
    height = int(message.metadata.size_y)
    resolution = float(message.metadata.resolution)
    origin = message.metadata.origin.position
    column = int((robot_pose["x_m"] - float(origin.x)) / resolution)
    row = int((robot_pose["y_m"] - float(origin.y)) / resolution)
    if not (0 <= column < width and 0 <= row < height):
        return {
            "frame_id": message.header.frame_id,
            "robot_cell_in_bounds": False,
        }
    cells = np.asarray(message.data, dtype=np.uint8).reshape(height, width)
    radius_cells = max(1, int(math.ceil(0.20 / resolution)))
    nearby = cells[
        max(0, row - radius_cells) : min(height, row + radius_cells + 1),
        max(0, column - radius_cells) : min(
            width,
            column + radius_cells + 1,
        ),
    ]
    return {
        "frame_id": message.header.frame_id,
        "robot_cell_in_bounds": True,
        "robot_cell_cost": int(cells[row, column]),
        "maximum_cost_inside_0_20m_square": int(nearby.max()),
    }


def scan_metrics(message: LaserScan) -> dict:
    ranges = np.asarray(message.ranges, dtype=np.float64)
    finite = ranges[np.isfinite(ranges)]
    return {
        "frame_id": message.header.frame_id,
        "sample_count": int(ranges.size),
        "finite_count": int(finite.size),
        "minimum_m": None if finite.size == 0 else float(finite.min()),
        "median_m": None if finite.size == 0 else float(np.median(finite)),
        "p90_m": None
        if finite.size == 0
        else float(np.quantile(finite, 0.90)),
        "returns_inside_configured_minimum": int(
            (finite < float(message.range_min)).sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--lidar-frame", default="hesai_lidar")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = LiveMapProbe()
    started = time.monotonic()
    next_pose_sample = started
    map_pose_samples: list[tuple[float, float, float]] = []
    try:
        while time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if now >= next_pose_sample:
                sample = node.lookup_pose("map", "base_link")
                if sample is not None:
                    map_pose_samples.append(
                        (
                            sample["x_m"],
                            sample["y_m"],
                            sample["yaw_rad"],
                        )
                    )
                next_pose_sample = now + 0.10
        robot_pose = node.lookup_pose("map", "base_link")
        odom_pose = node.lookup_pose("odom", "base_link")
        lidar_pose = node.lookup_pose("base_link", args.lidar_frame)
        local_costmap_service = node.get_costmap(
            node.local_costmap_client,
        )
        global_costmap_service = node.get_costmap(
            node.global_costmap_client,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    odom = np.asarray(node.odom, dtype=np.float64)
    if odom.shape[0] >= 2:
        translation = np.linalg.norm(odom[:, :2] - odom[0, :2], axis=1)
        yaw_delta = (odom[:, 2] - odom[0, 2] + np.pi) % (2 * np.pi) - np.pi
        odom_metrics = {
            "sample_count": int(odom.shape[0]),
            "maximum_translation_from_start_m": float(translation.max()),
            "maximum_yaw_from_start_deg": float(
                np.degrees(np.abs(yaw_delta).max())
            ),
        }
    else:
        odom_metrics = {"sample_count": int(odom.shape[0])}

    localization = np.asarray(map_pose_samples, dtype=np.float64)
    if localization.shape[0] >= 2:
        translation = np.linalg.norm(
            localization[:, :2] - localization[0, :2],
            axis=1,
        )
        yaw_delta = (
            localization[:, 2] - localization[0, 2] + np.pi
        ) % (2 * np.pi) - np.pi
        localization_metrics = {
            "sample_count": int(localization.shape[0]),
            "maximum_translation_from_start_m": float(translation.max()),
            "maximum_yaw_from_start_deg": float(
                np.degrees(np.abs(yaw_delta).max())
            ),
        }
    else:
        localization_metrics = {
            "sample_count": int(localization.shape[0]),
        }

    report = {
        "ok": node.map is not None
        and node.scan is not None
        and robot_pose is not None
        and lidar_pose is not None,
        "read_only": True,
        "duration_s": args.duration,
        "robot_pose": robot_pose,
        "odom_pose": odom_pose,
        "lidar_pose": lidar_pose,
        "odometry": odom_metrics,
        "localization": localization_metrics,
        "map": None
        if node.map is None
        else map_metrics(node.map, robot_pose),
        "local_costmap": grid_cost_at_pose(
            node.local_costmap,
            odom_pose,
        )
        or costmap_cost_at_pose(local_costmap_service, odom_pose),
        "global_costmap": grid_cost_at_pose(
            node.global_costmap,
            robot_pose,
        )
        or costmap_cost_at_pose(global_costmap_service, robot_pose),
        "scan": None if node.scan is None else scan_metrics(node.scan),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
