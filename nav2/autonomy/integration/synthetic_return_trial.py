#!/usr/bin/env python3
"""Exercise the real Nav2 gateway with a deterministic planar robot.

This is a local-only integration harness.  It publishes a free occupancy map,
fresh odometry, TF, and LaserScan data, and emulates the guarded collie-demo
navigation HTTP owner.  No Unitree SDK or hardware endpoint is used.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_z_w(yaw: float) -> tuple[float, float]:
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MotionState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.armed = False
        self.forward_mps = 0.0
        self.yaw_rps = 0.0
        self.command_count = 0
        self.forward_command_count = 0
        self.max_forward_mps = 0.0
        self.max_abs_yaw_rps = 0.0
        self.last_forward_mps = 0.0
        self.last_yaw_rps = 0.0
        self.stop_count = 0

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "armed": self.armed,
                "owner": "navigation" if self.armed else None,
                "fault": None,
                "mode": "direct_navigation" if self.armed else "disarmed",
            }


class MotionHandler(BaseHTTPRequestHandler):
    state: MotionState

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path == "/api/navigation/arm":
            body = self._body()
            if body.get("confirmation") != "MAP AND PATH CLEAR":
                self._respond(409, {"detail": "confirmation mismatch"})
                return
            with self.state.lock:
                self.state.armed = True
                self.state.forward_mps = 0.0
                self.state.yaw_rps = 0.0
            self._respond(200, self.state.payload())
            return
        if self.path == "/api/navigation/cmd":
            body = self._body()
            with self.state.lock:
                if not self.state.armed:
                    self._respond(409, {"detail": "motion is not armed"})
                    return
                forward = float(body.get("forward_mps", 0.0))
                yaw = float(body.get("yaw_rps", 0.0))
                if not 0.0 <= forward <= 0.30 or abs(yaw) > 0.50:
                    self._respond(409, {"detail": "command envelope exceeded"})
                    return
                self.state.forward_mps = forward
                self.state.yaw_rps = yaw
                self.state.command_count += 1
                self.state.forward_command_count += int(forward > 0.001)
                self.state.max_forward_mps = max(
                    self.state.max_forward_mps, forward
                )
                self.state.max_abs_yaw_rps = max(
                    self.state.max_abs_yaw_rps, abs(yaw)
                )
                self.state.last_forward_mps = forward
                self.state.last_yaw_rps = yaw
            self._respond(200, self.state.payload())
            return
        if self.path == "/api/navigation/stop":
            with self.state.lock:
                self.state.armed = False
                self.state.forward_mps = 0.0
                self.state.yaw_rps = 0.0
                self.state.stop_count += 1
            self._respond(200, self.state.payload())
            return
        self._respond(404, {"detail": "not found"})


class SyntheticStage(Node):
    def __init__(
        self,
        motion: MotionState,
        *,
        obstacle_mode: str = "none",
    ) -> None:
        super().__init__("collie_synthetic_return_stage")
        self.motion = motion
        self.obstacle_mode = obstacle_mode
        self.state_enabled = True
        self.hold_position = False
        self._lock = threading.RLock()
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.max_abs_y = 0.0
        self._last_step = time.monotonic()
        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._odom = self.create_publisher(Odometry, "/odom", 20)
        self._scan = self.create_publisher(
            LaserScan, "/scan", qos_profile_sensor_data
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self._publish_static_transform()
        self.create_timer(0.02, self._step)
        self.create_timer(0.05, self._publish_state)
        self.create_timer(0.25, self._publish_map)

    def teleport(self, x: float, y: float, yaw: float) -> None:
        with self._lock:
            self.x = x
            self.y = y
            self.yaw = normalize_angle(yaw)
            self._last_step = time.monotonic()

    def pose(self) -> tuple[float, float, float]:
        with self._lock:
            return self.x, self.y, self.yaw

    def _publish_static_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.transform.rotation.w = 1.0
        self._static_tf.sendTransform(transform)

    def _step(self) -> None:
        now = time.monotonic()
        dt = min(0.05, max(0.0, now - self._last_step))
        self._last_step = now
        with self.motion.lock:
            forward = self.motion.forward_mps if self.motion.armed else 0.0
            yaw_rate = self.motion.yaw_rps if self.motion.armed else 0.0
        if self.hold_position:
            forward = 0.0
            yaw_rate = 0.0
        with self._lock:
            self.x += math.cos(self.yaw) * forward * dt
            self.y += math.sin(self.yaw) * forward * dt
            self.yaw = normalize_angle(self.yaw + yaw_rate * dt)
            self.max_abs_y = max(self.max_abs_y, abs(self.y))

    def _publish_state(self) -> None:
        if not self.state_enabled:
            return
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            x, y, yaw = self.x, self.y, self.yaw
        with self.motion.lock:
            forward = self.motion.forward_mps if self.motion.armed else 0.0
            yaw_rate = self.motion.yaw_rps if self.motion.armed else 0.0
        if self.hold_position:
            forward = 0.0
            yaw_rate = 0.0
        z, w = quaternion_z_w(yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = z
        odom.pose.pose.orientation.w = w
        odom.twist.twist.linear.x = forward
        odom.twist.twist.angular.z = yaw_rate
        self._odom.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = z
        transform.transform.rotation.w = w
        self._tf.sendTransform(transform)

        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = "base_link"
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = math.radians(1.0)
        scan.time_increment = 0.0
        scan.scan_time = 0.05
        scan.range_min = 0.35
        scan.range_max = 8.0
        scan.ranges = [8.0] * 361
        self._scan.publish(scan)

    def _publish_map(self) -> None:
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"
        grid.info.map_load_time = grid.header.stamp
        grid.info.resolution = 0.05
        grid.info.width = 200
        grid.info.height = 200
        grid.info.origin.position.x = -5.0
        grid.info.origin.position.y = -5.0
        grid.info.origin.orientation.w = 1.0
        data = [0] * (grid.info.width * grid.info.height)
        if self.obstacle_mode in {"blocked", "detour"}:
            wall_x = int((1.0 - grid.info.origin.position.x) / 0.05)
            rows = range(grid.info.height)
            if self.obstacle_mode == "detour":
                first = int((-0.25 - grid.info.origin.position.y) / 0.05)
                last = int((0.25 - grid.info.origin.position.y) / 0.05)
                rows = range(first, last + 1)
            for row in rows:
                data[row * grid.info.width + wall_x] = 100
        grid.data = data
        self._map.publish(grid)


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 3.0,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode()
    request = Request(
        url,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def wait_for_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            status, payload = request_json("GET", f"{base_url}/api/health")
            last = payload
            if status == 200 and payload.get("ready"):
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"gateway did not become ready: {last}")


def wait_for_terminal(
    base_url: str, timeout_s: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status_code, payload = request_json(
            "GET", f"{base_url}/api/status"
        )
        if status_code != 200:
            raise RuntimeError(f"status endpoint failed: {payload}")
        state = payload["navigation"]["state"]
        if state in {"succeeded", "failed", "cancelled", "aborted"}:
            return payload
        time.sleep(0.2)
    raise RuntimeError("Home navigation did not reach a terminal state")


def wait_for_stop(motion: MotionState, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with motion.lock:
            if not motion.armed and motion.stop_count >= 1:
                return
        time.sleep(0.05)
    with motion.lock:
        raise RuntimeError(
            "motion owner did not confirm STOP: "
            f"armed={motion.armed}, stop_count={motion.stop_count}"
        )


def run_trial(
    base_url: str,
    node: SyntheticStage,
    motion: MotionState,
    scenario: str,
) -> None:
    wait_for_ready(base_url, 25.0)
    # Allow Nav2 lifecycle activation to settle after the action server appears.
    time.sleep(2.0)
    capture_status, capture = request_json(
        "POST",
        f"{base_url}/api/home/capture",
        {
            "duration_s": 0.7,
            "maximum_position_span_m": 0.02,
            "maximum_yaw_span_deg": 2.0,
        },
        timeout=3.0,
    )
    if capture_status != 200:
        raise RuntimeError(f"Home capture failed ({capture_status}): {capture}")
    start_x = 2.0 if scenario in {"detour", "no-path"} else 1.0
    node.teleport(start_x, 0.0, math.pi)
    node.hold_position = scenario == "stalled-motion"
    time.sleep(0.8)
    start_status, started = request_json(
        "POST",
        f"{base_url}/api/home/navigate",
        {
            "position_tolerance_m": 0.10,
            "heading_tolerance_deg": 5.0,
            "timeout_s": 45.0,
        },
        timeout=4.0,
    )
    if start_status != 200:
        raise RuntimeError(
            f"Home navigation failed to start ({start_status}): {started}"
        )

    if scenario == "localization-loss":
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with motion.lock:
                if motion.command_count >= 5:
                    break
            time.sleep(0.05)
        node.state_enabled = False
    terminal = wait_for_terminal(base_url, 48.0)
    wait_for_stop(motion)

    if scenario not in {"success", "detour"}:
        state = terminal["navigation"]["state"]
        reason = str(terminal["navigation"]["reason"])
        if state == "succeeded":
            raise RuntimeError(
                f"{scenario} unexpectedly reached Home: {terminal}"
            )
        with motion.lock:
            command_count = motion.command_count
            forward_command_count = motion.forward_command_count
            stop_count = motion.stop_count
            armed = motion.armed
        if scenario == "no-path" and forward_command_count:
            raise RuntimeError(
                "no-path scenario commanded forward translation before "
                f"failing: {forward_command_count=}"
            )
        print(
            json.dumps(
                {
                    "result": "passed",
                    "scenario": scenario,
                    "terminal_state": state,
                    "terminal_reason": reason,
                    "motion_command_count": command_count,
                    "forward_command_count": forward_command_count,
                    "stop_count": stop_count,
                    "armed_after_stop": armed,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if terminal["navigation"]["state"] != "succeeded":
        with motion.lock:
            evidence = {
                "command_count": motion.command_count,
                "forward_command_count": motion.forward_command_count,
                "max_forward_mps": motion.max_forward_mps,
                "max_abs_yaw_rps": motion.max_abs_yaw_rps,
                "last_forward_mps": motion.last_forward_mps,
                "last_yaw_rps": motion.last_yaw_rps,
                "max_abs_y_m": node.max_abs_y,
            }
        raise RuntimeError(
            f"Home navigation failed: {terminal}; motion={evidence}"
        )

    x, y, yaw = node.pose()
    position_error = math.hypot(x, y)
    heading_error_deg = abs(math.degrees(normalize_angle(yaw)))
    with motion.lock:
        command_count = motion.command_count
        stop_count = motion.stop_count
        armed = motion.armed
    if position_error > 0.10 or heading_error_deg > 5.0:
        raise RuntimeError(
            f"terminal pose outside tolerance: {position_error=:.3f}, "
            f"{heading_error_deg=:.2f}"
        )
    if scenario == "detour" and node.max_abs_y < 0.45:
        raise RuntimeError(
            "detour scenario did not route around the finite wall: "
            f"max_abs_y={node.max_abs_y:.3f}"
        )
    if command_count < 10 or stop_count < 1 or armed:
        raise RuntimeError(
            "motion relay evidence missing: "
            f"{command_count=}, {stop_count=}, {armed=}"
        )
    print(
        json.dumps(
            {
                "result": "passed",
                "scenario": scenario,
                "home": capture["home"],
                "terminal_navigation": terminal["navigation"],
                "final_pose": {
                    "x_m": round(x, 4),
                    "y_m": round(y, 4),
                    "yaw_deg": round(math.degrees(yaw), 3),
                },
                "motion_command_count": command_count,
                "forward_command_count": motion.forward_command_count,
                "max_forward_mps": motion.max_forward_mps,
                "max_abs_yaw_rps": motion.max_abs_yaw_rps,
                "stop_count": stop_count,
                "maximum_lateral_detour_m": round(node.max_abs_y, 4),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gateway-url", default="http://127.0.0.1:8100"
    )
    parser.add_argument("--runtime-port", type=int, default=8096)
    parser.add_argument(
        "--scenario",
        choices=[
            "success",
            "detour",
            "localization-loss",
            "no-path",
            "stalled-motion",
        ],
        default="success",
    )
    args = parser.parse_args()

    motion = MotionState()
    MotionHandler.state = motion
    server = ThreadingHTTPServer(("127.0.0.1", args.runtime_port), MotionHandler)
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="synthetic-collie-runtime",
        daemon=True,
    )
    server_thread.start()

    rclpy.init()
    node = SyntheticStage(
        motion,
        obstacle_mode=(
            "blocked"
            if args.scenario == "no-path"
            else args.scenario
            if args.scenario == "detour"
            else "none"
        ),
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    ros_thread = threading.Thread(
        target=executor.spin,
        name="synthetic-stage-ros",
        daemon=True,
    )
    ros_thread.start()
    try:
        run_trial(
            args.gateway_url.rstrip("/"),
            node,
            motion,
            args.scenario,
        )
    finally:
        with motion.lock:
            motion.armed = False
            motion.forward_mps = 0.0
            motion.yaw_rps = 0.0
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        ros_thread.join(timeout=2.0)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
