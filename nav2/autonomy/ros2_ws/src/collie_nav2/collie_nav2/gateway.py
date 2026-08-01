"""Guarded map-frame Home capture and Nav2 return gateway.

Nav2 plans in ROS domain 30. Velocity commands are relayed over localhost to
collie-demo, which remains the only Unitree motion owner and retains its
direct-Sport command envelope plus watchdog.
"""

from __future__ import annotations

import asyncio
from collections import deque
import math
import os
import threading
import time
from typing import Any

from action_msgs.msg import GoalStatus
from fastapi import FastAPI, HTTPException
import httpx
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from pydantic import BaseModel, Field
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from tf2_ros import Buffer, TransformException, TransformListener
import uvicorn

from collie_nav2.command_shaping import apply_measured_motion_floors


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


class CaptureRequest(BaseModel):
    duration_s: float = Field(ge=0.4, le=3.0)
    maximum_position_span_m: float = Field(gt=0.0, le=0.05)
    maximum_yaw_span_deg: float = Field(gt=0.0, le=5.0)


class NavigateRequest(BaseModel):
    position_tolerance_m: float = Field(gt=0.0, le=0.10)
    heading_tolerance_deg: float = Field(gt=0.0, le=5.0)
    timeout_s: float = Field(ge=10.0, le=90.0)


class CancelRequest(BaseModel):
    reason: str = Field(default="operator_stop", max_length=240)


class CollieNav2Gateway(Node):
    ODOM_MAX_AGE_S = 0.50
    SCAN_MAX_AGE_S = 0.50
    MAP_MAX_AGE_S = 5.0
    POSE_MAX_AGE_S = 0.50
    COMMAND_MAX_AGE_S = 0.35
    # Nav2 intentionally stops publishing /cmd_vel while it clears costmaps,
    # switches recovery behaviors, or executes the bounded two-second Wait in
    # navigate_home.xml.  Keep sending an explicit zero heartbeat during that
    # planner-only gap so collie-demo's independent 350 ms watchdog remains
    # satisfied without replaying a stale non-zero command.  A longer gap still
    # cancels the goal and revokes navigation ownership.
    COMMAND_TRANSITION_GRACE_S = 3.0
    COMMAND_START_GRACE_S = 1.50
    # Nav2's pose progress checker fails a stalled controller attempt at 5 s.
    # Keep this outer guard later so its recovery tree gets a bounded chance
    # to clear costmaps or begin another motion before we cancel the goal.
    STALL_TIMEOUT_S = 8.0
    STALL_PROGRESS_M = 0.04
    STALL_HEADING_PROGRESS_RAD = math.radians(3.0)
    ROTATION_ONLY_TIMEOUT_S = 12.0
    MAX_HOME_DISTANCE_M = 3.0
    MAX_FORWARD_MPS = 0.30
    # NAV2-DEADBAND-001: operator-observed direct SportClient pulses moved at
    # 0.25 and 0.50 m/s. Use the lowest confirmed working value as the
    # physical command floor while leaving zero/rotation-only commands alone.
    MIN_FORWARD_MPS = 0.25
    MAX_YAW_RPS = 0.50
    MIN_ROTATION_YAW_RPS = 0.35

    def __init__(self) -> None:
        super().__init__("collie_nav2_gateway")
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._collie_url = os.environ.get(
            "COLLIE_RUNTIME_URL", "http://127.0.0.1:8096"
        ).rstrip("/")
        self._mount_validated = (
            os.environ.get("COLLIE_LIDAR_TRANSFORM_VALIDATED", "0") == "1"
        )
        self._lidar_frame = os.environ.get(
            "COLLIE_LIDAR_FRAME", "hesai_lidar"
        )
        self._lidar_transform = {
            key: float(os.environ.get(environment, "0"))
            for key, environment in (
                ("x_m", "COLLIE_LIDAR_X"),
                ("y_m", "COLLIE_LIDAR_Y"),
                ("z_m", "COLLIE_LIDAR_Z"),
                ("roll_rad", "COLLIE_LIDAR_ROLL"),
                ("pitch_rad", "COLLIE_LIDAR_PITCH"),
                ("yaw_rad", "COLLIE_LIDAR_YAW"),
            )
        }
        self._odom_at: float | None = None
        self._scan_at: float | None = None
        self._map_at: float | None = None
        self._pose_at: float | None = None
        self._odom_speed_mps = math.inf
        self._odom_yaw_rps = math.inf
        self._odom_history: deque[tuple[float, float, float]] = deque(
            maxlen=300
        )
        self._pose: tuple[float, float, float] | None = None
        self._pose_history: deque[tuple[float, float, float, float]] = deque(
            maxlen=300
        )
        self._map_cells = 0
        self._home: dict[str, float | str] | None = None
        self._home_validation: dict[str, float | int] | None = None
        self._navigation: dict[str, Any] = {
            "state": "idle",
            "reason": "no Home navigation requested",
            "distance_remaining_m": None,
            "position_error_m": None,
            "heading_error_deg": None,
            "navigation_time_s": None,
            "recoveries": 0,
        }
        self._motion: dict[str, Any] = {
            "armed": False,
            "owner": None,
            "fault": None,
        }
        self._active = False
        self._started_at: float | None = None
        self._deadline: float | None = None
        self._best_distance: float | None = None
        self._best_heading_error_rad: float | None = None
        self._last_progress_at: float | None = None
        self._last_linear_progress_at: float | None = None
        self._last_linear_progress_pose: tuple[float, float] | None = None
        self._last_progress_yaw_rad: float | None = None
        self._position_tolerance_m = 0.10
        self._heading_tolerance_rad = math.radians(5.0)
        self._goal_handle = None
        self._latest_cmd: tuple[float, float, float, float] | None = None
        self._last_relay: dict[str, Any] | None = None
        self._command_counts = {
            "received": 0,
            "relay_attempts": 0,
            "forward_attempts": 0,
            "zero_attempts": 0,
            "boosted_forward_attempts": 0,
            "stale_zero_attempts": 0,
        }
        self._command_trace: deque[dict[str, Any]] = deque(maxlen=80)
        self._last_command_trace_at: float | None = None
        self._plan: dict[str, Any] | None = None
        self._nav2_active = False
        self._nav2_state_request = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.action_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self._nav2_state_client = self.create_client(
            GetState, "/bt_navigator/get_state"
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.create_subscription(
            LaserScan,
            "/scan",
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, 10)
        self.create_subscription(Path, "/plan", self._on_plan, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 20)
        self.create_timer(0.05, self._update_pose)
        self.create_timer(0.25, self._update_nav2_state)
        self.create_timer(0.10, self._safety_tick)
        self._motion_thread = threading.Thread(
            target=self._motion_loop,
            name="collie-nav2-motion-relay",
            daemon=True,
        )
        self._motion_thread.start()

    def _on_odom(self, msg: Odometry) -> None:
        now = time.monotonic()
        speed = math.hypot(
            msg.twist.twist.linear.x, msg.twist.twist.linear.y
        )
        yaw_rate = abs(msg.twist.twist.angular.z)
        with self._lock:
            self._odom_at = now
            self._odom_speed_mps = speed
            self._odom_yaw_rps = yaw_rate
            self._odom_history.append((now, speed, yaw_rate))

    def _on_scan(self, _msg: LaserScan) -> None:
        with self._lock:
            self._scan_at = time.monotonic()

    def _on_map(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._map_at = time.monotonic()
            self._map_cells = int(msg.info.width) * int(msg.info.height)

    def _on_plan(self, msg: Path) -> None:
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in msg.poses
        ]
        length_m = sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(points, points[1:])
        )
        with self._lock:
            self._plan = {
                "received_at": time.monotonic(),
                "frame_id": str(msg.header.frame_id),
                "pose_count": len(points),
                "length_m": round(length_m, 4),
                "start": None
                if not points
                else {
                    "x_m": round(points[0][0], 4),
                    "y_m": round(points[0][1], 4),
                },
                "end": None
                if not points
                else {
                    "x_m": round(points[-1][0], 4),
                    "y_m": round(points[-1][1], 4),
                },
            }

    def _on_cmd_vel(self, msg: Twist) -> None:
        now = time.monotonic()
        command = (
            float(msg.linear.x),
            float(msg.linear.y),
            float(msg.angular.z),
            now,
        )
        with self._lock:
            self._latest_cmd = command
            self._command_counts["received"] += 1

    def _update_pose(self) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", Time()
            )
        except TransformException:
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        pose = (
            float(translation.x),
            float(translation.y),
            quaternion_yaw(
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ),
        )
        now = time.monotonic()
        if not all(math.isfinite(value) for value in pose):
            return
        with self._lock:
            self._pose = pose
            self._pose_at = now
            self._pose_history.append((now, *pose))

    def _update_nav2_state(self) -> None:
        request = self._nav2_state_request
        if request is not None and not request.done():
            return
        if not self._nav2_state_client.service_is_ready():
            with self._lock:
                self._nav2_active = False
            return
        request = self._nav2_state_client.call_async(GetState.Request())
        self._nav2_state_request = request
        request.add_done_callback(self._nav2_state_callback)

    def _nav2_state_callback(self, future: Any) -> None:
        try:
            response = future.result()
            active = (
                int(response.current_state.id)
                == int(State.PRIMARY_STATE_ACTIVE)
            )
        except Exception:
            active = False
        with self._lock:
            self._nav2_active = active

    def _health_locked(self, now: float) -> dict[str, Any]:
        odom_age = None if self._odom_at is None else now - self._odom_at
        scan_age = None if self._scan_at is None else now - self._scan_at
        map_age = None if self._map_at is None else now - self._map_at
        pose_age = None if self._pose_at is None else now - self._pose_at
        odom_healthy = odom_age is not None and odom_age <= self.ODOM_MAX_AGE_S
        scan_healthy = scan_age is not None and scan_age <= self.SCAN_MAX_AGE_S
        map_healthy = (
            map_age is not None
            and map_age <= self.MAP_MAX_AGE_S
            and self._map_cells > 0
        )
        localization_healthy = (
            pose_age is not None
            and pose_age <= self.POSE_MAX_AGE_S
            and odom_healthy
        )
        nav2_action_ready = self.action_client.server_is_ready()
        nav2_healthy = nav2_action_ready and self._nav2_active
        ready = bool(
            odom_healthy
            and scan_healthy
            and map_healthy
            and localization_healthy
            and nav2_healthy
            and self._mount_validated
        )
        reasons: list[str] = []
        if not self._mount_validated:
            reasons.append(
                f"{self._lidar_frame} body transform is not validated"
            )
        if not odom_healthy:
            reasons.append("odometry is stale")
        if not scan_healthy:
            reasons.append("LiDAR scan is stale")
        if not map_healthy:
            reasons.append("map is stale or empty")
        if not localization_healthy:
            reasons.append("map localization is stale")
        if not nav2_action_ready:
            reasons.append("NavigateToPose server is unavailable")
        elif not self._nav2_active:
            reasons.append("NavigateToPose lifecycle is not active")
        return {
            "ready": ready,
            "reason": "ready" if ready else "; ".join(reasons),
            "odom_healthy": odom_healthy,
            "scan_healthy": scan_healthy,
            "map_healthy": map_healthy,
            "localization_healthy": localization_healthy,
            "nav2_healthy": nav2_healthy,
            "nav2_action_ready": nav2_action_ready,
            "nav2_active": self._nav2_active,
            "lidar_frame": self._lidar_frame,
            "lidar_transform": dict(self._lidar_transform),
            "lidar_transform_validated": self._mount_validated,
            "odom_age_s": None if odom_age is None else round(odom_age, 3),
            "scan_age_s": None if scan_age is None else round(scan_age, 3),
            "map_age_s": None if map_age is None else round(map_age, 3),
            "pose_age_s": None if pose_age is None else round(pose_age, 3),
        }

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            health = self._health_locked(time.monotonic())
        return {
            "ok": True,
            "ready": health["ready"],
            "reason": health["reason"],
            "health": health,
        }

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            health = self._health_locked(now)
            home = None if self._home is None else dict(self._home)
            validation = (
                None
                if self._home_validation is None
                else dict(self._home_validation)
            )
            navigation = dict(self._navigation)
            motion = dict(self._motion)
            pose = self._pose
            latest_cmd = self._latest_cmd
            last_relay = (
                None if self._last_relay is None else dict(self._last_relay)
            )
            command_counts = dict(self._command_counts)
            command_trace = list(self._command_trace)
            plan = None if self._plan is None else dict(self._plan)
            last_progress_at = self._last_progress_at
            last_linear_progress_at = self._last_linear_progress_at
        navigation["progress_age_s"] = (
            None
            if last_progress_at is None
            else round(now - last_progress_at, 3)
        )
        navigation["linear_progress_age_s"] = (
            None
            if last_linear_progress_at is None
            else round(now - last_linear_progress_at, 3)
        )
        if last_relay is not None:
            relayed_at = float(last_relay.pop("relayed_at"))
            last_relay["age_s"] = round(now - relayed_at, 3)
        if plan is not None:
            received_at = float(plan.pop("received_at"))
            plan["age_s"] = round(now - received_at, 3)
        return {
            "ok": True,
            "ready": health["ready"],
            "reason": health["reason"],
            "health": health,
            "home": home,
            "validation": validation,
            "pose": None
            if pose is None
            else {
                "x_m": round(pose[0], 4),
                "y_m": round(pose[1], 4),
                "yaw_rad": round(pose[2], 4),
                "frame_id": "map",
            },
            "navigation": navigation,
            "motion": motion,
            "command_diagnostics": {
                "raw_cmd_vel": None
                if latest_cmd is None
                else {
                    "forward_mps": round(latest_cmd[0], 4),
                    "lateral_mps": round(latest_cmd[1], 4),
                    "yaw_rps": round(latest_cmd[2], 4),
                    "age_s": round(now - latest_cmd[3], 3),
                },
                "last_relay": last_relay,
                "counts": command_counts,
                "trace": command_trace,
            },
            "global_plan": plan,
        }

    async def capture_home(self, request: CaptureRequest) -> dict[str, Any]:
        health = self.health_payload()
        if not health["ready"]:
            raise HTTPException(status_code=409, detail=health["reason"])
        started = time.monotonic()
        await asyncio.sleep(request.duration_s)
        ended = time.monotonic()
        with self._lock:
            final_health = self._health_locked(ended)
            samples = [
                sample
                for sample in self._pose_history
                if started <= sample[0] <= ended
            ]
            odom_samples = [
                sample
                for sample in self._odom_history
                if started <= sample[0] <= ended
            ]
        if not final_health["ready"]:
            raise HTTPException(
                status_code=409, detail=final_health["reason"]
            )
        minimum_samples = max(5, int(request.duration_s * 10.0))
        if len(samples) < minimum_samples:
            raise HTTPException(
                status_code=409,
                detail="not enough fresh map-frame poses to capture Home",
            )
        if len(odom_samples) < minimum_samples:
            raise HTTPException(
                status_code=409,
                detail="not enough fresh odometry to capture Home",
            )
        if any(
            speed > 0.03 or yaw_rate > 0.08
            for _, speed, yaw_rate in odom_samples
        ):
            raise HTTPException(
                status_code=409,
                detail="Woof must remain stationary while Home is captured",
            )
        maximum_position_span = 0.0
        maximum_yaw_span = 0.0
        for index, first in enumerate(samples):
            for second in samples[index + 1 :]:
                maximum_position_span = max(
                    maximum_position_span,
                    math.hypot(second[1] - first[1], second[2] - first[2]),
                )
                maximum_yaw_span = max(
                    maximum_yaw_span,
                    abs(normalize_angle(second[3] - first[3])),
                )
        if maximum_position_span > request.maximum_position_span_m:
            raise HTTPException(
                status_code=409,
                detail=(
                    "map localization drifted "
                    f"{maximum_position_span:.3f} m while capturing Home"
                ),
            )
        maximum_yaw_span_deg = math.degrees(maximum_yaw_span)
        if maximum_yaw_span_deg > request.maximum_yaw_span_deg:
            raise HTTPException(
                status_code=409,
                detail=(
                    "map heading drifted "
                    f"{maximum_yaw_span_deg:.2f} degrees while capturing Home"
                ),
            )
        mean_x = sum(sample[1] for sample in samples) / len(samples)
        mean_y = sum(sample[2] for sample in samples) / len(samples)
        mean_yaw = math.atan2(
            sum(math.sin(sample[3]) for sample in samples),
            sum(math.cos(sample[3]) for sample in samples),
        )
        home = {
            "x_m": mean_x,
            "y_m": mean_y,
            "yaw_rad": mean_yaw,
            "frame_id": "map",
        }
        validation = {
            "sample_count": len(samples),
            "maximum_position_span_m": maximum_position_span,
            "maximum_yaw_span_deg": maximum_yaw_span_deg,
        }
        with self._lock:
            self._home = home
            self._home_validation = validation
            self._navigation = {
                "state": "home_captured",
                "reason": "stable map-frame Home saved",
                "distance_remaining_m": 0.0,
                "position_error_m": 0.0,
                "heading_error_deg": 0.0,
                "navigation_time_s": 0.0,
                "recoveries": 0,
            }
        return self.status_payload()

    async def start_home_navigation(
        self, request: NavigateRequest
    ) -> dict[str, Any]:
        health = self.health_payload()
        if not health["ready"]:
            raise HTTPException(status_code=409, detail=health["reason"])
        with self._lock:
            if self._active:
                raise HTTPException(
                    status_code=409, detail="Home navigation is already active"
                )
            home = None if self._home is None else dict(self._home)
            pose = self._pose
        if home is None or pose is None:
            raise HTTPException(
                status_code=409, detail="capture map-frame Home first"
            )
        distance = math.hypot(
            float(home["x_m"]) - pose[0],
            float(home["y_m"]) - pose[1],
        )
        if distance > self.MAX_HOME_DISTANCE_M:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Home is {distance:.2f} m away, outside the "
                    f"{self.MAX_HOME_DISTANCE_M:.1f} m stage envelope"
                ),
            )
        server_ready = await asyncio.to_thread(
            self.action_client.wait_for_server, timeout_sec=2.0
        )
        if not server_ready:
            raise HTTPException(
                status_code=503,
                detail="NavigateToPose server did not become ready",
            )
        try:
            motion = await asyncio.to_thread(self._arm_motion_http)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"collie-demo motion handoff failed: {exc}",
            ) from exc
        if not motion.get("armed") or motion.get("owner") != "navigation":
            await asyncio.to_thread(self._stop_motion_http)
            raise HTTPException(
                status_code=409,
                detail=f"collie-demo rejected navigation ownership: {motion}",
            )

        now = time.monotonic()
        with self._lock:
            self._motion = motion
            self._active = True
            self._started_at = now
            self._deadline = now + request.timeout_s
            # Nav2 feedback is path length, not Euclidean displacement. Let
            # the first feedback sample establish the comparable baseline.
            self._best_distance = None
            self._best_heading_error_rad = None
            self._last_progress_at = now
            self._last_linear_progress_at = now
            self._last_linear_progress_pose = (
                None if pose is None else (pose[0], pose[1])
            )
            self._last_progress_yaw_rad = None if pose is None else pose[2]
            self._position_tolerance_m = request.position_tolerance_m
            self._heading_tolerance_rad = math.radians(
                request.heading_tolerance_deg
            )
            self._latest_cmd = None
            self._last_relay = None
            self._command_counts = {
                name: 0 for name in self._command_counts
            }
            self._command_trace.clear()
            self._last_command_trace_at = None
            self._plan = None
            self._navigation = {
                "state": "starting",
                "reason": "NavigateToPose goal submitted",
                "distance_remaining_m": distance,
                "position_error_m": None,
                "heading_error_deg": None,
                "navigation_time_s": 0.0,
                "recoveries": 0,
            }

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(home["x_m"])
        goal.pose.pose.position.y = float(home["y_m"])
        yaw = float(home["yaw_rad"])
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        try:
            future = self.action_client.send_goal_async(
                goal, feedback_callback=self._feedback_callback
            )
        except Exception as exc:
            self._fail_or_cancel(
                "failed", f"NavigateToPose goal submission failed: {exc}"
            )
            raise HTTPException(
                status_code=503,
                detail=f"NavigateToPose goal submission failed: {exc}",
            ) from exc
        future.add_done_callback(self._goal_response_callback)
        return self.status_payload()

    async def cancel_navigation(self, reason: str) -> dict[str, Any]:
        self._fail_or_cancel("cancelled", reason)
        return self.status_payload()

    def _goal_response_callback(self, future: Any) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._fail_or_cancel("failed", f"goal submission failed: {exc}")
            return
        if not goal_handle.accepted:
            self._fail_or_cancel("failed", "NavigateToPose goal was rejected")
            return
        with self._lock:
            if not self._active:
                goal_handle.cancel_goal_async()
                return
            self._goal_handle = goal_handle
            self._navigation["state"] = "running"
            self._navigation["reason"] = "following planned path to Home"
        result = goal_handle.get_result_async()
        result.add_done_callback(self._result_callback)

    def _feedback_callback(self, message: Any) -> None:
        feedback = message.feedback
        distance = float(feedback.distance_remaining)
        now = time.monotonic()
        with self._lock:
            if not self._active:
                return
            previous_best = self._best_distance
            if (
                previous_best is None
                or distance <= previous_best - self.STALL_PROGRESS_M
            ):
                self._best_distance = distance
            self._navigation.update(
                {
                    "state": "running",
                    "reason": "following planned path to Home",
                    "distance_remaining_m": round(distance, 4),
                    "navigation_time_s": (
                        None
                        if self._started_at is None
                        else round(now - self._started_at, 3)
                    ),
                    "recoveries": int(feedback.number_of_recoveries),
                }
            )

    def _result_callback(self, future: Any) -> None:
        try:
            wrapped = future.result()
            status = wrapped.status
        except Exception as exc:
            self._fail_or_cancel("failed", f"NavigateToPose result failed: {exc}")
            return
        with self._lock:
            if self._navigation["state"] in {
                "failed",
                "cancelled",
                "aborted",
            }:
                return
            home = None if self._home is None else dict(self._home)
            pose = self._pose
            position_tolerance = self._position_tolerance_m
            heading_tolerance = self._heading_tolerance_rad
            started = self._started_at
        if status != GoalStatus.STATUS_SUCCEEDED:
            label = {
                GoalStatus.STATUS_CANCELED: "cancelled",
                GoalStatus.STATUS_ABORTED: "aborted",
            }.get(status, "failed")
            self._fail_or_cancel(
                label, f"NavigateToPose finished with status {status}"
            )
            return
        if home is None or pose is None:
            self._fail_or_cancel(
                "failed", "map pose disappeared while verifying Home"
            )
            return
        position_error = math.hypot(
            float(home["x_m"]) - pose[0],
            float(home["y_m"]) - pose[1],
        )
        heading_error = normalize_angle(float(home["yaw_rad"]) - pose[2])
        if (
            position_error > position_tolerance
            or abs(heading_error) > heading_tolerance
        ):
            self._fail_or_cancel(
                "failed",
                "Nav2 result was outside the saved Home tolerances",
                position_error_m=position_error,
                heading_error_rad=heading_error,
            )
            return
        now = time.monotonic()
        with self._lock:
            health = self._health_locked(now)
            if not (
                health["localization_healthy"]
                and health["scan_healthy"]
                and health["map_healthy"]
            ):
                self._fail_or_cancel(
                    "failed",
                    "sensor health failed during final Home verification",
                )
                return
            self._active = False
            self._goal_handle = None
            self._navigation.update(
                {
                    "state": "succeeded",
                    "reason": "saved map-frame Home verified",
                    "distance_remaining_m": position_error,
                    "position_error_m": position_error,
                    "heading_error_deg": math.degrees(heading_error),
                    "navigation_time_s": (
                        None if started is None else now - started
                    ),
                }
            )
        self._stop_motion_in_background()

    def _safety_tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            if not self._active:
                return
            health = self._health_locked(now)
            state = self._navigation["state"]
            deadline = self._deadline
            home = None if self._home is None else dict(self._home)
            pose = self._pose
            near_goal = False
            if home is not None and pose is not None:
                position_error = math.hypot(
                    float(home["x_m"]) - pose[0],
                    float(home["y_m"]) - pose[1],
                )
                previous_position = self._last_linear_progress_pose
                if (
                    previous_position is None
                    or math.hypot(
                        pose[0] - previous_position[0],
                        pose[1] - previous_position[1],
                    )
                    >= self.STALL_PROGRESS_M
                ):
                    # Remaining path length can change when Nav2 replans
                    # without the robot translating. Only map-frame pose
                    # displacement proves physical linear progress.
                    self._last_linear_progress_pose = (pose[0], pose[1])
                    self._last_linear_progress_at = now
                    self._last_progress_at = now
                heading_error = abs(
                    normalize_angle(float(home["yaw_rad"]) - pose[2])
                )
                heading_progress_radius = max(
                    0.15, self._position_tolerance_m + 0.05
                )
                near_goal = position_error <= heading_progress_radius
                if near_goal and (
                    self._best_heading_error_rad is None
                    or heading_error
                    <= self._best_heading_error_rad
                    - self.STALL_HEADING_PROGRESS_RAD
                ):
                    self._best_heading_error_rad = heading_error
                    self._last_progress_at = now
                elif not near_goal:
                    previous_yaw = self._last_progress_yaw_rad
                    if (
                        previous_yaw is None
                        or abs(normalize_angle(pose[2] - previous_yaw))
                        >= self.STALL_HEADING_PROGRESS_RAD
                    ):
                        self._last_progress_yaw_rad = pose[2]
                        self._last_progress_at = now
            last_progress = self._last_progress_at
            last_linear_progress = self._last_linear_progress_at
        if not (
            health["localization_healthy"]
            and health["scan_healthy"]
            and health["map_healthy"]
        ):
            self._fail_or_cancel(
                "failed", f"navigation health gate failed: {health['reason']}"
            )
        elif deadline is not None and now >= deadline:
            self._fail_or_cancel("failed", "Home navigation timed out")
        elif (
            state == "running"
            and not near_goal
            and last_linear_progress is not None
            and now - last_linear_progress >= self.ROTATION_ONLY_TIMEOUT_S
        ):
            self._fail_or_cancel(
                "failed",
                "Home navigation made no linear progress after turning",
            )
        elif (
            state == "running"
            and last_progress is not None
            and now - last_progress >= self.STALL_TIMEOUT_S
        ):
            self._fail_or_cancel(
                "failed", "Home navigation made no measurable progress"
            )

    def _fail_or_cancel(
        self,
        state: str,
        reason: str,
        *,
        position_error_m: float | None = None,
        heading_error_rad: float | None = None,
    ) -> None:
        with self._lock:
            if not self._active and self._navigation["state"] in {
                "succeeded",
                "failed",
                "cancelled",
                "aborted",
            }:
                return
            goal_handle = self._goal_handle
            self._active = False
            self._goal_handle = None
            self._navigation.update(
                {
                    "state": state,
                    "reason": str(reason)[:400],
                    "position_error_m": position_error_m,
                    "heading_error_deg": (
                        None
                        if heading_error_rad is None
                        else math.degrees(heading_error_rad)
                    ),
                }
            )
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._stop_motion_in_background()

    def _arm_motion_http(self) -> dict[str, Any]:
        with httpx.Client(timeout=2.0) as client:
            response = client.post(
                f"{self._collie_url}/api/navigation/arm",
                json={"confirmation": "MAP AND PATH CLEAR"},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("collie-demo returned invalid motion state")
        return payload

    def _stop_motion_http(self) -> None:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.post(
                    f"{self._collie_url}/api/navigation/stop"
                )
                response.raise_for_status()
                payload = response.json()
            if isinstance(payload, dict):
                with self._lock:
                    self._motion = payload
        except Exception as exc:
            with self._lock:
                self._motion = {
                    # A failed STOP RPC means the physical motion state is
                    # unknown. Never render that as confirmed-disarmed.
                    "armed": True,
                    "owner": "unknown",
                    "fault": f"navigation STOP request failed: {exc}",
                }

    def _stop_motion_in_background(self) -> None:
        threading.Thread(
            target=self._stop_motion_http,
            name="collie-nav2-stop",
            daemon=True,
        ).start()

    def _motion_loop(self) -> None:
        with httpx.Client(timeout=0.8) as client:
            while not self._stop_event.wait(0.05):
                now = time.monotonic()
                with self._lock:
                    active = self._active
                    started = self._started_at
                    command = self._latest_cmd
                if not active:
                    continue
                if command is None:
                    if (
                        started is not None
                        and now - started > self.COMMAND_START_GRACE_S
                    ):
                        self._fail_or_cancel(
                            "failed", "Nav2 produced no velocity heartbeat"
                        )
                    continue
                forward, lateral, yaw, received_at = command
                raw_forward = forward
                raw_lateral = lateral
                raw_yaw = yaw
                command_age = now - received_at
                if command_age > self.COMMAND_TRANSITION_GRACE_S:
                    self._fail_or_cancel(
                        "failed", "Nav2 velocity heartbeat became stale"
                    )
                    continue
                if command_age > self.COMMAND_MAX_AGE_S:
                    # A stale non-zero command is never repeated.  This zero
                    # pulse brakes immediately but preserves the exclusive
                    # navigation lease through short, intentional Nav2
                    # recovery transitions.
                    forward = 0.0
                    lateral = 0.0
                    yaw = 0.0
                    stale_zero = True
                else:
                    stale_zero = False
                    # Apply measured floors only to fresh commands. Small
                    # forward commands can settle posture without a step;
                    # zero, reverse, stale, and rotation-only translation are
                    # not promoted.
                    forward, yaw = apply_measured_motion_floors(
                        forward,
                        yaw,
                        minimum_forward_mps=self.MIN_FORWARD_MPS,
                        minimum_rotation_yaw_rps=(
                            self.MIN_ROTATION_YAW_RPS
                        ),
                    )
                if (
                    forward < -0.001
                    or forward > self.MAX_FORWARD_MPS
                    or abs(lateral) > 0.02
                    or abs(yaw) > self.MAX_YAW_RPS
                ):
                    self._fail_or_cancel(
                        "failed",
                        "Nav2 command exceeded the guarded stage envelope",
                    )
                    continue
                boosted_forward = forward > raw_forward + 1.0e-6
                relay = {
                    "relayed_at": now,
                    "raw_forward_mps": round(raw_forward, 4),
                    "raw_lateral_mps": round(raw_lateral, 4),
                    "raw_yaw_rps": round(raw_yaw, 4),
                    "sent_forward_mps": round(max(0.0, forward), 4),
                    "sent_yaw_rps": round(yaw, 4),
                    "source_age_s": round(command_age, 4),
                    "stale_zero": stale_zero,
                    "boosted_forward": boosted_forward,
                }
                with self._lock:
                    self._last_relay = relay
                    self._command_counts["relay_attempts"] += 1
                    if forward > 0.001:
                        self._command_counts["forward_attempts"] += 1
                    elif abs(yaw) <= 0.001:
                        self._command_counts["zero_attempts"] += 1
                    if boosted_forward:
                        self._command_counts[
                            "boosted_forward_attempts"
                        ] += 1
                    if stale_zero:
                        self._command_counts["stale_zero_attempts"] += 1
                    if (
                        self._last_command_trace_at is None
                        or now - self._last_command_trace_at >= 0.20
                    ):
                        started = self._started_at
                        trace_item = dict(relay)
                        trace_item.pop("relayed_at")
                        trace_item["elapsed_s"] = (
                            None
                            if started is None
                            else round(now - started, 3)
                        )
                        self._command_trace.append(trace_item)
                        self._last_command_trace_at = now
                try:
                    response = client.post(
                        f"{self._collie_url}/api/navigation/cmd",
                        json={
                            "forward_mps": max(0.0, forward),
                            "yaw_rps": yaw,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise RuntimeError("invalid motion heartbeat response")
                    if (
                        not payload.get("armed")
                        or payload.get("owner") != "navigation"
                    ):
                        raise RuntimeError(
                            "collie-demo lost navigation motion ownership"
                        )
                    with self._lock:
                        self._motion = payload
                except Exception as exc:
                    self._fail_or_cancel(
                        "failed",
                        f"collie-demo motion heartbeat failed: {exc}",
                    )

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._stop_motion_http()
        self._motion_thread.join(timeout=2.0)
        return super().destroy_node()


def create_app(node: CollieNav2Gateway) -> FastAPI:
    app = FastAPI(title="Collie Nav2 return gateway", version="1")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return node.health_payload()

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return node.status_payload()

    @app.post("/api/home/capture")
    async def capture(request: CaptureRequest) -> dict[str, Any]:
        return await node.capture_home(request)

    @app.post("/api/home/navigate")
    async def navigate(request: NavigateRequest) -> dict[str, Any]:
        return await node.start_home_navigation(request)

    @app.post("/api/navigation/cancel")
    async def cancel(request: CancelRequest) -> dict[str, Any]:
        return await node.cancel_navigation(request.reason)

    return app


def main() -> None:
    rclpy.init()
    node = CollieNav2Gateway()
    app = create_app(node)
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": app,
            "host": "0.0.0.0",
            "port": int(os.environ.get("COLLIE_NAV2_PORT", "8100")),
            "log_level": "info",
        },
        name="collie-nav2-http",
        daemon=True,
    )
    server_thread.start()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
