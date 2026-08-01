from __future__ import annotations

import asyncio
from collections import deque
import math
from pathlib import Path
import time
from typing import Callable, Protocol
import uuid

import cv2

from .controller import ApproachController
from .fruit import FruitDetection, annotate_fruits, annotate_selected_produce
from .heading import HeadingProviderProtocol, directed_progress, normalize_angle
from .matcher import ClassCandidate, ClassMatchResult, FruitClassMatcher
from .memory import FruitMemory, crop_bbox, encode_jpeg
from .mission import MissionConfig, MissionPhase, MissionTelemetry
from .motion import MotionError, MotionNotReady, UnitreeMotionAdapter
from .nav2_return import (
    MapHomeCapture,
    Nav2ReturnClientProtocol,
    Nav2ReturnError,
)
from .pointing import (
    POINTING_PREPARE_CONFIRMATION,
    POINTING_RUN_CONFIRMATION,
    PointingPolicyError,
    PointingPolicyManager,
)
from .return_home import (
    Pose2D,
    PoseWindowAssessment,
    ReturnMode,
    ReturnPlannerConfig,
    ReturnTurnDirectionLatch,
    assess_pose_window,
    plan_return_step,
)
from .types import CameraFrame, TargetObservation, VelocityCommand


ARM_CONFIRMATION = "TARGET AND PATH CLEAR"
NAVIGATION_ARM_CONFIRMATION = "MAP AND PATH CLEAR"
DEMO_CONFIRMATION = "TARGET SAVED AND AREA CLEAR"
DEMO_GO_CONFIRMATION = "CLASS LOCKED AND PATH CLEAR"
VOICE_MISSION_CONFIRMATION = "VOICE COMMAND HEARD"
POSTURE_STAND_CONFIRMATION = "WOOF IS CLEAR TO STAND"
FORWARD_CALIBRATION_CONFIRMATION = "PATH CLEAR AND STOP READY"
NAV2_FORWARD_CALIBRATION_CONFIRMATION = (
    "PATH CLEAR NO AVOIDANCE STOP READY"
)
FORWARD_CALIBRATION_DURATION_S = 0.40


class CameraProtocol(Protocol):
    def read(self) -> CameraFrame: ...


class ProduceDetectorProtocol(Protocol):
    model_path: Path
    confidence: float
    class_thresholds: dict[str, float]
    names: dict[int, str]

    def detect(self, bgr: object) -> list[FruitDetection]: ...


class ProduceTrackerProtocol(Protocol):
    def update(
        self, bgr: object
    ) -> tuple[bool, tuple[float, float, float, float]]: ...


ProduceTrackerFactory = Callable[
    [object, tuple[int, int, int, int]], ProduceTrackerProtocol
]


def _detect_frame(
    detector: ProduceDetectorProtocol, frame: CameraFrame
) -> list[FruitDetection]:
    return detector.detect(frame.bgr.copy())


def _update_tracker_frame(
    tracker: ProduceTrackerProtocol, frame: CameraFrame
) -> tuple[bool, tuple[float, float, float, float]]:
    return tracker.update(frame.bgr)


def _create_tracker_for_frame(
    factory: ProduceTrackerFactory,
    frame: CameraFrame,
    bbox_xywh: tuple[int, int, int, int],
) -> ProduceTrackerProtocol:
    return factory(frame.bgr.copy(), bbox_xywh)


def _encode_legacy_frame(
    frame: CameraFrame,
    detections: list[FruitDetection],
    selected_name: str | None,
    selected_target: TargetObservation | None,
) -> bytes:
    produce_image = annotate_fruits(frame.bgr, detections)
    produce_image = annotate_selected_produce(
        produce_image, selected_name, selected_target
    )
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        produce_image,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not encoded_ok:
        raise RuntimeError("could not encode annotated frame")
    return encoded.tobytes()


class RuntimeCommandError(RuntimeError):
    pass


class _ReturnTurnNoResponse(RuntimeCommandError):
    """A yaw lease was accepted but produced no measured physical turn."""


class CollieRuntime:
    def __init__(
        self,
        *,
        camera: CameraProtocol,
        controller: ApproachController,
        motion: UnitreeMotionAdapter | None,
        motion_enabled: bool,
        allow_unranged_forward: bool,
        produce_detector: ProduceDetectorProtocol | None = None,
        produce_tracker_factory: ProduceTrackerFactory | None = None,
        loop_hz: float = 30.0,
        annotated_hz: float = 5.0,
        produce_revalidation_iou: float = 0.15,
        produce_revalidation_misses_required: int = 3,
        maximum_produce_age_s: float = 0.75,
        follow_period_s: float = 0.05,
        follow_start_timeout_s: float = 1.5,
        navigation_idle_arm_s: float = 30.0,
        navigation_command_lease_s: float = 0.75,
        class_matcher: FruitClassMatcher | None = None,
        heading_provider: HeadingProviderProtocol | None = None,
        mission_config: MissionConfig | None = None,
        pointing: PointingPolicyManager | None = None,
        nav2_return: Nav2ReturnClientProtocol | None = None,
    ) -> None:
        self.camera = camera
        self.controller = controller
        self.motion = motion
        self.motion_enabled = bool(motion_enabled)
        self.allow_unranged_forward = bool(allow_unranged_forward)
        self.produce_detector = produce_detector
        # A local OpenCV tracker is optional. On the Go2 Jetson, MIL tracker
        # updates and repeated tracker initialization can take longer than the
        # 350 ms target-freshness safety window. The production runtime uses
        # the GPU YOLO detections as the authoritative track instead. Tests or
        # other deployments can still opt into a local tracker explicitly.
        self.produce_tracker_factory = produce_tracker_factory
        if loop_hz <= 0.0:
            raise ValueError("loop_hz must be positive")
        self.loop_hz = float(loop_hz)
        if annotated_hz <= 0.0:
            raise ValueError("annotated_hz must be positive")
        self.annotated_hz = min(float(annotated_hz), self.loop_hz)
        if not 0.0 <= produce_revalidation_iou <= 1.0:
            raise ValueError("produce_revalidation_iou must be between 0 and 1")
        self.produce_revalidation_iou = float(produce_revalidation_iou)
        if produce_revalidation_misses_required < 1:
            raise ValueError("produce_revalidation_misses_required must be positive")
        self.produce_revalidation_misses_required = int(
            produce_revalidation_misses_required
        )
        if maximum_produce_age_s <= 0.0:
            raise ValueError("maximum_produce_age_s must be positive")
        self.maximum_produce_age_s = float(maximum_produce_age_s)
        if follow_period_s <= 0.0:
            raise ValueError("follow_period_s must be positive")
        self.follow_period_s = float(follow_period_s)
        if follow_start_timeout_s <= 0.0:
            raise ValueError("follow_start_timeout_s must be positive")
        self.follow_start_timeout_s = float(follow_start_timeout_s)
        if navigation_idle_arm_s <= 0.0:
            raise ValueError("navigation_idle_arm_s must be positive")
        self.navigation_idle_arm_s = float(navigation_idle_arm_s)
        if navigation_command_lease_s <= 0.0:
            raise ValueError("navigation_command_lease_s must be positive")
        self.navigation_command_lease_s = float(navigation_command_lease_s)
        self.class_matcher = class_matcher or FruitClassMatcher()
        self.heading_provider = heading_provider
        self.mission_config = mission_config or MissionConfig()
        self.pointing = pointing
        self.nav2_return = nav2_return
        self._state_lock = asyncio.Lock()
        self._stream_condition = asyncio.Condition()
        self._action_lock = asyncio.Lock()
        # Serializes long stock gestures/capture with the low-level pointing
        # handoff. A memory-save Hello must never race policy startup.
        self._exclusive_skill_lock = asyncio.Lock()
        self._forward_calibration_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._produce_task: asyncio.Task[None] | None = None
        self._closing = False
        self._jpeg: bytes | None = None
        self._stream_jpeg: bytes | None = None
        self._stream_frame_id = 0
        self._latest_frame: CameraFrame | None = None
        self._runtime_id = uuid.uuid4().hex
        self._round_id = uuid.uuid4().hex
        self._round_generation = 0
        self._target_lock_id = 0
        self._selected_target_name: str | None = None
        self._selected_target_hint: tuple[int, int] | None = None
        self._target: TargetObservation | None = None
        self._produce_tracker: ProduceTrackerProtocol | None = None
        self._produce_tracker_label: str | None = None
        self._produce_visible_frames = 0
        self._produce_verified_at: float | None = None
        self._produce_revalidation_failures = 0
        self._produce_detections: list[FruitDetection] = []
        self._produce_frame: CameraFrame | None = None
        self._produce_frame_id: int | None = None
        self._produce_last_at: float | None = None
        self._produce_inference_ms: float | None = None
        self._produce_error = "waiting for first inference" if produce_detector else "disabled"
        self._frame_width: int | None = None
        self._frame_height: int | None = None
        self._frame_count = 0
        self._camera_frame_times: deque[float] = deque(
            maxlen=max(8, int(round(self.loop_hz * 2.0)))
        )
        self._last_annotated_at: float | None = None
        self._last_frame_at: float | None = None
        self._last_error = "waiting for camera"
        self._camera_stream_generation: str | None = None
        self._lease: str | None = None
        self._motion_owner: str | None = None
        self._navigation_deadline: float | None = None
        self._navigation_watchdog_task: asyncio.Task[None] | None = None
        self._nav2_health_task: asyncio.Task[None] | None = None
        self._nav2_health: dict[str, object] = {
            "ready": False,
            "reason": (
                "Nav2 return is not configured"
                if nav2_return is None
                else "waiting for Nav2 health"
            ),
        }
        self._nav2_health_at: float | None = None
        self._last_pulse_at: float | None = None
        self._forward_elapsed_s = 0.0
        self._command = VelocityCommand(reason="disarmed")
        self._follow_task: asyncio.Task[None] | None = None
        self._follow_start_generation = 0
        self._fruit_memory: FruitMemory | None = None
        self._home_pose: tuple[float, float, float] | None = None
        self._map_home: MapHomeCapture | None = None
        self._nav2_return_active = False
        self._mission = MissionTelemetry()
        self._mission_task: asyncio.Task[None] | None = None
        self._demo_go_event = asyncio.Event()
        self._voice_autogo_task: asyncio.Task[None] | None = None
        self._voice_mission_generation: int | None = None
        self._voice_status: dict[str, object] = {
            "last_event": "waiting_for_voice_service",
            "last_heard": "",
            "last_target": None,
            "error": "",
            "mission_active": False,
        }

    async def start(self) -> None:
        if self.heading_provider is not None:
            self.heading_provider.start()
        if self.motion_enabled and self.motion is not None:
            try:
                await self.motion.initialize()
            except MotionNotReady as exc:
                self._last_error = str(exc)
        self._closing = False
        self._task = asyncio.create_task(self._camera_loop())
        if self.produce_detector is not None:
            self._produce_task = asyncio.create_task(self._produce_loop())
        self._navigation_watchdog_task = asyncio.create_task(
            self._navigation_watchdog_loop()
        )
        if self.nav2_return is not None:
            self._nav2_health_task = asyncio.create_task(
                self._nav2_health_loop()
            )

    async def close(self) -> None:
        self._closing = True
        async with self._stream_condition:
            self._stream_condition.notify_all()
        self._follow_start_generation += 1
        await self._cancel_voice_autogo()
        if self._mission_task is not None:
            self._mission_task.cancel()
            try:
                await self._mission_task
            except asyncio.CancelledError:
                pass
            self._mission_task = None
        if self._follow_task is not None:
            self._follow_task.cancel()
            try:
                await self._follow_task
            except asyncio.CancelledError:
                pass
            self._follow_task = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._produce_task is not None:
            self._produce_task.cancel()
            try:
                await self._produce_task
            except asyncio.CancelledError:
                pass
        if self._navigation_watchdog_task is not None:
            self._navigation_watchdog_task.cancel()
            try:
                await self._navigation_watchdog_task
            except asyncio.CancelledError:
                pass
            self._navigation_watchdog_task = None
        if self._nav2_health_task is not None:
            self._nav2_health_task.cancel()
            try:
                await self._nav2_health_task
            except asyncio.CancelledError:
                pass
            self._nav2_health_task = None
        if self.pointing is not None:
            await self.pointing.close()
        await self.stop("shutdown")
        if self.motion is not None:
            await self.motion.close()
        if self.heading_provider is not None:
            self.heading_provider.close()

    async def arm(
        self,
        confirmation: str,
        *,
        initial_forward_elapsed_s: float = 0.0,
    ) -> dict[str, object]:
        async with self._action_lock:
            self._require_pointing_idle()
            if confirmation.strip().upper() != ARM_CONFIRMATION:
                raise RuntimeCommandError(f'type exactly "{ARM_CONFIRMATION}"')
            if not self.motion_enabled or self.motion is None:
                raise RuntimeCommandError("motion backend is disabled")
            if not self.allow_unranged_forward:
                raise RuntimeCommandError("unranged demo motion is disabled")
            async with self._state_lock:
                readiness = self._follow_readiness_locked(time.monotonic())
            if readiness is not None:
                raise RuntimeCommandError(readiness)
            try:
                lease = await self.motion.arm()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._motion_owner = "fruit"
            self._navigation_deadline = None
            self._last_pulse_at = None
            self._forward_elapsed_s = max(
                0.0, float(initial_forward_elapsed_s)
            )
            self._command = VelocityCommand(reason="armed_waiting_for_hold")
            return await self.status()

    async def navigation_arm(self, confirmation: str) -> dict[str, object]:
        """Acquire the exclusive lease for the configured navigation backend."""

        async with self._action_lock:
            self._require_pointing_idle()
            if confirmation.strip().upper() != NAVIGATION_ARM_CONFIRMATION:
                raise RuntimeCommandError(
                    f'type exactly "{NAVIGATION_ARM_CONFIRMATION}"'
                )
            if not self.motion_enabled or self.motion is None:
                raise RuntimeCommandError("motion backend is disabled")
            if self._lease is not None or self.motion.armed:
                raise RuntimeCommandError("motion is already owned; stop it first")
            direct_navigation = (
                self.mission_config.return_backend == "nav2"
            )
            if direct_navigation:
                async with self._state_lock:
                    now = time.monotonic()
                    nav2_ready = bool(
                        self._nav2_health.get("ready")
                        and self._nav2_health_at is not None
                        and now - self._nav2_health_at < 1.5
                    )
                    nav2_reason = str(
                        self._nav2_health.get("reason")
                        or "Nav2 return is not ready"
                    )
                if not nav2_ready:
                    raise RuntimeCommandError(nav2_reason)
            async with self._state_lock:
                self._clear_selection_locked()
            try:
                lease = (
                    await self.motion.arm_direct_navigation()
                    if direct_navigation
                    else await self.motion.arm()
                )
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._motion_owner = "navigation"
            self._navigation_deadline = (
                time.monotonic() + self.navigation_idle_arm_s
            )
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="navigation_armed_zero")
            return await self.navigation_status()

    async def navigation_command(
        self, forward_mps: float, yaw_rps: float
    ) -> dict[str, object]:
        """Send one bounded map-navigation heartbeat for the active lease."""

        async with self._action_lock:
            if self._motion_owner != "navigation":
                raise RuntimeCommandError("navigation motion is not armed")
            if (
                self.motion is None
                or self._lease is None
                or not self.motion.armed
            ):
                self._lease = None
                self._motion_owner = None
                self._navigation_deadline = None
                raise RuntimeCommandError("navigation motion is not armed")
            command = VelocityCommand(
                forward_mps=float(forward_mps),
                yaw_rps=float(yaw_rps),
                reason="map_navigation",
            )
            try:
                self._command = (
                    await self.motion.send_direct_navigation(
                        self._lease, command
                    )
                    if self.mission_config.return_backend == "nav2"
                    else await self.motion.send(self._lease, command)
                )
            except (MotionError, ValueError) as exc:
                self._lease = None
                self._motion_owner = None
                self._navigation_deadline = None
                self._command = VelocityCommand(reason="navigation_motion_fault")
                raise RuntimeCommandError(str(exc)) from exc
            self._navigation_deadline = (
                time.monotonic() + self.navigation_command_lease_s
            )
            return await self.navigation_status()

    async def _send_final_push(self) -> dict[str, object]:
        """Send the fixed off-screen push without widening navigation limits."""

        async with self._action_lock:
            if self._motion_owner != "navigation":
                raise RuntimeCommandError("final push motion is not armed")
            if (
                self.motion is None
                or self._lease is None
                or not self.motion.armed
            ):
                self._lease = None
                self._motion_owner = None
                self._navigation_deadline = None
                raise RuntimeCommandError("final push motion is not armed")
            try:
                if self.mission_config.return_backend == "nav2":
                    self._command = await self.motion.send_direct_final_push(
                        self._lease,
                        self.mission_config.final_push_mps,
                    )
                else:
                    self._command = await self.motion.send_avoidance_final_push(
                        self._lease,
                        self.mission_config.final_push_mps,
                    )
            except (MotionError, ValueError) as exc:
                self._lease = None
                self._motion_owner = None
                self._navigation_deadline = None
                self._command = VelocityCommand(reason="final_push_motion_fault")
                raise RuntimeCommandError(str(exc)) from exc
            self._navigation_deadline = (
                time.monotonic() + self.navigation_command_lease_s
            )
            return await self.navigation_status()

    async def _direct_turn_arm(self) -> None:
        """Acquire the private yaw-only SportClient lease for the demo turn."""

        async with self._action_lock:
            self._require_pointing_idle()
            if not self.motion_enabled or self.motion is None:
                raise RuntimeCommandError("motion backend is disabled")
            if self._lease is not None or self.motion.armed:
                raise RuntimeCommandError("motion is already owned; stop it first")
            try:
                lease = await self.motion.arm_direct_yaw()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._motion_owner = "direct_turn"
            self._navigation_deadline = None
            self._last_pulse_at = None
            self._command = VelocityCommand(reason="direct_turn_armed_zero")

    async def _direct_turn_command(self, yaw_rps: float) -> None:
        """Renew one yaw-only heartbeat; translation is impossible in this mode."""

        async with self._action_lock:
            if self._motion_owner != "direct_turn":
                raise RuntimeCommandError("direct turn motion is not armed")
            if (
                self.motion is None
                or self._lease is None
                or not self.motion.armed
            ):
                self._lease = None
                self._motion_owner = None
                raise RuntimeCommandError("direct turn motion is not armed")
            try:
                self._command = await self.motion.send_direct_yaw(
                    self._lease,
                    yaw_rps,
                    "measured_direct_turn",
                )
            except (MotionError, ValueError) as exc:
                self._lease = None
                self._motion_owner = None
                self._command = VelocityCommand(reason="direct_turn_motion_fault")
                raise RuntimeCommandError(str(exc)) from exc

    async def navigation_status(self) -> dict[str, object]:
        now = time.monotonic()
        motion_status = None if self.motion is None else self.motion.status()
        available = bool(
            self.motion_enabled
            and motion_status is not None
            and motion_status["initialized"]
            and motion_status["fault"] is None
        )
        armed = bool(
            self._motion_owner == "navigation"
            and self._lease is not None
            and self.motion is not None
            and self.motion.armed
        )
        return {
            "available": available,
            "armed": armed,
            "owner": self._motion_owner,
            "fault": None if motion_status is None else motion_status["fault"],
            "motion_mode": (
                None if motion_status is None else motion_status.get("mode")
            ),
            "deadline_s": None
            if self._navigation_deadline is None
            else round(max(0.0, self._navigation_deadline - now), 3),
            "command": self._command.to_dict(),
            "limits": None if motion_status is None else motion_status["limits"],
        }

    async def prepare_pointing(self, confirmation: str) -> dict[str, object]:
        """Stop locomotion and prepare the isolated standing-point runner."""

        async with self._exclusive_skill_lock:
            return await self._prepare_pointing(confirmation)

    async def restore_standing(self, confirmation: str) -> dict[str, object]:
        """Stop all owners and use Unitree's paired StandUp transition."""

        if confirmation.strip().upper() != POSTURE_STAND_CONFIRMATION:
            raise RuntimeCommandError(
                f'type exactly "{POSTURE_STAND_CONFIRMATION}"'
            )
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")
        async with self._exclusive_skill_lock:
            await self.stop("posture_stand_prepare")
            async with self._action_lock:
                try:
                    await self.motion.perform_stand_up()
                except MotionError as exc:
                    raise RuntimeCommandError(str(exc)) from exc
                if self.pointing is not None:
                    self.pointing.clear_prepared()
                self._command = VelocityCommand(
                    reason="posture_stand_up_complete"
                )
        return await self.status()

    async def _prepare_pointing(self, confirmation: str) -> dict[str, object]:
        if confirmation.strip().upper() != POINTING_PREPARE_CONFIRMATION:
            raise RuntimeCommandError(
                f'type exactly "{POINTING_PREPARE_CONFIRMATION}"'
            )
        if self.pointing is None or not self.pointing.available:
            error = (
                "pointing policy is unavailable"
                if self.pointing is None
                else str(self.pointing.status().get("error") or "unavailable")
            )
            raise RuntimeCommandError(error)
        if self.pointing.active:
            raise RuntimeCommandError("pointing policy is already active")
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")
        await self.stop("pointing_prepare")
        async with self._action_lock:
            self._require_pointing_idle()
            self.pointing.mark_prepared()
            self._command = VelocityCommand(
                reason="standing_point_handoff_prepared"
            )
        return await self.status()

    async def start_pointing(self, confirmation: str) -> dict[str, object]:
        """Run the one-second, guarded full policy against the selected bbox."""

        async with self._exclusive_skill_lock:
            return await self._start_pointing(confirmation)

    async def _start_pointing(
        self,
        confirmation: str,
        *,
        allow_current_mission: bool = False,
    ) -> dict[str, object]:
        if confirmation.strip().upper() != POINTING_RUN_CONFIRMATION:
            raise RuntimeCommandError(f'type exactly "{POINTING_RUN_CONFIRMATION}"')
        if self.pointing is None:
            raise RuntimeCommandError("pointing policy is unavailable")
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")
        async with self._action_lock:
            self._require_pointing_idle()
            if self._lease is not None or self.motion.armed:
                raise RuntimeCommandError("motion is already owned; stop it first")
            mission_owns_call = bool(
                allow_current_mission
                and self._mission_task is asyncio.current_task()
            )
            if (
                self._mission_task is not None
                and not self._mission_task.done()
                and not mission_owns_call
            ):
                raise RuntimeCommandError("stop the fruit mission before pointing")
            async with self._state_lock:
                now = time.monotonic()
                readiness = self._follow_readiness_locked(now)
                target_label = self._selected_target_name
                target = self._target
                default_confidence = (
                    0.5
                    if self.produce_detector is None
                    else float(self.produce_detector.confidence)
                )
                class_thresholds = (
                    {}
                    if self.produce_detector is None
                    else getattr(self.produce_detector, "class_thresholds", {})
                )
            if readiness is not None:
                raise RuntimeCommandError(readiness)
            if target_label is None or target is None or target.confidence is None:
                raise RuntimeCommandError("selected target has no fresh YOLO confidence")
            minimum_confidence = float(
                class_thresholds.get(target_label, default_confidence)
            )
            if target.confidence < minimum_confidence:
                raise RuntimeCommandError(
                    f"selected {target_label} confidence is "
                    f"{target.confidence:.2f}; need {minimum_confidence:.2f}"
                )
            try:
                await self.motion.emergency_stop()
                await self.pointing.start(
                    target_label=target_label,
                    minimum_confidence=minimum_confidence,
                )
            except (MotionError, PointingPolicyError) as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._command = VelocityCommand(reason="pointing_policy_starting")
        return await self.status()

    async def stop_pointing(self) -> dict[str, object]:
        if self.pointing is not None:
            await self.pointing.stop()
        return await self.stop("pointing_operator_stop")

    async def start_follow(
        self,
        confirmation: str,
        *,
        initial_forward_elapsed_s: float = 0.0,
    ) -> dict[str, object]:
        generation = self._follow_start_generation
        deadline = time.monotonic() + self.follow_start_timeout_s
        print(
            "follow event=start_requested "
            f"generation={generation} "
            f"timeout_s={self.follow_start_timeout_s:.2f}",
            flush=True,
        )
        while True:
            if generation != self._follow_start_generation:
                print(
                    "follow event=start_cancelled "
                    f"expected_generation={generation} "
                    f"actual_generation={self._follow_start_generation}",
                    flush=True,
                )
                raise RuntimeCommandError("follow start cancelled")
            async with self._state_lock:
                selected_name = self._selected_target_name
                readiness = self._follow_readiness_locked(time.monotonic())
            if selected_name is None:
                raise RuntimeCommandError("select a detected fruit first")
            if readiness is None:
                break
            if readiness not in {
                "selected_target_not_stable",
                "selected_target_not_revalidated",
            }:
                raise RuntimeCommandError(readiness)
            if time.monotonic() >= deadline:
                raise RuntimeCommandError(readiness)
            await asyncio.sleep(0.025)

        if generation != self._follow_start_generation:
            raise RuntimeCommandError("follow start cancelled")
        await self.arm(
            confirmation,
            initial_forward_elapsed_s=initial_forward_elapsed_s,
        )
        if generation != self._follow_start_generation:
            await self.stop("follow_start_cancelled")
            raise RuntimeCommandError("follow start cancelled")
        if self._follow_task is not None and not self._follow_task.done():
            await self.stop("follow_already_active")
            raise RuntimeCommandError("follow is already active")
        self._follow_task = asyncio.create_task(self._follow_loop())
        print(
            "follow event=armed "
            f"generation={generation} "
            f"target={selected_name}",
            flush=True,
        )
        return await self.status()

    async def select_target(
        self,
        target_name: str,
        preferred_center: tuple[int, int] | None = None,
        *,
        confirmed_visible_frames: int = 1,
    ) -> dict[str, object]:
        canonical_name = self._canonical_target_name(target_name)
        confirmed_visible_frames = max(1, int(confirmed_visible_frames))
        self._follow_start_generation += 1
        async with self._action_lock:
            self._require_pointing_idle()
            if self.motion is not None and (
                self._lease is not None or self.motion.armed
            ):
                await self.motion.emergency_stop()
            self._lease = None
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="fruit_selected")
            frame: CameraFrame | None = None
            detection: FruitDetection | None = None
            selection_error: str | None = None
            async with self._state_lock:
                frame = self._latest_frame
                detection = self._best_produce_detection_locked(
                    canonical_name, preferred_center
                )
                produce_age = (
                    None
                    if self._produce_last_at is None
                    else time.monotonic() - self._produce_last_at
                )
                if (
                    frame is None
                    or detection is None
                    or produce_age is None
                    or produce_age > self.maximum_produce_age_s
                ):
                    self._clear_selection_locked()
                    selection_error = (
                        f"{canonical_name} is no longer freshly detected; select it again"
                    )
                else:
                    self._target_lock_id += 1
                    self._selected_target_name = canonical_name
                    self._selected_target_hint = preferred_center
                    self._clear_produce_tracker_locked()
                    self._target = None
            if selection_error is not None:
                raise RuntimeCommandError(selection_error)
            if frame is not None and detection is not None:
                tracker: ProduceTrackerProtocol | None = None
                try:
                    if self.produce_tracker_factory is not None:
                        tracker = await asyncio.to_thread(
                            _create_tracker_for_frame,
                            self.produce_tracker_factory,
                            frame,
                            self._detection_bbox_xywh(detection),
                        )
                except Exception as exc:
                    async with self._state_lock:
                        self._clear_selection_locked()
                    raise RuntimeCommandError(
                        f"could not initialize {canonical_name} tracker: {exc}"
                    ) from exc
                observation = self._observation_from_bbox(
                    frame,
                    self._detection_bbox_xywh(detection),
                    confidence=detection.confidence,
                    visible_frames=confirmed_visible_frames,
                )
                async with self._state_lock:
                    if self._selected_target_name == canonical_name:
                        self._produce_tracker = tracker
                        self._produce_tracker_label = (
                            canonical_name if tracker is not None else None
                        )
                        self._produce_visible_frames = confirmed_visible_frames
                        self._produce_verified_at = self._produce_last_at
                        self._produce_revalidation_failures = 0
                        self._target = observation
            return await self.status()

    async def remember_target(
        self,
        target_name: str,
        preferred_center: tuple[int, int] | None = None,
        expected_round_id: str | None = None,
    ) -> dict[str, object]:
        """Save a freshly detected YOLO class without arming motion."""

        async with self._exclusive_skill_lock:
            return await self._remember_target(
                target_name,
                preferred_center,
                expected_round_id,
            )

    async def _remember_target(
        self,
        target_name: str,
        preferred_center: tuple[int, int] | None = None,
        expected_round_id: str | None = None,
    ) -> dict[str, object]:
        self._require_pointing_idle()

        if not self.mission_config.enabled:
            raise RuntimeCommandError("fruit-memory demo is disabled")
        canonical_name = self._canonical_target_name(target_name)
        async with self._state_lock:
            if expected_round_id is not None and expected_round_id != self._round_id:
                raise RuntimeCommandError("this save belongs to an old round; save the fruit again")
            capture_generation = self._round_generation
        await self.stop("memory_capture")
        async with self._state_lock:
            if capture_generation != self._round_generation:
                raise RuntimeCommandError("round was reset; save the fruit again")
            # A new capture attempt starts a new round. Never leave an older
            # fruit silently armed as the fallback if this capture is poor.
            self._fruit_memory = None
            self._clear_selection_locked()
            self._mission = MissionTelemetry(
                phase=MissionPhase.LEARNING,
                reason="saving_target_class",
                started_monotonic_s=time.monotonic(),
            )

        reference_crop = None
        reference_bbox: tuple[int, int, int, int] | None = None
        last_frame_id: int | None = None
        hint = preferred_center
        deadline = time.monotonic() + self.mission_config.capture_timeout_s
        while (
            reference_crop is None
            and time.monotonic() < deadline
            and capture_generation == self._round_generation
        ):
            async with self._state_lock:
                frame = self._produce_frame
                frame_id = self._produce_frame_id
                detection = self._best_produce_detection_locked(
                    canonical_name, hint
                )
                produce_age = (
                    None
                    if self._produce_last_at is None
                    else time.monotonic() - self._produce_last_at
                )
            if (
                frame is None
                or frame_id is None
                or frame_id == last_frame_id
                or detection is None
                or produce_age is None
                or produce_age > self.maximum_produce_age_s
            ):
                await asyncio.sleep(0.02)
                continue
            last_frame_id = frame_id
            hint = detection.center
            try:
                reference_crop = await asyncio.to_thread(
                    crop_bbox,
                    frame.bgr,
                    detection.bbox_xyxy,
                )
            except ValueError:
                await asyncio.sleep(0)
                continue
            reference_bbox = detection.bbox_xyxy
            await asyncio.sleep(0)

        if capture_generation != self._round_generation:
            raise RuntimeCommandError("round was reset; save the fruit again")
        if (
            reference_crop is None
            or reference_bbox is None
        ):
            async with self._state_lock:
                self._mission.phase = MissionPhase.ABORTED
                self._mission.reason = "target_class_not_visible"
            raise RuntimeCommandError(
                f"keep a {canonical_name} visible, then save the class again"
            )
        memory = FruitMemory.create(
            label=canonical_name,
            reference_jpeg=await asyncio.to_thread(encode_jpeg, reference_crop),
            reference_bbox_xyxy=reference_bbox,
        )
        async with self._state_lock:
            if capture_generation != self._round_generation:
                raise RuntimeCommandError("round was reset; save the fruit again")
            self._fruit_memory = memory
            self._home_pose = None
            self._map_home = None
            self._clear_selection_locked()
            self._mission = MissionTelemetry(
                phase=MissionPhase.MEMORIZED,
                reason=(
                    "acknowledging_saved_fruit"
                    if self.mission_config.initial_hello_enabled
                    else "fruit_saved"
                ),
                initial_hello_status=(
                    "running"
                    if self.mission_config.initial_hello_enabled
                    else "not_requested"
                ),
            )
        if self.mission_config.initial_hello_enabled:
            gesture_error: str | None = None
            if not self.motion_enabled or self.motion is None:
                gesture_error = "motion backend is disabled"
            else:
                try:
                    await self.motion.perform_hello()
                except MotionError as exc:
                    gesture_error = str(exc)
            async with self._state_lock:
                if capture_generation != self._round_generation:
                    raise RuntimeCommandError("round was reset; save the fruit again")
                self._mission.reason = (
                    "fruit_saved"
                    if gesture_error is None
                    else "fruit_saved_hello_failed"
                )
                self._mission.initial_hello_status = (
                    "complete" if gesture_error is None else "failed"
                )
                self._mission.initial_hello_error = gesture_error
        return await self.status()

    async def reset_round(self) -> dict[str, object]:
        """Atomically stop every controller and invalidate the previous round."""

        await self._cancel_voice_autogo()
        async with self._state_lock:
            self._round_generation += 1
            self._round_id = uuid.uuid4().hex
            self._voice_mission_generation = None
            self._demo_go_event.clear()
        await self.stop("round_reset")
        follow_task = self._follow_task
        if (
            follow_task is not None
            and follow_task is not asyncio.current_task()
            and not follow_task.done()
        ):
            follow_task.cancel()
            try:
                await follow_task
            except asyncio.CancelledError:
                pass
        if self._follow_task is follow_task:
            self._follow_task = None
        async with self._action_lock:
            self._forward_elapsed_s = 0.0
            self._last_pulse_at = None
            self._lease = None
            self._motion_owner = None
            self._navigation_deadline = None
            self._command = VelocityCommand(reason="round_reset")
        async with self._state_lock:
            self._fruit_memory = None
            self._home_pose = None
            self._map_home = None
            self._target_lock_id += 1
            self._clear_selection_locked()
            if self._voice_status["mission_active"]:
                self._voice_status["last_event"] = "voice_mission_reset"
            self._voice_status["mission_active"] = False
            self._mission = MissionTelemetry(
                phase=MissionPhase.IDLE,
                reason="round_reset",
            )
        return await self.status()

    async def clear_memory(self) -> dict[str, object]:
        """Backward-compatible alias for the full round reset."""

        return await self.reset_round()

    async def memory_reference_jpeg(self) -> bytes | None:
        async with self._state_lock:
            memory = self._fruit_memory
        return (
            None
            if memory is None or not memory.reference_jpeg
            else memory.reference_jpeg
        )

    async def record_voice_event(
        self,
        *,
        event: str,
        transcript: str = "",
        target: str | None = None,
        error: str = "",
    ) -> dict[str, object]:
        """Record non-authoritative voice telemetry for the stage UI.

        This method never changes motion state. Only ``start_voice_mission``
        may turn a committed, allowlisted transcript into a guarded mission.
        """

        async with self._state_lock:
            self._voice_status["last_event"] = event.strip()[:120]
            if transcript.strip():
                self._voice_status["last_heard"] = transcript.strip()[:240]
            if target is not None:
                self._voice_status["last_target"] = target
            self._voice_status["error"] = error.strip()[:500]
        return await self.status()

    async def start_voice_mission(
        self,
        target_name: str,
        transcript: str,
        confirmation: str,
    ) -> dict[str, object]:
        """Start the normal guarded mission from an allowlisted voice label.

        Speech does not command the motors. It creates the same class memory
        used by the UI, starts the existing measured turn/search mission, and
        releases Go only after that mission reports a fresh class lock.
        """

        if confirmation.strip().upper() != VOICE_MISSION_CONFIRMATION:
            raise RuntimeCommandError(
                f'type exactly "{VOICE_MISSION_CONFIRMATION}"'
            )
        canonical_name = self._canonical_target_name(target_name)
        if canonical_name not in {"apple", "banana", "pear"}:
            raise RuntimeCommandError("voice target must be apple, banana, or pear")
        if self._mission_task is not None and not self._mission_task.done():
            raise RuntimeCommandError("a fruit mission is already active")
        if (
            self._voice_autogo_task is not None
            and not self._voice_autogo_task.done()
        ):
            raise RuntimeCommandError("a voice mission is already active")

        await self.reset_round()
        async with self._state_lock:
            generation = self._round_generation
            self._voice_mission_generation = generation
            self._fruit_memory = FruitMemory.create(
                label=canonical_name,
                reference_jpeg=b"",
                reference_bbox_xyxy=(0, 0, 1, 1),
            )
            self._mission = MissionTelemetry(
                phase=MissionPhase.MEMORIZED,
                reason="voice_target_class_set",
            )
            self._voice_status.update(
                {
                    "last_event": "voice_mission_starting",
                    "last_heard": transcript.strip()[:240],
                    "last_target": canonical_name,
                    "error": "",
                    "mission_active": True,
                }
            )

        try:
            await self.start_demo(DEMO_CONFIRMATION)
        except Exception as exc:
            async with self._state_lock:
                self._voice_status["last_event"] = "voice_mission_rejected"
                self._voice_status["error"] = str(exc)[:500]
                self._voice_status["mission_active"] = False
            raise
        self._voice_autogo_task = asyncio.create_task(
            self._voice_autogo_loop(generation)
        )
        return await self.status()

    async def _cancel_voice_autogo(self) -> None:
        task = self._voice_autogo_task
        if task is None:
            return
        self._voice_autogo_task = None
        if task is asyncio.current_task() or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _voice_autogo_loop(self, generation: int) -> None:
        current_task = asyncio.current_task()
        deadline = time.monotonic() + (
            self.mission_config.turn_timeout_s
            + self.mission_config.search_timeout_s
            + self.mission_config.match_reacquire_timeout_s
            + 10.0
        )
        try:
            while time.monotonic() < deadline:
                async with self._state_lock:
                    phase = self._mission.phase
                    active = bool(
                        self._mission_task is not None
                        and not self._mission_task.done()
                    )
                    same_round = generation == self._round_generation
                if not same_round:
                    return
                if phase == MissionPhase.WAITING_FOR_GO:
                    await self.approve_demo_go(DEMO_GO_CONFIRMATION)
                    async with self._state_lock:
                        self._voice_status["last_event"] = "voice_go_released"
                    while same_round:
                        async with self._state_lock:
                            phase = self._mission.phase
                            same_round = generation == self._round_generation
                        if phase in {MissionPhase.SUCCESS, MissionPhase.ABORTED}:
                            break
                        await asyncio.sleep(0.1)
                    async with self._state_lock:
                        self._voice_status["last_event"] = (
                            "voice_mission_complete"
                            if phase == MissionPhase.SUCCESS
                            else "voice_mission_aborted"
                        )
                        self._voice_status["mission_active"] = False
                        if phase == MissionPhase.ABORTED:
                            self._voice_status["error"] = self._mission.reason
                    return
                if phase == MissionPhase.ABORTED or not active:
                    async with self._state_lock:
                        self._voice_status["last_event"] = "voice_mission_aborted"
                        self._voice_status["mission_active"] = False
                        self._voice_status["error"] = self._mission.reason
                    return
                await asyncio.sleep(0.05)
            await self.stop("voice_mission_timeout")
            async with self._state_lock:
                self._mission.phase = MissionPhase.ABORTED
                self._mission.reason = "voice_mission_timeout"
                self._voice_status["last_event"] = "voice_mission_timeout"
                self._voice_status["mission_active"] = False
                self._voice_status["error"] = (
                    "voice mission timed out before class lock"
                )
        except asyncio.CancelledError:
            raise
        except RuntimeCommandError as exc:
            await self.stop(f"voice_mission_abort:{exc}")
            async with self._state_lock:
                self._voice_status["last_event"] = "voice_mission_aborted"
                self._voice_status["mission_active"] = False
                self._voice_status["error"] = str(exc)
        finally:
            async with self._state_lock:
                if self._voice_mission_generation == generation:
                    self._voice_mission_generation = None
            if self._voice_autogo_task is current_task:
                self._voice_autogo_task = None

    async def start_demo(self, confirmation: str) -> dict[str, object]:
        self._require_pointing_idle()
        if confirmation.strip().upper() != DEMO_CONFIRMATION:
            raise RuntimeCommandError(f'type exactly "{DEMO_CONFIRMATION}"')
        if not self.mission_config.enabled:
            raise RuntimeCommandError("fruit-memory demo is disabled")
        if not self.mission_config.autonomous_turn_enabled:
            raise RuntimeCommandError("autonomous turn is disabled")
        if self._mission_task is not None and not self._mission_task.done():
            raise RuntimeCommandError("fruit-memory demo is already active")
        if self._mission.initial_hello_status == "running":
            raise RuntimeCommandError("wait for Woof's hello gesture to finish")
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")
        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        pose = self.heading_provider.status()
        if not pose.healthy or pose.yaw_rad is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        if self.mission_config.return_home_enabled and not pose.pose_healthy:
            raise RuntimeCommandError("fresh Go2 local position is unavailable")
        if (
            self.mission_config.return_home_enabled
            and self.mission_config.return_backend == "nav2"
            and self.nav2_return is None
        ):
            raise RuntimeCommandError("Nav2 return service is not configured")
        async with self._state_lock:
            if self._fruit_memory is None:
                raise RuntimeCommandError("save a fruit first")
            memory_label = self._fruit_memory.label
            voice_mission = (
                self._voice_mission_generation == self._round_generation
            )
            now = time.monotonic()
            camera_fresh = (
                self._last_frame_at is not None
                and now - self._last_frame_at < 1.0
            )
            produce_fresh = (
                self._produce_last_at is not None
                and now - self._produce_last_at < self.maximum_produce_age_s
                and not self._produce_error
            )
        if not camera_fresh or not produce_fresh:
            raise RuntimeCommandError("camera or fruit detector is not fresh")
        if self._arrival_pointing_required(memory_label) and (
            self.pointing is None or not self.pointing.available
        ):
            raise RuntimeCommandError("arrival pointing policy is unavailable")
        await self.stop("demo_start_reset")
        home_pose = None
        home_pose_validation = None
        map_home = None
        if self.mission_config.return_home_enabled:
            # Capture Home only after the stop boundary has completed, so any
            # final deceleration before this mission cannot offset the origin.
            # Keep the robot-local pose even when Nav2 is the selected return
            # backend.  Nav2 remains responsible for mapped translation and
            # obstacle avoidance, while the local pose gives the proven
            # measured-turn controller an independent departure heading.
            home_assessment = await self._capture_stable_home_pose()
            home_pose = (
                home_assessment.pose.x_m,
                home_assessment.pose.y_m,
                home_assessment.pose.yaw_rad,
            )
            if self.mission_config.return_backend == "nav2":
                assert self.nav2_return is not None
                try:
                    health = await self.nav2_return.health()
                    if not health.get("ready"):
                        detail = health.get("reason") or health.get("health")
                        raise Nav2ReturnError(
                            f"Nav2 localization is not ready: {detail}"
                        )
                    map_home = await self.nav2_return.capture_home(
                        duration_s=(
                            self.mission_config.return_pose_capture_duration_s
                        ),
                        maximum_position_span_m=(
                            self.mission_config.return_pose_capture_max_drift_m
                        ),
                        maximum_yaw_span_rad=(
                            self.mission_config
                            .return_pose_capture_max_yaw_drift_rad
                        ),
                    )
                except Nav2ReturnError as exc:
                    raise RuntimeCommandError(
                        f"cannot capture map-frame Home: {exc}"
                    ) from exc
                home_pose_validation = {
                    **map_home.validation_dict(),
                    "local_odometry": home_assessment.to_dict(),
                }
            else:
                home_pose_validation = home_assessment.to_dict()
        async with self._state_lock:
            self._home_pose = home_pose
            self._map_home = map_home
            self._clear_selection_locked()
            self._demo_go_event.clear()
            self._mission = MissionTelemetry(
                phase=MissionPhase.TURNING,
                reason="starting_measured_turn",
                started_monotonic_s=time.monotonic(),
                home_pose=(
                    map_home.pose_dict()
                    if map_home is not None
                    else None
                    if home_pose is None
                    else {
                        "x_m": round(home_pose[0], 4),
                        "y_m": round(home_pose[1], 4),
                        "yaw_rad": round(home_pose[2], 4),
                    }
                ),
                home_pose_validation=home_pose_validation,
                return_backend=self.mission_config.return_backend,
                return_home_status=(
                    "pending"
                    if self.mission_config.return_home_enabled
                    else "not_requested"
                ),
                return_turn_status=(
                    "pending"
                    if self.mission_config.return_home_enabled
                    else "not_requested"
                ),
                arrival_pointing_status=(
                    "pending"
                    if self._arrival_pointing_required(memory_label)
                    else "not_requested"
                ),
                arrival_pointing_target=(
                    memory_label
                    if self._arrival_pointing_required(memory_label)
                    else None
                ),
                arrival_rest_status=(
                    "pending"
                    if self.mission_config.arrival_rest_enabled
                    else "not_requested"
                ),
                initial_hello_status=(
                    "skipped_for_voice"
                    if voice_mission
                    else "not_requested"
                ),
                match_stretch_status=(
                    "skipped_for_voice"
                    if voice_mission
                    else "not_requested"
                ),
                contact_status=(
                    "pending"
                    if self._arrival_pointing_required(memory_label)
                    else "not_requested"
                ),
            )
        if home_pose is not None:
            print(
                "return_home event=home_saved "
                f"x_m={home_pose[0]:.4f} y_m={home_pose[1]:.4f} "
                f"yaw_rad={home_pose[2]:.4f} "
                f"samples={home_pose_validation['sample_count']} "
                "position_span_m="
                f"{home_pose_validation['maximum_position_span_m']:.4f} "
                "yaw_span_deg="
                f"{home_pose_validation['maximum_yaw_span_deg']:.2f}",
                flush=True,
            )
        elif map_home is not None:
            print(
                "return_home event=map_home_saved "
                f"frame={map_home.frame_id} "
                f"x_m={map_home.x_m:.4f} y_m={map_home.y_m:.4f} "
                f"yaw_rad={map_home.yaw_rad:.4f} "
                f"samples={map_home.sample_count} "
                "position_span_m="
                f"{map_home.maximum_position_span_m:.4f} "
                "yaw_span_deg="
                f"{math.degrees(map_home.maximum_yaw_span_rad):.2f}",
                flush=True,
            )
        self._mission_task = asyncio.create_task(self._demo_loop())
        return await self.status()

    async def approve_demo_go(self, confirmation: str) -> dict[str, object]:
        """Release a locked class mission to revalidate and approach.

        The search and stretch stages never grant locomotion permission. This
        explicit operator action only wakes the mission task; that task must
        still reacquire the saved fruit class in fresh detector frames before
        it can arm the guarded follower.
        """

        self._require_pointing_idle()
        if confirmation.strip().upper() != DEMO_GO_CONFIRMATION:
            raise RuntimeCommandError(f'type exactly "{DEMO_GO_CONFIRMATION}"')
        if self._mission_task is None or self._mission_task.done():
            raise RuntimeCommandError("fruit-memory demo is not active")
        now = time.monotonic()
        async with self._state_lock:
            if self._mission.phase != MissionPhase.WAITING_FOR_GO:
                raise RuntimeCommandError("wait until the saved fruit class is locked")
            camera_fresh = (
                self._last_frame_at is not None
                and now - self._last_frame_at < 1.0
            )
            produce_fresh = (
                self._produce_last_at is not None
                and now - self._produce_last_at < self.maximum_produce_age_s
                and not self._produce_error
            )
            if not camera_fresh or not produce_fresh:
                raise RuntimeCommandError("camera or fruit detector is not fresh")
            self._mission.phase = MissionPhase.CONFIRMING
            self._mission.reason = "go_received_revalidating_remembered_fruit"
            self._demo_go_event.set()
        print("demo_go event=accepted", flush=True)
        return await self.status()

    async def stop_demo(self) -> dict[str, object]:
        await self.stop("demo_operator_stop")
        async with self._state_lock:
            self._mission.phase = MissionPhase.ABORTED
            self._mission.reason = "operator_stop"
        return await self.status()

    async def run_forward_calibration(
        self,
        amount_mps: float,
        confirmation: str,
    ) -> dict[str, object]:
        """Run one factory-avoidance deadband probe with automatic STOP."""

        return await self._run_forward_calibration(
            amount_mps=amount_mps,
            confirmation=confirmation,
            expected_confirmation=FORWARD_CALIBRATION_CONFIRMATION,
            owner="forward_calibration",
            direct_navigation=False,
        )

    async def run_nav2_forward_calibration(
        self,
        amount_mps: float,
        confirmation: str,
    ) -> dict[str, object]:
        """Probe the exact direct SportClient path used by Nav2 return."""

        return await self._run_forward_calibration(
            amount_mps=amount_mps,
            confirmation=confirmation,
            expected_confirmation=NAV2_FORWARD_CALIBRATION_CONFIRMATION,
            owner="nav2_forward_calibration",
            direct_navigation=True,
        )

    async def _run_forward_calibration(
        self,
        *,
        amount_mps: float,
        confirmation: str,
        expected_confirmation: str,
        owner: str,
        direct_navigation: bool,
    ) -> dict[str, object]:
        """Run one short forward-only probe through a selected motion path."""

        amount_mps = float(amount_mps)
        if confirmation.strip().upper() != expected_confirmation:
            raise RuntimeCommandError(
                f'type exactly "{expected_confirmation}"'
            )
        if not math.isfinite(amount_mps) or amount_mps < 0.0:
            raise RuntimeCommandError(
                "forward calibration amount must be finite and non-negative"
            )

        async with self._forward_calibration_lock:
            async with self._action_lock:
                self._require_pointing_idle()
                mission_active = bool(
                    self._mission_task is not None
                    and not self._mission_task.done()
                )
                if mission_active:
                    raise RuntimeCommandError(
                        "stop the fruit mission before forward calibration"
                    )
                if not self.motion_enabled or self.motion is None:
                    raise RuntimeCommandError("motion backend is disabled")
                if self._lease is not None or self.motion.armed:
                    raise RuntimeCommandError(
                        "motion is already owned; stop it first"
                    )
                try:
                    if direct_navigation:
                        lease = await self.motion.arm_direct_navigation()
                    else:
                        lease = await self.motion.arm()
                except MotionError as exc:
                    raise RuntimeCommandError(str(exc)) from exc
                self._lease = lease
                self._motion_owner = owner
                self._navigation_deadline = None
                self._last_pulse_at = None
                self._command = VelocityCommand(
                    reason=f"{owner}_armed_zero"
                )

            started_at = time.monotonic()
            command_count = 0
            sent_amount_mps = 0.0
            try:
                while True:
                    remaining_s = (
                        FORWARD_CALIBRATION_DURATION_S
                        - (time.monotonic() - started_at)
                    )
                    if remaining_s <= 0.0:
                        break
                    async with self._action_lock:
                        if (
                            self.motion is None
                            or self._lease != lease
                            or self._motion_owner != owner
                            or not self.motion.armed
                        ):
                            raise RuntimeCommandError(
                                "forward calibration was stopped"
                            )
                        try:
                            if direct_navigation:
                                self._command = (
                                    await self.motion.send_unbounded_direct_forward_calibration(
                                        lease,
                                        amount_mps,
                                        owner,
                                    )
                                )
                            else:
                                self._command = (
                                    await self.motion.send_unbounded_forward_calibration(
                                        lease,
                                        amount_mps,
                                        owner,
                                    )
                                )
                            sent_amount_mps = self._command.forward_mps
                        except (MotionError, ValueError) as exc:
                            raise RuntimeCommandError(str(exc)) from exc
                    command_count += 1
                    await asyncio.sleep(
                        min(self.follow_period_s, remaining_s)
                    )
            finally:
                await self.stop(f"{owner}_complete")

            navigation = await self.navigation_status()
            print(
                f"{owner} "
                f"requested_amount_mps={amount_mps} "
                f"sent_amount_mps={sent_amount_mps} "
                f"duration_s={FORWARD_CALIBRATION_DURATION_S:.2f} "
                f"commands={command_count} stopped={not navigation['armed']}",
                flush=True,
            )
            return {
                "direction": "forward",
                "motion_path": (
                    "direct_sportclient"
                    if direct_navigation
                    else "factory_avoidance"
                ),
                "requested_amount_mps": amount_mps,
                "amount_mps": sent_amount_mps,
                "pulse_duration_s": FORWARD_CALIBRATION_DURATION_S,
                "movement": None,
                "command_count": command_count,
                "factory_avoidance_enabled": not direct_navigation,
                "calibration_speed_limit_enabled": False,
                "configured_motion_limit_mps": (
                    self.motion.config.maximum_forward_mps
                    if direct_navigation and self.motion is not None
                    else None
                ),
                "stopped": not navigation["armed"],
            }

    async def pulse(self) -> dict[str, object]:
        async with self._action_lock:
            if self._motion_owner != "fruit":
                raise RuntimeCommandError("fruit motion is not armed")
            if (
                self.motion is None
                or self._lease is None
                or not self.motion.armed
            ):
                self._lease = None
                self._motion_owner = None
                if self._command.forward_mps != 0.0 or self._command.yaw_rps != 0.0:
                    self._command = VelocityCommand(reason="motion_not_armed")
                raise RuntimeCommandError("motion is not armed")
            now = time.monotonic()
            async with self._state_lock:
                target = self._target
                frame_width = self._frame_width
            if self._last_pulse_at is not None and self._command.forward_mps > 0.0:
                self._forward_elapsed_s += min(0.25, max(0.0, now - self._last_pulse_at))
            command = self.controller.plan(
                target,
                frame_width=frame_width,
                now_monotonic_s=now,
                forward_elapsed_s=self._forward_elapsed_s,
                allow_unranged_forward=self.allow_unranged_forward,
            )
            unsafe_reasons = {
                "selected_target_not_found",
                "selected_target_not_stable",
                "selected_target_stale",
                "frame_geometry_missing",
                "unranged_forward_disabled",
                "forward_budget_complete",
            }
            if command.reason in unsafe_reasons:
                await self._release_locked(command.reason)
                raise RuntimeCommandError(command.reason)
            try:
                self._command = await self.motion.send(self._lease, command)
            except MotionError as exc:
                self._lease = None
                self._command = VelocityCommand(reason="motion_fault")
                raise RuntimeCommandError(str(exc)) from exc
            self._last_pulse_at = now
            return await self.status()

    async def stop(self, reason: str = "user_stop") -> dict[str, object]:
        if self.pointing is not None and self.pointing.active:
            await self.pointing.stop()
        if self._nav2_return_active and self.nav2_return is not None:
            await self.nav2_return.cancel(reason)
            self._nav2_return_active = False
        current_task = asyncio.current_task()
        mission_task = self._mission_task
        # Target-loss stops are the safety brake for an active approach. Keep
        # the mission monitor alive just long enough to classify the stopped
        # run as reached-camera-edge versus lost-before-arrival. All operator,
        # watchdog, navigation, and shutdown stops still cancel it immediately.
        preserve_mission_for_classification = reason in {
            "selected_target_lost",
            "selected_target_not_revalidated",
        }
        cancelled_mission = bool(
            mission_task is not None
            and mission_task is not current_task
            and not mission_task.done()
            and not preserve_mission_for_classification
        )
        if cancelled_mission and mission_task is not None:
            mission_task.cancel()
            try:
                await mission_task
            except asyncio.CancelledError:
                pass
            if self._mission_task is mission_task:
                self._mission_task = None
        self._follow_start_generation += 1
        async with self._action_lock:
            # Recheck after acquiring the same lock used by policy startup.
            # This closes the race where Stop was requested while startup was
            # between its initial idle check and manager activation.
            if self.pointing is not None and self.pointing.active:
                await self.pointing.stop()
            if self.motion is not None:
                await self.motion.emergency_stop()
            self._lease = None
            self._motion_owner = None
            self._navigation_deadline = None
            self._last_pulse_at = None
            self._command = VelocityCommand(reason=reason)
        if cancelled_mission:
            async with self._state_lock:
                if self._mission.phase not in {
                    MissionPhase.SUCCESS,
                    MissionPhase.ABORTED,
                }:
                    self._mission.phase = MissionPhase.ABORTED
                    self._mission.reason = reason
        return await self.status()

    async def jpeg(self) -> bytes | None:
        async with self._state_lock:
            return self._jpeg

    async def wait_for_stream_frame(
        self, last_frame_id: int
    ) -> tuple[int, bytes] | None:
        """Wait for the next low-latency camera JPEG for an MJPEG client."""

        async with self._stream_condition:
            await self._stream_condition.wait_for(
                lambda: self._closing
                or (
                    self._stream_jpeg is not None
                    and self._stream_frame_id > last_frame_id
                )
            )
            if self._closing or self._stream_jpeg is None:
                return None
            return self._stream_frame_id, self._stream_jpeg

    async def raw_jpeg(self) -> bytes | None:
        """Return the latest unannotated frame for evaluation and capture."""
        async with self._state_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        if frame.source_jpeg is not None:
            return frame.source_jpeg
        encoded_ok, encoded = await asyncio.to_thread(
            lambda: cv2.imencode(
                ".jpg",
                frame.bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
        )
        if not encoded_ok:
            raise RuntimeError("could not encode raw camera frame")
        return encoded.tobytes()

    async def status(self) -> dict[str, object]:
        camera_telemetry: dict[str, object] = {}
        telemetry = getattr(self.camera, "telemetry", None)
        if callable(telemetry):
            try:
                rendered_telemetry = telemetry()
                if isinstance(rendered_telemetry, dict):
                    camera_telemetry = dict(rendered_telemetry)
            except Exception as exc:
                camera_telemetry = {"telemetry_error": str(exc)}
        camera_stream_source = str(camera_telemetry.get("source") or "").strip()
        camera_stream = {
            "unitree_video_client_rpc": "sdk_jpeg_passthrough",
            "voice_webrtc_camera_broker": "webrtc_broker_jpeg_passthrough",
        }.get(camera_stream_source, camera_stream_source or "sdk_jpeg_passthrough")
        heading = (
            None
            if self.heading_provider is None
            else self.heading_provider.status()
        )
        async with self._state_lock:
            now = time.monotonic()
            frame_age = None if self._last_frame_at is None else now - self._last_frame_at
            camera_fps = None
            if len(self._camera_frame_times) >= 2:
                camera_window_s = (
                    self._camera_frame_times[-1] - self._camera_frame_times[0]
                )
                if camera_window_s > 0.0:
                    camera_fps = (
                        len(self._camera_frame_times) - 1
                    ) / camera_window_s
            target_age = None if self._target is None else now - self._target.captured_monotonic_s
            produce_age = (
                None if self._produce_last_at is None else now - self._produce_last_at
            )
            target = self._target
            device_status = self._produce_device_status()
            camera_live = frame_age is not None and frame_age < 1.0
            produce_live = bool(
                self.produce_detector is not None
                and produce_age is not None
                and produce_age < 1.0
                and not self._produce_error
            )
            gpu_ready = bool(
                device_status.get("cuda_available")
                and str(device_status.get("resolved", "")).startswith("cuda")
            )
            motion_status = None if self.motion is None else self.motion.status()
            motion_ready = bool(
                self.motion_enabled
                and motion_status is not None
                and motion_status["initialized"]
                and motion_status["fault"] is None
            )
            nav2_required = bool(
                self.mission_config.return_home_enabled
                and self.mission_config.return_backend == "nav2"
            )
            nav2_ready = bool(
                not nav2_required
                or (
                    self._nav2_health.get("ready")
                    and self._nav2_health_at is not None
                    and now - self._nav2_health_at < 1.5
                )
            )
            pointing_active = bool(
                self.pointing is not None and self.pointing.active
            )
            follow_active = bool(
                self._follow_task is not None
                and not self._follow_task.done()
                and self._lease is not None
                and self.motion is not None
                and self.motion.armed
            )
            follow_readiness = self._follow_readiness_locked(now)
            can_follow = (
                not follow_active
                and not pointing_active
                and follow_readiness is None
                and motion_ready
            )
            stage_ready = (
                camera_live
                and produce_live
                and gpu_ready
                and motion_ready
                and nav2_ready
            )
            mission_active = bool(
                self._mission_task is not None and not self._mission_task.done()
            )
            if not self.mission_config.enabled:
                demo_readiness = "fruit-memory demo disabled"
            elif not self.mission_config.autonomous_turn_enabled:
                demo_readiness = "autonomous turn disabled"
            elif self._fruit_memory is None:
                demo_readiness = "save a fruit first"
            elif self._arrival_pointing_required(self._fruit_memory.label) and (
                self.pointing is None or not self.pointing.available
            ):
                demo_readiness = "arrival pointing policy unavailable"
            elif self._mission.initial_hello_status == "running":
                demo_readiness = "wait for Woof's hello gesture to finish"
            elif heading is None or not heading.healthy:
                demo_readiness = "fresh Go2 heading unavailable"
            elif (
                self.mission_config.return_home_enabled
                and not heading.pose_healthy
            ):
                demo_readiness = "fresh Go2 local position unavailable"
            elif not nav2_ready:
                demo_readiness = str(
                    self._nav2_health.get("reason")
                    or "Nav2 return is not ready"
                )
            elif not stage_ready:
                demo_readiness = "stage health is not ready"
            elif mission_active:
                demo_readiness = "demo already active"
            elif pointing_active:
                demo_readiness = "pointing policy owns motion"
            elif self._lease is not None or (self.motion is not None and self.motion.armed):
                demo_readiness = "motion is already armed"
            else:
                demo_readiness = "ready"
            mission_status = self._mission.to_dict(now)
            mission_status.update(
                {
                    "enabled": self.mission_config.enabled,
                    "autonomous_turn_enabled": self.mission_config.autonomous_turn_enabled,
                    "direct_turn_enabled": self.mission_config.direct_turn_enabled,
                    "target_policy": "same_class",
                    "turn_rate_rps": self.mission_config.turn_rate_rps,
                    "turn_timeout_s": self.mission_config.turn_timeout_s,
                    "turn_stall_timeout_s": self.mission_config.turn_stall_timeout_s,
                    "turn_stall_min_progress_deg": round(
                        math.degrees(
                            self.mission_config.turn_stall_min_progress_rad
                        ),
                        1,
                    ),
                    "search_rate_rps": self.mission_config.search_rate_rps,
                    "search_sweep_deg": round(
                        math.degrees(self.mission_config.search_sweep_rad), 1
                    ),
                    "search_timeout_s": self.mission_config.search_timeout_s,
                    "near_bottom_ratio": self.mission_config.near_bottom_ratio,
                    "near_center_ratio": self.mission_config.near_center_ratio,
                    "near_bbox_height_ratio": (
                        self.mission_config.near_bbox_height_ratio
                    ),
                    "near_confirmations_required": (
                        self.mission_config.near_confirmations_required
                    ),
                    "near_loss_grace_s": self.mission_config.near_loss_grace_s,
                    "arrival_pointing_enabled": (
                        self.mission_config.arrival_pointing_enabled
                    ),
                    "arrival_pointing_label": (
                        self.mission_config.arrival_pointing_label
                    ),
                    "arrival_pointing_timeout_s": (
                        self.mission_config.arrival_pointing_timeout_s
                    ),
                    "arrival_rest_enabled": (
                        self.mission_config.arrival_rest_enabled
                    ),
                    "arrival_rest_duration_s": (
                        self.mission_config.arrival_rest_duration_s
                    ),
                    "final_push_mps": self.mission_config.final_push_mps,
                    "final_push_duration_s": (
                        self.mission_config.final_push_duration_s
                    ),
                    "active": mission_active,
                    "return_home_enabled": self.mission_config.return_home_enabled,
                    "return_backend": self.mission_config.return_backend,
                    "nav2_health": dict(self._nav2_health),
                    "return_arrival_tolerance_m": (
                        self.mission_config.return_arrival_tolerance_m
                    ),
                    "return_timeout_s": self.mission_config.return_timeout_s,
                    "can_remember": bool(
                        self.mission_config.enabled
                        and produce_live
                        and self._produce_detections
                        and not mission_active
                    ),
                    "can_start": demo_readiness == "ready",
                    "readiness": demo_readiness,
                    "confirmation": DEMO_CONFIRMATION,
                    "can_go": bool(
                        mission_active
                        and self._mission.phase == MissionPhase.WAITING_FOR_GO
                        and stage_ready
                    ),
                    "go_readiness": (
                        "ready"
                        if (
                            mission_active
                            and self._mission.phase == MissionPhase.WAITING_FOR_GO
                            and stage_ready
                        )
                        else (
                            "waiting for the saved fruit class"
                            if self._mission.phase != MissionPhase.WAITING_FOR_GO
                            else "stage health is not ready"
                        )
                    ),
                    "go_confirmation": DEMO_GO_CONFIRMATION,
                    "heading": None if heading is None else heading.to_dict(),
                }
            )
            pointing_status: dict[str, object]
            if self.pointing is None:
                pointing_status = {
                    "available": False,
                    "enabled": False,
                    "active": False,
                    "prepared": False,
                    "phase": "unavailable",
                    "error": "pointing policy is not configured",
                    "can_prepare": False,
                    "can_run": False,
                    "readiness": "pointing policy is not configured",
                    "prepare_confirmation": POINTING_PREPARE_CONFIRMATION,
                    "run_confirmation": POINTING_RUN_CONFIRMATION,
                }
            else:
                pointing_status = dict(self.pointing.status())
                if not pointing_status["available"]:
                    pointing_readiness = str(
                        pointing_status.get("error")
                        or "pointing policy is unavailable"
                    )
                elif pointing_active:
                    pointing_readiness = "pointing policy is active"
                elif not motion_ready:
                    pointing_readiness = "motion backend is not ready"
                elif mission_active:
                    pointing_readiness = "stop the fruit mission first"
                elif self._lease is not None or (
                    self.motion is not None and self.motion.armed
                ):
                    pointing_readiness = "stop current motion first"
                elif not pointing_status["prepared"]:
                    pointing_readiness = "prepare the standing-point handoff"
                elif follow_readiness is not None:
                    pointing_readiness = follow_readiness
                elif target is None or target.confidence is None:
                    pointing_readiness = (
                        "selected target has no fresh YOLO confidence"
                    )
                else:
                    pointing_readiness = "ready"
                pointing_status.update(
                    {
                        "can_prepare": bool(
                            pointing_status["available"]
                            and motion_ready
                            and not pointing_active
                            and not mission_active
                            and self._lease is None
                            and not (
                                self.motion is not None and self.motion.armed
                            )
                        ),
                        "can_run": pointing_readiness == "ready",
                        "readiness": pointing_readiness,
                        "prepare_confirmation": POINTING_PREPARE_CONFIRMATION,
                        "run_confirmation": POINTING_RUN_CONFIRMATION,
                    }
                )
            return {
                "ok": camera_live,
                "runtime_id": self._runtime_id,
                "round_id": self._round_id,
                "stage_ready": stage_ready,
                "health": {
                    "camera_live": camera_live,
                    "produce_live": produce_live,
                    "gpu_ready": gpu_ready,
                    "motion_ready": motion_ready,
                    "nav2_ready": nav2_ready,
                },
                "frame_count": self._frame_count,
                "frame_width": self._frame_width,
                "frame_height": self._frame_height,
                "camera_fps": None if camera_fps is None else round(camera_fps, 1),
                "camera_stream": camera_stream,
                "camera_rpc": camera_telemetry,
                "camera_stream_generation": self._camera_stream_generation,
                "frame_age_s": None if frame_age is None else round(frame_age, 3),
                "last_error": self._last_error,
                "selected_target_name": self._selected_target_name,
                "target_lock_id": self._target_lock_id
                if self._selected_target_name is not None
                else None,
                "selected_target_kind": "produce"
                if self._selected_target_name is not None
                else None,
                "selected_target_hint": self._selected_target_hint,
                "selected_target": None if target is None else target.to_dict(),
                "selected_target_age_s": None
                if target_age is None
                else round(target_age, 3),
                "produce": None
                if self.produce_detector is None
                else {
                    "model_path": str(self.produce_detector.model_path),
                    "confidence_threshold": self.produce_detector.confidence,
                    "class_thresholds": getattr(
                        self.produce_detector, "class_thresholds", {}
                    ),
                    "classes": self.produce_detector.names,
                    "detections": [item.to_dict() for item in self._produce_detections],
                    "frame_id": self._produce_frame_id,
                    "age_s": None if produce_age is None else round(produce_age, 3),
                    "inference_ms": self._produce_inference_ms,
                    "error": self._produce_error,
                    "device": device_status,
                    "tracker": {
                        "mode": "yolo+local"
                        if self.produce_tracker_factory is not None
                        else "yolo",
                        "label": self._produce_tracker_label,
                        "target": None
                        if self._produce_tracker_label is None or target is None
                        else target.to_dict(),
                        "last_verified_age_s": None
                        if self._produce_verified_at is None
                        else round(now - self._produce_verified_at, 3),
                        "revalidation_failures": self._produce_revalidation_failures,
                        "revalidation_failures_required": self.produce_revalidation_misses_required,
                    },
                },
                "motion_enabled": self.motion_enabled,
                "allow_unranged_forward": self.allow_unranged_forward,
                "can_follow": can_follow,
                "follow_readiness": "ready"
                if follow_readiness is None
                else follow_readiness,
                "follow_active": follow_active,
                "armed": self._lease is not None and self.motion is not None and self.motion.armed,
                "motion_owner": "pointing" if pointing_active else self._motion_owner,
                "memory": None
                if self._fruit_memory is None
                else self._fruit_memory.to_status(now),
                "voice": dict(self._voice_status),
                "mission": mission_status,
                "pointing": pointing_status,
                "navigation": await self.navigation_status(),
                "command": self._command.to_dict(),
                "forward_budget_s": self.controller.config.forward_budget_s,
                "forward_elapsed_s": round(self._forward_elapsed_s, 3),
                "forward_remaining_s": round(
                    max(0.0, self.controller.config.forward_budget_s - self._forward_elapsed_s),
                    3,
                ),
                "arm_confirmation": ARM_CONFIRMATION,
                "motion": motion_status,
            }

    async def _camera_loop(self) -> None:
        period = 1.0 / self.loop_hz
        annotated_period = 1.0 / self.annotated_hz
        while not self._closing:
            started = time.monotonic()
            lost_while_armed = False
            try:
                frame = await asyncio.to_thread(self.camera.read)
                generation_changed = False
                if frame.stream_generation is not None:
                    async with self._state_lock:
                        previous_generation = self._camera_stream_generation
                        if previous_generation is None:
                            self._camera_stream_generation = frame.stream_generation
                        elif previous_generation != frame.stream_generation:
                            generation_changed = True
                            self._camera_stream_generation = frame.stream_generation
                            self._follow_start_generation += 1
                            self._target_lock_id += 1
                            self._latest_frame = None
                            self._jpeg = None
                            self._frame_width = None
                            self._frame_height = None
                            self._last_frame_at = None
                            self._camera_frame_times.clear()
                            self._produce_detections = []
                            self._produce_frame = None
                            self._produce_frame_id = None
                            self._produce_last_at = None
                            self._produce_error = (
                                "camera generation changed; waiting for fresh inference"
                            )
                            self._clear_selection_locked()
                            self._last_error = "camera generation changed"
                    if generation_changed:
                        async with self._stream_condition:
                            self._stream_jpeg = None
                            self._stream_frame_id = 0
                            self._stream_condition.notify_all()
                        # Discard the first frame from the new connection. Stop
                        # and invalidate the old mission before accepting any
                        # camera-derived state from the replacement stream.
                        await self.stop("camera_generation_changed")
                        continue
                async with self._state_lock:
                    self._latest_frame = frame
                    self._frame_width = frame.width
                    self._frame_height = frame.height
                    self._frame_count += 1
                    self._last_frame_at = frame.captured_monotonic_s
                    self._camera_frame_times.append(frame.captured_monotonic_s)
                    self._last_error = ""
                    produce_detections = list(self._produce_detections)
                    selected_name = self._selected_target_name
                    tracker = self._produce_tracker
                    tracker_label = self._produce_tracker_label
                    tracker_visible_frames = self._produce_visible_frames
                    current_target = self._target
                if frame.source_jpeg is not None:
                    async with self._stream_condition:
                        self._stream_jpeg = frame.source_jpeg
                        self._stream_frame_id = frame.frame_id
                        self._stream_condition.notify_all()
                tracked_produce: TargetObservation | None = None
                if (
                    selected_name is not None
                    and tracker is not None
                    and tracker_label == selected_name
                ):
                    tracker_ok, tracker_bbox = await asyncio.to_thread(
                        _update_tracker_frame, tracker, frame
                    )
                    if tracker_ok:
                        tracked_produce = self._observation_from_bbox(
                            frame,
                            tuple(int(round(value)) for value in tracker_bbox),
                            confidence=None,
                            visible_frames=tracker_visible_frames + 1,
                        )
                now = time.monotonic()
                should_annotate = (
                    self._last_annotated_at is None
                    or now - self._last_annotated_at >= annotated_period
                )
                jpeg = frame.source_jpeg
                if should_annotate and jpeg is None:
                    jpeg = await asyncio.to_thread(
                        _encode_legacy_frame,
                        frame,
                        produce_detections,
                        selected_name,
                        tracked_produce
                        if tracked_produce is not None
                        else current_target,
                    )
                    self._last_annotated_at = time.monotonic()
                async with self._state_lock:
                    current_name = self._selected_target_name
                    if (
                        current_name is not None
                        and current_name == selected_name
                        and self._produce_tracker is tracker
                        and self._produce_tracker_label == current_name
                    ):
                        selected_target = tracked_produce
                        self._produce_visible_frames = (
                            0 if tracked_produce is None else tracked_produce.visible_frames
                        )
                        if tracked_produce is not None:
                            # Keep the reacquisition hint attached to the chosen
                            # detection as the tracker follows it between YOLO
                            # inference frames.
                            self._selected_target_hint = tracked_produce.center
                        if tracked_produce is None:
                            if (
                                self._motion_owner == "fruit"
                                and self._lease is not None
                            ):
                                self._clear_selection_locked()
                            else:
                                self._mark_selection_stale_locked()
                    else:
                        selected_target = self._target
                    if jpeg is not None:
                        self._jpeg = jpeg
                    self._target = selected_target
                    self._last_error = ""
                if frame.source_jpeg is None and jpeg is not None:
                    async with self._stream_condition:
                        self._stream_jpeg = jpeg
                        self._stream_frame_id = frame.frame_id
                        self._stream_condition.notify_all()
                lost_while_armed = self._motion_owner == "fruit" and self._lease is not None and (
                    selected_target is None
                    or selected_target.visible_frames
                    < self.controller.config.stable_frames_required
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with self._state_lock:
                    self._last_error = str(exc)
                    # Unitree's camera occasionally returns one malformed JPEG.
                    # A single bad sample must not erase an otherwise fresh,
                    # YOLO-verified selection.  Preserve the last observation
                    # until the same bounded target-age rule used by the motion
                    # controller says the camera outage is genuinely stale.
                    now = time.monotonic()
                    last_frame_age = (
                        None
                        if self._last_frame_at is None
                        else now - self._last_frame_at
                    )
                    camera_is_stale = (
                        last_frame_age is None
                        or last_frame_age
                        > self.controller.config.maximum_target_age_s
                    )
                    if camera_is_stale:
                        if (
                            self._motion_owner == "fruit"
                            and self._lease is not None
                        ):
                            self._clear_selection_locked()
                        else:
                            self._mark_selection_stale_locked()
                lost_while_armed = (
                    camera_is_stale
                    and self._motion_owner == "fruit"
                    and self._lease is not None
                )
            if lost_while_armed:
                await self.stop("selected_target_lost")
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    async def _produce_loop(self) -> None:
        assert self.produce_detector is not None
        last_frame_id = 0
        while not self._closing:
            async with self._state_lock:
                frame = self._latest_frame
            if frame is None or frame.frame_id == last_frame_id:
                await asyncio.sleep(0.01)
                continue
            last_frame_id = frame.frame_id
            started = time.monotonic()
            try:
                detections = await asyncio.to_thread(
                    _detect_frame, self.produce_detector, frame
                )
                inference_ms = (time.monotonic() - started) * 1000.0
                for detection in detections:
                    print(detection.log_line(), flush=True)
                refresh_tracker_for: tuple[str, FruitDetection, int] | None = None
                revalidation_failed = False
                async with self._state_lock:
                    if (
                        frame.stream_generation is not None
                        and frame.stream_generation
                        != self._camera_stream_generation
                    ):
                        continue
                    self._produce_detections = detections
                    self._produce_frame = frame
                    self._produce_frame_id = frame.frame_id
                    self._produce_last_at = time.monotonic()
                    self._produce_inference_ms = round(inference_ms, 1)
                    self._produce_error = ""
                    selected_name = self._selected_target_name
                    selected_target = self._target
                    selected_detection = None
                    if selected_name is not None:
                        preferred_center = (
                            selected_target.center
                            if selected_target is not None
                            else self._selected_target_hint
                        )
                        selected_detection = self._best_produce_detection_locked(
                            selected_name, preferred_center
                        )
                        tracker_is_active = (
                            self._produce_tracker is not None
                            and self._produce_tracker_label == selected_name
                        )
                        detection_matches_track = (
                            selected_detection is not None
                            and (
                                selected_target is None
                                # The production Go2 path intentionally has no
                                # CPU tracker: YOLO is the authoritative
                                # class-level track.  During a fast approach a
                                # small fruit can move by more than one box
                                # width between 8 Hz detections, so requiring
                                # IoU here falsely expires a continuously
                                # detected target.  A configured local tracker
                                # still keeps the stricter spatial check.
                                or not tracker_is_active
                                or self._target_detection_iou(
                                    selected_target, selected_detection
                                )
                                >= self.produce_revalidation_iou
                            )
                        )
                        if selected_detection is None or (
                            selected_target is not None and not detection_matches_track
                        ):
                            self._produce_revalidation_failures += 1
                            if self._produce_revalidation_expired_locked(
                                time.monotonic()
                            ):
                                if (
                                    self._motion_owner == "fruit"
                                    and self._lease is not None
                                ):
                                    self._clear_selection_locked()
                                else:
                                    self._mark_selection_stale_locked()
                                revalidation_failed = True
                        else:
                            self._produce_revalidation_failures = 0
                            visible_frames = (
                                1
                                if selected_target is None
                                else selected_target.visible_frames + 1
                            )
                            observation = self._observation_from_bbox(
                                frame,
                                self._detection_bbox_xywh(selected_detection),
                                confidence=selected_detection.confidence,
                                visible_frames=visible_frames,
                            )
                            if (
                                self._target is None
                                or observation.captured_monotonic_s
                                >= self._target.captured_monotonic_s
                            ):
                                self._target = observation
                            self._produce_visible_frames = visible_frames
                            self._produce_verified_at = time.monotonic()
                            self._selected_target_hint = selected_detection.center
                            if (
                                self.produce_tracker_factory is not None
                                and not tracker_is_active
                            ):
                                refresh_tracker_for = (
                                    selected_name,
                                    selected_detection,
                                    visible_frames,
                                )
                if revalidation_failed:
                    await self._handle_produce_revalidation_failure()
                if refresh_tracker_for is not None:
                    refresh_name, selected_detection, visible_frames = refresh_tracker_for
                    tracker_factory = self.produce_tracker_factory
                    if tracker_factory is None:
                        continue
                    tracker = await asyncio.to_thread(
                        _create_tracker_for_frame,
                        tracker_factory,
                        frame,
                        self._detection_bbox_xywh(selected_detection),
                    )
                    async with self._state_lock:
                        if (
                            self._selected_target_name == refresh_name
                            and refresh_name is not None
                            and (
                                frame.stream_generation is None
                                or frame.stream_generation
                                == self._camera_stream_generation
                            )
                        ):
                            if self._produce_tracker is None:
                                self._produce_tracker = tracker
                                self._produce_tracker_label = refresh_name
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                revalidation_failed = False
                async with self._state_lock:
                    if (
                        frame.stream_generation is not None
                        and frame.stream_generation
                        != self._camera_stream_generation
                    ):
                        continue
                    self._produce_detections = []
                    self._produce_frame = frame
                    self._produce_frame_id = frame.frame_id
                    self._produce_last_at = time.monotonic()
                    self._produce_inference_ms = round(
                        (time.monotonic() - started) * 1000.0, 1
                    )
                    self._produce_error = str(exc)
                    if self._selected_target_name is not None:
                        self._produce_revalidation_failures += 1
                        if self._produce_revalidation_expired_locked(
                            time.monotonic()
                        ):
                            if (
                                self._motion_owner == "fruit"
                                and self._lease is not None
                            ):
                                self._clear_selection_locked()
                            else:
                                self._mark_selection_stale_locked()
                            revalidation_failed = True
                if revalidation_failed:
                    await self._handle_produce_revalidation_failure()
            await asyncio.sleep(0)

    async def _handle_produce_revalidation_failure(self) -> None:
        """Leave bbox-loss enforcement to the isolated pointing runner.

        The runner owns low-level motion while a pointing policy is active and
        independently aborts unless it receives a fresh, same-class box within
        0.8 seconds. Calling the general runtime stop path here races policy
        startup and can interrupt a cold Torch import before the runner has
        created its safety report. Locomotion still stops immediately through
        the existing runtime path.
        """

        if self.pointing is not None and self.pointing.active:
            print(
                "produce event=revalidation_failure "
                "action=delegated_to_pointing_runner",
                flush=True,
            )
            return
        await self.stop("selected_target_not_revalidated")

    async def _demo_loop(self) -> None:
        current_task = asyncio.current_task()
        try:
            await self._run_measured_turn()
            match = await self._search_for_memory()
            match = await self._acknowledge_memory_match(match)
            match = await self._wait_for_demo_go(match)
            await self._start_memory_approach(match)
            approach_reason = await self._monitor_memory_approach()
            await self.stop("demo_visual_approach_complete")
            used_pointing_policy = (
                approach_reason == "target_visible_in_pointing_range"
            )
            if used_pointing_policy:
                approach_reason = await self._run_arrival_pointing()
            else:
                approach_reason = await self._run_final_approach(approach_reason)
            await self.stop("demo_target_reached")
            if used_pointing_policy:
                async with self._state_lock:
                    self._mission.arrival_hello_status = (
                        "replaced_by_pointing_policy"
                    )
            if self.mission_config.arrival_rest_enabled:
                await self._rest_at_target()
                if not used_pointing_policy:
                    async with self._state_lock:
                        self._mission.arrival_hello_status = (
                            "replaced_by_arrival_rest"
                        )
            elif not used_pointing_policy:
                await self._celebrate_target_reached()
            if self.mission_config.return_home_enabled:
                await self._return_home()
                success_reason = (
                    "pointing_reach_attempt_complete_returned_home_"
                    "contact_unverified"
                    if used_pointing_policy
                    else "remembered_fruit_reached_and_returned_home"
                )
            else:
                success_reason = approach_reason
            async with self._state_lock:
                self._mission.phase = MissionPhase.SUCCESS
                self._mission.reason = success_reason
        except asyncio.CancelledError:
            raise
        except RuntimeCommandError as exc:
            reason = str(exc)
            await self.stop(f"demo_abort:{reason}")
            async with self._state_lock:
                if self._mission.phase == MissionPhase.FINAL_APPROACHING:
                    self._mission.final_approach_status = "failed"
                if self._mission.phase == MissionPhase.POINTING:
                    self._mission.arrival_pointing_status = "failed"
                    self._mission.arrival_pointing_error = reason
                    self._mission.contact_status = "not_confirmed"
                if self._mission.phase == MissionPhase.RETURNING_HOME:
                    self._mission.return_home_status = "failed"
                self._mission.phase = MissionPhase.ABORTED
                self._mission.reason = reason
        except Exception as exc:
            reason = f"demo_internal_error:{exc}"
            await self.stop(reason)
            async with self._state_lock:
                if self._mission.phase == MissionPhase.FINAL_APPROACHING:
                    self._mission.final_approach_status = "failed"
                if self._mission.phase == MissionPhase.POINTING:
                    self._mission.arrival_pointing_status = "failed"
                    self._mission.arrival_pointing_error = reason
                    self._mission.contact_status = "not_confirmed"
                if self._mission.phase == MissionPhase.RETURNING_HOME:
                    self._mission.return_home_status = "failed"
                self._mission.phase = MissionPhase.ABORTED
                self._mission.reason = reason
                self._last_error = reason
        finally:
            if self._mission_task is current_task:
                self._mission_task = None

    async def _run_arrival_pointing(self) -> str:
        """Run the standing point while the selected fruit box is still live."""

        async with self._state_lock:
            memory = self._fruit_memory
        if memory is None:
            raise RuntimeCommandError("saved fruit memory disappeared")
        if not self._arrival_pointing_required(memory.label):
            raise RuntimeCommandError(
                f"arrival pointing is not enabled for {memory.label}"
            )
        if self.pointing is None or not self.pointing.available:
            raise RuntimeCommandError("arrival pointing policy is unavailable")
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")

        async with self._state_lock:
            self._mission.phase = MissionPhase.POINTING
            self._mission.reason = "preparing_guarded_standing_point"
            self._mission.arrival_pointing_status = "preparing"
            self._mission.arrival_pointing_error = None
            self._mission.arrival_pointing_target = memory.label
            self._mission.contact_status = "pending"
        print(
            "arrival_pointing event=prepare "
            f"label={memory.label} contact_sensor=false",
            flush=True,
        )

        try:
            async with self._exclusive_skill_lock:
                await self._prepare_pointing(POINTING_PREPARE_CONFIRMATION)
                async with self._state_lock:
                    self._mission.reason = "running_locked_standing_point_policy"
                    self._mission.arrival_pointing_status = "running"
                await self._start_pointing(
                    POINTING_RUN_CONFIRMATION,
                    allow_current_mission=True,
                )
                status = await self.pointing.wait(
                    timeout_s=self.mission_config.arrival_pointing_timeout_s
                )
                report = status.get("last_report")
                controller_restored = bool(
                    isinstance(report, dict)
                    and report.get("controller_restored")
                )
                if status.get("phase") != "complete" or not controller_restored:
                    raise PointingPolicyError(
                        str(
                            status.get("error")
                            or "pointing policy did not complete with Sport restored"
                        )
                    )
                if self.mission_config.return_home_enabled:
                    await self.motion.perform_stand_up()
        except (MotionError, PointingPolicyError, RuntimeCommandError) as exc:
            self.pointing.clear_prepared()
            async with self._state_lock:
                self._mission.arrival_pointing_status = "failed"
                self._mission.arrival_pointing_error = str(exc)
                self._mission.contact_status = "not_confirmed"
            print(
                f"arrival_pointing event=failed error={exc}",
                flush=True,
            )
            raise RuntimeCommandError(f"arrival pointing failed: {exc}") from exc

        async with self._state_lock:
            self._mission.arrival_pointing_status = "complete"
            self._mission.arrival_pointing_error = None
            self._mission.contact_status = "unverified"
            self._mission.reason = (
                "pointing_reach_attempt_complete_contact_unverified"
            )
        print(
            "arrival_pointing event=complete "
            f"label={memory.label} contact_status=unverified",
            flush=True,
        )
        return "pointing_reach_attempt_complete_contact_unverified"

    async def _rest_at_target(self) -> None:
        """Lie down visibly, hold, then stand before the return-home leg."""

        if not self.mission_config.arrival_rest_enabled:
            return
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")
        async with self._state_lock:
            self._mission.phase = MissionPhase.CELEBRATING
            self._mission.reason = "laying_down_at_reached_fruit"
            self._mission.arrival_rest_status = "laying_down"
            self._mission.arrival_rest_error = None
        print(
            "arrival_rest event=standdown "
            f"hold_s={self.mission_config.arrival_rest_duration_s:.2f}",
            flush=True,
        )
        try:
            async with self._exclusive_skill_lock:
                await self.motion.perform_standdown()
                async with self._state_lock:
                    self._mission.reason = "resting_at_reached_fruit"
                    self._mission.arrival_rest_status = "holding"
                await asyncio.sleep(self.mission_config.arrival_rest_duration_s)
                async with self._state_lock:
                    self._mission.reason = "standing_for_return_home"
                    self._mission.arrival_rest_status = "standing_up"
                await self.motion.perform_stand_up()
        except MotionError as exc:
            async with self._state_lock:
                self._mission.arrival_rest_status = "failed"
                self._mission.arrival_rest_error = str(exc)
            print(f"arrival_rest event=failed error={exc}", flush=True)
            raise RuntimeCommandError(f"arrival rest failed: {exc}") from exc
        async with self._state_lock:
            self._mission.reason = "arrival_rest_complete"
            self._mission.arrival_rest_status = "complete"
            self._mission.arrival_rest_error = None
        print("arrival_rest event=complete", flush=True)

    async def _celebrate_target_reached(self) -> None:
        """Acknowledge a verified arrival while locomotion remains disarmed."""

        if not self.mission_config.arrival_hello_enabled:
            return
        async with self._state_lock:
            self._mission.phase = MissionPhase.CELEBRATING
            self._mission.reason = "celebrating_verified_fruit_arrival"
            self._mission.arrival_hello_status = "running"
            self._mission.arrival_hello_error = None
        print(
            "arrival_hello event=start "
            f"settle_s={self.mission_config.arrival_hello_settle_s:.2f}",
            flush=True,
        )
        error: str | None = None
        try:
            if not self.motion_enabled or self.motion is None:
                raise MotionNotReady("motion backend is disabled")
            if self.mission_config.arrival_hello_settle_s:
                await asyncio.sleep(self.mission_config.arrival_hello_settle_s)
            await self.motion.perform_hello()
        except MotionError as exc:
            error = str(exc)
        async with self._state_lock:
            self._mission.arrival_hello_status = (
                "complete" if error is None else "failed"
            )
            self._mission.arrival_hello_error = error
            self._mission.reason = (
                "verified_fruit_arrival_acknowledged"
                if error is None
                else "verified_fruit_arrival_hello_failed"
            )
        if error is None:
            print("arrival_hello event=complete", flush=True)
        else:
            # Arrival is already established and all velocity owners are gone.
            # A cosmetic SDK action failure must not block the safe return path.
            print(
                f"arrival_hello event=failed nonfatal=true error={error}",
                flush=True,
            )

    async def _wait_for_demo_go(self, match: ClassCandidate) -> ClassCandidate:
        """Hold at zero motion until the operator explicitly presses Go."""

        await self.stop("demo_waiting_for_go")
        self._demo_go_event.clear()
        async with self._state_lock:
            self._mission.phase = MissionPhase.WAITING_FOR_GO
            self._mission.reason = "remembered_fruit_locked_waiting_for_go"
        print(
            "demo_go event=waiting "
            f"label={match.detection.label} "
            f"center={match.detection.center} "
            f"confidence={match.detection.confidence:.4f}",
            flush=True,
        )
        await self._demo_go_event.wait()
        async with self._state_lock:
            voice_mission = (
                self._voice_mission_generation == self._round_generation
            )
        if voice_mission:
            # Voice auto-Go is released immediately after the search stage
            # established a fresh, stable multi-frame class lock. Requiring a
            # second pair of detector frames here is redundant and creates a
            # race with Unitree VideoClient's occasional multi-second RPC
            # stall. Manual Go may have an arbitrary operator delay and still
            # uses the stricter fresh-frame reacquisition path below.
            async with self._state_lock:
                self._mission.phase = MissionPhase.CONFIRMING
                self._mission.reason = (
                    "voice_target_class_lock_reused_after_auto_go"
                )
            print(
                "demo_go event=voice_lock_reused "
                f"label={match.detection.label} "
                f"center={match.detection.center} "
                f"confidence={match.detection.confidence:.4f}",
                flush=True,
            )
            return match
        return await self._reacquire_memory(
            event_name="demo_go",
            success_reason="target_class_reacquired_after_go",
            timeout_reason=(
                "saved fruit class was not reacquired after Go; approach blocked"
            ),
        )

    async def _run_measured_turn(self) -> None:
        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        initial = self.heading_provider.status()
        if not initial.healthy or initial.yaw_rad is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        async with self._state_lock:
            self._mission.phase = MissionPhase.TURNING
            self._mission.reason = "measured_180_degree_turn"
            self._mission.turn_progress_rad = 0.0
        if self.mission_config.direct_turn_enabled:
            await self._direct_turn_arm()
        else:
            await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        direction = 1.0
        turn_started = time.monotonic()
        deadline = turn_started + self.mission_config.turn_timeout_s
        last_log_at = 0.0
        print(
            "turn event=start "
            f"initial_yaw_rad={initial.yaw_rad:.4f} "
            f"command_yaw_rps={direction * self.mission_config.turn_rate_rps:.3f} "
            f"target_deg={math.degrees(self.mission_config.turn_angle_rad):.1f} "
            f"timeout_s={self.mission_config.turn_timeout_s:.2f} "
            f"mode={'direct_yaw' if self.mission_config.direct_turn_enabled else 'avoidance'}",
            flush=True,
        )
        while time.monotonic() < deadline:
            now = time.monotonic()
            sample = self.heading_provider.status()
            if not sample.healthy or sample.yaw_rad is None:
                raise RuntimeCommandError("Go2 heading became stale during turn")
            progress = directed_progress(initial.yaw_rad, sample.yaw_rad, direction)
            async with self._state_lock:
                self._mission.turn_progress_rad = progress
            if (
                progress
                >= self.mission_config.turn_angle_rad
                - self.mission_config.turn_tolerance_rad
            ):
                print(
                    "turn event=complete "
                    f"elapsed_s={now - turn_started:.3f} "
                    f"current_yaw_rad={sample.yaw_rad:.4f} "
                    f"progress_deg={math.degrees(progress):.1f}",
                    flush=True,
                )
                await self.stop("demo_turn_complete")
                return
            elapsed = now - turn_started
            if (
                self.mission_config.direct_turn_enabled
                and elapsed >= self.mission_config.turn_stall_timeout_s
                and progress < self.mission_config.turn_stall_min_progress_rad
            ):
                print(
                    "turn event=stalled "
                    f"elapsed_s={elapsed:.3f} "
                    f"current_yaw_rad={sample.yaw_rad:.4f} "
                    f"progress_deg={math.degrees(progress):.1f}",
                    flush=True,
                )
                raise RuntimeCommandError(
                    "direct turn stalled at "
                    f"{math.degrees(progress):.1f} degrees"
                )
            if self.mission_config.direct_turn_enabled:
                await self._direct_turn_command(
                    direction * self.mission_config.turn_rate_rps
                )
            else:
                await self.navigation_command(
                    0.0, direction * self.mission_config.turn_rate_rps
                )
            if now - last_log_at >= 0.25:
                last_log_at = now
                motion_status = None if self.motion is None else self.motion.status()
                print(
                    "turn event=progress "
                    f"elapsed_s={elapsed:.3f} "
                    f"current_yaw_rad={sample.yaw_rad:.4f} "
                    f"progress_deg={math.degrees(progress):.1f} "
                    f"command_yaw_rps={direction * self.mission_config.turn_rate_rps:.3f} "
                    f"motion_armed={None if motion_status is None else motion_status['armed']} "
                    f"motion_mode={None if motion_status is None else motion_status['mode']}",
                    flush=True,
                )
            await asyncio.sleep(0.05)
        final = self.heading_provider.status()
        print(
            "turn event=timeout "
            f"elapsed_s={time.monotonic() - turn_started:.3f} "
            f"current_yaw_rad={final.yaw_rad} "
            f"progress_deg={math.degrees(self._mission.turn_progress_rad):.1f}",
            flush=True,
        )
        raise RuntimeCommandError("measured turn timed out")

    async def _search_for_memory(self) -> ClassCandidate:
        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        initial = self.heading_provider.status()
        if not initial.healthy or initial.yaw_rad is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        async with self._state_lock:
            self._mission.phase = MissionPhase.SEARCHING
            self._mission.reason = "searching_for_saved_fruit_class"
            self._mission.match_confirmations = 0
            self._mission.match_failures = 0
            self._mission.search_progress_rad = 0.0
        await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        deadline = time.monotonic() + self.mission_config.search_timeout_s
        last_frame_id: int | None = None
        last_center: tuple[int, int] | None = None
        confirmations = 0
        direction = 1.0
        previous_yaw = initial.yaw_rad
        accumulated_progress = 0.0
        while time.monotonic() < deadline:
            result, frame_id = await self._match_latest_frame(last_frame_id)
            if frame_id is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                accepted = bool(result is not None and result.accepted and result.best)
                if accepted and result is not None and result.best is not None:
                    center = result.best.detection.center
                    box = result.best.detection.bbox_xyxy
                    width = max(1, box[2] - box[0])
                    stable_candidate = bool(
                        last_center is None
                        or math.hypot(
                            center[0] - last_center[0], center[1] - last_center[1]
                        )
                        <= max(90.0, width * 1.75)
                    )
                    confirmations = confirmations + 1 if stable_candidate else 1
                    last_center = center
                else:
                    confirmations = 0
                    last_center = None
                async with self._state_lock:
                    self._mission.last_match = (
                        None if result is None else result.to_dict()
                    )
                    self._mission.match_confirmations = confirmations
                    self._mission.match_failures = (
                        0 if accepted else self._mission.match_failures + 1
                    )
                if result is not None:
                    print(
                        "search event=class_match "
                        f"accepted={accepted} "
                        f"reason={result.reason} "
                        f"confirmations={confirmations} "
                        f"candidate_count={result.candidate_count} "
                        "best_confidence="
                        f"{None if result.best is None else result.best.detection.confidence}",
                        flush=True,
                    )
                if (
                    accepted
                    and result is not None
                    and result.best is not None
                    and confirmations
                    >= self.mission_config.match_confirmations_required
                ):
                    await self.stop("demo_match_locked")
                    return result.best
                if accepted:
                    # Freeze as soon as the saved class appears. Continuing the
                    # sweep while confirming it can push the fruit out of frame.
                    await self.navigation_command(0.0, 0.0)
                    await asyncio.sleep(0.05)
                    continue

            sample = self.heading_provider.status()
            if not sample.healthy or sample.yaw_rad is None:
                raise RuntimeCommandError("Go2 heading became stale during search")
            # Accumulate short, wrap-safe yaw deltas instead of comparing the
            # current heading with the start heading. A start/end comparison
            # aliases after 180 degrees and cannot represent a bounded full
            # revolution.
            accumulated_progress += directed_progress(
                previous_yaw, sample.yaw_rad, direction
            )
            previous_yaw = sample.yaw_rad
            async with self._state_lock:
                self._mission.search_progress_rad = accumulated_progress
            if accumulated_progress >= self.mission_config.search_sweep_rad:
                raise RuntimeCommandError("saved fruit class not found in search sweep")
            await self.navigation_command(
                0.0, direction * self.mission_config.search_rate_rps
            )
            await asyncio.sleep(0.05)
        raise RuntimeCommandError("saved fruit class search timed out")

    async def _acknowledge_memory_match(
        self, match: ClassCandidate
    ) -> ClassCandidate:
        """Stretch once for a class lock, then require fresh visual proof."""

        async with self._state_lock:
            voice_mission = (
                self._voice_mission_generation == self._round_generation
            )
            if voice_mission:
                self._mission.match_stretch_status = "skipped_for_voice"
                self._mission.match_stretch_error = None
                self._mission.reason = "voice_target_class_locked_no_gesture"
        if voice_mission:
            print("match_stretch event=skipped source=voice", flush=True)
            return match
        if not self.mission_config.match_stretch_enabled:
            return match
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")
        async with self._state_lock:
            self._mission.phase = MissionPhase.ACKNOWLEDGING
            self._mission.reason = "stretching_for_saved_fruit_class"
            self._mission.match_stretch_status = "running"
            self._mission.match_stretch_error = None
        print(
            "match_stretch event=start "
            f"settle_s={self.mission_config.match_stretch_settle_s:.2f}",
            flush=True,
        )
        try:
            await self.motion.perform_stretch(
                settle_s=self.mission_config.match_stretch_settle_s
            )
        except MotionError as exc:
            error = str(exc)
            async with self._state_lock:
                self._mission.match_stretch_status = "failed"
                self._mission.match_stretch_error = error
                self._mission.reason = "stretch_failed_continuing_to_reacquire"
            # Stretch is cosmetic, not a locomotion prerequisite. Keep the
            # failure visible, then demand fresh class evidence and continue
            # through the same stopped, operator-Go-gated path.
            print(
                f"match_stretch event=failed nonfatal=true error={error}",
                flush=True,
            )
            return await self._reacquire_memory_after_stretch()
        async with self._state_lock:
            self._mission.match_stretch_status = "complete"
            self._mission.reason = "reacquiring_saved_fruit_class_after_stretch"
        print("match_stretch event=complete", flush=True)
        return await self._reacquire_memory_after_stretch()

    async def _reacquire_memory_after_stretch(self) -> ClassCandidate:
        """Confirm the saved class in new frames before locomotion arms."""

        return await self._reacquire_memory(
            event_name="match_stretch",
            success_reason="target_class_reacquired_after_stretch",
            timeout_reason=(
                "saved fruit class was not reacquired after stretch; approach blocked"
            ),
        )

    async def _reacquire_memory(
        self,
        *,
        event_name: str,
        success_reason: str,
        timeout_reason: str,
        timeout_s: float | None = None,
    ) -> ClassCandidate:
        """Require a new multi-frame class detection before locomotion."""

        async with self._state_lock:
            last_frame_id = self._produce_frame_id
        deadline = time.monotonic() + (
            self.mission_config.match_reacquire_timeout_s
            if timeout_s is None
            else timeout_s
        )
        confirmations = 0
        last_center: tuple[int, int] | None = None
        while time.monotonic() < deadline:
            result, frame_id = await self._match_latest_frame(last_frame_id)
            if frame_id is None or frame_id == last_frame_id:
                await asyncio.sleep(0.02)
                continue
            last_frame_id = frame_id
            accepted = bool(result is not None and result.accepted and result.best)
            if accepted and result is not None and result.best is not None:
                center = result.best.detection.center
                box = result.best.detection.bbox_xyxy
                width = max(1, box[2] - box[0])
                stable_candidate = bool(
                    last_center is None
                    or math.hypot(
                        center[0] - last_center[0], center[1] - last_center[1]
                    )
                    <= max(90.0, width * 1.75)
                )
                confirmations = confirmations + 1 if stable_candidate else 1
                last_center = center
            else:
                confirmations = 0
                last_center = None
            async with self._state_lock:
                self._mission.last_match = (
                    None if result is None else result.to_dict()
                )
                self._mission.match_confirmations = confirmations
                self._mission.match_failures = (
                    0 if accepted else self._mission.match_failures + 1
                )
            print(
                f"{event_name} event=reacquire "
                f"accepted={accepted} "
                f"confirmations={confirmations} "
                f"reason={None if result is None else result.reason}",
                flush=True,
            )
            if (
                accepted
                and result is not None
                and result.best is not None
                and confirmations
                >= self.mission_config.match_confirmations_required
            ):
                async with self._state_lock:
                    self._mission.phase = MissionPhase.CONFIRMING
                    self._mission.reason = success_reason
                return result.best
            await asyncio.sleep(0)
        raise RuntimeCommandError(timeout_reason)

    async def _start_memory_approach(
        self,
        match: ClassCandidate,
        *,
        initial_forward_elapsed_s: float = 0.0,
    ) -> None:
        async with self._state_lock:
            memory = self._fruit_memory
            self._mission.phase = MissionPhase.CONFIRMING
            self._mission.reason = "locking_saved_fruit_class_track"
        if memory is None:
            raise RuntimeCommandError("saved fruit memory disappeared")
        # The search stage already required several consecutive, spatially
        # stable class detections. Preserve that evidence across the handoff
        # instead of resetting the target to one visible frame and waiting for
        # another full detector cycle. On Jetson, that redundant wait created
        # a race where the follow request could cancel itself before arming.
        confirmed_frames = max(
            self.controller.config.stable_frames_required,
            self.mission_config.match_confirmations_required,
        )
        print(
            "approach event=target_lock "
            f"label={memory.label} "
            f"center={match.detection.center} "
            f"confirmed_frames={confirmed_frames} "
            f"confidence={match.detection.confidence:.4f}",
            flush=True,
        )
        await self.select_target(
            memory.label,
            match.detection.center,
            confirmed_visible_frames=confirmed_frames,
        )
        await self.start_follow(
            ARM_CONFIRMATION,
            initial_forward_elapsed_s=initial_forward_elapsed_s,
        )
        async with self._state_lock:
            self._mission.phase = MissionPhase.APPROACHING
            self._mission.reason = "approaching_saved_fruit_class"
            self._mission.match_failures = 0

    async def _resume_memory_approach(
        self,
        *,
        pause_reason: str,
        pause_count: int,
    ) -> float:
        """Brake, reacquire the same class, and resume without blind motion."""

        prior_forward_elapsed_s = self._forward_elapsed_s
        paused_at = time.monotonic()
        await self.stop(f"approach_pause:{pause_reason}")
        async with self._state_lock:
            self._mission.phase = MissionPhase.CONFIRMING
            self._mission.reason = "approach_paused_waiting_for_fresh_camera"
            self._mission.approach_pause_count = pause_count
            self._mission.approach_pause_status = "reacquiring"
        print(
            "approach event=paused "
            f"reason={pause_reason} attempt={pause_count} "
            f"forward_elapsed_s={prior_forward_elapsed_s:.3f}",
            flush=True,
        )
        match = await self._reacquire_memory(
            event_name="approach",
            success_reason="approach_same_class_reacquired",
            timeout_reason=(
                "camera or saved fruit class did not recover during approach"
            ),
            timeout_s=self.mission_config.approach_reacquire_timeout_s,
        )
        await self._start_memory_approach(
            match,
            initial_forward_elapsed_s=prior_forward_elapsed_s,
        )
        pause_elapsed_s = time.monotonic() - paused_at
        async with self._state_lock:
            self._mission.approach_pause_status = "resumed"
        print(
            "approach event=resumed "
            f"attempt={pause_count} pause_elapsed_s={pause_elapsed_s:.3f} "
            f"forward_elapsed_s={self._forward_elapsed_s:.3f}",
            flush=True,
        )
        return pause_elapsed_s

    def _near_target_geometry(
        self,
        target: TargetObservation | None,
        frame_height: int | None,
    ) -> tuple[bool, float | None]:
        if target is None or not frame_height:
            return False, None
        _, y, _, height = target.bbox_xywh
        bottom_ratio = (y + height) / frame_height
        center_ratio = target.center[1] / frame_height
        bbox_height_ratio = height / frame_height
        # Floor objects can leave the camera through the lower edge before
        # their boxes grow large. Bottom-center travel is therefore the arrival
        # evidence; box size remains a separate pointing-policy range guard.
        return (
            bottom_ratio >= self.mission_config.near_bottom_ratio
            and center_ratio >= self.mission_config.near_center_ratio,
            bbox_height_ratio,
        )

    async def _monitor_memory_approach(self) -> str:
        deadline = time.monotonic() + self.controller.config.forward_budget_s + 4.0
        last_frame_id: int | None = None
        failures = 0
        near_target_seen = False
        near_confirmations = 0
        last_near_at: float | None = None
        pause_count = 0
        transient_stop_reasons = {
            "selected_target_lost",
            "selected_target_not_found",
            "selected_target_not_revalidated",
            "selected_target_stale",
        }
        while time.monotonic() < deadline:
            now = time.monotonic()
            async with self._state_lock:
                memory = self._fruit_memory
                frame = self._produce_frame
                frame_id = self._produce_frame_id
                detections = list(self._produce_detections)
                target = self._target
                selected_name = self._selected_target_name
                frame_width = self._frame_width
                frame_height = None if self._latest_frame is None else self._latest_frame.height
                follow_active = bool(
                    self._follow_task is not None
                    and not self._follow_task.done()
                    and self._lease is not None
                    and self.motion is not None
                    and self.motion.armed
                )
                motion_status = (
                    None if self.motion is None else self.motion.status()
                )
                motion_fault = (
                    None
                    if motion_status is None
                    else motion_status.get("fault")
                )
                command_reason = self._command.reason
            if memory is None:
                raise RuntimeCommandError("saved fruit memory disappeared")
            near_geometry, bbox_height_ratio = self._near_target_geometry(
                target, frame_height
            )
            new_produce_frame = bool(
                frame is not None
                and frame_id is not None
                and frame_id != last_frame_id
            )
            if new_produce_frame:
                if near_geometry:
                    near_confirmations += 1
                    if (
                        near_confirmations
                        >= self.mission_config.near_confirmations_required
                    ):
                        near_target_seen = True
                        last_near_at = now
                else:
                    near_confirmations = 0
            near_target_recent = bool(
                near_target_seen
                and last_near_at is not None
                and now - last_near_at <= self.mission_config.near_loss_grace_s
            )
            async with self._state_lock:
                self._mission.near_target_seen = near_target_seen
                self._mission.near_target_recent = near_target_recent
                self._mission.near_target_confirmations = near_confirmations
                self._mission.near_target_bbox_height_ratio = bbox_height_ratio

            if new_produce_frame:
                assert frame is not None and frame_id is not None
                last_frame_id = frame_id
                result = self.class_matcher.match(
                    memory,
                    detections,
                    None if target is None else target.center,
                )
                associated = False
                if result.accepted and result.best is not None and target is not None:
                    candidate_center = result.best.detection.center
                    associated = math.hypot(
                        candidate_center[0] - target.center[0],
                        candidate_center[1] - target.center[1],
                    ) <= max(90.0, target.bbox_xywh[2] * 1.75)
                failures = 0 if associated else failures + 1
                async with self._state_lock:
                    self._mission.last_match = result.to_dict()
                    self._mission.match_failures = failures
                if (
                    self._arrival_pointing_required(memory.label)
                    and near_target_seen
                    and associated
                    and bbox_height_ratio is not None
                    and bbox_height_ratio
                    >= self.mission_config.near_bbox_height_ratio
                ):
                    return "target_visible_in_pointing_range"
                if near_target_recent and result.best is None:
                    return "target_class_reached_camera_edge"
                if failures >= self.mission_config.approach_misses_allowed:
                    command_reason = "selected_target_not_revalidated"

            if motion_fault:
                raise RuntimeCommandError(
                    f"approach motion fault: {motion_fault}"
                )
            should_reacquire = bool(
                command_reason in transient_stop_reasons
                and not near_target_recent
                and (
                    selected_name is None
                    or not follow_active
                    or failures
                    >= self.mission_config.approach_misses_allowed
                )
            )
            if should_reacquire:
                pause_count += 1
                if (
                    pause_count
                    > self.mission_config.approach_reacquire_attempts
                ):
                    raise RuntimeCommandError(
                        "approach camera recovery attempts exhausted"
                    )
                pause_elapsed_s = await self._resume_memory_approach(
                    pause_reason=command_reason,
                    pause_count=pause_count,
                )
                # Camera stalls do not consume the commanded-motion budget or
                # the monitor deadline. The dog remains disarmed throughout
                # the pause and resumes from its accumulated forward time.
                deadline += pause_elapsed_s
                failures = 0
                last_frame_id = None
                continue

            if selected_name is None:
                if near_target_recent:
                    return "target_class_reached_camera_edge"
                raise RuntimeCommandError("saved fruit class lost before arrival")
            if not follow_active:
                if near_target_recent and command_reason in {
                    "selected_target_not_revalidated",
                    "selected_target_lost",
                }:
                    return "target_class_reached_camera_edge"
                if near_target_recent and command_reason == "forward_budget_complete":
                    return "target_visible_near_forward_budget_complete"
                raise RuntimeCommandError(f"approach stopped: {command_reason}")
            if frame_width is None:
                raise RuntimeCommandError("camera geometry unavailable during approach")
            await asyncio.sleep(0.025)
        raise RuntimeCommandError("saved fruit approach timed out")

    async def _run_final_approach(self, approach_reason: str) -> str:
        """Send one bounded push after a near fruit leaves the camera edge."""

        if approach_reason != "target_class_reached_camera_edge":
            raise RuntimeCommandError(
                "final push requires the confirmed fruit to leave the "
                f"camera edge, got {approach_reason}"
            )
        if self.heading_provider is None:
            raise RuntimeCommandError(
                "fresh Go2 local pose is unavailable for final push"
            )
        initial = self.heading_provider.status()
        if not initial.pose_healthy:
            raise RuntimeCommandError(
                "fresh Go2 local pose is unavailable for final push"
            )
        assert initial.x_m is not None and initial.y_m is not None
        start_x = initial.x_m
        start_y = initial.y_m
        async with self._state_lock:
            self._mission.phase = MissionPhase.FINAL_APPROACHING
            self._mission.reason = "fruit_offscreen_final_push"
            self._mission.final_approach_status = "running"
            self._mission.final_approach_elapsed_s = 0.0
            self._mission.final_approach_commanded_distance_m = 0.0
            self._mission.final_approach_measured_distance_m = 0.0
        print(
            "final_push event=start "
            f"trigger={approach_reason} "
            f"speed_mps={self.mission_config.final_push_mps:.3f} "
            f"duration_s={self.mission_config.final_push_duration_s:.2f}",
            flush=True,
        )
        await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        started_at = time.monotonic()
        deadline = started_at + self.mission_config.final_push_duration_s
        last_tick_at = started_at
        last_command_mps = 0.0
        commanded_distance = 0.0
        measured_distance = 0.0
        while True:
            now = time.monotonic()
            tick_s = max(0.0, now - last_tick_at)
            commanded_distance += last_command_mps * tick_s
            last_tick_at = now
            if now >= deadline:
                break
            async with self._state_lock:
                camera_fresh = bool(
                    self._last_frame_at is not None
                    and now - self._last_frame_at < 1.0
                )
            if not camera_fresh:
                raise RuntimeCommandError(
                    "camera became stale during final push"
                )
            sample = self.heading_provider.status()
            if not sample.pose_healthy:
                raise RuntimeCommandError(
                    "Go2 local pose became stale during final push"
                )
            assert sample.x_m is not None and sample.y_m is not None
            measured_distance = math.hypot(
                sample.x_m - start_x,
                sample.y_m - start_y,
            )
            command_status = await self._send_final_push()
            last_command_mps = float(
                command_status["command"]["forward_mps"]  # type: ignore[index]
            )
            async with self._state_lock:
                self._mission.final_approach_elapsed_s = now - started_at
                self._mission.final_approach_commanded_distance_m = (
                    commanded_distance
                )
                self._mission.final_approach_measured_distance_m = (
                    measured_distance
                )
            await asyncio.sleep(min(0.05, max(0.0, deadline - now)))
        await self.navigation_command(0.0, 0.0)
        elapsed = time.monotonic() - started_at
        async with self._state_lock:
            self._mission.reason = "final_push_complete_continuing_to_standdown"
            self._mission.final_approach_status = "complete"
            self._mission.final_approach_elapsed_s = elapsed
            self._mission.final_approach_commanded_distance_m = commanded_distance
            self._mission.final_approach_measured_distance_m = measured_distance
        print(
            "final_push event=complete "
            f"elapsed_s={elapsed:.3f} "
            f"commanded_distance_m={commanded_distance:.3f} "
            f"measured_distance_m={measured_distance:.3f} "
            "next=standdown",
            flush=True,
        )
        return "offscreen_final_push_complete"

    async def _return_home(self) -> None:
        """Return to the captured start pose using fresh local odometry.

        This is deliberately a short-range stage controller, not a global map
        planner. After the post-arrival StandUp/BalanceStand transition, Woof
        turns normally toward Home and then translates through the factory
        obstacle-avoidance lease. Stale pose, lack of progress, or timeout
        immediately aborts the run.
        """

        if self.mission_config.return_backend == "nav2":
            await self._return_home_nav2()
            return

        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 local pose is unavailable")
        home = self._home_pose
        if home is None:
            raise RuntimeCommandError("home pose was not captured")
        started_at = time.monotonic()
        deadline = started_at + self.mission_config.return_timeout_s
        async with self._state_lock:
            self._mission.phase = MissionPhase.RETURNING_HOME
            self._mission.reason = "waiting_for_return_pose_to_settle"
            self._mission.return_home_status = "running"
        initial_assessment = await self._wait_for_stable_return_pose(
            deadline=deadline
        )
        home_pose = Pose2D(*home)
        initial_pose = initial_assessment.pose
        yaw_limit = (
            0.8
            if self.motion is None
            else self.motion.config.maximum_yaw_rps
        )
        planner_config = ReturnPlannerConfig(
            arrival_tolerance_m=(
                self.mission_config.return_arrival_tolerance_m
            ),
            heading_tolerance_rad=(
                self.mission_config.return_heading_tolerance_rad
            ),
            heading_gate_rad=self.mission_config.return_heading_gate_rad,
            maximum_forward_mps=self.mission_config.return_forward_mps,
            yaw_gain=self.mission_config.return_yaw_gain,
            maximum_yaw_rps=yaw_limit,
        )
        initial_step = plan_return_step(
            home=home_pose,
            current=initial_pose,
            config=planner_config,
        )
        initial_distance = initial_step.distance_m
        if initial_distance > self.mission_config.return_max_distance_m:
            raise RuntimeCommandError(
                "saved Home is outside the bounded return envelope "
                f"({initial_distance:.2f} m > "
                f"{self.mission_config.return_max_distance_m:.2f} m)"
            )
        best_distance = initial_distance
        last_progress_at = time.monotonic()
        async with self._state_lock:
            self._mission.reason = "returning_to_saved_start_pose"
            self._mission.return_distance_m = initial_distance
            self._mission.return_progress_m = 0.0
        print(
            "return_home event=start "
            f"distance_m={initial_distance:.3f} "
            f"timeout_s={self.mission_config.return_timeout_s:.2f}",
            flush=True,
        )

        # Woof normally faces the reached fruit, about 180 degrees away from
        # Home. Reorient before translation so obstacle avoidance only has to
        # make small steering corrections while walking.
        if (
            initial_step.distance_m
            > self.mission_config.return_arrival_tolerance_m
        ):
            await self._orient_for_return(
                initial_step.target_yaw_rad,
                deadline=deadline,
                stage="departure",
            )

        # Rotation is not a translation stall. Start the progress window only
        # after the departure heading is established.
        best_distance = initial_distance
        last_progress_at = time.monotonic()
        if (
            initial_step.distance_m
            > self.mission_config.return_arrival_tolerance_m
        ):
            await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        last_pose = self.heading_provider.status()
        if not last_pose.pose_healthy:
            raise RuntimeCommandError(
                "Go2 local pose became stale before return translation"
            )
        assert last_pose.x_m is not None and last_pose.y_m is not None
        last_log_at = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            sample = self.heading_provider.status()
            if not sample.pose_healthy:
                raise RuntimeCommandError("Go2 local pose became stale during return")
            assert (
                sample.x_m is not None
                and sample.y_m is not None
                and sample.yaw_rad is not None
            )
            odometry_step_m = math.hypot(
                sample.x_m - last_pose.x_m,
                sample.y_m - last_pose.y_m,
            )
            assert last_pose.yaw_rad is not None
            odometry_yaw_step_rad = abs(
                normalize_angle(sample.yaw_rad - last_pose.yaw_rad)
            )
            if (
                odometry_step_m
                > self.mission_config.return_max_odometry_step_m
            ):
                raise RuntimeCommandError(
                    "Go2 local odometry jumped "
                    f"{odometry_step_m:.2f} m during return"
                )
            if (
                odometry_yaw_step_rad
                > self.mission_config.return_max_odometry_yaw_step_rad
            ):
                raise RuntimeCommandError(
                    "Go2 local heading jumped "
                    f"{math.degrees(odometry_yaw_step_rad):.1f} degrees "
                    "during return"
                )
            last_pose = sample
            step = plan_return_step(
                home=home_pose,
                current=Pose2D(
                    sample.x_m,
                    sample.y_m,
                    sample.yaw_rad,
                ),
                config=planner_config,
            )
            distance = step.distance_m
            heading_error = step.heading_error_rad
            if distance > self.mission_config.return_max_distance_m:
                raise RuntimeCommandError(
                    "return odometry left the bounded stage envelope "
                    f"({distance:.2f} m)"
                )
            if step.mode == ReturnMode.COMPLETE:
                await self.stop("return_home_complete")
                async with self._state_lock:
                    self._mission.return_home_status = "complete"
                    self._mission.return_distance_m = distance
                    self._mission.return_heading_error_rad = heading_error
                    self._mission.return_progress_m = max(
                        0.0, initial_distance - distance
                    )
                print(
                    "return_home event=complete "
                    f"elapsed_s={time.monotonic() - started_at:.3f} "
                    f"distance_m={distance:.3f} "
                    f"heading_error_deg={math.degrees(heading_error):.1f}",
                    flush=True,
                )
                return
            if step.mode == ReturnMode.RESTORE_HEADING:
                await self.stop("return_home_position_reached")
                await self._orient_for_return(
                    step.target_yaw_rad,
                    deadline=deadline,
                    stage="final_heading",
                )
                # Re-evaluate both position and heading after the turn. A turn
                # may translate slightly; never report success from stale
                # pre-turn odometry.
                last_pose = self.heading_provider.status()
                if not last_pose.pose_healthy:
                    raise RuntimeCommandError(
                        "Go2 local pose became stale at Home"
                    )
                assert last_pose.x_m is not None and last_pose.y_m is not None
                continue
            if step.mode == ReturnMode.TURN_TO_HOME:
                # Do not ask obstacle avoidance for a pure rotation. Release it,
                # correct yaw with the translation-impossible direct lease, then
                # reacquire obstacle-protected translation.
                await self.stop("return_home_reorient")
                await self._orient_for_return(
                    step.target_yaw_rad,
                    deadline=deadline,
                    stage="course_correction",
                )
                await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
                last_progress_at = time.monotonic()
                last_pose = self.heading_provider.status()
                if not last_pose.pose_healthy:
                    raise RuntimeCommandError(
                        "Go2 local pose became stale after return reorientation"
                    )
                assert last_pose.x_m is not None and last_pose.y_m is not None
                continue
            if self._motion_owner != "navigation":
                await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
            forward_mps = step.forward_mps
            yaw_rps = step.yaw_rps
            await self.navigation_command(forward_mps, yaw_rps)

            if distance <= best_distance - self.mission_config.return_stall_min_progress_m:
                best_distance = distance
                last_progress_at = now
            elif forward_mps <= 0.0:
                # Time spent rotating in place is not a translation stall.
                last_progress_at = now
            elif now - last_progress_at >= self.mission_config.return_stall_timeout_s:
                raise RuntimeCommandError(
                    f"return home stalled {distance:.2f} m from start"
                )

            async with self._state_lock:
                self._mission.return_distance_m = distance
                self._mission.return_heading_error_rad = heading_error
                self._mission.return_progress_m = max(
                    0.0, initial_distance - distance
                )
            if now - last_log_at >= 0.25:
                last_log_at = now
                print(
                    "return_home event=progress "
                    f"elapsed_s={now - started_at:.3f} "
                    f"distance_m={distance:.3f} "
                    f"heading_error_deg={math.degrees(heading_error):.1f} "
                    f"forward_mps={forward_mps:.3f} yaw_rps={yaw_rps:.3f}",
                    flush=True,
                )
            await asyncio.sleep(0.05)
        raise RuntimeCommandError("return home timed out")

    async def _return_home_nav2(self) -> None:
        """Pre-align with measured odometry, then delegate the mapped path."""

        if self.nav2_return is None:
            raise RuntimeCommandError("Nav2 return service is not configured")
        if self._map_home is None:
            raise RuntimeCommandError("map-frame Home was not captured")

        await self.stop("nav2_return_handoff")
        started_at = time.monotonic()
        deadline = started_at + self.mission_config.return_timeout_s
        initial_distance: float | None = None
        local_home = self._home_pose
        async with self._state_lock:
            self._mission.phase = MissionPhase.RETURNING_HOME
            self._mission.reason = (
                "nav2_preparing_measured_departure_turn"
                if local_home is not None
                else "nav2_planning_return_home"
            )
            self._mission.return_home_status = "running"
            self._mission.return_backend = "nav2"
            self._mission.return_turn_status = (
                "prealigning" if local_home is not None else "owned_by_nav2"
            )
            self._mission.nav2_status = {
                "state": "starting",
                "reason": "submitting saved map-frame Home",
            }

        if local_home is not None:
            assessment = await self._wait_for_stable_return_pose(
                deadline=deadline
            )
            current = assessment.pose
            local_distance = math.hypot(
                local_home[0] - current.x_m,
                local_home[1] - current.y_m,
            )
            if local_distance > self.mission_config.return_max_distance_m:
                raise RuntimeCommandError(
                    "saved local Home is outside the bounded return envelope "
                    f"({local_distance:.2f} m > "
                    f"{self.mission_config.return_max_distance_m:.2f} m)"
                )
            if local_distance > self.mission_config.return_arrival_tolerance_m:
                target_yaw = math.atan2(
                    local_home[1] - current.y_m,
                    local_home[0] - current.x_m,
                )
                print(
                    "return_home event=nav2_prealign "
                    f"local_distance_m={local_distance:.3f} "
                    f"target_yaw_rad={target_yaw:.4f}",
                    flush=True,
                )
                await self._orient_for_return(
                    target_yaw,
                    deadline=deadline,
                    stage="nav2_departure",
                )
            async with self._state_lock:
                self._mission.reason = "nav2_planning_return_home"
                if self._mission.return_turn_status == "not_needed":
                    self._mission.return_turn_status = "prealignment_not_needed"
                else:
                    self._mission.return_turn_status = "prealigned_for_nav2"

        print(
            "return_home event=nav2_start "
            f"frame={self._map_home.frame_id} "
            f"x_m={self._map_home.x_m:.4f} "
            f"y_m={self._map_home.y_m:.4f} "
            f"yaw_rad={self._map_home.yaw_rad:.4f} "
            f"position_tolerance_m="
            f"{self.mission_config.return_arrival_tolerance_m:.3f} "
            "heading_tolerance_deg="
            f"{math.degrees(self.mission_config.return_heading_tolerance_rad):.1f}",
            flush=True,
        )

        try:
            first = await self.nav2_return.start_return(
                position_tolerance_m=(
                    self.mission_config.return_arrival_tolerance_m
                ),
                heading_tolerance_rad=(
                    self.mission_config.return_heading_tolerance_rad
                ),
                timeout_s=self.mission_config.return_timeout_s,
            )
            self._nav2_return_active = not first.terminal
            status = first
            last_log_at = 0.0
            while True:
                if status.distance_remaining_m is not None:
                    if initial_distance is None:
                        initial_distance = status.distance_remaining_m
                    progress = max(
                        0.0,
                        initial_distance - status.distance_remaining_m,
                    )
                else:
                    progress = 0.0
                async with self._state_lock:
                    self._mission.reason = f"nav2_{status.reason}"
                    self._mission.nav2_status = status.to_dict()
                    self._mission.return_distance_m = (
                        status.distance_remaining_m
                        if status.distance_remaining_m is not None
                        else status.position_error_m
                    )
                    self._mission.return_heading_error_rad = (
                        status.heading_error_rad
                    )
                    self._mission.return_progress_m = progress

                if status.terminal:
                    self._nav2_return_active = False
                    if status.state != "succeeded":
                        raise RuntimeCommandError(
                            f"Nav2 return {status.state}: {status.reason}"
                        )
                    if (
                        status.position_error_m is None
                        or status.position_error_m
                        > self.mission_config.return_arrival_tolerance_m
                    ):
                        raise RuntimeCommandError(
                            "Nav2 reported success outside the Home position "
                            "tolerance"
                        )
                    if (
                        status.heading_error_rad is None
                        or abs(status.heading_error_rad)
                        > self.mission_config.return_heading_tolerance_rad
                    ):
                        raise RuntimeCommandError(
                            "Nav2 reported success outside the Home heading "
                            "tolerance"
                        )
                    if not (
                        status.localization_healthy
                        and status.map_healthy
                        and status.scan_healthy
                    ):
                        raise RuntimeCommandError(
                            "Nav2 reached Home with unhealthy localization "
                            "or obstacle sensing"
                        )
                    await self.stop("nav2_return_home_complete")
                    async with self._state_lock:
                        self._mission.return_home_status = "complete"
                        self._mission.reason = "nav2_returned_to_saved_home"
                    print(
                        "return_home event=nav2_complete "
                        f"elapsed_s={time.monotonic() - started_at:.3f} "
                        f"position_error_m={status.position_error_m:.3f} "
                        "heading_error_deg="
                        f"{math.degrees(status.heading_error_rad):.1f} "
                        f"recoveries={status.recoveries}",
                        flush=True,
                    )
                    return

                now = time.monotonic()
                if now >= deadline:
                    raise RuntimeCommandError("Nav2 return home timed out")
                if now - last_log_at >= 0.25:
                    last_log_at = now
                    print(
                        "return_home event=nav2_progress "
                        f"elapsed_s={now - started_at:.3f} "
                        f"state={status.state} "
                        f"distance_m={status.distance_remaining_m} "
                        f"recoveries={status.recoveries}",
                        flush=True,
                    )
                await asyncio.sleep(self.mission_config.nav2_poll_period_s)
                status = await self.nav2_return.status()
        except Nav2ReturnError as exc:
            raise RuntimeCommandError(f"Nav2 return unavailable: {exc}") from exc
        finally:
            if self._nav2_return_active:
                await self.nav2_return.cancel("collie_return_finished")
                self._nav2_return_active = False

    async def _wait_for_stable_return_pose(
        self, *, deadline: float
    ) -> PoseWindowAssessment:
        """Wait for post-skill odometry to become stationary before returning."""

        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 local pose is unavailable")
        started_at = time.monotonic()
        settle_deadline = min(
            deadline,
            started_at + self.mission_config.return_pose_settle_timeout_s,
        )
        stable_started_at = started_at
        samples: list[Pose2D] = []
        last_assessment: PoseWindowAssessment | None = None
        print(
            "return_home_pose_settle event=start "
            f"stable_window_s="
            f"{self.mission_config.return_pose_settle_duration_s:.2f} "
            f"timeout_s={settle_deadline - started_at:.2f}",
            flush=True,
        )
        while time.monotonic() < settle_deadline:
            sample = self.heading_provider.status()
            if not sample.pose_healthy:
                raise RuntimeCommandError(
                    "Go2 local pose became stale while settling for return"
                )
            assert (
                sample.x_m is not None
                and sample.y_m is not None
                and sample.yaw_rad is not None
            )
            pose = Pose2D(sample.x_m, sample.y_m, sample.yaw_rad)
            samples.append(pose)
            now = time.monotonic()
            if len(samples) >= 2:
                last_assessment = assess_pose_window(samples)
                if (
                    last_assessment.maximum_position_span_m
                    > self.mission_config.return_pose_settle_max_drift_m
                    or last_assessment.maximum_yaw_span_rad
                    > self.mission_config.return_pose_settle_max_yaw_drift_rad
                ):
                    samples = [pose]
                    stable_started_at = now
                    last_assessment = None
                elif (
                    now - stable_started_at
                    >= self.mission_config.return_pose_settle_duration_s
                ):
                    print(
                        "return_home_pose_settle event=complete "
                        f"elapsed_s={now - started_at:.3f} "
                        f"position_span_m="
                        f"{last_assessment.maximum_position_span_m:.4f} "
                        f"yaw_span_deg="
                        f"{math.degrees(last_assessment.maximum_yaw_span_rad):.2f}",
                        flush=True,
                    )
                    return last_assessment
            await asyncio.sleep(0.05)

        position_span = (
            "unknown"
            if last_assessment is None
            else f"{last_assessment.maximum_position_span_m:.3f} m"
        )
        raise RuntimeCommandError(
            "Go2 local odometry did not settle before return "
            f"(position span {position_span})"
        )

    async def _capture_stable_home_pose(self) -> PoseWindowAssessment:
        """Capture Home only from a fresh, stationary odometry window."""

        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 local pose is unavailable")
        samples: list[Pose2D] = []
        started_at = time.monotonic()
        deadline = (
            started_at
            + self.mission_config.return_pose_capture_duration_s
        )
        while True:
            sample = self.heading_provider.status()
            if not sample.pose_healthy:
                raise RuntimeCommandError(
                    "Go2 local pose became stale while capturing Home"
                )
            assert (
                sample.x_m is not None
                and sample.y_m is not None
                and sample.yaw_rad is not None
            )
            samples.append(
                Pose2D(sample.x_m, sample.y_m, sample.yaw_rad)
            )
            now = time.monotonic()
            if now >= deadline and len(samples) >= 2:
                break
            await asyncio.sleep(min(0.05, max(0.0, deadline - now)))

        assessment = assess_pose_window(samples)
        if (
            assessment.maximum_position_span_m
            > self.mission_config.return_pose_capture_max_drift_m
        ):
            raise RuntimeCommandError(
                "cannot capture Home: local odometry drifted "
                f"{assessment.maximum_position_span_m:.3f} m while stopped"
            )
        if (
            assessment.maximum_yaw_span_rad
            > self.mission_config.return_pose_capture_max_yaw_drift_rad
        ):
            raise RuntimeCommandError(
                "cannot capture Home: local heading drifted "
                f"{math.degrees(assessment.maximum_yaw_span_rad):.1f} "
                "degrees while stopped"
            )
        return assessment

    async def _orient_for_return(
        self,
        target_yaw_rad: float,
        *,
        deadline: float,
        stage: str,
    ) -> None:
        """Turn toward Home, recovering the Sport posture handoff once.

        Unitree can acknowledge the first avoidance yaw lease after
        StandDown/StandUp without physically rotating. A stationary pose is
        not sufficient evidence that locomotion is ready, so the first turn
        attempt must demonstrate real yaw progress. If it does not, stop,
        reissue the same StandUp/BalanceStand recovery that made the normal
        stage turn responsive on Woof, settle odometry, and retry exactly
        once. A second non-response remains a hard, fail-closed abort.
        """

        for attempt in range(2):
            try:
                await self._orient_for_return_once(
                    target_yaw_rad,
                    deadline=deadline,
                    stage=stage,
                    attempt=attempt,
                )
                async with self._state_lock:
                    self._mission.return_turn_status = "complete"
                return
            except _ReturnTurnNoResponse as exc:
                await self.stop(
                    f"return_home_{stage}_turn_no_response_attempt_{attempt + 1}"
                )
                if attempt >= 1:
                    async with self._state_lock:
                        self._mission.return_turn_status = "failed"
                    raise RuntimeCommandError(
                        f"return home {stage} turn did not respond "
                        "after Sport recovery"
                    ) from exc
                if self.motion is None:
                    raise RuntimeCommandError(
                        "motion backend disappeared during return recovery"
                    ) from exc
                async with self._state_lock:
                    self._mission.reason = (
                        f"return_home_{stage}_recovering_sport_mode"
                    )
                    self._mission.return_turn_status = "recovering_sport_mode"
                    self._mission.return_turn_recovery_count += 1
                print(
                    "return_home_turn event=recover "
                    f"stage={stage} attempt={attempt + 1} "
                    f"reason={exc}",
                    flush=True,
                )
                async with self._exclusive_skill_lock:
                    try:
                        await self.motion.perform_stand_up(
                            settle_s=(
                                self.mission_config
                                .return_turn_recovery_settle_s
                            )
                        )
                    except MotionError as recovery_exc:
                        async with self._state_lock:
                            self._mission.return_turn_status = "failed"
                        raise RuntimeCommandError(
                            "return turn Sport recovery failed: "
                            f"{recovery_exc}"
                        ) from recovery_exc
                await self._wait_for_stable_return_pose(deadline=deadline)

    async def _orient_for_return_once(
        self,
        target_yaw_rad: float,
        *,
        deadline: float,
        stage: str,
        attempt: int,
    ) -> None:
        """Turn in place to a measured return-home heading.

        Translation is impossible under this lease. The same watchdog,
        heading freshness, stall guard, and global return deadline remain in
        force.
        """

        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        sample = self.heading_provider.status()
        if not sample.healthy or sample.yaw_rad is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        turn_direction = ReturnTurnDirectionLatch.create(
            target_yaw_rad=target_yaw_rad,
            current_yaw_rad=sample.yaw_rad,
            release_progress_rad=max(
                self.mission_config.turn_stall_min_progress_rad,
                self.mission_config.return_heading_tolerance_rad,
            ),
            activation_margin_rad=max(
                math.radians(10.0),
                2.0 * self.mission_config.turn_stall_min_progress_rad,
            ),
        )
        initial_error = turn_direction.update(sample.yaw_rad)
        if abs(initial_error) <= self.mission_config.return_heading_tolerance_rad:
            async with self._state_lock:
                self._mission.return_turn_status = "not_needed"
            return

        use_direct_yaw = self.mission_config.direct_turn_enabled
        if use_direct_yaw:
            await self._direct_turn_arm()
        else:
            # Use the same factory-avoidance yaw path as the mission's initial
            # measured turn.  Return-home previously ignored this setting and
            # always selected SportClient direct yaw, which can acknowledge
            # Move() without producing rotation on Woof's current sport state.
            await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        started_at = time.monotonic()
        initial_yaw_rad = sample.yaw_rad
        response_confirmed = False
        best_error = abs(initial_error)
        last_progress_at = started_at
        last_log_at = 0.0
        last_yaw_rad = sample.yaw_rad
        print(
            "return_home_turn event=start "
            f"stage={stage} "
            f"heading_error_deg={math.degrees(initial_error):.1f} "
            f"direction={'positive' if turn_direction.direction > 0 else 'negative'} "
            f"direction_latched={str(turn_direction.active).lower()} "
            f"attempt={attempt + 1} "
            f"mode={'direct_yaw' if use_direct_yaw else 'avoidance'}",
            flush=True,
        )
        async with self._state_lock:
            self._mission.return_turn_status = (
                "running" if attempt == 0 else "retrying_after_sport_recovery"
            )
        while time.monotonic() < deadline:
            now = time.monotonic()
            sample = self.heading_provider.status()
            if not sample.healthy or sample.yaw_rad is None:
                raise RuntimeCommandError(
                    "Go2 heading became stale during return turn"
                )
            yaw_step_rad = abs(
                normalize_angle(sample.yaw_rad - last_yaw_rad)
            )
            if (
                yaw_step_rad
                > self.mission_config.return_max_odometry_yaw_step_rad
            ):
                raise RuntimeCommandError(
                    "Go2 local heading jumped "
                    f"{math.degrees(yaw_step_rad):.1f} degrees "
                    "during return turn"
                )
            last_yaw_rad = sample.yaw_rad
            shortest_heading_error = normalize_angle(
                target_yaw_rad - sample.yaw_rad
            )
            heading_error = turn_direction.update(sample.yaw_rad)
            error_magnitude = abs(heading_error)
            response_progress = directed_progress(
                initial_yaw_rad,
                sample.yaw_rad,
                turn_direction.direction,
            )
            if (
                response_progress
                >= self.mission_config.return_turn_response_min_progress_rad
            ):
                response_confirmed = True
            async with self._state_lock:
                self._mission.return_heading_error_rad = shortest_heading_error
                self._mission.reason = f"return_home_{stage}_turn"
            if (
                abs(shortest_heading_error)
                <= self.mission_config.return_heading_tolerance_rad
            ):
                await self.stop(f"return_home_{stage}_turn_complete")
                print(
                    "return_home_turn event=complete "
                    f"stage={stage} "
                    f"elapsed_s={now - started_at:.3f} "
                    "heading_error_deg="
                    f"{math.degrees(shortest_heading_error):.1f}",
                    flush=True,
                )
                return

            if (
                not response_confirmed
                and now - started_at
                >= self.mission_config.return_turn_response_timeout_s
            ):
                raise _ReturnTurnNoResponse(
                    "yaw command was accepted but measured only "
                    f"{math.degrees(response_progress):.1f} degrees in "
                    f"{now - started_at:.2f} seconds"
                )

            if (
                best_error - error_magnitude
                >= self.mission_config.turn_stall_min_progress_rad
            ):
                best_error = error_magnitude
                last_progress_at = now
            elif (
                now - last_progress_at
                >= self.mission_config.turn_stall_timeout_s
            ):
                raise RuntimeCommandError(
                    "return home "
                    f"{stage} turn stalled at "
                    f"{math.degrees(shortest_heading_error):.1f} degrees"
                )

            # Return-home has its own proportional controller.  Its upper
            # bound is the motion adapter's verified yaw envelope, not the
            # mission's initial/search turn rate (which may intentionally be
            # much slower).
            yaw_limit = (
                0.8
                if self.motion is None
                else self.motion.config.maximum_yaw_rps
            )
            minimum_yaw = min(
                yaw_limit,
                self.mission_config.return_turn_minimum_yaw_rps,
            )
            yaw_magnitude = min(
                yaw_limit,
                max(
                    minimum_yaw,
                    self.mission_config.return_yaw_gain
                    * abs(heading_error),
                ),
            )
            yaw_rps = math.copysign(yaw_magnitude, heading_error)
            if use_direct_yaw:
                await self._direct_turn_command(yaw_rps)
            else:
                await self.navigation_command(0.0, yaw_rps)
            if now - last_log_at >= 0.25:
                last_log_at = now
                print(
                    "return_home_turn event=progress "
                    f"stage={stage} "
                    f"elapsed_s={now - started_at:.3f} "
                    "heading_error_deg="
                    f"{math.degrees(shortest_heading_error):.1f} "
                    f"command_yaw_rps={yaw_rps:.3f} "
                    f"response_progress_deg="
                    f"{math.degrees(response_progress):.1f} "
                    "direction_latched="
                    f"{str(turn_direction.active).lower()}",
                    flush=True,
                )
            await asyncio.sleep(0.05)
        raise RuntimeCommandError("return home timed out during heading correction")

    async def _match_latest_frame(
        self, last_frame_id: int | None
    ) -> tuple[ClassMatchResult | None, int | None]:
        async with self._state_lock:
            memory = self._fruit_memory
            frame = self._produce_frame
            frame_id = self._produce_frame_id
            detections = list(self._produce_detections)
            produce_age = (
                None
                if self._produce_last_at is None
                else time.monotonic() - self._produce_last_at
            )
        if (
            memory is None
            or frame is None
            or frame_id is None
            or frame_id == last_frame_id
            or produce_age is None
            or produce_age > self.maximum_produce_age_s
        ):
            return None, frame_id
        result = self.class_matcher.match(memory, detections)
        return result, frame_id

    async def _follow_loop(self) -> None:
        current_task = asyncio.current_task()
        try:
            while not self._closing:
                try:
                    await self.pulse()
                except RuntimeCommandError:
                    break
                await asyncio.sleep(self.follow_period_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"follow event=error error={exc!r}", flush=True)
            async with self._state_lock:
                self._last_error = f"follow loop failed: {exc}"
            await self.stop("follow_loop_error")
        finally:
            if self._follow_task is current_task:
                self._follow_task = None

    async def _navigation_watchdog_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(0.10)
            deadline = self._navigation_deadline
            if (
                self._motion_owner == "navigation"
                and deadline is not None
                and time.monotonic() >= deadline
            ):
                await self.stop("navigation_lease_expired")

    async def _nav2_health_loop(self) -> None:
        assert self.nav2_return is not None
        while not self._closing:
            try:
                payload = await self.nav2_return.health()
                health = payload.get("health")
                if not isinstance(health, dict):
                    health = {}
                rendered: dict[str, object] = {
                    **health,
                    "ready": bool(payload.get("ready")),
                    "reason": str(
                        payload.get("reason")
                        or ("ready" if payload.get("ready") else "not ready")
                    ),
                }
            except Exception as exc:
                rendered = {
                    "ready": False,
                    "reason": f"Nav2 return unavailable: {exc}",
                }
            async with self._state_lock:
                self._nav2_health = rendered
                self._nav2_health_at = time.monotonic()
            await asyncio.sleep(0.50)

    async def _release_locked(self, reason: str) -> None:
        if self.motion is not None and self._lease is not None:
            try:
                await self.motion.release(self._lease)
            except MotionError:
                await self.motion.emergency_stop()
        self._lease = None
        self._motion_owner = None
        self._navigation_deadline = None
        self._last_pulse_at = None
        self._command = VelocityCommand(reason=reason)

    def _best_produce_detection_locked(
        self,
        label: str,
        preferred_center: tuple[int, int] | None = None,
    ) -> FruitDetection | None:
        matches = [
            item
            for item in self._produce_detections
            if item.label.casefold() == label.casefold()
        ]
        if not matches:
            return None
        if preferred_center is None:
            return max(matches, key=lambda item: item.confidence)
        target_x, target_y = preferred_center
        return min(
            matches,
            key=lambda item: (
                (item.center[0] - target_x) ** 2
                + (item.center[1] - target_y) ** 2,
                -item.confidence,
            ),
        )

    def _canonical_target_name(self, requested_name: str) -> str:
        if self.produce_detector is None:
            raise RuntimeCommandError("produce detector is disabled")
        requested = requested_name.strip()
        available_names = list(self.produce_detector.names.values())
        if requested in available_names:
            return requested
        folded_names = {name.casefold(): name for name in available_names}
        if not requested or requested.casefold() not in folded_names:
            raise RuntimeCommandError("target is not a class in the produce model")
        return folded_names[requested.casefold()]

    def _clear_produce_tracker_locked(self) -> None:
        self._produce_tracker = None
        self._produce_tracker_label = None
        self._produce_visible_frames = 0
        self._produce_verified_at = None
        self._produce_revalidation_failures = 0

    def _clear_selection_locked(self) -> None:
        self._selected_target_name = None
        self._selected_target_hint = None
        self._target = None
        self._clear_produce_tracker_locked()

    def _mark_selection_stale_locked(self) -> None:
        """Keep the user's explicit choice while removing motion readiness.

        A detector or camera gap while disarmed must not silently turn a click
        into a different round.  The stale selection keeps its process/lock ID
        and reacquisition hint, but cannot pass ``can_follow`` until a new YOLO
        observation rebuilds the target and resets the revalidation failures.
        Active motion still uses ``_clear_selection_locked`` and stops.
        """

        self._target = None
        self._produce_tracker = None
        self._produce_tracker_label = None
        self._produce_visible_frames = 0
        self._produce_verified_at = None

    def _produce_device_status(self) -> dict[str, object]:
        if self.produce_detector is None:
            return {"requested": "disabled", "resolved": "disabled"}
        status = getattr(self.produce_detector, "device_status", None)
        if callable(status):
            try:
                return status()
            except Exception as exc:
                return {
                    "requested": "unknown",
                    "resolved": "error",
                    "error": str(exc),
                }
        return {"requested": "test", "resolved": "test"}

    def _follow_readiness_locked(self, now: float) -> str | None:
        if self._selected_target_name is None:
            return "select a detected fruit first"
        if self._target is None:
            return "selected_target_not_found"
        if self._target.visible_frames < self.controller.config.stable_frames_required:
            return "selected_target_not_stable"
        target_age = now - self._target.captured_monotonic_s
        if target_age < 0.0 or target_age > self.controller.config.maximum_target_age_s:
            return "selected_target_stale"
        if (
            self._produce_verified_at is None
            or now - self._produce_verified_at > self.maximum_produce_age_s
            or self._produce_revalidation_failures > 0
        ):
            return "selected_target_not_revalidated"
        return None

    def _require_pointing_idle(self) -> None:
        if self.pointing is not None and self.pointing.active:
            raise RuntimeCommandError(
                "pointing policy owns motion; stop it and wait for recovery"
            )

    def _arrival_pointing_required(self, label: str) -> bool:
        configured = self.mission_config.arrival_pointing_label.strip().lower()
        return bool(
            self.mission_config.arrival_pointing_enabled
            and (
                configured == "all"
                or label.strip().lower() == configured
            )
        )

    def _produce_revalidation_expired_locked(self, now: float) -> bool:
        """Bound target loss by both misses and elapsed time.

        Jetson inference cadence varies, so three quick low-confidence results
        must not erase a still-fresh click.  Conversely, the elapsed-time bound
        guarantees that a genuinely missing target is cleared independently of
        frame rate.
        """
        if (
            self._produce_revalidation_failures
            < self.produce_revalidation_misses_required
        ):
            return False
        return (
            self._produce_verified_at is None
            or now - self._produce_verified_at > self.maximum_produce_age_s
        )

    @staticmethod
    def _target_detection_iou(
        target: TargetObservation, detection: FruitDetection
    ) -> float:
        target_x, target_y, target_width, target_height = target.bbox_xywh
        target_x2 = target_x + target_width
        target_y2 = target_y + target_height
        detection_x1, detection_y1, detection_x2, detection_y2 = detection.bbox_xyxy
        intersection_width = max(
            0, min(target_x2, detection_x2) - max(target_x, detection_x1)
        )
        intersection_height = max(
            0, min(target_y2, detection_y2) - max(target_y, detection_y1)
        )
        intersection = intersection_width * intersection_height
        target_area = target_width * target_height
        detection_area = max(0, detection_x2 - detection_x1) * max(
            0, detection_y2 - detection_y1
        )
        union = target_area + detection_area - intersection
        return 0.0 if union <= 0 else intersection / union

    @staticmethod
    def _detection_bbox_xywh(
        detection: FruitDetection,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = detection.bbox_xyxy
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _observation_from_bbox(
        frame: CameraFrame,
        bbox_xywh: tuple[int, int, int, int],
        *,
        confidence: float | None,
        visible_frames: int,
    ) -> TargetObservation:
        x, y, width, height = bbox_xywh
        x = max(0, min(frame.width - 1, x))
        y = max(0, min(frame.height - 1, y))
        width = max(1, min(frame.width - x, width))
        height = max(1, min(frame.height - y, height))
        return TargetObservation(
            frame_id=frame.frame_id,
            captured_monotonic_s=frame.captured_monotonic_s,
            bbox_xywh=(x, y, width, height),
            center=(x + width // 2, y + height // 2),
            confidence=None if confidence is None else round(confidence, 4),
            visible_frames=visible_frames,
        )
