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
from .pointing import (
    POINTING_PREPARE_CONFIRMATION,
    POINTING_RUN_CONFIRMATION,
    PointingPolicyError,
    PointingPolicyManager,
)
from .types import CameraFrame, TargetObservation, VelocityCommand


ARM_CONFIRMATION = "TARGET AND PATH CLEAR"
NAVIGATION_ARM_CONFIRMATION = "MAP AND PATH CLEAR"
DEMO_CONFIRMATION = "TARGET SAVED AND AREA CLEAR"
DEMO_GO_CONFIRMATION = "CLASS LOCKED AND PATH CLEAR"
VOICE_MISSION_CONFIRMATION = "VOICE COMMAND HEARD"


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


def create_produce_tracker(
    bgr: object, bbox_xywh: tuple[int, int, int, int]
) -> ProduceTrackerProtocol:
    tracker = cv2.TrackerMIL_create()
    tracker.init(bgr, bbox_xywh)
    return tracker


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
        self._final_approach_distance_m = (
            self.mission_config.final_approach_distance_m
        )
        self._state_lock = asyncio.Lock()
        self._stream_condition = asyncio.Condition()
        self._action_lock = asyncio.Lock()
        # Serializes long stock gestures/capture with the low-level pointing
        # handoff. A memory-save Hello must never race policy startup.
        self._exclusive_skill_lock = asyncio.Lock()
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
        self._lease: str | None = None
        self._motion_owner: str | None = None
        self._navigation_deadline: float | None = None
        self._navigation_watchdog_task: asyncio.Task[None] | None = None
        self._last_pulse_at: float | None = None
        self._forward_elapsed_s = 0.0
        self._command = VelocityCommand(reason="disarmed")
        self._follow_task: asyncio.Task[None] | None = None
        self._follow_start_generation = 0
        self._fruit_memory: FruitMemory | None = None
        self._home_pose: tuple[float, float, float] | None = None
        self._mission = MissionTelemetry()
        self._mission_task: asyncio.Task[None] | None = None
        self._demo_go_event = asyncio.Event()
        self._voice_autogo_task: asyncio.Task[None] | None = None
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
        if self.pointing is not None:
            await self.pointing.close()
        await self.stop("shutdown")
        if self.motion is not None:
            await self.motion.close()
        if self.heading_provider is not None:
            self.heading_provider.close()

    async def arm(self, confirmation: str) -> dict[str, object]:
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
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="armed_waiting_for_hold")
            return await self.status()

    async def navigation_arm(self, confirmation: str) -> dict[str, object]:
        """Acquire the factory-avoidance lease for map navigation only."""

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
            async with self._state_lock:
                self._clear_selection_locked()
            try:
                lease = await self.motion.arm()
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
                self._command = await self.motion.send(self._lease, command)
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
            "deadline_s": None
            if self._navigation_deadline is None
            else round(max(0.0, self._navigation_deadline - now), 3),
            "command": self._command.to_dict(),
            "limits": None if motion_status is None else motion_status["limits"],
        }

    async def prepare_pointing(self, confirmation: str) -> dict[str, object]:
        """Stop every locomotion owner and request Unitree's StandDown pose."""

        async with self._exclusive_skill_lock:
            return await self._prepare_pointing(confirmation)

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
            try:
                await self.motion.perform_standdown()
            except MotionError as exc:
                self.pointing.clear_prepared()
                raise RuntimeCommandError(str(exc)) from exc
            self.pointing.mark_prepared()
            self._command = VelocityCommand(reason="pointing_standdown_complete")
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

    async def start_follow(self, confirmation: str) -> dict[str, object]:
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
        await self.arm(confirmation)
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
                    lambda: crop_bbox(frame.bgr, detection.bbox_xyxy)
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
        async with self._state_lock:
            if self._fruit_memory is None:
                raise RuntimeCommandError("save a fruit first")
            memory_label = self._fruit_memory.label
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
        if self.mission_config.return_home_enabled:
            # Capture Home only after the stop boundary has completed, so any
            # final deceleration before this mission cannot offset the origin.
            pose = self.heading_provider.status()
            if not pose.pose_healthy:
                raise RuntimeCommandError("fresh Go2 local position is unavailable")
            assert (
                pose.x_m is not None
                and pose.y_m is not None
                and pose.yaw_rad is not None
            )
            home_pose = (pose.x_m, pose.y_m, pose.yaw_rad)
        async with self._state_lock:
            self._home_pose = home_pose
            self._clear_selection_locked()
            self._demo_go_event.clear()
            self._mission = MissionTelemetry(
                phase=MissionPhase.TURNING,
                reason="starting_measured_turn",
                started_monotonic_s=time.monotonic(),
                home_pose=None
                if home_pose is None
                else {
                    "x_m": round(home_pose[0], 4),
                    "y_m": round(home_pose[1], 4),
                    "yaw_rad": round(home_pose[2], 4),
                },
                return_home_status=(
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
                f"yaw_rad={home_pose[2]:.4f}",
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

    async def set_final_approach_distance(
        self, distance_m: float
    ) -> dict[str, object]:
        """Set the measured post-vision creep distance while Woof is idle."""

        distance_m = float(distance_m)
        if not math.isfinite(distance_m) or not 0.02 <= distance_m <= 0.30:
            raise RuntimeCommandError(
                "final approach distance must be between 0.02 and 0.30 m"
            )
        async with self._action_lock:
            self._require_pointing_idle()
            mission_active = bool(
                self._mission_task is not None and not self._mission_task.done()
            )
            if (
                self._lease is not None
                or self._motion_owner is not None
                or mission_active
                or (self.motion is not None and self.motion.armed)
            ):
                raise RuntimeCommandError(
                    "stop Woof before changing final approach calibration"
                )
            self._final_approach_distance_m = distance_m
        print(
            "final_approach calibration "
            f"distance_m={distance_m:.3f}",
            flush=True,
        )
        return await self.status()

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
            stage_ready = camera_live and produce_live and gpu_ready and motion_ready
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
                    "final_approach_distance_m": self._final_approach_distance_m,
                    "final_approach_mps": self.mission_config.final_approach_mps,
                    "final_approach_timeout_s": (
                        self.mission_config.final_approach_timeout_s
                    ),
                    "active": mission_active,
                    "return_home_enabled": self.mission_config.return_home_enabled,
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
                    pointing_readiness = "lay Woof down first"
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
                },
                "frame_count": self._frame_count,
                "frame_width": self._frame_width,
                "frame_height": self._frame_height,
                "camera_fps": None if camera_fps is None else round(camera_fps, 1),
                "camera_stream": "sdk_jpeg_passthrough",
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
                    await self.stop("selected_target_not_revalidated")
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
                        ):
                            if self._produce_tracker is None:
                                self._produce_tracker = tracker
                                self._produce_tracker_label = refresh_name
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                revalidation_failed = False
                async with self._state_lock:
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
                    await self.stop("selected_target_not_revalidated")
            await asyncio.sleep(0)

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
            else:
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
        """Run the bounded reach while the selected pear box is still live."""

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
            self._mission.reason = "laying_down_for_live_bbox_reach"
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
                    self._mission.reason = "running_live_bbox_reach_policy"
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
                    await self.motion.perform_balance_stand()
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
    ) -> ClassCandidate:
        """Require a new multi-frame class detection before locomotion."""

        async with self._state_lock:
            last_frame_id = self._produce_frame_id
        deadline = (
            time.monotonic() + self.mission_config.match_reacquire_timeout_s
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

    async def _start_memory_approach(self, match: ClassCandidate) -> None:
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
        await self.start_follow(ARM_CONFIRMATION)
        async with self._state_lock:
            self._mission.phase = MissionPhase.APPROACHING
            self._mission.reason = "approaching_saved_fruit_class"
            self._mission.match_failures = 0

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
        return (
            bottom_ratio >= self.mission_config.near_bottom_ratio
            and center_ratio >= self.mission_config.near_center_ratio
            and bbox_height_ratio >= self.mission_config.near_bbox_height_ratio,
            bbox_height_ratio,
        )

    async def _monitor_memory_approach(self) -> str:
        deadline = time.monotonic() + self.controller.config.forward_budget_s + 4.0
        last_frame_id: int | None = None
        failures = 0
        near_target_seen = False
        near_confirmations = 0
        last_near_at: float | None = None
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
                ):
                    return "target_visible_in_pointing_range"
                if near_target_recent and result.best is None:
                    return "target_class_reached_camera_edge"
                if failures >= self.mission_config.approach_misses_allowed:
                    raise RuntimeCommandError("saved fruit class lost during approach")

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
        """Advance a measured, calibrated distance after a verified edge loss."""

        if approach_reason != "target_class_reached_camera_edge":
            raise RuntimeCommandError(
                f"final approach unavailable after {approach_reason}"
            )
        if self.heading_provider is None:
            raise RuntimeCommandError(
                "fresh Go2 local pose is unavailable for final approach"
            )
        initial = self.heading_provider.status()
        if not initial.pose_healthy:
            raise RuntimeCommandError(
                "fresh Go2 local pose is unavailable for final approach"
            )
        assert initial.x_m is not None and initial.y_m is not None
        start_x = initial.x_m
        start_y = initial.y_m
        target_distance = self._final_approach_distance_m
        started_at = time.monotonic()
        deadline = started_at + self.mission_config.final_approach_timeout_s
        last_tick_at = started_at
        best_distance = 0.0
        last_progress_at = started_at
        commanded_distance = 0.0
        async with self._state_lock:
            self._mission.phase = MissionPhase.FINAL_APPROACHING
            self._mission.reason = "measured_touch_range_approach"
            self._mission.final_approach_status = "running"
            self._mission.final_approach_elapsed_s = 0.0
            self._mission.final_approach_commanded_distance_m = 0.0
            self._mission.final_approach_measured_distance_m = 0.0
        print(
            "final_approach event=start "
            f"target_distance_m={target_distance:.3f} "
            f"speed_mps={self.mission_config.final_approach_mps:.3f} "
            f"timeout_s={self.mission_config.final_approach_timeout_s:.2f}",
            flush=True,
        )
        await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        while time.monotonic() < deadline:
            now = time.monotonic()
            async with self._state_lock:
                camera_fresh = bool(
                    self._last_frame_at is not None
                    and now - self._last_frame_at < 1.0
                )
            if not camera_fresh:
                raise RuntimeCommandError(
                    "camera became stale during final approach"
                )
            sample = self.heading_provider.status()
            if not sample.pose_healthy:
                raise RuntimeCommandError(
                    "Go2 local pose became stale during final approach"
                )
            assert sample.x_m is not None and sample.y_m is not None
            measured_distance = math.hypot(
                sample.x_m - start_x,
                sample.y_m - start_y,
            )
            if measured_distance >= target_distance:
                await self.navigation_command(0.0, 0.0)
                elapsed = now - started_at
                async with self._state_lock:
                    self._mission.final_approach_status = "complete"
                    self._mission.final_approach_elapsed_s = elapsed
                    self._mission.final_approach_commanded_distance_m = (
                        commanded_distance
                    )
                    self._mission.final_approach_measured_distance_m = (
                        measured_distance
                    )
                print(
                    "final_approach event=complete "
                    f"elapsed_s={elapsed:.3f} "
                    f"commanded_distance_m={commanded_distance:.3f} "
                    f"measured_distance_m={measured_distance:.3f}",
                    flush=True,
                )
                return "measured_final_approach_complete"
            if (
                measured_distance
                >= best_distance
                + self.mission_config.final_approach_stall_min_progress_m
            ):
                best_distance = measured_distance
                last_progress_at = now
            elif (
                now - last_progress_at
                >= self.mission_config.final_approach_stall_timeout_s
            ):
                raise RuntimeCommandError(
                    "final approach stalled before touch range"
                )
            remaining = target_distance - measured_distance
            command_mps = min(
                self.mission_config.final_approach_mps,
                max(0.04, remaining * 1.2),
            )
            await self.navigation_command(command_mps, 0.0)
            tick_s = max(0.0, now - last_tick_at)
            commanded_distance += command_mps * tick_s
            last_tick_at = now
            async with self._state_lock:
                self._mission.final_approach_elapsed_s = now - started_at
                self._mission.final_approach_commanded_distance_m = (
                    commanded_distance
                )
                self._mission.final_approach_measured_distance_m = (
                    measured_distance
                )
            await asyncio.sleep(0.05)
        raise RuntimeCommandError("final approach timed out before touch range")

    async def _return_home(self) -> None:
        """Return to the captured start pose using fresh local odometry.

        This is deliberately a short-range stage controller, not a global map
        planner. Every command uses the factory obstacle-avoidance lease, and
        stale pose, lack of progress, or timeout immediately aborts the run.
        """

        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 local pose is unavailable")
        home = self._home_pose
        if home is None:
            raise RuntimeCommandError("home pose was not captured")
        initial = self.heading_provider.status()
        if not initial.pose_healthy:
            raise RuntimeCommandError("fresh Go2 local pose is unavailable")
        assert initial.x_m is not None and initial.y_m is not None
        initial_distance = math.hypot(home[0] - initial.x_m, home[1] - initial.y_m)
        best_distance = initial_distance
        last_progress_at = time.monotonic()
        started_at = time.monotonic()
        deadline = started_at + self.mission_config.return_timeout_s
        async with self._state_lock:
            self._mission.phase = MissionPhase.RETURNING_HOME
            self._mission.reason = "returning_to_saved_start_pose"
            self._mission.return_home_status = "running"
            self._mission.return_distance_m = initial_distance
            self._mission.return_progress_m = 0.0
        print(
            "return_home event=start "
            f"distance_m={initial_distance:.3f} "
            f"timeout_s={self.mission_config.return_timeout_s:.2f}",
            flush=True,
        )
        await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
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
            dx = home[0] - sample.x_m
            dy = home[1] - sample.y_m
            distance = math.hypot(dx, dy)
            if distance <= self.mission_config.return_arrival_tolerance_m:
                heading_error = normalize_angle(home[2] - sample.yaw_rad)
                if (
                    abs(heading_error)
                    <= self.mission_config.return_heading_tolerance_rad
                ):
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
                        f"elapsed_s={now - started_at:.3f} "
                        f"distance_m={distance:.3f} "
                        f"heading_error_deg={math.degrees(heading_error):.1f}",
                        flush=True,
                    )
                    return
                forward_mps = 0.0
            else:
                goal_heading = math.atan2(dy, dx)
                heading_error = normalize_angle(goal_heading - sample.yaw_rad)
                if abs(heading_error) > self.mission_config.return_heading_gate_rad:
                    forward_mps = 0.0
                else:
                    remaining = max(
                        0.0,
                        distance - self.mission_config.return_arrival_tolerance_m,
                    )
                    forward_mps = min(
                        self.mission_config.return_forward_mps,
                        max(0.06, remaining * 0.8),
                    ) * max(0.25, math.cos(heading_error))

            yaw_rps = self.mission_config.return_yaw_gain * heading_error
            if self.motion is not None:
                yaw_limit = self.motion.config.maximum_yaw_rps
                yaw_rps = max(-yaw_limit, min(yaw_limit, yaw_rps))
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
        return bool(
            self.mission_config.arrival_pointing_enabled
            and label.strip().lower()
            == self.mission_config.arrival_pointing_label.strip().lower()
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
