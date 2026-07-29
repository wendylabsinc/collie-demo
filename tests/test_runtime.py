from __future__ import annotations

import asyncio
import math
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from collie_demo.controller import ApproachConfig, ApproachController
from collie_demo.fruit import FruitDetection
from collie_demo.heading import HeadingSample, normalize_angle
from collie_demo.mission import MissionConfig
from collie_demo.motion import UnitreeMotionAdapter
from collie_demo.runtime import (
    ARM_CONFIRMATION,
    DEMO_CONFIRMATION,
    DEMO_GO_CONFIRMATION,
    NAVIGATION_ARM_CONFIRMATION,
    VOICE_MISSION_CONFIRMATION,
    CollieRuntime,
    RuntimeCommandError,
)
from collie_demo.types import CameraFrame

from test_motion import FakeAvoidance, FakeSport


class StaticFruitCamera:
    def __init__(self) -> None:
        self.frame_id = 0

    def read(self) -> CameraFrame:
        self.frame_id += 1
        image = np.full((720, 1280, 3), 120, dtype=np.uint8)
        return CameraFrame(self.frame_id, time.monotonic(), image)


class OneBadFrameCamera(StaticFruitCamera):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next = False

    def read(self) -> CameraFrame:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("camera returned an invalid JPEG")
        return super().read()


class FakeProduceDetector:
    model_path = Path("/models/snapstock.pt")
    confidence = 0.5
    names = {1: "apple", 6: "banana", 30: "Strawberry", 57: "strawberry"}

    def device_status(self) -> dict[str, object]:
        return {
            "requested": "0",
            "resolved": "cuda:0",
            "cuda_available": True,
        }

    def detect(self, _image: object) -> list[FruitDetection]:
        return [
            FruitDetection(
                class_id=1,
                label="apple",
                confidence=0.91,
                bbox_xyxy=(100, 200, 180, 300),
                center=(140, 250),
            ),
            FruitDetection(
                class_id=6,
                label="banana",
                confidence=0.88,
                bbox_xyxy=(300, 200, 400, 280),
                center=(350, 240),
            ),
            FruitDetection(
                class_id=1,
                label="apple",
                confidence=0.82,
                bbox_xyxy=(960, 200, 1060, 280),
                center=(1010, 240),
            ),
            FruitDetection(
                class_id=30,
                label="Strawberry",
                confidence=0.77,
                bbox_xyxy=(560, 200, 660, 280),
                center=(610, 240),
            ),
        ]


class ToggleProduceDetector(FakeProduceDetector):
    def __init__(self) -> None:
        self.visible = True

    def detect(self, image: object) -> list[FruitDetection]:
        return super().detect(image) if self.visible else []


class BurstMissProduceDetector(FakeProduceDetector):
    def __init__(self) -> None:
        self.misses_remaining = 0

    def detect(self, image: object) -> list[FruitDetection]:
        if self.misses_remaining > 0:
            self.misses_remaining -= 1
            return []
        return super().detect(image)


class SlowProduceDetector(FakeProduceDetector):
    def detect(self, image: object) -> list[FruitDetection]:
        time.sleep(0.4)
        return super().detect(image)


class JetsonSpeedProduceDetector(FakeProduceDetector):
    def detect(self, image: object) -> list[FruitDetection]:
        time.sleep(0.12)
        return super().detect(image)


class FastMovingProduceDetector(FakeProduceDetector):
    def __init__(self) -> None:
        self.x = 300

    def detect(self, _image: object) -> list[FruitDetection]:
        self.x = min(900, self.x + 80)
        return [
            FruitDetection(
                class_id=6,
                label="banana",
                confidence=0.88,
                bbox_xyxy=(self.x, 200, self.x + 50, 280),
                center=(self.x + 25, 240),
            )
        ]


class FakeProduceTracker:
    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox

    def update(
        self, _image: object
    ) -> tuple[bool, tuple[float, float, float, float]]:
        return True, tuple(float(value) for value in self.bbox)


def fake_tracker_factory(
    _image: object, bbox: tuple[int, int, int, int]
) -> FakeProduceTracker:
    return FakeProduceTracker(bbox)


class NearBananaCamera(StaticFruitCamera):
    def __init__(self) -> None:
        super().__init__()
        self.show_different_banana = False

    def read(self) -> CameraFrame:
        self.frame_id += 1
        image = np.full((720, 1280, 3), 28, dtype=np.uint8)
        if self.show_different_banana:
            cv2.circle(image, (640, 645), 65, (210, 45, 35), -1)
        else:
            image[580:710, 540:740] = (20, 210, 235)
        return CameraFrame(self.frame_id, time.monotonic(), image)


class NearBananaDetector:
    model_path = Path("/models/collie.pt")
    confidence = 0.2
    class_thresholds = {"banana": 0.2}
    names = {0: "banana"}

    def device_status(self) -> dict[str, object]:
        return {
            "requested": "0",
            "resolved": "cuda:0",
            "cuda_available": True,
        }

    def detect(self, _image: object) -> list[FruitDetection]:
        return [
            FruitDetection(
                class_id=0,
                label="banana",
                confidence=0.92,
                bbox_xyxy=(540, 580, 740, 710),
                center=(640, 645),
            )
        ]


class NearBananaThenLostDetector(NearBananaDetector):
    """Simulate a small floor fruit leaving the lower edge after motion."""

    def __init__(self, avoidance: FakeAvoidance) -> None:
        self.avoidance = avoidance

    def detect(self, image: object) -> list[FruitDetection]:
        if any(move[0] > 0.0 for move in self.avoidance.moves):
            return []
        return [
            FruitDetection(
                class_id=0,
                label="banana",
                confidence=0.92,
                bbox_xyxy=(620, 670, 660, 710),
                center=(640, 690),
            )
        ]


class NearPearDetector(NearBananaDetector):
    class_thresholds = {"pear": 0.2}
    names = {0: "pear"}

    def detect(self, _image: object) -> list[FruitDetection]:
        return [
            FruitDetection(
                class_id=0,
                label="pear",
                confidence=0.93,
                bbox_xyxy=(540, 580, 740, 710),
                center=(640, 645),
            )
        ]


class MotionCoupledHeading:
    def __init__(self, avoidance: FakeAvoidance, sport: FakeSport | None = None) -> None:
        self.avoidance = avoidance
        self.sport = sport
        self.yaw = 0.0

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def status(self) -> HeadingSample:
        yaw_rate = 0.0
        if self.sport is not None:
            yaw_rate = self.sport.current_move[2]
        if not yaw_rate and self.avoidance.moves:
            yaw_rate = self.avoidance.moves[-1][2]
        if yaw_rate:
            self.yaw = normalize_angle(
                self.yaw + (0.36 if yaw_rate > 0.0 else -0.36)
            )
        return HeadingSample(self.yaw, 0.0, True)


class MotionCoupledPose(MotionCoupledHeading):
    def __init__(self, avoidance: FakeAvoidance, sport: FakeSport | None = None) -> None:
        super().__init__(avoidance, sport)
        self.x_m = 0.0
        self.y_m = 0.0
        self.last_update_at = time.monotonic()
        self.simulation_speed = 10.0

    def status(self) -> HeadingSample:
        now = time.monotonic()
        elapsed_s = min(0.1, now - self.last_update_at) * self.simulation_speed
        self.last_update_at = now
        yaw_rate = 0.0
        if self.sport is not None:
            yaw_rate = self.sport.current_move[2]
        if not yaw_rate and self.avoidance.moves:
            yaw_rate = self.avoidance.moves[-1][2]
        self.yaw = normalize_angle(self.yaw + yaw_rate * elapsed_s)
        forward_mps = self.avoidance.moves[-1][0] if self.avoidance.moves else 0.0
        if forward_mps:
            self.x_m += forward_mps * math.cos(self.yaw) * elapsed_s
            self.y_m += forward_mps * math.sin(self.yaw) * elapsed_s
        return HeadingSample(
            self.yaw,
            0.0,
            True,
            None,
            self.x_m,
            self.y_m,
            True,
            None,
        )


class OutwardStallingPose(MotionCoupledPose):
    """Allow a little outward travel, then block until commands return home."""

    def __init__(
        self,
        avoidance: FakeAvoidance,
        sport: FakeSport | None = None,
        *,
        maximum_outward_distance_m: float = 0.04,
    ) -> None:
        super().__init__(avoidance, sport)
        self.maximum_outward_distance_m = maximum_outward_distance_m

    def status(self) -> HeadingSample:
        previous_x = self.x_m
        previous_y = self.y_m
        sample = super().status()
        previous_distance = math.hypot(previous_x, previous_y)
        proposed_distance = math.hypot(self.x_m, self.y_m)
        if (
            previous_distance >= self.maximum_outward_distance_m
            and proposed_distance > previous_distance
        ):
            self.x_m = previous_x
            self.y_m = previous_y
            return HeadingSample(
                self.yaw,
                0.0,
                True,
                None,
                self.x_m,
                self.y_m,
                True,
                None,
            )
        return sample


class CallbackStretchSport(FakeSport):
    def __init__(self, on_stretch: Callable[[], None]) -> None:
        super().__init__()
        self.on_stretch = on_stretch

    def Stretch(self) -> int:
        result = super().Stretch()
        self.on_stretch()
        return result


class FailingStretchSport(FakeSport):
    def Stretch(self) -> int:
        self.stretch_calls += 1
        return 3104


class FakePointingPolicy:
    def __init__(self) -> None:
        self.available = True
        self.active = False
        self.prepared = False
        self.started: tuple[str, float] | None = None
        self.phase = "idle"
        self.last_report: dict[str, object] | None = None

    def mark_prepared(self) -> None:
        self.prepared = True
        self.phase = "prepared"

    def clear_prepared(self) -> None:
        self.prepared = False
        if not self.active and self.phase == "prepared":
            self.phase = "idle"

    def status(self) -> dict[str, object]:
        return {
            "available": self.available,
            "enabled": True,
            "active": self.active,
            "prepared": self.prepared,
            "phase": self.phase,
            "duration_s": 1.0,
            "error": None,
            "last_report": self.last_report,
        }

    async def start(
        self, *, target_label: str, minimum_confidence: float
    ) -> dict[str, object]:
        if not self.prepared:
            raise AssertionError("pointing started without StandDown preparation")
        self.started = (target_label, minimum_confidence)
        self.active = True
        self.phase = "running"
        self.last_report = None
        return self.status()

    async def wait(self, *, timeout_s: float = 20.0) -> dict[str, object]:
        assert timeout_s > 0.0
        assert self.active
        self.active = False
        self.prepared = False
        self.phase = "complete"
        self.last_report = {
            "outcome": "completed_full_policy",
            "controller_restored": True,
            "selected_label": None if self.started is None else self.started[0],
        }
        return self.status()

    async def stop(self) -> dict[str, object]:
        self.active = False
        self.prepared = False
        if self.phase in {"running", "prepared"}:
            self.phase = "idle"
        return self.status()

    async def close(self) -> None:
        await self.stop()


def test_runtime_requires_confirmation_then_pulses_forward() -> None:
    async def scenario() -> None:
        avoidance = FakeAvoidance()
        motion = UnitreeMotionAdapter(FakeSport(), avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.22)
            try:
                await runtime.arm("wrong")
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("wrong confirmation unexpectedly armed motion")
            try:
                await runtime.arm(ARM_CONFIRMATION)
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("motion armed without a selected fruit")
            status = await runtime.status()
            assert status["camera_fps"] is not None
            assert status["frame_width"] == 1280
            assert status["frame_height"] == 720
            assert status["camera_stream"] == "sdk_jpeg_passthrough"
            assert status["produce"]["detections"][0]["label"] == "apple"
            assert status["produce"]["inference_ms"] is not None
            assert status["selected_target_name"] is None

            apple = await runtime.select_target("apple", (1010, 240))
            assert apple["selected_target_name"] == "apple"
            assert apple["runtime_id"]
            apple_lock_id = apple["target_lock_id"]
            assert isinstance(apple_lock_id, int)
            assert apple["selected_target_kind"] == "produce"
            assert apple["selected_target_hint"] == (1010, 240)
            assert apple["armed"] is False
            await asyncio.sleep(0.18)
            await runtime.arm(ARM_CONFIRMATION)
            apple_status = await runtime.pulse()
            assert apple_status["command"]["reason"] == "curving_to_selected_target"
            assert avoidance.moves[-1][2] < 0.0

            banana = await runtime.select_target("banana", (350, 240))
            assert banana["selected_target_name"] == "banana"
            assert banana["target_lock_id"] == apple_lock_id + 1
            assert banana["armed"] is False
            await asyncio.sleep(0.18)
            await runtime.arm(ARM_CONFIRMATION)
            banana_status = await runtime.pulse()
            assert banana_status["command"]["reason"] == "curving_to_selected_target"
            assert avoidance.moves[-1][2] > 0.0

            strawberry = await runtime.select_target("Strawberry", (610, 240))
            assert strawberry["selected_target_name"] == "Strawberry"

            try:
                await runtime.select_target("purple")
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("invalid target color was accepted")
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_pointing_ui_flow_prepares_locks_and_stops_one_motion_owner() -> None:
    async def scenario() -> None:
        sport = FakeSport()
        pointing = FakePointingPolicy()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=UnitreeMotionAdapter(sport, FakeAvoidance()),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            pointing=pointing,  # type: ignore[arg-type]
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.select_target("banana", (350, 240))
            await asyncio.sleep(0.10)

            prepared = await runtime.prepare_pointing(
                "WOOF IS CLEAR TO LIE DOWN"
            )
            assert sport.standdown_calls == 1
            assert prepared["pointing"]["prepared"] is True
            for _ in range(100):
                prepared = await runtime.status()
                if prepared["pointing"]["can_run"]:
                    break
                await asyncio.sleep(0.01)
            assert prepared["pointing"]["can_run"] is True

            started = await runtime.start_pointing(
                "WOOF IS LYING DOWN AND TARGET AREA IS CLEAR"
            )
            assert pointing.started == ("banana", 0.5)
            assert started["pointing"]["active"] is True
            assert started["motion_owner"] == "pointing"
            assert started["can_follow"] is False

            try:
                await runtime.select_target("apple", (140, 250))
            except RuntimeCommandError as exc:
                assert "pointing policy owns motion" in str(exc)
            else:
                raise AssertionError("target lock changed during low-level control")

            stopped = await runtime.stop_pointing()
            assert stopped["pointing"]["active"] is False
            assert stopped["motion_owner"] is None
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_navigation_owner_accepts_bounded_commands_without_a_fruit() -> None:
    async def scenario() -> None:
        avoidance = FakeAvoidance()
        motion = UnitreeMotionAdapter(FakeSport(), avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            navigation_command_lease_s=0.05,
        )
        await runtime.start()
        try:
            try:
                await runtime.navigation_arm("wrong")
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("wrong navigation confirmation armed motion")

            armed = await runtime.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
            assert armed["armed"] is True
            assert armed["owner"] == "navigation"
            commanded = await runtime.navigation_command(0.3, -0.2)
            assert commanded["command"]["forward_mps"] == 0.08
            assert commanded["command"]["yaw_rps"] == -0.2
            try:
                await runtime.pulse()
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("fruit pulse stole the navigation lease")
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_navigation_lease_expires_to_hardware_stop() -> None:
    async def scenario() -> None:
        sport = FakeSport()
        motion = UnitreeMotionAdapter(sport, FakeAvoidance())
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            loop_hz=60.0,
            navigation_command_lease_s=0.05,
        )
        await runtime.start()
        try:
            await runtime.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
            await runtime.navigation_command(0.03, 0.0)
            await asyncio.sleep(0.18)
            status = await runtime.navigation_status()
            assert status["armed"] is False
            assert sport.stop_calls >= 1
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_sustained_yolo_loss_preserves_disarmed_choice_but_removes_readiness() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
            maximum_produce_age_s=0.05,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("produce detector never returned a frame")

            selected = await runtime.select_target("banana", (350, 240))
            assert selected["selected_target_name"] == "banana"
            detector.visible = False

            for _ in range(100):
                status = await runtime.status()
                if (
                    status["selected_target_name"] == "banana"
                    and status["selected_target"] is None
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("missing banana did not become stale")

            assert status["selected_target_name"] == "banana"
            assert status["target_lock_id"] == 1
            assert status["selected_target"] is None
            assert status["produce"]["tracker"]["label"] is None
            assert status["produce"]["tracker"]["revalidation_failures"] >= 3
            assert (
                status["produce"]["tracker"]["revalidation_failures_required"]
                == 3
            )
            assert status["command"]["reason"] == "selected_target_not_revalidated"
            assert status["can_follow"] is False

            detector.visible = True
            for _ in range(100):
                status = await runtime.status()
                if (
                    status["selected_target"] is not None
                    and status["produce"]["tracker"]["revalidation_failures"] == 0
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("same selected banana was not reacquired")

            assert status["selected_target_name"] == "banana"
            assert status["target_lock_id"] == 1
            assert status["produce"]["tracker"]["revalidation_failures"] == 0
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_one_bad_camera_frame_does_not_clear_fresh_selection() -> None:
    async def scenario() -> None:
        camera = OneBadFrameCamera()
        runtime = CollieRuntime(
            camera=camera,
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=3,
                    maximum_target_age_s=0.75,
                )
            ),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("produce detector never returned a frame")

            await runtime.select_target("banana", (350, 240))
            camera.fail_next = True
            await asyncio.sleep(0.08)

            status = await runtime.status()
            assert status["selected_target_name"] == "banana"
            assert status["selected_target"] is not None
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_three_fast_yolo_misses_do_not_clear_fresh_selection() -> None:
    async def scenario() -> None:
        detector = BurstMissProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
            produce_revalidation_misses_required=3,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.select_target("banana", (350, 240))
            detector.misses_remaining = 3
            await asyncio.sleep(0.08)

            status = await runtime.status()
            assert detector.misses_remaining == 0
            assert status["selected_target_name"] == "banana"
            assert status["selected_target"] is not None
            assert status["produce"]["tracker"]["revalidation_failures"] == 0
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_robot_side_follow_continues_without_browser_pulses() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
            follow_period_s=0.02,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.08)
            await runtime.select_target("banana", (350, 240))
            status = await runtime.start_follow(ARM_CONFIRMATION)
            assert status["follow_active"] is True
            await asyncio.sleep(0.12)

            status = await runtime.status()
            nonzero_moves = [move for move in avoidance.moves if move != (0.0, 0.0, 0.0)]
            assert status["stage_ready"] is True
            assert all(status["health"].values())
            assert status["follow_active"] is True
            assert status["armed"] is True
            assert len(nonzero_moves) >= 2

            await runtime.stop("test_stop")
            await asyncio.sleep(0.05)
            status = await runtime.status()
            assert status["follow_active"] is False
            assert status["armed"] is False
            assert status["command"]["forward_mps"] == 0.0
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_confirmed_mission_lock_arms_without_rewaiting_for_yolo() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=3)
            ),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            follow_start_timeout_s=0.01,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("test detector did not publish")

            await runtime.select_target(
                "banana",
                (350, 240),
                confirmed_visible_frames=3,
            )
            status = await runtime.start_follow(ARM_CONFIRMATION)

            assert status["follow_active"] is True
            assert status["armed"] is True
            assert status["selected_target"]["visible_frames"] >= 3
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_stale_detection_cannot_be_selected() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            detector.visible = False
            for _ in range(100):
                status = await runtime.status()
                if not status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)

            try:
                await runtime.select_target("banana", (350, 240))
            except RuntimeCommandError as exc:
                assert "no longer freshly detected" in str(exc)
            else:
                raise AssertionError("stale banana detection was selectable")
            assert (await runtime.status())["selected_target_name"] is None
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_stop_cancels_a_follow_request_waiting_for_target_stability() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=100)
            ),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.04)
            await runtime.select_target("banana", (350, 240))
            pending_follow = asyncio.create_task(
                runtime.start_follow(ARM_CONFIRMATION)
            )
            await asyncio.sleep(0.04)
            await runtime.stop("operator_stop")
            try:
                await pending_follow
            except RuntimeCommandError as exc:
                assert str(exc) == "follow start cancelled"
            else:
                raise AssertionError("cancelled follow request unexpectedly started")

            status = await runtime.status()
            assert status["follow_active"] is False
            assert status["armed"] is False
            assert status["command"]["reason"] == "operator_stop"
            assert not [
                move for move in avoidance.moves if move != (0.0, 0.0, 0.0)
            ]
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_sustained_yolo_loss_stops_an_active_robot_side_follow() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
            maximum_produce_age_s=0.05,
            follow_period_s=0.02,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.08)
            await runtime.select_target("banana", (350, 240))
            await runtime.start_follow(ARM_CONFIRMATION)
            detector.visible = False

            for _ in range(100):
                status = await runtime.status()
                if not status["armed"] and status["selected_target_name"] is None:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("active follow did not stop after sustained loss")

            assert status["follow_active"] is False
            assert status["command"]["reason"] == "selected_target_not_revalidated"
            assert sport.stop_calls >= 1
            assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_old_yolo_result_does_not_replace_fresh_tracker_observation() -> None:
    async def scenario() -> None:
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=3)
            ),
            motion=UnitreeMotionAdapter(FakeSport(), FakeAvoidance()),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=SlowProduceDetector(),
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("slow detector never returned a frame")

            await runtime.select_target("banana", (350, 240))
            await asyncio.sleep(0.1)
            target_ages: list[float] = []
            readiness: list[str] = []
            for _ in range(60):
                status = await runtime.status()
                if status["selected_target_age_s"] is not None:
                    target_ages.append(status["selected_target_age_s"])
                readiness.append(status["follow_readiness"])
                await asyncio.sleep(0.02)

            assert target_ages
            assert max(target_ages) < 0.35
            assert "selected_target_stale" not in readiness
            assert status["can_follow"] is True
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_saved_fruit_memory_survives_visual_target_loss() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            loop_hz=60.0,
            maximum_produce_age_s=0.1,
            mission_config=MissionConfig(
                enabled=True,
                capture_timeout_s=0.6,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            saved = await runtime.remember_target("banana", (350, 240))
            assert saved["memory"]["label"] == "banana"
            assert saved["memory"]["role"] == "target_class"

            detector.visible = False
            for _ in range(100):
                status = await runtime.status()
                if not status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)

            assert status["selected_target_name"] is None
            assert status["memory"]["label"] == "banana"
            assert status["mission"]["phase"] == "memorized"
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_saving_initial_fruit_runs_one_stock_hello_gesture() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            mission_config=MissionConfig(
                enabled=True,
                initial_hello_enabled=True,
                capture_timeout_s=0.6,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            saved = await runtime.remember_target("banana", (350, 240))

            assert saved["memory"]["label"] == "banana"
            assert saved["mission"]["initial_hello_status"] == "complete"
            assert saved["mission"]["initial_hello_error"] is None
            assert sport.hello_calls == 1
            assert saved["armed"] is False
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_saving_initial_fruit_accepts_already_stopped_hello_boundary() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        sport.stop_result = -1
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            mission_config=MissionConfig(
                enabled=True,
                initial_hello_enabled=True,
                capture_timeout_s=0.6,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            saved = await runtime.remember_target("banana", (350, 240))

            assert saved["mission"]["initial_hello_status"] == "complete"
            assert saved["mission"]["can_start"] is False
            assert saved["motion"]["fault"] is None
            assert sport.hello_calls == 1
        finally:
            sport.stop_result = 0
            await runtime.close()

    asyncio.run(scenario())


def test_start_over_invalidates_an_inflight_save_and_resets_the_whole_round() -> None:
    class BlockingHelloMotion:
        def __init__(self) -> None:
            self.initialized = False
            self.hello_started = asyncio.Event()
            self.release_hello = asyncio.Event()

        @property
        def armed(self) -> bool:
            return False

        async def initialize(self) -> None:
            self.initialized = True

        async def perform_hello(self) -> None:
            self.hello_started.set()
            await self.release_hello.wait()

        async def emergency_stop(self) -> list[str]:
            if self.hello_started.is_set():
                self.release_hello.set()
            return []

        async def close(self) -> list[str]:
            self.release_hello.set()
            return []

        def status(self) -> dict[str, object]:
            return {
                "initialized": self.initialized,
                "fault": None,
                "limits": {
                    "forward_mps": 0.0,
                    "yaw_rps": 0.0,
                    "lateral_mps": 0.0,
                },
                "last_command": {
                    "forward_mps": 0.0,
                    "yaw_rps": 0.0,
                    "reason": "test",
                },
            }

    async def scenario() -> None:
        motion = BlockingHelloMotion()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,  # type: ignore[arg-type]
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            mission_config=MissionConfig(
                enabled=True,
                initial_hello_enabled=True,
                capture_timeout_s=0.6,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            old_round_id = status["round_id"]
            save_task = asyncio.create_task(
                runtime.remember_target(
                    "banana",
                    (350, 240),
                    expected_round_id=old_round_id,
                )
            )
            await asyncio.wait_for(motion.hello_started.wait(), timeout=1.0)

            reset = await runtime.reset_round()

            try:
                await save_task
            except RuntimeCommandError as exc:
                assert "round was reset" in str(exc)
            else:
                raise AssertionError("an old save repopulated a reset round")
            assert reset["round_id"] != old_round_id
            assert reset["memory"] is None
            assert reset["selected_target_name"] is None
            assert reset["mission"]["phase"] == "idle"
            assert reset["mission"]["reason"] == "round_reset"
            assert reset["follow_active"] is False
            assert reset["armed"] is False
            assert reset["forward_elapsed_s"] == 0.0
            assert reset["command"] == {
                "forward_mps": 0.0,
                "yaw_rps": 0.0,
                "reason": "round_reset",
            }
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_return_home_demo_refuses_to_start_without_fresh_local_position() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        runtime = CollieRuntime(
            camera=NearBananaCamera(),
            controller=ApproachController(),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaDetector(),
            loop_hz=60.0,
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                return_home_enabled=True,
                capture_timeout_s=0.6,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (640, 645))

            try:
                await runtime.start_demo(DEMO_CONFIRMATION)
            except RuntimeCommandError as exc:
                assert "local position" in str(exc)
            else:
                raise AssertionError("demo started without a fresh local position")

            status = await runtime.status()
            assert status["mission"]["can_start"] is False
            assert "local position" in status["mission"]["readiness"]
            assert status["armed"] is False
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_memory_demo_turns_searches_and_reuses_guarded_follow() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        camera = NearBananaCamera()
        pose = MotionCoupledPose(avoidance, sport)
        runtime = CollieRuntime(
            camera=camera,
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                    forward_mps=0.08,
                    forward_budget_s=0.08,
                )
            ),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaThenLostDetector(avoidance),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            heading_provider=pose,
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                match_stretch_enabled=True,
                match_stretch_settle_s=0.0,
                match_reacquire_timeout_s=1.0,
                arrival_rest_enabled=True,
                arrival_rest_duration_s=0.01,
                return_home_enabled=True,
                return_arrival_tolerance_m=0.02,
                return_heading_tolerance_rad=0.10,
                return_heading_gate_rad=0.55,
                return_forward_mps=0.08,
                return_yaw_gain=1.2,
                return_timeout_s=3.0,
                return_stall_timeout_s=0.75,
                return_stall_min_progress_m=0.005,
                capture_timeout_s=0.6,
                match_confirmations_required=2,
                approach_misses_allowed=2,
                turn_angle_rad=0.65,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.5,
                search_rate_rps=0.10,
                search_sweep_rad=2.5,
                search_timeout_s=1.5,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (640, 645))
            started = await runtime.start_demo(DEMO_CONFIRMATION)
            assert started["mission"]["active"] is True
            assert started["mission"]["direct_turn_enabled"] is True

            for _ in range(300):
                status = await runtime.status()
                if status["mission"]["phase"] in {"waiting_for_go", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("memory demo did not wait for Go")

            assert status["mission"]["phase"] == "waiting_for_go", repr(
                (status["mission"], status["last_error"])
            )
            assert status["mission"]["can_go"] is True
            assert not any(move[0] > 0.0 for move in avoidance.moves)
            assert status["armed"] is False
            assert status["command"]["forward_mps"] == 0.0

            released = await runtime.approve_demo_go(DEMO_GO_CONFIRMATION)
            assert released["mission"]["phase"] == "confirming"
            assert released["mission"]["can_go"] is False

            for _ in range(600):
                status = await runtime.status()
                if status["mission"]["phase"] in {"success", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("memory demo did not finish after Go")

            assert status["mission"]["phase"] == "success", repr(
                (status["mission"], status["last_error"])
            )
            assert status["mission"]["match_stretch_status"] == "complete"
            assert status["mission"]["match_stretch_error"] is None
            assert (
                status["mission"]["arrival_hello_status"]
                == "replaced_by_arrival_rest"
            )
            assert status["mission"]["arrival_rest_status"] == "complete"
            assert status["mission"]["arrival_rest_error"] is None
            assert status["mission"]["near_target_seen"] is True
            assert status["mission"]["final_approach_status"] == "complete"
            assert status["mission"]["final_approach_measured_distance_m"] >= 0.10
            assert status["mission"]["return_home_status"] == "complete"
            assert status["mission"]["return_distance_m"] <= 0.02
            assert status["mission"]["home_pose"] == {
                "x_m": 0.0,
                "y_m": 0.0,
                "yaw_rad": 0.0,
            }
            assert status["memory"]["label"] == "banana"
            assert status["armed"] is False
            assert status["command"]["forward_mps"] == 0.0
            assert any(move[2] > 0.0 for move in sport.moves)
            assert sport.stretch_calls == 1
            assert sport.hello_calls == 0
            assert sport.standdown_calls == 1
            assert sport.stand_up_calls == 1
            assert all(move[0] == 0.0 and move[1] == 0.0 for move in sport.moves)
            assert all(move[2] != 0.20 for move in avoidance.moves)
            assert any(move[0] > 0.0 for move in avoidance.moves)
            assert any(abs(move[2]) > 0.0 for move in avoidance.moves)
            assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_pear_mission_hands_live_near_bbox_to_pointing_policy() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        pointing = FakePointingPolicy()
        pose = MotionCoupledPose(avoidance, sport)
        runtime = CollieRuntime(
            camera=NearBananaCamera(),
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                    forward_mps=0.08,
                    forward_budget_s=0.5,
                )
            ),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearPearDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            heading_provider=pose,
            pointing=pointing,  # type: ignore[arg-type]
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                arrival_hello_enabled=True,
                arrival_pointing_enabled=True,
                arrival_pointing_label="pear",
                arrival_pointing_timeout_s=2.0,
                return_home_enabled=True,
                return_arrival_tolerance_m=0.02,
                return_heading_tolerance_rad=0.10,
                return_heading_gate_rad=0.55,
                return_forward_mps=0.08,
                return_yaw_gain=1.2,
                return_timeout_s=3.0,
                return_stall_timeout_s=0.75,
                return_stall_min_progress_m=0.005,
                capture_timeout_s=0.6,
                match_confirmations_required=2,
                approach_misses_allowed=2,
                near_confirmations_required=4,
                turn_angle_rad=0.65,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.5,
                search_rate_rps=0.10,
                search_sweep_rad=2.5,
                search_timeout_s=1.5,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("pear", (640, 645))
            await runtime.start_demo(DEMO_CONFIRMATION)

            for _ in range(300):
                status = await runtime.status()
                if status["mission"]["phase"] in {"waiting_for_go", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            assert status["mission"]["phase"] == "waiting_for_go", repr(
                status["mission"]
            )

            await runtime.approve_demo_go(DEMO_GO_CONFIRMATION)
            for _ in range(600):
                status = await runtime.status()
                if status["mission"]["phase"] in {"success", "aborted"}:
                    break
                await asyncio.sleep(0.01)

            assert status["mission"]["phase"] == "success", repr(
                (status["mission"], status["pointing"])
            )
            assert status["mission"]["arrival_pointing_status"] == "complete"
            assert status["mission"]["arrival_pointing_target"] == "pear"
            assert status["mission"]["contact_status"] == "unverified"
            assert (
                status["mission"]["arrival_hello_status"]
                == "replaced_by_pointing_policy"
            )
            assert status["mission"]["final_approach_status"] == "not_requested"
            assert pointing.started == ("pear", 0.2)
            assert sport.standdown_calls == 1
            assert sport.stand_up_calls == 1
            assert sport.hello_calls == 0
            assert any(move[0] > 0.0 for move in avoidance.moves)
            assert status["mission"]["return_home_status"] == "complete"
            assert status["armed"] is False
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_voice_mission_sets_class_releases_go_and_returns_home() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        pose = OutwardStallingPose(avoidance, sport)
        runtime = CollieRuntime(
            camera=NearBananaCamera(),
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                    forward_mps=0.08,
                    forward_budget_s=0.08,
                )
            ),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaThenLostDetector(avoidance),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            heading_provider=pose,
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                match_stretch_enabled=True,
                match_stretch_settle_s=0.0,
                match_reacquire_timeout_s=1.0,
                arrival_hello_enabled=True,
                arrival_hello_settle_s=0.0,
                arrival_rest_enabled=True,
                arrival_rest_duration_s=0.01,
                return_home_enabled=True,
                return_arrival_tolerance_m=0.02,
                return_heading_tolerance_rad=0.10,
                return_heading_gate_rad=0.55,
                return_forward_mps=0.08,
                return_yaw_gain=1.2,
                return_timeout_s=3.0,
                return_stall_timeout_s=0.75,
                return_stall_min_progress_m=0.005,
                final_approach_stall_timeout_s=0.10,
                final_approach_stall_min_progress_m=0.001,
                match_confirmations_required=2,
                approach_misses_allowed=2,
                turn_angle_rad=0.65,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.5,
                search_rate_rps=0.10,
                search_sweep_rad=2.5,
                search_timeout_s=1.5,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)

            started = await runtime.start_voice_mission(
                "banana",
                "Find the banana",
                VOICE_MISSION_CONFIRMATION,
            )
            assert started["memory"]["label"] == "banana"
            assert started["memory"]["has_reference"] is False
            assert started["voice"]["last_heard"] == "Find the banana"
            assert started["voice"]["mission_active"] is True

            for _ in range(800):
                status = await runtime.status()
                if (
                    status["mission"]["phase"] == "aborted"
                    or (
                        status["mission"]["phase"] == "success"
                        and not status["voice"]["mission_active"]
                    )
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("voice mission did not finish")

            assert status["mission"]["phase"] == "success", repr(
                (status["mission"], status["last_error"], status["voice"])
            )
            assert status["voice"]["last_event"] == "voice_mission_complete"
            assert status["voice"]["mission_active"] is False
            assert status["mission"]["return_home_status"] == "complete"
            assert status["mission"]["initial_hello_status"] == "skipped_for_voice"
            assert status["mission"]["match_stretch_status"] == "skipped_for_voice"
            assert status["mission"]["arrival_rest_status"] == "complete"
            assert status["mission"]["final_approach_status"] == "partial"
            assert (
                status["mission"]["final_approach_commanded_distance_m"]
                > 0.0
            )
            assert (
                status["mission"]["near_target_bbox_height_ratio"]
                < runtime.mission_config.near_bbox_height_ratio
            )
            assert sport.hello_calls == 0
            assert sport.stretch_calls == 0
            assert sport.standdown_calls == 1
            assert sport.stand_up_calls == 1
            assert any(move[0] > 0.0 for move in avoidance.moves)
            assert avoidance.moves[-1] == (0.0, 0.0, 0.0)

            second = await runtime.start_voice_mission(
                "banana",
                "banana",
                VOICE_MISSION_CONFIRMATION,
            )
            assert second["voice"]["mission_active"] is True
            assert second["voice"]["last_heard"] == "banana"
            assert second["mission"]["initial_hello_status"] == "skipped_for_voice"
            await runtime.stop_demo()
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_measured_turn_uses_avoidance_motion_when_direct_turn_is_disabled() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=False,
                turn_angle_rad=0.35,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.0,
            ),
        )
        await runtime.start()
        try:
            await runtime._run_measured_turn()

            assert any(move == (0.0, 0.0, 0.20) for move in avoidance.moves)
            assert sport.moves == []
            assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
            assert runtime._mission.turn_progress_rad >= 0.30
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_memory_demo_accepts_a_different_banana_of_the_saved_class() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        camera = NearBananaCamera()
        runtime = CollieRuntime(
            camera=camera,
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                    forward_mps=0.08,
                    forward_budget_s=0.08,
                )
            ),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                capture_timeout_s=0.6,
                match_confirmations_required=2,
                approach_misses_allowed=2,
                turn_angle_rad=0.35,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.0,
                search_rate_rps=0.10,
                search_sweep_rad=0.50,
                search_timeout_s=1.0,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (640, 645))
            camera.show_different_banana = True
            await runtime.start_demo(DEMO_CONFIRMATION)

            for _ in range(200):
                status = await runtime.status()
                if status["mission"]["phase"] in {"waiting_for_go", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("class-based mission did not lock")

            assert status["mission"]["phase"] == "waiting_for_go"
            assert status["mission"]["target_policy"] == "same_class"
            assert (
                status["mission"]["last_match"]["reason"]
                == "target_class_detected"
            )
            assert not any(move[0] > 0.0 for move in avoidance.moves)
            assert status["armed"] is False
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_memory_demo_reacquires_same_class_after_stretch() -> None:
    async def scenario() -> None:
        camera = NearBananaCamera()
        avoidance = FakeAvoidance()
        sport = CallbackStretchSport(
            lambda: setattr(camera, "show_different_banana", True)
        )
        runtime = CollieRuntime(
            camera=camera,
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                    forward_mps=0.08,
                    forward_budget_s=0.08,
                )
            ),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                match_stretch_enabled=True,
                match_stretch_settle_s=0.0,
                match_reacquire_timeout_s=0.4,
                capture_timeout_s=0.6,
                match_confirmations_required=2,
                approach_misses_allowed=2,
                turn_angle_rad=0.65,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.5,
                search_rate_rps=0.10,
                search_sweep_rad=2.5,
                search_timeout_s=1.5,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (640, 645))
            await runtime.start_demo(DEMO_CONFIRMATION)

            for _ in range(300):
                status = await runtime.status()
                if status["mission"]["phase"] in {"waiting_for_go", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("post-stretch class reacquisition did not finish")

            assert status["mission"]["phase"] == "waiting_for_go"
            assert status["mission"]["target_policy"] == "same_class"
            assert status["mission"]["last_match"]["reason"] == "target_class_detected"
            assert status["mission"]["match_stretch_status"] == "complete"
            assert sport.stretch_calls == 1
            assert not any(move[0] > 0.0 for move in avoidance.moves)
            assert status["armed"] is False
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_memory_demo_keeps_optional_stretch_failure_nonfatal() -> None:
    async def scenario() -> None:
        camera = NearBananaCamera()
        avoidance = FakeAvoidance()
        sport = FailingStretchSport()
        runtime = CollieRuntime(
            camera=camera,
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                )
            ),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                match_stretch_enabled=True,
                match_stretch_settle_s=0.0,
                match_reacquire_timeout_s=0.5,
                capture_timeout_s=0.6,
                match_confirmations_required=2,
                turn_angle_rad=0.35,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.0,
                search_rate_rps=0.10,
                search_sweep_rad=1.0,
                search_timeout_s=1.0,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (640, 645))
            await runtime.start_demo(DEMO_CONFIRMATION)

            for _ in range(300):
                status = await runtime.status()
                if status["mission"]["phase"] in {"waiting_for_go", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("mission did not recover from Stretch failure")

            assert status["mission"]["phase"] == "waiting_for_go", repr(
                status["mission"]
            )
            assert status["mission"]["match_stretch_status"] == "failed"
            assert "3104" in status["mission"]["match_stretch_error"]
            assert status["mission"]["can_go"] is True
            assert status["armed"] is False
            assert not any(move[0] > 0.0 for move in avoidance.moves)
            assert sport.stretch_calls == 1
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_search_sweep_accumulates_past_heading_wrap() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        avoidance = FakeAvoidance()
        sport = FakeSport()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=detector,
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=False,
                capture_timeout_s=0.6,
                search_rate_rps=0.20,
                search_sweep_rad=4.5,
                search_timeout_s=1.5,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (350, 240))
            detector.visible = False

            try:
                await runtime._search_for_memory()
            except RuntimeCommandError as exc:
                assert str(exc) == "saved fruit class not found in search sweep"
            else:
                raise AssertionError("bounded sweep did not stop after crossing wrap")

            status = await runtime.status()
            assert status["mission"]["search_progress_deg"] >= math.degrees(4.5)
            assert status["armed"] is True
            await runtime.stop("test_complete")
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_detector_only_tracking_stays_fresh_at_jetson_cadence() -> None:
    async def scenario() -> None:
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=3)
            ),
            motion=UnitreeMotionAdapter(FakeSport(), FakeAvoidance()),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=JetsonSpeedProduceDetector(),
            loop_hz=8.0,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("detector never returned a frame")

            await runtime.select_target("banana", (350, 240))
            await asyncio.sleep(0.5)
            statuses = []
            for _ in range(60):
                statuses.append(await runtime.status())
                await asyncio.sleep(0.02)

            assert all(
                item["produce"]["tracker"]["mode"] == "yolo"
                for item in statuses
            )
            assert all(
                item["selected_target"]["confidence"] is not None
                for item in statuses
            )
            assert max(item["selected_target_age_s"] for item in statuses) < 0.35
            assert all(item["follow_readiness"] == "ready" for item in statuses)
            assert all(item["can_follow"] is True for item in statuses)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_detector_only_tracking_accepts_fast_same_class_motion() -> None:
    async def scenario() -> None:
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(ApproachConfig(stable_frames_required=1)),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=FastMovingProduceDetector(),
            loop_hz=30.0,
            maximum_produce_age_s=0.2,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.select_target("banana", (380, 240))
            await asyncio.sleep(0.45)
            status = await runtime.status()

            assert status["selected_target_name"] == "banana"
            assert status["selected_target"] is not None
            assert status["produce"]["tracker"]["revalidation_failures"] == 0
            assert status["follow_readiness"] == "ready"
        finally:
            await runtime.close()

    asyncio.run(scenario())
