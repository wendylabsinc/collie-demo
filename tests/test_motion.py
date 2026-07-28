from __future__ import annotations

import asyncio
import time

from collie_demo.motion import MotionConfig, MotionNotReady, UnitreeMotionAdapter
from collie_demo.types import VelocityCommand


class FakeSport:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.stop_result = 0
        self.balance_stand_calls = 0
        self.hello_calls = 0
        self.stretch_calls = 0
        self.standdown_calls = 0
        self.timeout_s: float | None = None
        self.moves: list[tuple[float, float, float]] = []
        self.current_move = (0.0, 0.0, 0.0)

    def SetTimeout(self, value: float) -> None: self.timeout_s = value
    def Init(self) -> None: pass
    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.current_move = (vx, vy, vyaw)
        self.moves.append(self.current_move)
        return 0
    def BalanceStand(self) -> int:
        self.balance_stand_calls += 1
        return 0
    def Hello(self) -> int:
        self.hello_calls += 1
        return 0
    def StandDown(self) -> int:
        self.standdown_calls += 1
        return 0
    def Stretch(self) -> int:
        self.stretch_calls += 1
        return 0
    def StopMove(self) -> int:
        self.stop_calls += 1
        self.current_move = (0.0, 0.0, 0.0)
        return self.stop_result


class FakeAvoidance:
    def __init__(self) -> None:
        self.enabled = False
        self.remote = False
        self.timeout_s: float | None = None
        self.moves: list[tuple[float, float, float]] = []

    def SetTimeout(self, value: float) -> None: self.timeout_s = value
    def Init(self) -> None: pass
    def SwitchSet(self, enabled: bool) -> int:
        self.enabled = enabled
        return 0
    def SwitchGet(self) -> tuple[int, bool]: return (0, self.enabled)
    def UseRemoteCommandFromApi(self, enabled: bool) -> int:
        self.remote = enabled
        return 0
    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.moves.append((vx, vy, vyaw))
        return 0


class DelayedAvoidance(FakeAvoidance):
    def __init__(self) -> None:
        super().__init__()
        self.move_delay_s = 0.0

    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        time.sleep(self.move_delay_s)
        return super().Move(vx, vy, vyaw)


def test_motion_uses_avoidance_and_clamps_forward_speed() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        assert sport.timeout_s == 12.0
        assert avoidance.timeout_s == 12.0
        lease = await motion.arm()
        sent = await motion.send(lease, VelocityCommand(1.0, 0.0, "test"))
        assert sent.forward_mps == 0.08
        assert avoidance.moves[-1] == (0.08, 0.0, 0.0)
        await motion.close()

    asyncio.run(scenario())


def test_stale_command_watchdog_brakes_and_revokes_lease() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(
            sport,
            avoidance,
            MotionConfig(command_watchdog_s=0.03),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.send(lease, VelocityCommand(0.05, 0.0, "test"))
        await asyncio.sleep(0.08)
        assert not motion.armed
        assert sport.stop_calls >= 1
        assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        await motion.close()

    asyncio.run(scenario())


def test_idle_emergency_stop_accepts_sdk_already_stopped_response() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        sport.stop_result = -1

        errors = await motion.emergency_stop()

        assert errors == []
        assert motion.status()["fault"] is None
        assert motion.armed is False
        await motion.close()

    asyncio.run(scenario())


def test_active_emergency_stop_faults_if_stopmove_returns_minus_one() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        lease = await motion.arm()
        await motion.send(lease, VelocityCommand(0.05, 0.0, "moving"))
        sport.stop_result = -1

        errors = await motion.emergency_stop()

        assert errors
        assert "StopMove returned -1" in errors[0]
        assert motion.status()["fault"] is not None
        assert motion.armed is False
        await motion.close()

    asyncio.run(scenario())


def test_fresh_inflight_command_does_not_race_previous_watchdog() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), DelayedAvoidance()
        motion = UnitreeMotionAdapter(
            sport,
            avoidance,
            MotionConfig(command_watchdog_s=0.03, rpc_timeout_s=0.2),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.send(lease, VelocityCommand(0.05, 0.0, "first"))
        avoidance.move_delay_s = 0.06
        await motion.send(lease, VelocityCommand(0.05, 0.0, "fresh"))
        assert motion.armed
        assert sport.stop_calls == 0
        await motion.close()

    asyncio.run(scenario())


def test_direct_yaw_bypasses_avoidance_and_hard_locks_translation() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        lease = await motion.arm_direct_yaw()

        sent = await motion.send_direct_yaw(lease, 1.0, "measured_turn")

        assert sent.forward_mps == 0.0
        assert sent.yaw_rps == motion.config.maximum_yaw_rps
        assert sport.moves == [(0.0, 0.0, motion.config.maximum_yaw_rps)]
        assert sport.balance_stand_calls == 0
        assert avoidance.moves == []
        assert motion.status()["mode"] == "direct_yaw"
        try:
            await motion.send(lease, VelocityCommand(0.08, 0.0, "forbidden"))
        except MotionNotReady:
            pass
        else:
            raise AssertionError("direct-yaw lease accepted translation")
        assert avoidance.moves == []
        await motion.close()

    asyncio.run(scenario())


def test_direct_yaw_accepts_sdk_already_stopped_response_before_arming() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        sport.stop_result = -1

        lease = await motion.arm_direct_yaw()

        assert lease
        assert motion.armed is True
        assert motion.status()["mode"] == "direct_yaw"
        assert sport.stop_calls == 1
        assert sport.balance_stand_calls == 0
        assert avoidance.enabled is False
        assert avoidance.remote is False
        sport.stop_result = 0
        await motion.close()

    asyncio.run(scenario())


def test_direct_yaw_rejects_unexpected_stopmove_failure() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        sport.stop_result = -2

        try:
            await motion.arm_direct_yaw()
        except MotionNotReady as exc:
            assert "StopMove returned -2" in str(exc)
        else:
            raise AssertionError("direct yaw ignored unexpected StopMove failure")
        assert motion.armed is False
        assert sport.balance_stand_calls == 0
        assert avoidance.enabled is False
        assert avoidance.remote is False
        sport.stop_result = 0
        await motion.close()

    asyncio.run(scenario())


def test_hello_uses_stock_skill_while_motion_remains_disarmed() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()

        await motion.perform_hello()

        assert sport.hello_calls == 1
        assert sport.stop_calls == 1
        assert sport.balance_stand_calls == 0
        assert avoidance.enabled is False
        assert avoidance.remote is False
        assert motion.armed is False
        assert motion.status()["last_command"]["reason"] == "hello_gesture_complete"
        await motion.close()

    asyncio.run(scenario())


def test_standdown_prepares_the_pointing_pose_while_motion_is_disarmed() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()

        await motion.perform_standdown(settle_s=0.0)

        assert sport.standdown_calls == 1
        assert sport.stop_calls == 1
        assert avoidance.enabled is False
        assert avoidance.remote is False
        assert motion.armed is False
        assert (
            motion.status()["last_command"]["reason"]
            == "pointing_standdown_complete"
        )
        await motion.close()

    asyncio.run(scenario())


def test_balance_stand_recovers_posture_before_return_home() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()

        await motion.perform_balance_stand(settle_s=0.0)

        assert sport.balance_stand_calls == 1
        assert sport.stop_calls == 1
        assert avoidance.enabled is False
        assert avoidance.remote is False
        assert motion.armed is False
        assert (
            motion.status()["last_command"]["reason"]
            == "pointing_balance_stand_complete"
        )
        await motion.close()

    asyncio.run(scenario())


def test_hello_accepts_sdk_already_stopped_response() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        sport.stop_result = -1

        await motion.perform_hello()

        assert sport.hello_calls == 1
        assert motion.status()["fault"] is None
        assert motion.armed is False
        sport.stop_result = 0
        await motion.close()

    asyncio.run(scenario())


def test_hello_refuses_to_overlap_an_active_motion_lease() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        await motion.arm()

        try:
            await motion.perform_hello()
        except MotionNotReady as exc:
            assert "lease already active" in str(exc)
        else:
            raise AssertionError("Hello overlapped an active motion lease")
        assert sport.hello_calls == 0
        await motion.close()

    asyncio.run(scenario())


def test_stretch_runs_once_then_recovers_while_motion_remains_disarmed() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()

        await motion.perform_stretch(settle_s=0.0)

        assert sport.stretch_calls == 1
        assert sport.stop_calls == 2
        assert sport.balance_stand_calls == 0
        assert avoidance.enabled is False
        assert avoidance.remote is False
        assert motion.armed is False
        assert motion.status()["last_command"]["reason"] == "stretch_gesture_complete"
        await motion.close()

    asyncio.run(scenario())


def test_stretch_refuses_to_overlap_an_active_motion_lease() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        await motion.arm()

        try:
            await motion.perform_stretch(settle_s=0.0)
        except MotionNotReady as exc:
            assert "lease already active" in str(exc)
        else:
            raise AssertionError("Stretch overlapped an active motion lease")
        assert sport.stretch_calls == 0
        await motion.close()

    asyncio.run(scenario())


def test_direct_yaw_watchdog_uses_stopmove_and_revokes_lease() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(
            sport,
            avoidance,
            MotionConfig(command_watchdog_s=0.03),
        )
        await motion.initialize()
        lease = await motion.arm_direct_yaw()
        await motion.send_direct_yaw(lease, 0.2)
        stop_calls_after_arm = sport.stop_calls

        await asyncio.sleep(0.08)

        assert not motion.armed
        assert sport.stop_calls > stop_calls_after_arm
        assert motion.status()["mode"] is None
        await motion.close()

    asyncio.run(scenario())
