"""Exclusive, watchdog-protected Unitree motion boundary.

Adapted from go2-follow-clean. Translational commands only use Unitree's
factory ObstaclesAvoidClient. A separately armed, yaw-only lease may use
SportClient.Move for an in-place turn; that mode cannot accept translation and
shares the same watchdog and independent StopMove brake.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import secrets
import time
from typing import Any, Protocol

from .types import VelocityCommand


class MotionError(RuntimeError):
    pass


class MotionNotReady(MotionError):
    pass


class LeaseMismatch(MotionError):
    pass


class SportClientProtocol(Protocol):
    def SetTimeout(self, timeout_s: float) -> Any: ...
    def Init(self) -> Any: ...
    def BalanceStand(self) -> int: ...
    def StandDown(self) -> int: ...
    def Hello(self) -> int: ...
    def Stretch(self) -> int: ...
    def Move(self, vx: float, vy: float, vyaw: float) -> int: ...
    def StopMove(self) -> int: ...


class AvoidanceClientProtocol(Protocol):
    def SetTimeout(self, timeout_s: float) -> Any: ...
    def Init(self) -> Any: ...
    def SwitchSet(self, enabled: bool) -> int: ...
    def SwitchGet(self) -> tuple[int, bool]: ...
    def UseRemoteCommandFromApi(self, enabled: bool) -> int: ...
    def Move(self, vx: float, vy: float, vyaw: float) -> int: ...


@dataclass(frozen=True, slots=True)
class MotionConfig:
    maximum_forward_mps: float = 0.08
    maximum_yaw_rps: float = 0.25
    command_watchdog_s: float = 0.35
    rpc_timeout_s: float = 0.75
    # Unitree's visible stock skills can take several seconds before their RPC
    # returns. Keep both the SDK-side and our asyncio-side timeouts above the
    # longest stage gesture so a completed animation is not reported as a
    # transport failure.
    skill_timeout_s: float = 12.0
    client_timeout_s: float = 12.0
    avoidance_verify_interval_s: float = 0.30
    remote_api_settle_s: float = 0.0


class UnitreeMotionAdapter:
    def __init__(
        self,
        sport: SportClientProtocol,
        avoidance: AvoidanceClientProtocol,
        config: MotionConfig | None = None,
    ) -> None:
        self.sport = sport
        self.avoidance = avoidance
        self.config = config or MotionConfig()
        self._lock = asyncio.Lock()
        self._rpc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collie-sdk")
        self._stop_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collie-stop")
        self._initialized = False
        self._fault: str | None = None
        self._lease: str | None = None
        self._mode: str | None = None
        self._avoidance_enabled = False
        self._remote_api_enabled = False
        self._last_verify_at: float | None = None
        self._last_command = VelocityCommand(reason="disarmed")
        self._watchdog: asyncio.Task[None] | None = None
        self._watchdog_generation = 0

    @property
    def armed(self) -> bool:
        if not self._initialized or self._fault is not None or not self._lease:
            return False
        if self._mode == "avoidance":
            return self._avoidance_enabled and self._remote_api_enabled
        return self._mode == "direct_yaw"

    def status(self) -> dict[str, object]:
        return {
            "initialized": self._initialized,
            "armed": self.armed,
            "mode": self._mode,
            "fault": self._fault,
            "avoidance_required": True,
            "avoidance_enabled": self._avoidance_enabled,
            "remote_api_enabled": self._remote_api_enabled,
            "watchdog_s": self.config.command_watchdog_s,
            "limits": {
                "forward_mps": self.config.maximum_forward_mps,
                "yaw_rps": self.config.maximum_yaw_rps,
                "lateral_mps": 0.0,
            },
            "last_command": self._last_command.to_dict(),
        }

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            try:
                await self._call(self.sport.SetTimeout, self.config.client_timeout_s)
                await self._call(self.sport.Init)
                await self._call(self.avoidance.SetTimeout, self.config.client_timeout_s)
                await self._call(self.avoidance.Init)
            except Exception as exc:
                self._fault = f"initialization failed: {exc}"
                raise MotionNotReady(self._fault) from exc
            self._initialized = True

    async def arm(self) -> str:
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            try:
                await self._success(self.avoidance.SwitchSet, True)
                response = await self._call(self.avoidance.SwitchGet)
                if response != (0, True):
                    raise MotionError(f"avoidance not confirmed: {response!r}")
                self._avoidance_enabled = True
                self._last_verify_at = time.monotonic()
                await self._success(self.avoidance.UseRemoteCommandFromApi, True)
                self._remote_api_enabled = True
                if self.config.remote_api_settle_s:
                    await asyncio.sleep(self.config.remote_api_settle_s)
                await self._success(self.avoidance.Move, 0.0, 0.0, 0.0)
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"arm failed: {exc}") from exc
            self._lease = secrets.token_urlsafe(32)
            self._mode = "avoidance"
            self._last_command = VelocityCommand(reason="armed_zero")
            return self._lease

    async def arm_direct_yaw(self) -> str:
        """Acquire an exclusive SportClient lease that permits yaw only."""

        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            try:
                # Ensure the factory-avoidance API is not concurrently holding
                # the motion channel before handing yaw control to SportClient.
                await self._success(self.avoidance.UseRemoteCommandFromApi, False)
                await self._success(self.avoidance.SwitchSet, False)
                await self._idle_stop()
                # Unitree's Go2 high-level example issues Move directly;
                # BalanceStand is a separate posture action, not a prerequisite
                # for velocity control. Calling it here can be rejected with
                # -1 when Woof is already standing, which must not prevent the
                # guarded yaw command from being attempted.
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"direct yaw arm failed: {exc}") from exc
            self._avoidance_enabled = False
            self._remote_api_enabled = False
            self._last_verify_at = None
            self._lease = secrets.token_urlsafe(32)
            self._mode = "direct_yaw"
            self._last_command = VelocityCommand(reason="direct_yaw_armed_zero")
            return self._lease

    async def perform_hello(self) -> None:
        """Run the stock paw-forward gesture while locomotion is disarmed."""

        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            self._cancel_watchdog()
            try:
                # Explicitly remove both locomotion owners before invoking a
                # high-level SportClient skill. The lock prevents a turn or
                # approach lease from being acquired until Hello returns.
                await self._success(self.avoidance.UseRemoteCommandFromApi, False)
                await self._success(self.avoidance.SwitchSet, False)
                await self._idle_stop()
                await self._success(
                    self.sport.Hello,
                    timeout_s=self.config.skill_timeout_s,
                )
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"hello gesture failed: {exc}") from exc
            self._avoidance_enabled = False
            self._remote_api_enabled = False
            self._last_verify_at = None
            self._last_command = VelocityCommand(reason="hello_gesture_complete")

    async def perform_standdown(self, *, settle_s: float = 1.0) -> None:
        """Put Woof in the measured low StandDown pose with no motion lease."""

        settle_s = float(settle_s)
        if not math.isfinite(settle_s) or settle_s < 0.0:
            raise ValueError("StandDown settle time must be finite and non-negative")
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            self._cancel_watchdog()
            try:
                await self._success(self.avoidance.UseRemoteCommandFromApi, False)
                await self._success(self.avoidance.SwitchSet, False)
                await self._idle_stop()
                await self._success(
                    self.sport.StandDown,
                    timeout_s=self.config.skill_timeout_s,
                )
                # The RPC can return before LowState shows the completed
                # posture. The runner independently verifies pose/stillness;
                # this pause prevents an eager stage click from racing it.
                if settle_s:
                    await asyncio.sleep(settle_s)
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"StandDown failed: {exc}") from exc
            self._avoidance_enabled = False
            self._remote_api_enabled = False
            self._last_verify_at = None
            self._last_command = VelocityCommand(
                reason="pointing_standdown_complete"
            )

    async def perform_stretch(self, *, settle_s: float = 0.0) -> None:
        """Run the stock stretch once while locomotion remains disarmed."""

        settle_s = float(settle_s)
        if not math.isfinite(settle_s) or settle_s < 0.0:
            raise ValueError("stretch settle time must be finite and non-negative")
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            self._cancel_watchdog()
            try:
                # A high-level skill and a locomotion owner must never share
                # the sport channel. Keep the exclusive lock through the
                # visible animation and recovery stand.
                await self._success(self.avoidance.UseRemoteCommandFromApi, False)
                await self._success(self.avoidance.SwitchSet, False)
                await self._idle_stop()
                await self._success(
                    self.sport.Stretch,
                    timeout_s=self.config.skill_timeout_s,
                )
                if settle_s:
                    await asyncio.sleep(settle_s)
                await self._idle_stop()
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"stretch gesture failed: {exc}") from exc
            self._avoidance_enabled = False
            self._remote_api_enabled = False
            self._last_verify_at = None
            self._last_command = VelocityCommand(reason="stretch_gesture_complete")

    async def send(self, lease: str, command: VelocityCommand) -> VelocityCommand:
        forward = self._bounded_forward(command.forward_mps)
        yaw = self._bounded_yaw(command.yaw_rps)
        async with self._lock:
            self._require_owner(lease, required_mode="avoidance")
            # A fresh command has arrived. Do not let the previous command's
            # watchdog race the in-flight avoidance verification/Move RPC.
            # Each RPC remains bounded by rpc_timeout_s and faults to StopMove.
            self._cancel_watchdog()
            try:
                if forward != 0.0 or yaw != 0.0:
                    await self._verify_avoidance_if_due()
                await self._success(self.avoidance.Move, forward, 0.0, yaw)
            except Exception as exc:
                self._fault = f"velocity command failed: {exc}"
                await self._release_locked(use_stop=True)
                raise MotionNotReady(self._fault) from exc
            self._last_command = VelocityCommand(forward, yaw, command.reason)
            if forward != 0.0 or yaw != 0.0:
                self._arm_watchdog()
            else:
                self._cancel_watchdog()
            return self._last_command

    async def send_direct_yaw(
        self, lease: str, yaw_rps: float, reason: str = "direct_yaw"
    ) -> VelocityCommand:
        """Send a watchdog-protected SportClient turn with zero translation."""

        yaw = self._bounded_yaw(yaw_rps)
        async with self._lock:
            self._require_owner(lease, required_mode="direct_yaw")
            self._cancel_watchdog()
            try:
                await self._success(self.sport.Move, 0.0, 0.0, yaw)
            except Exception as exc:
                self._fault = f"direct yaw command failed: {exc}"
                await self._release_locked(use_stop=True)
                raise MotionNotReady(self._fault) from exc
            self._last_command = VelocityCommand(0.0, yaw, reason)
            if yaw != 0.0:
                self._arm_watchdog()
            return self._last_command

    async def release(self, lease: str) -> None:
        async with self._lock:
            self._require_owner(lease)
            errors = await self._release_locked(use_stop=True)
            if errors:
                raise MotionError("; ".join(errors))

    async def emergency_stop(self) -> list[str]:
        self._cancel_watchdog()
        errors: list[str] = []
        was_active = bool(
            self._lease is not None
            or self._mode is not None
            or self._last_command.forward_mps != 0.0
            or self._last_command.yaw_rps != 0.0
        )
        if self._initialized:
            try:
                result = await self._call_stop(self.sport.StopMove)
                # Go2 SportClient returns -1 when StopMove is repeated while
                # the robot is already idle. Start Over must be idempotent in
                # that state, but the same response while we own an active
                # lease remains a hard fault because motion may not have
                # stopped.
                already_stopped = result == -1 and not was_active
                if result != 0 and not already_stopped:
                    raise MotionError(f"StopMove returned {result!r}")
            except Exception as exc:
                errors.append(f"StopMove: {exc}")
                if self._fault is None:
                    self._fault = f"StopMove failed: {exc}"
        async with self._lock:
            errors.extend(await self._release_locked(use_stop=False))
        return errors

    async def close(self) -> list[str]:
        errors = await self.emergency_stop()
        self._rpc_executor.shutdown(wait=False, cancel_futures=False)
        self._stop_executor.shutdown(wait=False, cancel_futures=False)
        return errors

    async def _verify_avoidance_if_due(self) -> None:
        if (
            self._last_verify_at is not None
            and time.monotonic() - self._last_verify_at
            < self.config.avoidance_verify_interval_s
        ):
            return
        response = await self._call(self.avoidance.SwitchGet)
        if response != (0, True):
            self._avoidance_enabled = False
            raise MotionError(f"avoidance switched off: {response!r}")
        self._last_verify_at = time.monotonic()

    async def _release_locked(self, *, use_stop: bool) -> list[str]:
        self._cancel_watchdog()
        errors: list[str] = []
        calls: list[tuple[str, Any, tuple[Any, ...], bool]] = []
        if self._initialized and use_stop:
            calls.append(("StopMove", self.sport.StopMove, (), True))
        if self._initialized:
            calls.extend(
                [
                    ("avoidance zero", self.avoidance.Move, (0.0, 0.0, 0.0), False),
                    ("remote API disable", self.avoidance.UseRemoteCommandFromApi, (False,), False),
                    ("avoidance disable", self.avoidance.SwitchSet, (False,), False),
                ]
            )
        for label, method, args, emergency in calls:
            try:
                result = await (self._call_stop(method, *args) if emergency else self._call(method, *args))
                if result != 0:
                    raise MotionError(f"returned {result!r}")
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        self._lease = None
        self._mode = None
        self._avoidance_enabled = False
        self._remote_api_enabled = False
        self._last_verify_at = None
        self._last_command = VelocityCommand(reason="released")
        return errors

    def _require_ready(self) -> None:
        if not self._initialized:
            raise MotionNotReady("motion adapter is not initialized")
        if self._fault is not None:
            raise MotionNotReady(self._fault)

    def _require_owner(self, lease: str, *, required_mode: str | None = None) -> None:
        self._require_ready()
        if not self.armed:
            raise MotionNotReady("motion is disarmed")
        if not isinstance(lease, str) or self._lease is None or not secrets.compare_digest(lease, self._lease):
            raise LeaseMismatch("motion lease is stale or does not match")
        if required_mode is not None and self._mode != required_mode:
            raise MotionNotReady(
                f"motion lease mode is {self._mode!r}, expected {required_mode!r}"
            )

    def _arm_watchdog(self) -> None:
        self._cancel_watchdog()
        generation = self._watchdog_generation

        async def expire() -> None:
            try:
                await asyncio.sleep(self.config.command_watchdog_s)
                if generation == self._watchdog_generation:
                    await self.emergency_stop()
            except asyncio.CancelledError:
                return

        self._watchdog = asyncio.create_task(expire())

    def _cancel_watchdog(self) -> None:
        self._watchdog_generation += 1
        task, self._watchdog = self._watchdog, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _success(
        self,
        method: Any,
        *args: Any,
        timeout_s: float | None = None,
    ) -> None:
        result = await self._call(method, *args, timeout_s=timeout_s)
        if result != 0:
            raise MotionError(f"{method.__name__} returned {result!r}")

    async def _idle_stop(self) -> None:
        """Apply an idempotent brake when no motion lease exists.

        Woof's Sport service returns ``-1`` when StopMove is repeated while
        already idle. This helper is deliberately restricted to call sites
        that hold the adapter lock and have confirmed there is no active lease.
        Active-motion emergency stops retain their stricter result handling.
        """

        if self._lease is not None or self._mode is not None:
            raise MotionError("idle StopMove requested while motion is active")
        result = await self._call_stop(self.sport.StopMove)
        if result not in (0, -1):
            raise MotionError(f"StopMove returned {result!r}")

    async def _call(
        self,
        method: Any,
        *args: Any,
        timeout_s: float | None = None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._rpc_executor, method, *args)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                self.config.rpc_timeout_s if timeout_s is None else timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise MotionNotReady(f"{method.__name__} timed out") from exc

    async def _call_stop(self, method: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._stop_executor, method, *args)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.config.rpc_timeout_s)
        except asyncio.TimeoutError as exc:
            raise MotionNotReady(f"{method.__name__} timed out") from exc

    def _bounded_forward(self, value: float) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("forward speed must be finite and non-negative")
        return min(number, self.config.maximum_forward_mps)

    def _bounded_yaw(self, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("yaw speed must be finite")
        return max(-self.config.maximum_yaw_rps, min(self.config.maximum_yaw_rps, number))


def initialize_dds(network_interface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if network_interface:
        ChannelFactoryInitialize(0, network_interface)
    else:
        ChannelFactoryInitialize(0)


def create_motion(config: MotionConfig | None = None) -> UnitreeMotionAdapter:
    from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    return UnitreeMotionAdapter(SportClient(), ObstaclesAvoidClient(), config)
