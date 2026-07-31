#!/usr/bin/env python3
"""Run the guarded ``locked_point_v19`` standing-point policy on a Go2.

The runner requires an explicitly selected Collie target.  Its stage path
continuously publishes Woof's measured Sport-standing pose before releasing
``mcf``, then runs the 47-input/9-action balance policy while the front-right
leg follows its deterministic point schedule.  Every exit path restores
Unitree's ``mcf`` Sport controller; firmware that refuses a standing-to-standing
controller transfer uses a controlled StandDown/StandUp recovery.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import struct
from threading import Event, Lock, Thread
import time
from typing import Any, Sequence

import torch

from .pointing_contract import (
    ISAAC_GROUNDED_STAND_RAD,
    ISAAC_JOINT_ORDER,
    LOCKED_POINT_ACTION_SIZE,
    LOCKED_POINT_MAX_DELTA_RAD,
    LOCKED_POINT_MIN_DELTA_RAD,
    LOCKED_POINT_OBSERVATION_SIZE,
    WOOF_BUILTIN_STANDDOWN_RAD,
    build_locked_point_observation,
    guarded_policy_target,
    isaac_to_unitree,
    joint_limit_violations,
    locked_point_joint_targets,
    projected_gravity_wxyz,
    standdown_error_rad,
    unitree_to_isaac,
)
from .pointing_shadow import (
    BBoxProvider,
    EXPECTED_POLICY_SHA256,
    LiveState,
    sha256_file,
)
from .standing_pose import (
    LOCKED_POINT_RAMP_S,
    LOCKED_POINT_SETUP_S,
    locked_point_phase,
)


PUBLISH_RATE_HZ = 500.0
POLICY_RATE_HZ = 50.0
COMMAND_TRACKING_SLACK_RAD = 0.45
POLICY_TORQUE_GUARD_NM = 22.0
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--network-interface", default="enP8p1s0")
    parser.add_argument("--status-url", default="http://127.0.0.1:8096/api/status")
    parser.add_argument(
        "--target-label",
        required=True,
        choices=("apple", "banana", "pear"),
        help="the explicit Collie target lock that this run must retain",
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--max-rate-rad-s", type=float, default=0.60)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=5.0)
    parser.add_argument("--standup-seconds", type=float, default=3.0)
    parser.add_argument("--standup-kp", type=float, default=60.0)
    parser.add_argument("--standup-kd", type=float, default=5.0)
    parser.add_argument(
        "--direct-standing-handoff",
        action="store_true",
        help=(
            "continuously hold the measured Sport-standing pose during the "
            "controller handoff instead of starting from StandDown"
        ),
    )
    parser.add_argument(
        "--full-power",
        action="store_true",
        help="allow the full actor gain and the staged full-policy joint envelope",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/go2_pointing_guarded.json"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required to release Sport mode and publish low-level commands",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help='must be exactly "AREA IS CLEAR AND WOOF MAY MOVE"',
    )
    return parser.parse_args()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _maximum_abs(values: Sequence[float]) -> float:
    return max(abs(float(value)) for value in values)


def lowcmd_crc(message: Any) -> int:
    """Compute Unitree's Go2 LowCmd CRC without the optional native library."""

    values: list[int | float] = []
    values.extend(message.head)
    values.append(message.level_flag)
    values.append(message.frame_reserve)
    values.extend(message.sn)
    values.extend(message.version)
    values.append(message.bandwidth)
    for motor in message.motor_cmd:
        values.append(motor.mode)
        values.append(motor.q)
        values.append(motor.dq)
        values.append(motor.tau)
        values.append(motor.kp)
        values.append(motor.kd)
        values.extend(motor.reserve)
    values.append(message.bms_cmd.off)
    values.extend(message.bms_cmd.reserve)
    values.extend(message.wireless_remote)
    values.extend(message.led)
    values.extend(message.fan)
    values.append(message.gpio)
    values.append(message.reserve)
    values.append(message.crc)

    packed = struct.pack(
        "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I",
        *values,
    )
    words = [
        int.from_bytes(packed[offset : offset + 4], "little")
        for offset in range(0, len(packed) - 4, 4)
    ]
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for current in words:
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if current & bit:
                crc ^= polynomial
            bit >>= 1
    return crc


def _check_live_guards(
    *,
    low: dict[str, Any],
    now: float,
    start_q_isaac: Sequence[float],
    require_standdown: bool,
    minimum_delta_rad: Sequence[float] = LOCKED_POINT_MIN_DELTA_RAD,
    maximum_delta_rad: Sequence[float] = LOCKED_POINT_MAX_DELTA_RAD,
    maximum_joint_speed_rad_s: float = 1.0,
    maximum_estimated_torque: float = 20.0,
    maximum_roll_rad: float | None = 0.30,
    tracking_slack_rad: float = COMMAND_TRACKING_SLACK_RAD,
    observed_deviation: dict[str, float] | None = None,
) -> None:
    low_age_s = now - float(low["received_at"])
    if low_age_s > 0.05:
        raise RuntimeError(f"low-state timeout: {low_age_s:.3f}s")
    q_isaac = unitree_to_isaac(low["q"])
    dq_isaac = unitree_to_isaac(low["dq"])
    if not all(math.isfinite(value) for value in (*q_isaac, *dq_isaac)):
        raise RuntimeError("non-finite joint state")
    roll, pitch, _yaw = (float(value) for value in low["rpy"])
    if (
        (maximum_roll_rad is not None and abs(roll) > maximum_roll_rad)
        or abs(pitch) > 0.35
    ):
        raise RuntimeError(f"tilt guard: roll={roll:.3f}, pitch={pitch:.3f}")
    if _maximum_abs(dq_isaac) > maximum_joint_speed_rad_s:
        raise RuntimeError(f"joint-speed guard: {_maximum_abs(dq_isaac):.3f} rad/s")
    if _maximum_abs(low["tau_est"]) > maximum_estimated_torque:
        raise RuntimeError(f"estimated-torque guard: {_maximum_abs(low['tau_est']):.3f}")
    if require_standdown:
        rms, maximum = standdown_error_rad(q_isaac)
        if rms > 0.08 or maximum > 0.16:
            raise RuntimeError(
                f"not in StandDown pose: rms={rms:.3f}, max={maximum:.3f} rad"
            )
    for index, (value, start, minimum_delta, maximum_delta) in enumerate(
        zip(
            q_isaac,
            start_q_isaac,
            minimum_delta_rad,
            maximum_delta_rad,
            strict=True,
        )
    ):
        delta = float(value) - float(start)
        name = ISAAC_JOINT_ORDER[index]
        if observed_deviation is not None:
            outside = max(minimum_delta - delta, delta - maximum_delta, 0.0)
            if outside > observed_deviation.get(name, 0.0):
                observed_deviation[name] = outside
        if (
            delta < minimum_delta - tracking_slack_rad
            or delta > maximum_delta + tracking_slack_rad
        ):
            raise RuntimeError(
                f"joint envelope guard: {name} moved {delta:.3f} rad"
            )


class LowCommandPublisher:
    def __init__(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        self._message = unitree_go_msg_dds__LowCmd_()
        self._message.head[0] = 0xFE
        self._message.head[1] = 0xEF
        self._message.level_flag = 0xFF
        self._message.gpio = 0
        for motor in self._message.motor_cmd:
            motor.mode = 0x01
            motor.q = POS_STOP_F
            motor.kp = 0.0
            motor.dq = VEL_STOP_F
            motor.kd = 0.0
            motor.tau = 0.0
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()

    def write(self, target_unitree: Sequence[float], *, kp: float, kd: float) -> None:
        if len(target_unitree) != 12:
            raise ValueError("low-level target must contain 12 values")
        for index, target in enumerate(target_unitree):
            motor = self._message.motor_cmd[index]
            motor.mode = 0x01
            motor.q = float(target)
            motor.kp = float(kp)
            motor.dq = 0.0
            motor.kd = float(kd)
            motor.tau = 0.0
        self._message.crc = lowcmd_crc(self._message)
        self._publisher.Write(self._message)

    def close(self) -> None:
        self._publisher.Close()


class ContinuousJointHold:
    """Continuously publish one measured pose during a controller handoff."""

    def __init__(
        self,
        *,
        publisher: LowCommandPublisher,
        target_isaac: Sequence[float],
        kp: float,
        kd: float,
    ) -> None:
        if len(target_isaac) != 12:
            raise ValueError("continuous hold target must contain 12 joints")
        self._publisher = publisher
        self._lock = Lock()
        self._target_isaac = tuple(float(value) for value in target_isaac)
        self._kp = float(kp)
        self._kd = float(kd)
        self._stop = Event()
        self._started = Event()
        self._thread: Thread | None = None
        self._error: BaseException | None = None
        self.ticks = 0

    def set_target(self, target_isaac: Sequence[float]) -> None:
        if len(target_isaac) != 12:
            raise ValueError("continuous hold target must contain 12 joints")
        with self._lock:
            self._target_isaac = tuple(float(value) for value in target_isaac)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("continuous joint hold is already started")
        self._thread = Thread(
            target=self._run,
            name="continuous-lowcmd-hold",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(1.0):
            raise RuntimeError("continuous low-level publisher did not start")
        self.raise_if_failed()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("continuous low-level publisher did not stop")
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"continuous low-level publisher failed: {self._error}"
            ) from self._error

    def _run(self) -> None:
        deadline = time.perf_counter()
        period_s = 1.0 / PUBLISH_RATE_HZ
        try:
            while not self._stop.is_set():
                deadline += period_s
                with self._lock:
                    target_isaac = self._target_isaac
                    kp = self._kp
                    kd = self._kd
                self._publisher.write(
                    isaac_to_unitree(target_isaac),
                    kp=kp,
                    kd=kd,
                )
                self.ticks += 1
                self._started.set()
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
        except BaseException as exc:
            self._error = exc
            self._started.set()
            self._stop.set()


def _publish_for(
    *,
    publisher: LowCommandPublisher,
    live: LiveState,
    target_isaac: Sequence[float],
    start_q_isaac: Sequence[float],
    duration_s: float,
    kp: float,
    kd: float,
    require_standdown: bool = False,
    minimum_delta_rad: Sequence[float] = LOCKED_POINT_MIN_DELTA_RAD,
    maximum_delta_rad: Sequence[float] = LOCKED_POINT_MAX_DELTA_RAD,
    maximum_joint_speed_rad_s: float = 1.0,
    maximum_estimated_torque: float = 20.0,
    maximum_roll_rad: float | None = 0.30,
) -> None:
    period_s = 1.0 / PUBLISH_RATE_HZ
    deadline = time.perf_counter()
    ticks = int(round(duration_s * PUBLISH_RATE_HZ))
    target_unitree = isaac_to_unitree(target_isaac)
    for _ in range(ticks):
        deadline += period_s
        low, _sport = live.sample()
        _check_live_guards(
            low=low,
            now=time.monotonic(),
            start_q_isaac=start_q_isaac,
            require_standdown=require_standdown,
            minimum_delta_rad=minimum_delta_rad,
            maximum_delta_rad=maximum_delta_rad,
            maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
            maximum_estimated_torque=maximum_estimated_torque,
            maximum_roll_rad=maximum_roll_rad,
        )
        publisher.write(target_unitree, kp=kp, kd=kd)
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)


def ensure_standdown(sport: Any, live: LiveState, settle_s: float = 4.0) -> None:
    """Reach and verify StandDown before Sport-mode handoff."""

    low, _ = live.sample()
    rms, maximum = standdown_error_rad(unitree_to_isaac(low["q"]))
    if (
        rms <= 0.08
        and maximum <= 0.16
        and _maximum_abs(unitree_to_isaac(low["dq"])) <= 0.20
    ):
        print("[standing-point] already in StandDown pose", flush=True)
        return

    print(
        "[standing-point] commanding StandDown before low-level handoff "
        f"(rms={rms:.3f} rad)",
        flush=True,
    )
    code = sport.StandDown()
    if code != 0:
        raise RuntimeError(f"SportClient.StandDown() returned {code}")

    deadline = time.monotonic() + settle_s
    while time.monotonic() < deadline:
        time.sleep(0.1)
        low, _ = live.sample()
        rms, maximum = standdown_error_rad(unitree_to_isaac(low["q"]))
        if (
            rms <= 0.08
            and maximum <= 0.16
            and _maximum_abs(unitree_to_isaac(low["dq"])) <= 0.20
        ):
            print(
                f"[standing-point] StandDown verified, rms={rms:.3f} rad",
                flush=True,
            )
            return
    raise RuntimeError(
        f"StandDown did not settle: rms={rms:.3f}, max={maximum:.3f} rad"
    )


def release_sport_mode(switcher: Any, attempts: int = 12) -> None:
    """Release Sport mode and verify that its motion service is inactive."""

    for _ in range(attempts):
        code, mode = switcher.CheckMode()
        if code == 0 and (not mode or not mode.get("name")):
            return
        switcher.ReleaseMode()
        time.sleep(0.3)
    code, mode = switcher.CheckMode()
    raise RuntimeError(
        f"Sport mode still active after release: code={code}, mode={mode}"
    )


def _select_sport_mode(switcher: Any, attempts: int = 8) -> bool:
    """Request ``mcf`` and verify that Sport mode is actually active."""

    for _ in range(attempts):
        check_code, mode = switcher.CheckMode()
        if (
            check_code == 0
            and mode
            and mode.get("name") == "mcf"
        ):
            return True
        switcher.SelectMode("mcf")
        time.sleep(0.25)
    return False


def _validate_direct_standing_start(low: dict[str, Any]) -> tuple[float, ...]:
    """Validate a still, level Sport-standing pose for direct handoff."""

    age_s = time.monotonic() - float(low["received_at"])
    if age_s > 0.05:
        raise RuntimeError(f"low-state is stale: {age_s:.3f}s")
    q_isaac = unitree_to_isaac(low["q"])
    dq_isaac = unitree_to_isaac(low["dq"])
    violations = joint_limit_violations(q_isaac)
    if violations:
        raise RuntimeError(f"measured joint limit violation: {violations[0]}")
    maximum_speed = _maximum_abs(dq_isaac)
    if maximum_speed > 0.20:
        raise RuntimeError(
            "Woof is not still enough for direct handoff: "
            f"{maximum_speed:.3f} rad/s"
        )
    roll, pitch, _yaw = (float(value) for value in low["rpy"])
    if abs(roll) > 0.12 or abs(pitch) > 0.12:
        raise RuntimeError(
            f"Woof is not level enough: roll={roll:.3f}, pitch={pitch:.3f}"
        )
    standdown_rms, _standdown_max = standdown_error_rad(q_isaac)
    if standdown_rms < 0.40:
        raise RuntimeError("Woof is not in a standing posture")
    errors = tuple(
        actual - expected
        for actual, expected in zip(
            q_isaac,
            ISAAC_GROUNDED_STAND_RAD,
            strict=True,
        )
    )
    standing_rms = math.sqrt(
        sum(value * value for value in errors) / len(errors)
    )
    standing_max = _maximum_abs(errors)
    if standing_rms > 0.35 or standing_max > 0.65:
        raise RuntimeError(
            "Sport standing posture is too far from the policy stance: "
            f"rms={standing_rms:.3f}, max={standing_max:.3f}"
        )
    return q_isaac


def _monitor_continuous_target(
    *,
    controller: ContinuousJointHold,
    live: LiveState,
    target_isaac: Sequence[float],
    start_q_isaac: Sequence[float],
    duration_s: float,
    minimum_delta_rad: Sequence[float] = LOCKED_POINT_MIN_DELTA_RAD,
    maximum_delta_rad: Sequence[float] = LOCKED_POINT_MAX_DELTA_RAD,
    maximum_joint_speed_rad_s: float = 4.0,
    maximum_estimated_torque: float = POLICY_TORQUE_GUARD_NM,
    maximum_roll_rad: float | None = 0.30,
) -> None:
    """Set one threaded 500 Hz target while checking fresh low-state guards."""

    controller.set_target(target_isaac)
    period_s = 1.0 / PUBLISH_RATE_HZ
    deadline = time.perf_counter()
    ticks = max(1, int(round(duration_s * PUBLISH_RATE_HZ)))
    for _ in range(ticks):
        deadline += period_s
        controller.raise_if_failed()
        low, _sport = live.sample()
        _check_live_guards(
            low=low,
            now=time.monotonic(),
            start_q_isaac=start_q_isaac,
            require_standdown=False,
            minimum_delta_rad=minimum_delta_rad,
            maximum_delta_rad=maximum_delta_rad,
            maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
            maximum_estimated_torque=maximum_estimated_torque,
            maximum_roll_rad=maximum_roll_rad,
        )
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)


def _lower_continuous_to_standdown(
    *,
    controller: ContinuousJointHold,
    live: LiveState,
    start_q_isaac: Sequence[float],
    duration_s: float = 2.5,
) -> None:
    """Lower from a supported standing pose before closing the lowcmd writer."""

    relaxed_minimum = tuple(-2.5 for _ in range(12))
    relaxed_maximum = tuple(2.5 for _ in range(12))
    steps = max(1, int(round(duration_s * POLICY_RATE_HZ)))
    for step in range(1, steps + 1):
        alpha = step / steps
        blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
        target = tuple(
            start + (standdown - start) * blend
            for start, standdown in zip(
                start_q_isaac,
                WOOF_BUILTIN_STANDDOWN_RAD,
                strict=True,
            )
        )
        _monitor_continuous_target(
            controller=controller,
            live=live,
            target_isaac=target,
            start_q_isaac=start_q_isaac,
            duration_s=1.0 / POLICY_RATE_HZ,
            minimum_delta_rad=relaxed_minimum,
            maximum_delta_rad=relaxed_maximum,
            maximum_joint_speed_rad_s=4.0,
            maximum_estimated_torque=POLICY_TORQUE_GUARD_NM,
            maximum_roll_rad=0.30,
        )
    _monitor_continuous_target(
        controller=controller,
        live=live,
        target_isaac=WOOF_BUILTIN_STANDDOWN_RAD,
        start_q_isaac=start_q_isaac,
        duration_s=0.5,
        minimum_delta_rad=relaxed_minimum,
        maximum_delta_rad=relaxed_maximum,
        maximum_joint_speed_rad_s=4.0,
        maximum_estimated_torque=POLICY_TORQUE_GUARD_NM,
        maximum_roll_rad=0.30,
    )


def standup_from_lying(
    *,
    publisher: LowCommandPublisher,
    live: LiveState,
    duration_s: float,
    kp: float,
    kd: float,
) -> tuple[float, ...]:
    """Raise Woof smoothly into the policy's nominal standing stance."""

    low, _ = live.sample()
    start = unitree_to_isaac(low["q"])
    steps = max(1, int(round(duration_s * POLICY_RATE_HZ)))
    print(
        f"[standing-point] lifting over {duration_s:.1f}s "
        f"at kp={kp:.0f} kd={kd:.1f}",
        flush=True,
    )
    for step in range(1, steps + 1):
        alpha = step / steps
        blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
        interpolated = tuple(
            current + (target - current) * blend
            for current, target in zip(
                start,
                ISAAC_GROUNDED_STAND_RAD,
                strict=True,
            )
        )
        _publish_for(
            publisher=publisher,
            live=live,
            target_isaac=interpolated,
            start_q_isaac=start,
            duration_s=1.0 / POLICY_RATE_HZ,
            kp=kp,
            kd=kd,
            require_standdown=False,
            minimum_delta_rad=tuple(-2.5 for _ in range(12)),
            maximum_delta_rad=tuple(2.5 for _ in range(12)),
            maximum_joint_speed_rad_s=4.0,
            maximum_estimated_torque=POLICY_TORQUE_GUARD_NM,
            maximum_roll_rad=0.30,
        )
    _publish_for(
        publisher=publisher,
        live=live,
        target_isaac=ISAAC_GROUNDED_STAND_RAD,
        start_q_isaac=start,
        duration_s=0.6,
        kp=kp,
        kd=kd,
        require_standdown=False,
        minimum_delta_rad=tuple(-2.5 for _ in range(12)),
        maximum_delta_rad=tuple(2.5 for _ in range(12)),
        maximum_joint_speed_rad_s=4.0,
        maximum_estimated_torque=POLICY_TORQUE_GUARD_NM,
        maximum_roll_rad=0.30,
    )
    low, _ = live.sample()
    reached = unitree_to_isaac(low["q"])
    worst = max(
        abs(actual - target)
        for actual, target in zip(
            reached,
            ISAAC_GROUNDED_STAND_RAD,
            strict=True,
        )
    )
    print(
        f"[standing-point] lift complete; worst joint error={worst:.3f} rad",
        flush=True,
    )
    return reached


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("--execute is required; use shadow_runner.py for no-motion tests")
    expected_confirmation = "AREA IS CLEAR AND WOOF MAY MOVE"
    if args.confirm != expected_confirmation:
        raise RuntimeError("physical confirmation text does not match")
    if not (0.0 < args.minimum_confidence <= 1.0):
        raise ValueError("--minimum-confidence must be in (0, 1]")
    maximum_duration_s = 8.0
    maximum_action_gain = 1.0 if args.full_power else 0.20
    maximum_target_rate = 0.60 if args.full_power else 0.25
    maximum_kp = 60.0
    maximum_kd = 5.0
    if not (2.0 <= args.duration <= maximum_duration_s):
        raise ValueError(
            f"--duration must be between 2.0 and {maximum_duration_s} seconds"
        )
    if not (0.0 < args.action_gain <= maximum_action_gain):
        raise ValueError(f"--action-gain must be in (0, {maximum_action_gain}]")
    if not (0.05 <= args.max_rate_rad_s <= maximum_target_rate):
        raise ValueError(f"--max-rate-rad-s must be in [0.05, {maximum_target_rate}]")
    if not (
        0.0 < args.kp <= maximum_kp
        and 0.0 < args.kd <= maximum_kd
    ):
        raise ValueError("gain guard rejected Kp/Kd")
    if not (2.0 <= args.standup_seconds <= 5.0):
        raise ValueError("--standup-seconds must be between 2.0 and 5.0")
    if not (
        0.0 < args.standup_kp <= 60.0
        and 0.0 < args.standup_kd <= 5.0
    ):
        raise ValueError("stand-up gain guard rejected Kp/Kd")
    if sha256_file(args.policy) != EXPECTED_POLICY_SHA256:
        raise RuntimeError("policy SHA-256 mismatch")

    stop = Event()
    signal.signal(signal.SIGINT, lambda *_args: stop.set())
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())

    torch.set_num_threads(1)
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    with torch.inference_mode():
        warm = torch.zeros(
            (1, LOCKED_POINT_OBSERVATION_SIZE),
            dtype=torch.float32,
        )
        for _ in range(50):
            policy(warm)

    live = LiveState(args.network_interface)
    boxes = BBoxProvider(
        status_url=args.status_url,
        rate_hz=15.0,
        synthetic_bbox=None,
        maximum_age_s=0.8,
        # A real gesture must never continue from a remembered box after the
        # selected fruit leaves the detector. Every policy tick below requires
        # a fresh lock with the same lock id.
        hold_last_valid_s=0.0,
    )
    boxes.start()
    publisher: LowCommandPublisher | None = None
    controller: ContinuousJointHold | None = None
    released = False
    original_mode = "mcf"
    selected_label: str | None = None
    start_q_isaac: tuple[float, ...] | None = None
    records: list[dict[str, Any]] = []
    droop: dict[str, float] = {}
    peak_torque: dict[str, float] = {}
    outcome = "not_started"
    error: str | None = None
    recovery_used_standdown = False
    minimum_delta_rad = LOCKED_POINT_MIN_DELTA_RAD
    maximum_delta_rad = LOCKED_POINT_MAX_DELTA_RAD
    maximum_joint_speed_rad_s = 4.0
    maximum_estimated_torque = POLICY_TORQUE_GUARD_NM
    maximum_bbox_age_s = 0.8
    maximum_roll_rad = 0.30

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    switcher = MotionSwitcherClient()
    switcher.SetTimeout(5.0)
    switcher.Init()
    sport = SportClient()
    sport.SetTimeout(8.0)
    sport.Init()

    try:
        live.wait()
        bbox_deadline = time.monotonic() + 2.0
        box = boxes.sample()
        while (
            box["label"] is None
            or box["error"] is not None
            or box["received_at"] is None
        ):
            if time.monotonic() >= bbox_deadline:
                raise RuntimeError(f"no fresh selected target: {box['error']}")
            time.sleep(0.02)
            box = boxes.sample()
        if box["label"] != args.target_label:
            raise RuntimeError(
                f"selected target is {box['label']!r}, expected {args.target_label!r}"
            )
        selected_label = str(box["label"])
        if float(box["confidence"] or 0.0) < args.minimum_confidence:
            raise RuntimeError(
                f"selected {args.target_label} confidence is only "
                f"{box['confidence']}; need {args.minimum_confidence}"
            )

        mode_code, mode = switcher.CheckMode()
        if mode_code != 0 or not mode or mode.get("name") != original_mode:
            raise RuntimeError(f"unexpected controller mode: code={mode_code}, mode={mode}")

        if args.direct_standing_handoff:
            low, _sport_state = live.sample()
            start_q_isaac = _validate_direct_standing_start(low)
            publisher = LowCommandPublisher()
            controller = ContinuousJointHold(
                publisher=publisher,
                target_isaac=start_q_isaac,
                kp=args.kp,
                kd=args.kd,
            )
            controller.start()
            # Prove lowcmd frames exist before releasing the controller that is
            # currently carrying Woof's weight.
            time.sleep(0.10)
            controller.raise_if_failed()
            release_sport_mode(switcher)
            released = True
            _monitor_continuous_target(
                controller=controller,
                live=live,
                target_isaac=start_q_isaac,
                start_q_isaac=start_q_isaac,
                duration_s=0.25,
                maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                maximum_estimated_torque=maximum_estimated_torque,
                maximum_roll_rad=maximum_roll_rad,
            )
            print(
                "[standing-point] direct Sport-standing handoff verified",
                flush=True,
            )
        else:
            ensure_standdown(sport, live)
            low, _sport_state = live.sample()
            start_q_isaac = unitree_to_isaac(low["q"])
            _check_live_guards(
                low=low,
                now=time.monotonic(),
                start_q_isaac=start_q_isaac,
                require_standdown=True,
                minimum_delta_rad=minimum_delta_rad,
                maximum_delta_rad=maximum_delta_rad,
                maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                maximum_estimated_torque=maximum_estimated_torque,
                maximum_roll_rad=maximum_roll_rad,
            )
            if _maximum_abs(unitree_to_isaac(low["dq"])) > 0.20:
                raise RuntimeError("Woof is not still enough for handoff")

            publisher = LowCommandPublisher()
            release_sport_mode(switcher)
            released = True

            # Immediately take over at the measured pose with intentionally low
            # gains, then prove the low-state stream remains healthy.
            _publish_for(
                publisher=publisher,
                live=live,
                target_isaac=start_q_isaac,
                start_q_isaac=start_q_isaac,
                duration_s=0.25,
                kp=args.kp,
                kd=args.kd,
                require_standdown=True,
                minimum_delta_rad=minimum_delta_rad,
                maximum_delta_rad=maximum_delta_rad,
                maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                maximum_estimated_torque=maximum_estimated_torque,
                maximum_roll_rad=maximum_roll_rad,
            )

            start_q_isaac = standup_from_lying(
                publisher=publisher,
                live=live,
                duration_s=args.standup_seconds,
                kp=args.standup_kp,
                kd=args.standup_kd,
            )

        period_s = 1.0 / PUBLISH_RATE_HZ
        policy_stride = int(round(PUBLISH_RATE_HZ / POLICY_RATE_HZ))
        total_ticks = int(round(args.duration * PUBLISH_RATE_HZ))
        next_deadline = time.perf_counter()
        command_q = tuple(start_q_isaac)
        previous_action = (0.0,) * LOCKED_POINT_ACTION_SIZE
        lock_id = box.get("lock_id")
        latest_action = (0.0,) * LOCKED_POINT_ACTION_SIZE
        policy_started = time.monotonic()

        with torch.inference_mode():
            for tick in range(total_ticks):
                if stop.is_set():
                    raise RuntimeError("operator stop requested")
                if controller is not None:
                    controller.raise_if_failed()
                next_deadline += period_s
                low, _sport_state = live.sample()
                now = time.monotonic()
                _check_live_guards(
                    low=low,
                    now=now,
                    start_q_isaac=start_q_isaac,
                    require_standdown=False,
                    minimum_delta_rad=minimum_delta_rad,
                    maximum_delta_rad=maximum_delta_rad,
                    maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                    maximum_estimated_torque=maximum_estimated_torque,
                    maximum_roll_rad=maximum_roll_rad,
                    observed_deviation=droop,
                )
                for name, torque in zip(
                    ISAAC_JOINT_ORDER,
                    unitree_to_isaac(low["tau_est"]),
                    strict=True,
                ):
                    if abs(torque) > peak_torque.get(name, 0.0):
                        peak_torque[name] = abs(torque)

                if tick % policy_stride == 0:
                    box = boxes.sample()
                    bbox_age_s = (
                        math.inf
                        if box["received_at"] is None
                        else now
                        - float(box["received_at"])
                        + float(box["frame_age_s"] or 0.0)
                    )
                    if (
                        box["error"] is not None
                        or box["label"] != args.target_label
                        or box.get("lock_id") != lock_id
                        or bbox_age_s > maximum_bbox_age_s
                    ):
                        raise RuntimeError(
                            f"selected {args.target_label} lock was lost or became stale: "
                            f"label={box['label']!r}, age={bbox_age_s:.3f}, "
                            f"error={box['error']!r}"
                        )
                    q_isaac = unitree_to_isaac(low["q"])
                    dq_isaac = unitree_to_isaac(low["dq"])
                    phase = locked_point_phase(
                        time.monotonic() - policy_started,
                        LOCKED_POINT_SETUP_S,
                        LOCKED_POINT_RAMP_S,
                    )
                    observation = build_locked_point_observation(
                        # SportModeState stops updating after handoff. The
                        # standing point is stationary, so body linear velocity
                        # is explicitly zero.
                        base_linear_velocity_body=(0.0, 0.0, 0.0),
                        base_angular_velocity_body=low["gyroscope"],
                        gravity_body=projected_gravity_wxyz(low["quaternion"]),
                        joint_position_isaac=q_isaac,
                        joint_velocity_isaac=dq_isaac,
                        previous_action=previous_action,
                        point_phase=phase,
                        bbox_xyxy_normalized=box["bbox"],
                    )
                    output = policy(
                        torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    ).squeeze(0)
                    latest_action = tuple(float(value) for value in output)
                    if len(latest_action) != LOCKED_POINT_ACTION_SIZE or not all(
                        math.isfinite(value) for value in latest_action
                    ):
                        raise RuntimeError("actor returned invalid action")
                    raw_target = locked_point_joint_targets(
                        latest_action,
                        point_phase=phase,
                    )
                    command_q = guarded_policy_target(
                        raw_target=raw_target,
                        start_target=start_q_isaac,
                        previous_target=command_q,
                        action_gain=args.action_gain,
                        maximum_step_rad=args.max_rate_rad_s / POLICY_RATE_HZ,
                        minimum_delta_rad=minimum_delta_rad,
                        maximum_delta_rad=maximum_delta_rad,
                    )
                    previous_action = tuple(
                        (
                            command_q[ISAAC_JOINT_ORDER.index(name)]
                            - ISAAC_GROUNDED_STAND_RAD[
                                ISAAC_JOINT_ORDER.index(name)
                            ]
                        )
                        / 0.5
                        for name in ISAAC_JOINT_ORDER
                        if not name.startswith("FR_")
                    )
                    records.append(
                        {
                            "policy_tick": tick // policy_stride,
                            "point_phase": phase,
                            "bbox": box["bbox"],
                            "confidence": box["confidence"],
                            "bbox_age_s": bbox_age_s,
                            "bbox_held": bool(box.get("held")),
                            "bbox_source": box.get("source"),
                            "action": latest_action,
                            "command_q_isaac": command_q,
                            "measured_q_isaac": q_isaac,
                            "rpy": low["rpy"],
                            "max_joint_speed_rad_s": _maximum_abs(dq_isaac),
                            "max_estimated_torque": _maximum_abs(
                                low["tau_est"]
                            ),
                        }
                    )

                if controller is None:
                    publisher.write(
                        isaac_to_unitree(command_q),
                        kp=args.kp,
                        kd=args.kd,
                    )
                else:
                    controller.set_target(command_q)
                remaining = next_deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
        outcome = (
            "completed_full_policy"
            if args.full_power
            else "completed_reduced_amplitude_policy"
        )
    except Exception as exc:
        error = str(exc)
        outcome = "aborted"
    finally:
        if publisher is not None and released:
            try:
                low, _sport_state = live.sample()
                current_q = unitree_to_isaac(low["q"])
                return_period = 1.0 / POLICY_RATE_HZ
                maximum_return_distance = max(
                    abs(current - target)
                    for current, target in zip(
                        current_q, start_q_isaac, strict=True
                    )
                )
                return_ticks = max(
                    1,
                    int(
                        math.ceil(
                            maximum_return_distance
                            / (args.max_rate_rad_s * return_period)
                        )
                    ),
                )
                return_deadline = time.perf_counter()
                return_command = tuple(current_q)
                for _ in range(return_ticks):
                    return_deadline += return_period
                    return_command = tuple(
                        previous
                        + _clamp(
                            target - previous,
                            -args.max_rate_rad_s * return_period,
                            args.max_rate_rad_s * return_period,
                        )
                        for previous, target in zip(
                            return_command, start_q_isaac, strict=True
                        )
                    )
                    if controller is None:
                        _publish_for(
                            publisher=publisher,
                            live=live,
                            target_isaac=return_command,
                            start_q_isaac=start_q_isaac,
                            duration_s=return_period,
                            kp=args.kp,
                            kd=args.kd,
                            minimum_delta_rad=minimum_delta_rad,
                            maximum_delta_rad=maximum_delta_rad,
                            maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                            maximum_estimated_torque=maximum_estimated_torque,
                            maximum_roll_rad=maximum_roll_rad,
                        )
                    else:
                        _monitor_continuous_target(
                            controller=controller,
                            live=live,
                            target_isaac=return_command,
                            start_q_isaac=start_q_isaac,
                            duration_s=return_period,
                            minimum_delta_rad=minimum_delta_rad,
                            maximum_delta_rad=maximum_delta_rad,
                            maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                            maximum_estimated_torque=maximum_estimated_torque,
                            maximum_roll_rad=maximum_roll_rad,
                        )
                    remaining = return_deadline - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
                if controller is None:
                    _publish_for(
                        publisher=publisher,
                        live=live,
                        target_isaac=start_q_isaac,
                        start_q_isaac=start_q_isaac,
                        duration_s=0.20,
                        kp=args.kp,
                        kd=args.kd,
                        minimum_delta_rad=minimum_delta_rad,
                        maximum_delta_rad=maximum_delta_rad,
                        maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                        maximum_estimated_torque=maximum_estimated_torque,
                        maximum_roll_rad=maximum_roll_rad,
                    )
                else:
                    _monitor_continuous_target(
                        controller=controller,
                        live=live,
                        target_isaac=start_q_isaac,
                        start_q_isaac=start_q_isaac,
                        duration_s=0.20,
                        minimum_delta_rad=minimum_delta_rad,
                        maximum_delta_rad=maximum_delta_rad,
                        maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                        maximum_estimated_torque=maximum_estimated_torque,
                        maximum_roll_rad=maximum_roll_rad,
                    )
            except Exception as return_exc:
                error = f"{error or ''}; return guard: {return_exc}".strip("; ")
                outcome = "aborted_during_return"
            if controller is not None:
                try:
                    # Some firmware accepts mcf while the lowcmd writer is
                    # still holding the measured standing pose.  Woof's tested
                    # build does not, so the fallback lowers under continuous
                    # control before closing the writer.
                    restored = _select_sport_mode(switcher, attempts=3)
                    if not restored:
                        recovery_used_standdown = True
                        low, _sport_state = live.sample()
                        _lower_continuous_to_standdown(
                            controller=controller,
                            live=live,
                            start_q_isaac=unitree_to_isaac(low["q"]),
                        )
                    controller.stop()
                    controller = None
                    publisher.close()
                    publisher = None
                    if not restored:
                        restored = _select_sport_mode(switcher)
                    if not restored:
                        raise RuntimeError("failed to restore mcf")
                    released = False
                    if recovery_used_standdown:
                        standdown_code = sport.StandDown()
                        if standdown_code != 0:
                            raise RuntimeError(
                                "StandDown after controller restore failed: "
                                f"{standdown_code}"
                            )
                        time.sleep(0.75)
                        standup_code = sport.StandUp()
                        if standup_code != 0:
                            raise RuntimeError(
                                f"StandUp after controller restore failed: {standup_code}"
                            )
                        posture_deadline = time.monotonic() + 5.0
                        while True:
                            low, _sport_state = live.sample()
                            try:
                                _validate_direct_standing_start(low)
                                break
                            except RuntimeError:
                                if time.monotonic() >= posture_deadline:
                                    raise RuntimeError(
                                        "Sport standing pose did not settle "
                                        "after controller restoration"
                                    )
                                time.sleep(0.05)
                except Exception as restore_exc:
                    error = (
                        f"{error or ''}; restore exception: {restore_exc}"
                    ).strip("; ")
                    outcome = "restore_failed"
                    if controller is not None:
                        try:
                            controller.stop()
                        except Exception:
                            pass
                        controller = None
                    if publisher is not None:
                        try:
                            publisher.close()
                        except Exception:
                            pass
                        publisher = None
                    if _select_sport_mode(switcher):
                        released = False
            else:
                try:
                    publisher.close()
                    publisher = None
                    if not _select_sport_mode(switcher):
                        raise RuntimeError("failed to restore mcf")
                    released = False
                    time.sleep(0.4)
                    standdown_code = sport.StandDown()
                    if standdown_code != 0:
                        raise RuntimeError(
                            f"StandDown after controller restore failed: {standdown_code}"
                        )
                    # StandDown returns before the low-state stream necessarily
                    # reflects the completed action.  Wait for the measured
                    # joints instead of logging a transient mcf transition.
                    posture_deadline = time.monotonic() + 4.0
                    while True:
                        low, _sport_state = live.sample()
                        restored_rms, restored_max = standdown_error_rad(
                            unitree_to_isaac(low["q"])
                        )
                        restored_speed = _maximum_abs(unitree_to_isaac(low["dq"]))
                        if (
                            time.monotonic() - float(low["received_at"]) < 0.05
                            and restored_rms < 0.08
                            and restored_max < 0.16
                            and restored_speed < 0.20
                        ):
                            break
                        if time.monotonic() >= posture_deadline:
                            raise RuntimeError(
                                "StandDown did not reach the verified pose after mcf restore"
                            )
                        time.sleep(0.05)
                except Exception as restore_exc:
                    error = (
                        f"{error or ''}; restore exception: {restore_exc}"
                    ).strip("; ")
                    outcome = "restore_failed"
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass
        if publisher is not None:
            try:
                publisher.close()
            except Exception:
                pass
        boxes.close()

    final_low, _final_sport = live.sample()
    final_q_isaac = unitree_to_isaac(final_low["q"])
    final_rms, final_max = standdown_error_rad(final_q_isaac)
    final_standing_errors = tuple(
        actual - expected
        for actual, expected in zip(
            final_q_isaac,
            ISAAC_GROUNDED_STAND_RAD,
            strict=True,
        )
    )
    final_standing_rms = math.sqrt(
        sum(value * value for value in final_standing_errors)
        / len(final_standing_errors)
    )
    final_standing_max = _maximum_abs(final_standing_errors)
    report = {
        "outcome": outcome,
        "error": error,
        "policy_sha256": EXPECTED_POLICY_SHA256,
        "selected_label": selected_label,
        "policy_ticks": len(records),
        "policy_kind": "locked_point_v19",
        "handoff_mode": (
            "sport_standing_direct"
            if args.direct_standing_handoff
            else "standdown_then_lift"
        ),
        "recovery_used_standdown": recovery_used_standdown,
        "observation_size": LOCKED_POINT_OBSERVATION_SIZE,
        "action_size": LOCKED_POINT_ACTION_SIZE,
        "tracking_slack_rad": COMMAND_TRACKING_SLACK_RAD,
        "peak_estimated_torque_nm": {
            name: round(value, 2)
            for name, value in sorted(
                peak_torque.items(),
                key=lambda item: -item[1],
            )
        },
        "worst_droop_outside_envelope_rad": {
            name: round(value, 4)
            for name, value in sorted(
                droop.items(),
                key=lambda item: -item[1],
            )
        },
        "requested_duration_s": args.duration,
        "action_gain": args.action_gain,
        "max_rate_rad_s": args.max_rate_rad_s,
        "kp": args.kp,
        "kd": args.kd,
        "full_power": args.full_power,
        "maximum_estimated_torque_guard": maximum_estimated_torque,
        "roll_guard_bypassed": False,
        "controller_restored": not released,
        "start_q_isaac": start_q_isaac,
        "final_q_isaac": final_q_isaac,
        "final_lowstate_age_s": time.monotonic() - float(final_low["received_at"]),
        "final_standdown_rms_error_rad": final_rms,
        "final_standdown_max_error_rad": final_max,
        "final_standing_rms_error_rad": final_standing_rms,
        "final_standing_max_error_rad": final_standing_max,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return (
        0
        if outcome
        in {"completed_reduced_amplitude_policy", "completed_full_policy"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
