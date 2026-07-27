#!/usr/bin/env python3
"""Run a deliberately reduced Go2 pointing-policy hardware proof.

This is not the full-strength stage skill.  It requires an explicitly selected
Collie target, verifies Woof is in the measured StandDown pose, warms the actor,
hands off from Sport mode, and applies only a small, rate-limited fraction of
the actor target.  Every exit path attempts to return to the captured
StandDown pose and restore Unitree's ``mcf`` Sport controller.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import struct
from threading import Event
import time
from typing import Any, Sequence

import torch

from .pointing_contract import (
    ISAAC_GROUNDED_SIT_RAD,
    ISAAC_JOINT_ORDER,
    FULL_POLICY_MAX_DELTA_RAD,
    FULL_POLICY_MIN_DELTA_RAD,
    REDUCED_PROOF_MIN_DELTA_RAD,
    REDUCED_PROOF_MAX_DELTA_RAD,
    actor_action_to_joint_target,
    build_actor_observation,
    guarded_policy_target,
    isaac_to_unitree,
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


PUBLISH_RATE_HZ = 500.0
POLICY_RATE_HZ = 50.0
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
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--action-gain", type=float, default=0.15)
    parser.add_argument("--max-rate-rad-s", type=float, default=0.18)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=1.0)
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
        help='must be exactly "WOOF IS LYING DOWN AND AREA IS CLEAR"',
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
    minimum_delta_rad: Sequence[float] = REDUCED_PROOF_MIN_DELTA_RAD,
    maximum_delta_rad: Sequence[float] = REDUCED_PROOF_MAX_DELTA_RAD,
    maximum_joint_speed_rad_s: float = 1.0,
    maximum_estimated_torque: float = 20.0,
    maximum_roll_rad: float | None = 0.30,
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
        if delta < minimum_delta - 0.08 or delta > maximum_delta + 0.08:
            raise RuntimeError(
                f"joint envelope guard: {ISAAC_JOINT_ORDER[index]} moved "
                f"{delta:.3f} rad"
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
    minimum_delta_rad: Sequence[float] = REDUCED_PROOF_MIN_DELTA_RAD,
    maximum_delta_rad: Sequence[float] = REDUCED_PROOF_MAX_DELTA_RAD,
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


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("--execute is required; use shadow_runner.py for no-motion tests")
    expected_confirmation = (
        "WOOF IS LYING DOWN AND FULL POLICY AREA IS CLEAR"
        if args.full_power
        else "WOOF IS LYING DOWN AND AREA IS CLEAR"
    )
    if args.confirm != expected_confirmation:
        raise RuntimeError("physical confirmation text does not match")
    if not (0.0 < args.minimum_confidence <= 1.0):
        raise ValueError("--minimum-confidence must be in (0, 1]")
    maximum_duration_s = 3.0 if args.full_power else 2.0
    maximum_action_gain = 1.0 if args.full_power else 0.20
    maximum_target_rate = 0.60 if args.full_power else 0.25
    maximum_kp = 25.0 if args.full_power else 10.0
    if not (0.1 <= args.duration <= maximum_duration_s):
        raise ValueError(f"--duration must be between 0.1 and {maximum_duration_s} seconds")
    if not (0.0 < args.action_gain <= maximum_action_gain):
        raise ValueError(f"--action-gain must be in (0, {maximum_action_gain}]")
    if not (0.05 <= args.max_rate_rad_s <= maximum_target_rate):
        raise ValueError(f"--max-rate-rad-s must be in [0.05, {maximum_target_rate}]")
    if not (0.0 < args.kp <= maximum_kp and 0.0 < args.kd <= 1.0):
        raise ValueError("gain guard rejected Kp/Kd")
    if sha256_file(args.policy) != EXPECTED_POLICY_SHA256:
        raise RuntimeError("policy SHA-256 mismatch")

    stop = Event()
    signal.signal(signal.SIGINT, lambda *_args: stop.set())
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())

    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    with torch.inference_mode():
        warm = torch.zeros((1, 49), dtype=torch.float32)
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
    released = False
    original_mode = "mcf"
    selected_label: str | None = None
    start_q_isaac: tuple[float, ...] | None = None
    records: list[dict[str, Any]] = []
    outcome = "not_started"
    error: str | None = None
    minimum_delta_rad = (
        FULL_POLICY_MIN_DELTA_RAD if args.full_power else REDUCED_PROOF_MIN_DELTA_RAD
    )
    maximum_delta_rad = (
        FULL_POLICY_MAX_DELTA_RAD if args.full_power else REDUCED_PROOF_MAX_DELTA_RAD
    )
    maximum_joint_speed_rad_s = 2.0 if args.full_power else 1.0
    maximum_estimated_torque = 12.0 if args.full_power else 20.0
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

        mode_code, mode = switcher.CheckMode()
        if mode_code != 0 or not mode or mode.get("name") != original_mode:
            raise RuntimeError(f"unexpected controller mode: code={mode_code}, mode={mode}")

        publisher = LowCommandPublisher()
        release_code, _ = switcher.ReleaseMode()
        if release_code != 0:
            raise RuntimeError(f"failed to release Sport mode: {release_code}")
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

        period_s = 1.0 / PUBLISH_RATE_HZ
        policy_stride = int(round(PUBLISH_RATE_HZ / POLICY_RATE_HZ))
        total_ticks = int(round(args.duration * PUBLISH_RATE_HZ))
        next_deadline = time.perf_counter()
        command_q = tuple(start_q_isaac)
        previous_action = (0.0,) * 12
        lock_id = box.get("lock_id")
        latest_action = (0.0,) * 12

        with torch.inference_mode():
            for tick in range(total_ticks):
                if stop.is_set():
                    raise RuntimeError("operator stop requested")
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
                )

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
                    observation = build_actor_observation(
                        # SportModeState stops updating after handoff.  Woof is
                        # grounded and guarded stationary, so this first proof
                        # explicitly uses zero body linear velocity.
                        base_linear_velocity_body=(0.0, 0.0, 0.0),
                        base_angular_velocity_body=low["gyroscope"],
                        gravity_body=projected_gravity_wxyz(low["quaternion"]),
                        joint_position_isaac=q_isaac,
                        joint_velocity_isaac=dq_isaac,
                        previous_action=previous_action,
                        bbox_xyxy_normalized=box["bbox"],
                    )
                    output = policy(
                        torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    ).squeeze(0)
                    latest_action = tuple(float(value) for value in output)
                    if len(latest_action) != 12 or not all(
                        math.isfinite(value) for value in latest_action
                    ):
                        raise RuntimeError("actor returned invalid action")
                    raw_target = actor_action_to_joint_target(latest_action)
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
                        (command - default) / 0.5
                        for command, default in zip(
                            command_q, ISAAC_GROUNDED_SIT_RAD, strict=True
                        )
                    )
                    records.append(
                        {
                            "policy_tick": tick // policy_stride,
                            "bbox": box["bbox"],
                            "confidence": box["confidence"],
                            "bbox_age_s": bbox_age_s,
                            "bbox_held": bool(box.get("held")),
                            "action": latest_action,
                            "command_q_isaac": command_q,
                            "measured_q_isaac": q_isaac,
                            "rpy": low["rpy"],
                            "max_joint_speed_rad_s": _maximum_abs(dq_isaac),
                            "max_estimated_torque": _maximum_abs(low["tau_est"]),
                        }
                    )

                publisher.write(
                    isaac_to_unitree(command_q),
                    kp=args.kp,
                    kd=args.kd,
                )
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
                    remaining = return_deadline - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
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
            except Exception as return_exc:
                error = f"{error or ''}; return guard: {return_exc}".strip("; ")
                outcome = "aborted_during_return"
            try:
                publisher.close()
            except Exception:
                pass
            try:
                select_code = -1
                restored_mode: dict[str, Any] | None = None
                for _ in range(8):
                    check_code, restored_mode = switcher.CheckMode()
                    if (
                        check_code == 0
                        and restored_mode
                        and restored_mode.get("name") == original_mode
                    ):
                        select_code = 0
                        break
                    select_code, _ = switcher.SelectMode(original_mode)
                    if select_code == 0:
                        time.sleep(0.5)
                        continue
                    # Unitree code 7002 is transient while the just-closed
                    # lowcmd writer is still disappearing from DDS discovery.
                    time.sleep(0.5)
                check_code, restored_mode = switcher.CheckMode()
                if not (
                    select_code == 0
                    and check_code == 0
                    and restored_mode
                    and restored_mode.get("name") == original_mode
                ):
                    error = f"{error or ''}; failed to restore mcf: {select_code}".strip("; ")
                    outcome = "restore_failed"
                else:
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
                error = f"{error or ''}; restore exception: {restore_exc}".strip("; ")
                outcome = "restore_failed"
        boxes.close()

    final_low, _final_sport = live.sample()
    final_q_isaac = unitree_to_isaac(final_low["q"])
    final_rms, final_max = standdown_error_rad(final_q_isaac)
    report = {
        "outcome": outcome,
        "error": error,
        "policy_sha256": EXPECTED_POLICY_SHA256,
        "selected_label": selected_label,
        "policy_ticks": len(records),
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
