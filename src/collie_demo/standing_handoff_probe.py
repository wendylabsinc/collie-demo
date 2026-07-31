#!/usr/bin/env python3
"""Bounded Sport-standing to low-level-standing handoff probe.

This probe does not run a learned policy and cannot walk. It continuously
publishes Woof's measured standing joint positions before asking the Unitree
motion switcher to release Sport mode, holds that exact pose briefly, and then
restores Sport mode. If seamless restoration is rejected while the low-level
writer exists, it lowers to verified StandDown before closing the writer and
restoring Sport mode.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from .pointing_contract import (
    ISAAC_GROUNDED_STAND_RAD,
    WOOF_BUILTIN_STANDDOWN_RAD,
    joint_limit_violations,
    standdown_error_rad,
    unitree_to_isaac,
)
from .pointing_runner import (
    ContinuousJointHold,
    LowCommandPublisher,
    _maximum_abs,
    release_sport_mode,
)
from .pointing_shadow import LiveState


CONFIRMATION = "WOOF IS SUPPORTED FOR DIRECT STANDING HANDOFF"
MAXIMUM_TORQUE_NM = 22.0
MAXIMUM_JOINT_SPEED_RAD_S = 4.0
MAXIMUM_JOINT_DEVIATION_RAD = 0.45
MAXIMUM_ROLL_RAD = 0.20
MAXIMUM_PITCH_RAD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", default="enP8p1s0")
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/collie-standing-handoff.json"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


def standing_reference_error_rad(
    joint_position_isaac: Sequence[float],
) -> tuple[float, float]:
    if len(joint_position_isaac) != 12:
        raise ValueError("standing pose must contain 12 joint positions")
    errors = tuple(
        float(actual) - reference
        for actual, reference in zip(
            joint_position_isaac,
            ISAAC_GROUNDED_STAND_RAD,
            strict=True,
        )
    )
    rms = math.sqrt(sum(value * value for value in errors) / len(errors))
    return rms, _maximum_abs(errors)


def validate_direct_handoff_start(low: dict[str, Any]) -> tuple[float, ...]:
    age_s = time.monotonic() - float(low["received_at"])
    if age_s > 0.05:
        raise RuntimeError(f"low-state is stale: {age_s:.3f}s")
    q_isaac = unitree_to_isaac(low["q"])
    dq_isaac = unitree_to_isaac(low["dq"])
    violations = joint_limit_violations(q_isaac)
    if violations:
        raise RuntimeError(f"measured joint limit violation: {violations[0]}")
    if _maximum_abs(dq_isaac) > 0.20:
        raise RuntimeError("Woof is not still enough for direct handoff")
    roll, pitch, _yaw = (float(value) for value in low["rpy"])
    if abs(roll) > 0.12 or abs(pitch) > 0.12:
        raise RuntimeError(
            f"Woof is not level enough: roll={roll:.3f}, pitch={pitch:.3f}"
        )
    standdown_rms, _standdown_max = standdown_error_rad(q_isaac)
    if standdown_rms < 0.40:
        raise RuntimeError("Woof is not in a standing posture")
    standing_rms, standing_max = standing_reference_error_rad(q_isaac)
    if standing_rms > 0.35 or standing_max > 0.65:
        raise RuntimeError(
            "Sport standing posture is too far from the policy stance: "
            f"rms={standing_rms:.3f}, max={standing_max:.3f}"
        )
    return q_isaac


def validate_held_stance(
    *,
    low: dict[str, Any],
    captured_q_isaac: Sequence[float],
) -> dict[str, float]:
    age_s = time.monotonic() - float(low["received_at"])
    if age_s > 0.05:
        raise RuntimeError(f"low-state timeout during handoff: {age_s:.3f}s")
    q_isaac = unitree_to_isaac(low["q"])
    dq_isaac = unitree_to_isaac(low["dq"])
    deviation = max(
        abs(actual - captured)
        for actual, captured in zip(
            q_isaac,
            captured_q_isaac,
            strict=True,
        )
    )
    speed = _maximum_abs(dq_isaac)
    torque = _maximum_abs(low["tau_est"])
    roll, pitch, _yaw = (float(value) for value in low["rpy"])
    if abs(roll) > MAXIMUM_ROLL_RAD or abs(pitch) > MAXIMUM_PITCH_RAD:
        raise RuntimeError(
            f"tilt guard: roll={roll:.3f}, pitch={pitch:.3f}"
        )
    if speed > MAXIMUM_JOINT_SPEED_RAD_S:
        raise RuntimeError(f"joint-speed guard: {speed:.3f} rad/s")
    if torque > MAXIMUM_TORQUE_NM:
        raise RuntimeError(f"estimated-torque guard: {torque:.3f} Nm")
    if deviation > MAXIMUM_JOINT_DEVIATION_RAD:
        raise RuntimeError(
            f"standing-hold deviation guard: {deviation:.3f} rad"
        )
    return {
        "lowstate_age_s": age_s,
        "maximum_joint_deviation_rad": deviation,
        "maximum_joint_speed_rad_s": speed,
        "maximum_estimated_torque_nm": torque,
        "roll_rad": roll,
        "pitch_rad": pitch,
    }


def _select_sport_mode(switcher: Any, attempts: int = 8) -> bool:
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


def _lower_to_standdown(
    *,
    controller: ContinuousJointHold,
    start_q_isaac: Sequence[float],
    duration_s: float = 2.5,
) -> None:
    steps = max(1, int(round(duration_s * 50.0)))
    for step in range(1, steps + 1):
        alpha = step / steps
        blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
        controller.set_target(
            tuple(
                start + (target - start) * blend
                for start, target in zip(
                    start_q_isaac,
                    WOOF_BUILTIN_STANDDOWN_RAD,
                    strict=True,
                )
            )
        )
        controller.raise_if_failed()
        time.sleep(1.0 / 50.0)
    controller.set_target(WOOF_BUILTIN_STANDDOWN_RAD)
    time.sleep(0.5)


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("--execute is required")
    if args.confirm != CONFIRMATION:
        raise RuntimeError("physical confirmation text does not match")
    if not (0.5 <= args.hold_seconds <= 2.0):
        raise ValueError("--hold-seconds must be between 0.5 and 2.0")
    if not (20.0 <= args.kp <= 60.0 and 1.0 <= args.kd <= 5.0):
        raise ValueError("handoff gains are outside the bounded range")

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    live = LiveState(args.network_interface)
    live.wait()
    switcher = MotionSwitcherClient()
    switcher.SetTimeout(3.0)
    switcher.Init()
    sport = SportClient()
    sport.SetTimeout(5.0)
    sport.Init()

    code, mode = switcher.CheckMode()
    if code != 0 or not mode or mode.get("name") != "mcf":
        raise RuntimeError(f"expected Sport mcf mode, got code={code}, mode={mode}")
    low, _sport_state = live.sample()
    captured_q_isaac = validate_direct_handoff_start(low)

    publisher = LowCommandPublisher()
    controller = ContinuousJointHold(
        publisher=publisher,
        target_isaac=captured_q_isaac,
        kp=args.kp,
        kd=args.kd,
    )
    controller_started = False
    released = False
    restored = False
    recovery_used_standdown = False
    outcome = "not_started"
    error: str | None = None
    peaks = {
        "maximum_joint_deviation_rad": 0.0,
        "maximum_joint_speed_rad_s": 0.0,
        "maximum_estimated_torque_nm": 0.0,
        "maximum_abs_roll_rad": 0.0,
        "maximum_abs_pitch_rad": 0.0,
        "maximum_lowstate_age_s": 0.0,
    }

    try:
        controller.start()
        controller_started = True
        time.sleep(0.10)
        release_sport_mode(switcher)
        released = True
        started_at = time.monotonic()
        while time.monotonic() - started_at < args.hold_seconds:
            controller.raise_if_failed()
            low, _sport_state = live.sample()
            values = validate_held_stance(
                low=low,
                captured_q_isaac=captured_q_isaac,
            )
            peaks["maximum_joint_deviation_rad"] = max(
                peaks["maximum_joint_deviation_rad"],
                values["maximum_joint_deviation_rad"],
            )
            peaks["maximum_joint_speed_rad_s"] = max(
                peaks["maximum_joint_speed_rad_s"],
                values["maximum_joint_speed_rad_s"],
            )
            peaks["maximum_estimated_torque_nm"] = max(
                peaks["maximum_estimated_torque_nm"],
                values["maximum_estimated_torque_nm"],
            )
            peaks["maximum_abs_roll_rad"] = max(
                peaks["maximum_abs_roll_rad"],
                abs(values["roll_rad"]),
            )
            peaks["maximum_abs_pitch_rad"] = max(
                peaks["maximum_abs_pitch_rad"],
                abs(values["pitch_rad"]),
            )
            peaks["maximum_lowstate_age_s"] = max(
                peaks["maximum_lowstate_age_s"],
                values["lowstate_age_s"],
            )
            time.sleep(0.01)
        outcome = "standing_handoff_held"
    except Exception as exc:
        error = str(exc)
        outcome = "aborted"
    finally:
        if released:
            # First try the desired standing-to-standing restoration while the
            # measured pose remains continuously supported.
            restored = _select_sport_mode(switcher, attempts=3)
            if not restored and controller_started:
                recovery_used_standdown = True
                try:
                    low, _sport_state = live.sample()
                    _lower_to_standdown(
                        controller=controller,
                        start_q_isaac=unitree_to_isaac(low["q"]),
                    )
                except Exception as recovery_exc:
                    error = (
                        f"{error or ''}; StandDown recovery failed: "
                        f"{recovery_exc}"
                    ).strip("; ")
                    outcome = "recovery_failed"
        if controller_started:
            try:
                controller.stop()
            except Exception as stop_exc:
                error = f"{error or ''}; publisher stop failed: {stop_exc}".strip("; ")
                outcome = "recovery_failed"
        publisher.close()
        if released and not restored:
            restored = _select_sport_mode(switcher)
        if restored and recovery_used_standdown:
            try:
                code = sport.StandDown()
                if code != 0:
                    raise RuntimeError(f"SportClient.StandDown returned {code}")
                time.sleep(0.75)
                code = sport.StandUp()
                if code != 0:
                    raise RuntimeError(f"SportClient.StandUp returned {code}")
                time.sleep(3.0)
            except Exception as settle_exc:
                error = f"{error or ''}; Sport settle failed: {settle_exc}".strip("; ")
                outcome = "recovery_failed"
        if released and not restored:
            error = f"{error or ''}; Sport mode restoration failed".strip("; ")
            outcome = "restore_failed"

    final_low, _sport_state = live.sample()
    report = {
        "outcome": outcome,
        "error": error,
        "hold_seconds": args.hold_seconds,
        "kp": args.kp,
        "kd": args.kd,
        "controller_publish_ticks": controller.ticks,
        "controller_restored": restored,
        "recovery_used_standdown": recovery_used_standdown,
        "final_rpy": final_low["rpy"],
        **{key: round(value, 5) for key, value in peaks.items()},
    }
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if outcome == "standing_handoff_held" and restored else 1


if __name__ == "__main__":
    raise SystemExit(main())
