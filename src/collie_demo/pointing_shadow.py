#!/usr/bin/env python3
"""Run the exported pointing actor against live Go2 state without motor output.

Safety invariant: this file imports no LowCmd message and creates no publisher.
It only subscribes to LowState/SportModeState, polls the existing Collie status
endpoint for a fruit box, and records what the actor *would* request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any
from urllib.request import urlopen

import torch

from .pointing_contract import (
    ISAAC_JOINT_ORDER,
    actor_action_to_joint_target,
    build_actor_observation,
    joint_limit_violations,
    projected_gravity_wxyz,
    selected_target_bbox_from_status,
    standdown_error_rad,
    unitree_to_isaac,
)


EXPECTED_POLICY_SHA256 = "5ac866353150b82309a083827aefd2f43e779a5ba67c8d617a5b612b89fe1938"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--network-interface", default="enP8p1s0")
    parser.add_argument("--status-url", default="http://127.0.0.1:8096/api/status")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--bbox-rate", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/go2_pointing_shadow.json"))
    parser.add_argument(
        "--synthetic-bbox",
        help="normalized x1,y1,x2,y2; marks the run synthetic instead of using Collie",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LiveState:
    def __init__(self, network_interface: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

        self._lock = Lock()
        self._low: dict[str, Any] | None = None
        self._sport: dict[str, Any] | None = None
        self._low_event = Event()
        self._sport_event = Event()

        ChannelFactoryInitialize(0, network_interface)
        self._low_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_subscriber.Init(self._on_lowstate, 10)
        self._sport_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._sport_subscriber.Init(self._on_sport_state, 10)

    def _on_lowstate(self, message: Any) -> None:
        now = time.monotonic()
        try:
            sample = {
                "received_at": now,
                "tick": int(message.tick),
                "quaternion": tuple(float(value) for value in message.imu_state.quaternion),
                "gyroscope": tuple(float(value) for value in message.imu_state.gyroscope),
                "rpy": tuple(float(value) for value in message.imu_state.rpy),
                "q": tuple(float(state.q) for state in message.motor_state[:12]),
                "dq": tuple(float(state.dq) for state in message.motor_state[:12]),
                "tau_est": tuple(float(state.tau_est) for state in message.motor_state[:12]),
            }
        except Exception:
            return
        with self._lock:
            self._low = sample
        self._low_event.set()

    def _on_sport_state(self, message: Any) -> None:
        now = time.monotonic()
        try:
            sample = {
                "received_at": now,
                "velocity": tuple(float(value) for value in message.velocity),
                "position": tuple(float(value) for value in message.position),
            }
        except Exception:
            return
        with self._lock:
            self._sport = sample
        self._sport_event.set()

    def wait(self, timeout: float = 5.0) -> None:
        if not self._low_event.wait(timeout):
            raise TimeoutError("no rt/lowstate sample")
        if not self._sport_event.wait(timeout):
            raise TimeoutError("no rt/sportmodestate sample for base linear velocity")

    def sample(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            if self._low is None or self._sport is None:
                raise RuntimeError("live state is incomplete")
            return dict(self._low), dict(self._sport)


class BBoxProvider:
    def __init__(
        self,
        *,
        status_url: str,
        rate_hz: float,
        synthetic_bbox: tuple[float, float, float, float] | None,
        maximum_age_s: float = 0.5,
        hold_last_valid_s: float = 0.0,
    ) -> None:
        self.status_url = status_url
        self.period_s = 1.0 / rate_hz
        self.synthetic_bbox = synthetic_bbox
        self.maximum_age_s = maximum_age_s
        self.hold_last_valid_s = hold_last_valid_s
        self._lock = Lock()
        self._sample: dict[str, Any] = {
            "bbox": synthetic_bbox or (0.0, 0.0, 0.0, 0.0),
            "label": "synthetic" if synthetic_bbox else None,
            "confidence": 1.0 if synthetic_bbox else None,
            "received_at": time.monotonic() if synthetic_bbox else None,
            "frame_age_s": 0.0 if synthetic_bbox else None,
            "error": None,
            "held": False,
        }
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self.synthetic_bbox is not None:
            return
        self._thread = Thread(target=self._run, name="bbox-poller", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sample(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._sample)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with urlopen(self.status_url, timeout=0.5) as response:
                    status = json.load(response)
                selected = selected_target_bbox_from_status(
                    status,
                    maximum_age_s=self.maximum_age_s,
                )
                sample = {
                    "bbox": selected["bbox"],
                    "label": selected["label"],
                    "confidence": selected["confidence"],
                    "lock_id": selected["lock_id"],
                    "received_at": time.monotonic(),
                    "frame_age_s": selected["frame_age_s"],
                    "error": None,
                    "held": False,
                }
            except Exception as exc:
                now = time.monotonic()
                with self._lock:
                    previous = dict(self._sample)
                previous_received_at = previous.get("received_at")
                if (
                    self.hold_last_valid_s > 0.0
                    and previous.get("label") is not None
                    and previous_received_at is not None
                    and now - float(previous_received_at) <= self.hold_last_valid_s
                ):
                    sample = previous
                    sample["error"] = None
                    sample["held"] = True
                else:
                    sample = {
                        "bbox": (0.0, 0.0, 0.0, 0.0),
                        "label": None,
                        "confidence": None,
                        "lock_id": None,
                        "received_at": now,
                        "frame_age_s": None,
                        "error": str(exc),
                        "held": False,
                    }
            with self._lock:
                self._sample = sample
            self._stop.wait(max(0.0, self.period_s - (time.monotonic() - started)))


def parse_synthetic_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    values = tuple(float(item) for item in value.split(","))
    if len(values) != 4:
        raise ValueError("--synthetic-bbox requires x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("--synthetic-bbox must be a positive normalized xyxy box")
    return values


def main() -> int:
    args = parse_args()
    if args.duration <= 0.0 or args.rate <= 0.0 or args.bbox_rate <= 0.0:
        raise ValueError("duration and rates must be positive")
    actual_sha = sha256_file(args.policy)
    if actual_sha != EXPECTED_POLICY_SHA256:
        raise RuntimeError(
            f"policy SHA-256 mismatch: {actual_sha} != {EXPECTED_POLICY_SHA256}"
        )

    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    # Pay TorchScript's one-time graph initialization cost before a timed
    # control loop.  The first cold inference on Woof measured about 100 ms;
    # warm calls are around 3 ms.
    with torch.inference_mode():
        warm_observation = torch.zeros((1, 49), dtype=torch.float32)
        for _ in range(50):
            policy(warm_observation)
    synthetic_bbox = parse_synthetic_bbox(args.synthetic_bbox)
    live = LiveState(args.network_interface)
    boxes = BBoxProvider(
        status_url=args.status_url,
        rate_hz=args.bbox_rate,
        synthetic_bbox=synthetic_bbox,
    )
    boxes.start()
    try:
        live.wait()
        tick_count = int(round(args.duration * args.rate))
        period_s = 1.0 / args.rate
        previous_action = (0.0,) * 12
        records: list[dict[str, Any]] = []
        deadline = time.perf_counter()

        with torch.inference_mode():
            for tick in range(tick_count):
                deadline += period_s
                low, sport = live.sample()
                box = boxes.sample()
                now = time.monotonic()
                q_isaac = unitree_to_isaac(low["q"])
                dq_isaac = unitree_to_isaac(low["dq"])
                gravity = projected_gravity_wxyz(low["quaternion"])
                observation = build_actor_observation(
                    base_linear_velocity_body=sport["velocity"],
                    base_angular_velocity_body=low["gyroscope"],
                    gravity_body=gravity,
                    joint_position_isaac=q_isaac,
                    joint_velocity_isaac=dq_isaac,
                    previous_action=previous_action,
                    bbox_xyxy_normalized=box["bbox"],
                )

                inference_started = time.perf_counter()
                output = policy(torch.tensor(observation).unsqueeze(0)).squeeze(0)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                action = tuple(float(value) for value in output)
                if len(action) != 12 or not all(math.isfinite(value) for value in action):
                    raise RuntimeError("actor returned an invalid action")
                target = actor_action_to_joint_target(action)
                violations = joint_limit_violations(target)
                standdown_rms, standdown_max = standdown_error_rad(q_isaac)
                records.append(
                    {
                        "tick": tick,
                        "inference_ms": inference_ms,
                        "lowstate_age_s": now - float(low["received_at"]),
                        "sportstate_age_s": now - float(sport["received_at"]),
                        "bbox_age_s": None
                        if box["received_at"] is None
                        else now - float(box["received_at"]) + float(box["frame_age_s"] or 0.0),
                        "bbox": box["bbox"],
                        "label": box["label"],
                        "confidence": box["confidence"],
                        "target_lock_id": box.get("lock_id"),
                        "bbox_error": box["error"],
                        "q_isaac": q_isaac,
                        "dq_isaac": dq_isaac,
                        "rpy": low["rpy"],
                        "action": action,
                        "raw_target_isaac": target,
                        "joint_limit_violations": violations,
                        "standdown_rms_error_rad": standdown_rms,
                        "standdown_max_error_rad": standdown_max,
                        "max_joint_speed_rad_s": max(abs(value) for value in dq_isaac),
                    }
                )
                previous_action = action
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

        elapsed_s = records[-1]["tick"] / args.rate + period_s if records else 0.0
        all_violations = sorted(
            {
                violation
                for record in records
                for violation in record["joint_limit_violations"]
            }
        )
        latest = records[-1]
        summary = {
            "mode": "SHADOW_NO_MOTOR_PUBLISHER",
            "policy_sha256": actual_sha,
            "ticks": len(records),
            "requested_rate_hz": args.rate,
            "nominal_elapsed_s": elapsed_s,
            "inference_ms_mean": sum(item["inference_ms"] for item in records)
            / max(1, len(records)),
            "inference_ms_max": max(item["inference_ms"] for item in records),
            "lowstate_age_s_max": max(item["lowstate_age_s"] for item in records),
            "sportstate_age_s_max": max(item["sportstate_age_s"] for item in records),
            "bbox_source": "synthetic" if synthetic_bbox is not None else args.status_url,
            "bbox_seen_ticks": sum(
                1 for item in records if any(float(value) != 0.0 for value in item["bbox"])
            ),
            "latest_label": latest["label"],
            "latest_confidence": latest["confidence"],
            "latest_bbox": latest["bbox"],
            "latest_standdown_rms_error_rad": latest["standdown_rms_error_rad"],
            "latest_standdown_max_error_rad": latest["standdown_max_error_rad"],
            "latest_max_joint_speed_rad_s": latest["max_joint_speed_rad_s"],
            "raw_target_limit_violation_ticks": sum(
                1 for item in records if item["joint_limit_violations"]
            ),
            "distinct_raw_target_limit_violations": all_violations,
            "ready_for_motor_output": False,
            "blockers": [
                "shadow runner intentionally has no motor publisher",
                "raw actor targets must be clamped and rate-limited",
                "hardware gains and Sport-to-low-level handoff are not validated",
            ],
            "joint_order": list(ISAAC_JOINT_ORDER),
        }
        report = {"summary": summary, "records": records}
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        boxes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
