"""Process boundary for Woof's guarded bounding-box pointing policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any
import uuid

from .pointing_shadow import EXPECTED_POLICY_SHA256, sha256_file


POINTING_PREPARE_CONFIRMATION = "WOOF IS CLEAR FOR STANDING POINT"
POINTING_RUN_CONFIRMATION = "AREA IS CLEAR AND WOOF MAY MOVE"


class PointingPolicyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PointingPolicyConfig:
    enabled: bool
    policy_path: Path
    network_interface: str = "enP8p1s0"
    status_url: str = "http://127.0.0.1:8096/api/status"
    duration_s: float = 6.0
    action_gain: float = 1.0
    maximum_target_rate_rad_s: float = 0.60
    kp: float = 60.0
    kd: float = 5.0
    standup_seconds: float = 3.0
    standup_kp: float = 60.0
    standup_kd: float = 5.0
    direct_standing_handoff: bool = True
    output_directory: Path = Path("/tmp/collie-pointing")
    runner_module: str = "collie_demo.pointing_runner"

    def __post_init__(self) -> None:
        if not (2.0 <= self.duration_s <= 8.0):
            raise ValueError("standing-point duration must be between 2.0 and 8.0 s")
        if not (0.0 < self.action_gain <= 1.0):
            raise ValueError("pointing action gain must be in (0, 1]")
        if not (0.05 <= self.maximum_target_rate_rad_s <= 0.60):
            raise ValueError("pointing target rate must be in [0.05, 0.60] rad/s")
        if not (0.0 < self.kp <= 60.0 and 0.0 < self.kd <= 5.0):
            raise ValueError("pointing gains exceed the validated envelope")
        if not (2.0 <= self.standup_seconds <= 5.0):
            raise ValueError("standing-point lift must take between 2.0 and 5.0 s")
        if not (
            0.0 < self.standup_kp <= 60.0
            and 0.0 < self.standup_kd <= 5.0
        ):
            raise ValueError("standing-point lift gains exceed the validated envelope")


class PointingPolicyManager:
    """Launch one isolated policy run and expose concise UI telemetry.

    The motor publisher remains inside ``pointing_runner``. This manager owns
    its lifecycle, never starts two runs, and only requests an interrupt that
    lets the runner execute its guarded pose/controller restoration.
    """

    def __init__(self, config: PointingPolicyConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._phase = "idle"
        self._target_label: str | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._prepared_at: float | None = None
        self._stop_requested = False
        self._last_error: str | None = None
        self._last_report: dict[str, Any] | None = None
        self._stdout_tail = ""
        self._stderr_tail = ""
        self._policy_sha256: str | None = None
        self._configuration_error: str | None = None
        if self.config.enabled:
            try:
                if not self.config.policy_path.is_file():
                    raise FileNotFoundError(
                        f"policy not found: {self.config.policy_path}"
                    )
                self._policy_sha256 = sha256_file(self.config.policy_path)
                if self._policy_sha256 != EXPECTED_POLICY_SHA256:
                    raise ValueError(
                        "policy SHA-256 mismatch: "
                        f"{self._policy_sha256} != {EXPECTED_POLICY_SHA256}"
                    )
            except Exception as exc:
                self._configuration_error = str(exc)

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def available(self) -> bool:
        return bool(
            self.config.enabled
            and self._configuration_error is None
            and self._policy_sha256 == EXPECTED_POLICY_SHA256
        )

    def mark_prepared(self) -> None:
        if self.active:
            raise PointingPolicyError("pointing policy is already active")
        self._prepared_at = time.monotonic()
        self._phase = "prepared"
        self._last_error = None

    def clear_prepared(self) -> None:
        self._prepared_at = None
        if not self.active and self._phase == "prepared":
            self._phase = "idle"

    def status(self) -> dict[str, object]:
        now = time.monotonic()
        started_age = (
            None if self._started_at is None else max(0.0, now - self._started_at)
        )
        prepared_age = (
            None if self._prepared_at is None else max(0.0, now - self._prepared_at)
        )
        return {
            "available": self.available,
            "enabled": self.config.enabled,
            "active": self.active,
            "prepared": self._prepared_at is not None,
            "phase": self._phase,
            "target_label": self._target_label,
            "duration_s": self.config.duration_s,
            "policy_kind": "locked_point_v19",
            "gesture": "standing_front_right_point",
            "handoff_mode": (
                "sport_standing_direct"
                if self.config.direct_standing_handoff
                else "standdown_then_lift"
            ),
            "policy_rate_hz": 50,
            "motor_publish_rate_hz": 500,
            "started_age_s": None
            if started_age is None
            else round(started_age, 2),
            "prepared_age_s": None
            if prepared_age is None
            else round(prepared_age, 2),
            "stop_requested": self._stop_requested,
            "policy_sha256": self._policy_sha256,
            "expected_policy_sha256": EXPECTED_POLICY_SHA256,
            "error": self._configuration_error or self._last_error,
            "last_report": self._last_report,
            "stdout_tail": self._stdout_tail,
            "stderr_tail": self._stderr_tail,
            "safety": {
                "roll_guard_rad": 0.30,
                "pitch_guard_rad": 0.35,
                "maximum_estimated_torque": 22.0,
                "maximum_joint_speed_rad_s": 4.0,
                "maximum_target_rate_rad_s": (
                    self.config.maximum_target_rate_rad_s
                ),
                "stale_bbox_abort_s": 0.8,
                "bypass_available": False,
            },
        }

    async def start(
        self,
        *,
        target_label: str,
        minimum_confidence: float,
    ) -> dict[str, object]:
        target_label = target_label.strip().lower()
        if target_label not in {"apple", "banana", "pear"}:
            raise PointingPolicyError("pointing target must be apple, banana, or pear")
        if not math.isfinite(minimum_confidence) or not (
            0.0 < minimum_confidence <= 1.0
        ):
            raise PointingPolicyError("minimum confidence must be in (0, 1]")
        async with self._lock:
            if not self.available:
                raise PointingPolicyError(
                    self._configuration_error or "pointing policy is unavailable"
                )
            if self.active:
                raise PointingPolicyError("pointing policy is already active")
            if self._prepared_at is None:
                raise PointingPolicyError(
                    "prepare the guarded standing-point handoff first"
                )
            self._target_label = target_label
            self._phase = "starting"
            self._started_at = time.monotonic()
            self._finished_at = None
            self._stop_requested = False
            self._last_error = None
            self._last_report = None
            self._stdout_tail = ""
            self._stderr_tail = ""
            self._task = asyncio.create_task(
                self._run(target_label, minimum_confidence),
                name="collie-pointing-policy",
            )
        return self.status()

    async def stop(self) -> dict[str, object]:
        async with self._lock:
            task = self._task
            process = self._process
            if task is None or task.done():
                return self.status()
            self._stop_requested = True
            self._phase = "stopping_and_restoring"
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
        except TimeoutError:
            # Never SIGKILL a runner that may own low-level motor control.
            # Leave it alive to finish the guarded return and mcf restore.
            self._last_error = "stop requested; controller restoration is still running"
        return self.status()

    async def wait(self, *, timeout_s: float = 20.0) -> dict[str, object]:
        """Wait for a run without cancelling the motor owner on timeout."""

        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("pointing wait timeout must be positive")
        task = self._task
        if task is None:
            raise PointingPolicyError("pointing policy has not been started")
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except TimeoutError as exc:
            raise PointingPolicyError(
                "pointing policy is still running; request Stop and wait for restoration"
            ) from exc
        return self.status()

    async def close(self) -> None:
        await self.stop()
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
            except TimeoutError:
                self._last_error = (
                    "shutdown is waiting for pointing controller restoration"
                )

    def _build_command(
        self,
        *,
        target_label: str,
        minimum_confidence: float,
        output_path: Path,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            self.config.runner_module,
            "--policy",
            str(self.config.policy_path),
            "--network-interface",
            self.config.network_interface,
            "--status-url",
            self.config.status_url,
            "--target-label",
            target_label,
            "--minimum-confidence",
            f"{minimum_confidence:.6f}",
            "--duration",
            f"{self.config.duration_s:.3f}",
            "--action-gain",
            f"{self.config.action_gain:.3f}",
            "--max-rate-rad-s",
            f"{self.config.maximum_target_rate_rad_s:.3f}",
            "--kp",
            f"{self.config.kp:.3f}",
            "--kd",
            f"{self.config.kd:.3f}",
            "--standup-seconds",
            f"{self.config.standup_seconds:.3f}",
            "--standup-kp",
            f"{self.config.standup_kp:.3f}",
            "--standup-kd",
            f"{self.config.standup_kd:.3f}",
            "--output",
            str(output_path),
            "--full-power",
            "--execute",
            "--confirm",
            "AREA IS CLEAR AND WOOF MAY MOVE",
        ]
        if self.config.direct_standing_handoff:
            command.append("--direct-standing-handoff")
        return command

    async def _run(self, target_label: str, minimum_confidence: float) -> None:
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_directory / (
            f"pointing-{int(time.time())}-{uuid.uuid4().hex[:8]}.json"
        )
        command = self._build_command(
            target_label=target_label,
            minimum_confidence=minimum_confidence,
            output_path=output_path,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            async with self._lock:
                self._process = process
                stop_requested = self._stop_requested
                self._phase = (
                    "stopping_and_restoring" if stop_requested else "running"
                )
                if stop_requested and process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
            stdout, stderr = await process.communicate()
            self._stdout_tail = stdout.decode("utf-8", errors="replace")[-2000:]
            self._stderr_tail = stderr.decode("utf-8", errors="replace")[-2000:]
            report = self._read_report(output_path)
            self._last_report = self._summarize_report(report)
            outcome = str(report.get("outcome") or "")
            controller_restored = bool(report.get("controller_restored"))
            if (
                process.returncode == 0
                and outcome == "completed_full_policy"
                and controller_restored
            ):
                self._phase = "complete"
                self._last_error = None
            else:
                self._phase = (
                    "restore_failed"
                    if not controller_restored
                    else "aborted_safely"
                )
                self._last_error = str(
                    report.get("error")
                    or self._stderr_tail
                    or f"pointing runner exited {process.returncode}"
                )
        except Exception as exc:
            self._phase = "fault"
            self._last_error = str(exc)
        finally:
            self._finished_at = time.monotonic()
            self._prepared_at = None
            async with self._lock:
                self._process = None

    @staticmethod
    def _read_report(output_path: Path) -> dict[str, Any]:
        if not output_path.is_file():
            raise PointingPolicyError("pointing runner produced no safety report")
        rendered = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(rendered, dict):
            raise PointingPolicyError("pointing safety report is invalid")
        return rendered

    @staticmethod
    def _summarize_report(report: dict[str, Any]) -> dict[str, object]:
        records = report.get("records")
        if not isinstance(records, list):
            records = []
        rolls = [
            abs(float(record["rpy"][0]))
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("rpy"), (list, tuple))
            and record["rpy"]
        ]
        speeds = [
            float(record["max_joint_speed_rad_s"])
            for record in records
            if isinstance(record, dict)
            and record.get("max_joint_speed_rad_s") is not None
        ]
        torques = [
            float(record["max_estimated_torque"])
            for record in records
            if isinstance(record, dict)
            and record.get("max_estimated_torque") is not None
        ]
        confidences = [
            float(record["confidence"])
            for record in records
            if isinstance(record, dict) and record.get("confidence") is not None
        ]
        return {
            "outcome": report.get("outcome"),
            "error": report.get("error"),
            "target_label": report.get("selected_label"),
            "policy_ticks": report.get("policy_ticks"),
            "requested_duration_s": report.get("requested_duration_s"),
            "controller_restored": report.get("controller_restored"),
            "handoff_mode": report.get("handoff_mode"),
            "recovery_used_standdown": report.get(
                "recovery_used_standdown"
            ),
            "final_standing_rms_error_rad": report.get(
                "final_standing_rms_error_rad"
            ),
            "final_standdown_rms_error_rad": report.get(
                "final_standdown_rms_error_rad"
            ),
            "maximum_roll_rad": None if not rolls else round(max(rolls), 4),
            "maximum_joint_speed_rad_s": (
                None if not speeds else round(max(speeds), 4)
            ),
            "maximum_estimated_torque": (
                None if not torques else round(max(torques), 4)
            ),
            "peak_estimated_torque_nm": report.get(
                "peak_estimated_torque_nm"
            ),
            "worst_droop_outside_envelope_rad": report.get(
                "worst_droop_outside_envelope_rad"
            ),
            "confidence_range": (
                None
                if not confidences
                else [round(min(confidences), 4), round(max(confidences), 4)]
            ),
        }
