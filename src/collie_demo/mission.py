"""Configuration and telemetry for the learn-turn-find-approach demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class MissionPhase(str, Enum):
    IDLE = "idle"
    LEARNING = "learning"
    MEMORIZED = "memorized"
    TURNING = "turning"
    SEARCHING = "searching"
    ACKNOWLEDGING = "acknowledging"
    WAITING_FOR_GO = "waiting_for_go"
    CONFIRMING = "confirming"
    APPROACHING = "approaching"
    FINAL_APPROACHING = "final_approaching"
    POINTING = "pointing"
    CELEBRATING = "celebrating"
    RETURNING_HOME = "returning_home"
    SUCCESS = "success"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class MissionConfig:
    enabled: bool = False
    autonomous_turn_enabled: bool = False
    direct_turn_enabled: bool = False
    initial_hello_enabled: bool = False
    match_stretch_enabled: bool = False
    match_stretch_settle_s: float = 3.5
    match_reacquire_timeout_s: float = 3.0
    arrival_hello_enabled: bool = False
    arrival_hello_settle_s: float = 0.35
    arrival_rest_enabled: bool = False
    arrival_rest_duration_s: float = 5.0
    arrival_pointing_enabled: bool = False
    arrival_pointing_label: str = "all"
    arrival_pointing_timeout_s: float = 20.0
    return_home_enabled: bool = False
    return_backend: str = "local_odometry"
    nav2_poll_period_s: float = 0.10
    return_arrival_tolerance_m: float = 0.25
    return_heading_tolerance_rad: float = math.radians(10.0)
    return_heading_gate_rad: float = math.radians(30.0)
    return_forward_mps: float = 0.20
    return_yaw_gain: float = 1.2
    return_turn_minimum_yaw_rps: float = 0.35
    return_turn_response_timeout_s: float = 0.75
    return_turn_response_min_progress_rad: float = math.radians(2.0)
    return_turn_recovery_settle_s: float = 1.0
    return_timeout_s: float = 20.0
    return_stall_timeout_s: float = 3.0
    return_stall_min_progress_m: float = 0.06
    return_pose_capture_duration_s: float = 0.40
    return_pose_capture_max_drift_m: float = 0.04
    return_pose_capture_max_yaw_drift_rad: float = math.radians(4.0)
    return_pose_settle_duration_s: float = 0.50
    return_pose_settle_timeout_s: float = 3.0
    return_pose_settle_max_drift_m: float = 0.01
    return_pose_settle_max_yaw_drift_rad: float = math.radians(2.0)
    return_max_distance_m: float = 3.0
    return_max_odometry_step_m: float = 0.15
    return_max_odometry_yaw_step_rad: float = math.radians(30.0)
    capture_timeout_s: float = 2.0
    match_confirmations_required: int = 3
    approach_misses_allowed: int = 2
    approach_reacquire_attempts: int = 3
    approach_reacquire_timeout_s: float = 10.0
    turn_angle_rad: float = math.pi
    turn_rate_rps: float = 0.30
    turn_tolerance_rad: float = math.radians(7.0)
    turn_timeout_s: float = 15.0
    turn_stall_timeout_s: float = 2.0
    turn_stall_min_progress_rad: float = math.radians(10.0)
    search_rate_rps: float = 0.20
    search_sweep_rad: float = math.radians(75.0)
    search_timeout_s: float = 9.0
    near_bottom_ratio: float = 0.86
    near_center_ratio: float = 0.72
    near_bbox_height_ratio: float = 0.15
    near_confirmations_required: int = 3
    near_loss_grace_s: float = 0.75
    final_approach_distance_m: float = 0.10
    final_approach_mps: float = 0.10
    final_approach_timeout_s: float = 3.0
    final_approach_stall_timeout_s: float = 1.0
    final_approach_stall_min_progress_m: float = 0.001

    def __post_init__(self) -> None:
        if self.capture_timeout_s <= 0.0:
            raise ValueError("capture_timeout_s must be positive")
        if self.match_confirmations_required < 1:
            raise ValueError("match_confirmations_required must be positive")
        if self.approach_misses_allowed < 1:
            raise ValueError("approach_misses_allowed must be positive")
        if self.approach_reacquire_attempts < 1:
            raise ValueError("approach_reacquire_attempts must be positive")
        if self.near_confirmations_required < 1:
            raise ValueError("near_confirmations_required must be positive")
        for name in (
            "near_bottom_ratio",
            "near_center_ratio",
            "near_bbox_height_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name in (
            "turn_angle_rad",
            "turn_rate_rps",
            "turn_tolerance_rad",
            "turn_timeout_s",
            "turn_stall_timeout_s",
            "turn_stall_min_progress_rad",
            "search_rate_rps",
            "search_sweep_rad",
            "search_timeout_s",
            "match_reacquire_timeout_s",
            "approach_reacquire_timeout_s",
            "nav2_poll_period_s",
            "return_arrival_tolerance_m",
            "return_heading_tolerance_rad",
            "return_heading_gate_rad",
            "return_forward_mps",
            "return_yaw_gain",
            "return_turn_minimum_yaw_rps",
            "return_turn_response_timeout_s",
            "return_turn_response_min_progress_rad",
            "return_turn_recovery_settle_s",
            "return_timeout_s",
            "return_stall_timeout_s",
            "return_stall_min_progress_m",
            "return_pose_capture_duration_s",
            "return_pose_capture_max_drift_m",
            "return_pose_capture_max_yaw_drift_rad",
            "return_pose_settle_duration_s",
            "return_pose_settle_timeout_s",
            "return_pose_settle_max_drift_m",
            "return_pose_settle_max_yaw_drift_rad",
            "return_max_distance_m",
            "return_max_odometry_step_m",
            "return_max_odometry_yaw_step_rad",
            "near_loss_grace_s",
            "final_approach_distance_m",
            "final_approach_mps",
            "final_approach_timeout_s",
            "final_approach_stall_timeout_s",
            "final_approach_stall_min_progress_m",
            "arrival_pointing_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.match_stretch_settle_s)
            or self.match_stretch_settle_s < 0.0
        ):
            raise ValueError("match_stretch_settle_s must be non-negative")
        if (
            not math.isfinite(self.arrival_hello_settle_s)
            or self.arrival_hello_settle_s < 0.0
        ):
            raise ValueError("arrival_hello_settle_s must be non-negative")
        if (
            not math.isfinite(self.arrival_rest_duration_s)
            or self.arrival_rest_duration_s <= 0.0
        ):
            raise ValueError("arrival_rest_duration_s must be positive")
        if self.final_approach_distance_m > 0.30:
            raise ValueError("final_approach_distance_m cannot exceed 0.30 m")
        if self.arrival_pointing_label.strip().lower() not in {
            "all",
            "apple",
            "banana",
            "pear",
        }:
            raise ValueError(
                "arrival_pointing_label must be all, apple, banana, or pear"
            )
        if self.return_backend not in {"local_odometry", "nav2"}:
            raise ValueError(
                "return_backend must be local_odometry or nav2"
            )


@dataclass(slots=True)
class MissionTelemetry:
    phase: MissionPhase = MissionPhase.IDLE
    reason: str = "idle"
    started_monotonic_s: float | None = None
    turn_progress_rad: float = 0.0
    match_confirmations: int = 0
    match_failures: int = 0
    last_match: dict[str, object] | None = None
    near_target_seen: bool = False
    near_target_recent: bool = False
    near_target_confirmations: int = 0
    near_target_bbox_height_ratio: float | None = None
    approach_pause_count: int = 0
    approach_pause_status: str = "not_requested"
    final_approach_status: str = "not_requested"
    final_approach_elapsed_s: float = 0.0
    final_approach_commanded_distance_m: float = 0.0
    final_approach_measured_distance_m: float = 0.0
    initial_hello_status: str = "not_requested"
    initial_hello_error: str | None = None
    match_stretch_status: str = "not_requested"
    match_stretch_error: str | None = None
    arrival_hello_status: str = "not_requested"
    arrival_hello_error: str | None = None
    arrival_rest_status: str = "not_requested"
    arrival_rest_error: str | None = None
    arrival_pointing_status: str = "not_requested"
    arrival_pointing_error: str | None = None
    arrival_pointing_target: str | None = None
    contact_status: str = "not_requested"
    search_progress_rad: float = 0.0
    home_pose: dict[str, float | str] | None = None
    home_pose_validation: dict[str, float | int | str] | None = None
    return_backend: str = "local_odometry"
    nav2_status: dict[str, object] | None = None
    return_home_status: str = "not_requested"
    return_distance_m: float | None = None
    return_heading_error_rad: float | None = None
    return_progress_m: float = 0.0
    return_turn_status: str = "not_requested"
    return_turn_recovery_count: int = 0

    def to_dict(self, now: float) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "reason": self.reason,
            "elapsed_s": None
            if self.started_monotonic_s is None
            else round(max(0.0, now - self.started_monotonic_s), 3),
            "turn_progress_deg": round(math.degrees(self.turn_progress_rad), 1),
            "match_confirmations": self.match_confirmations,
            "match_failures": self.match_failures,
            "last_match": self.last_match,
            "near_target_seen": self.near_target_seen,
            "near_target_recent": self.near_target_recent,
            "near_target_confirmations": self.near_target_confirmations,
            "near_target_bbox_height_ratio": None
            if self.near_target_bbox_height_ratio is None
            else round(self.near_target_bbox_height_ratio, 3),
            "approach_pause_count": self.approach_pause_count,
            "approach_pause_status": self.approach_pause_status,
            "final_approach_status": self.final_approach_status,
            "final_approach_elapsed_s": round(self.final_approach_elapsed_s, 3),
            "final_approach_commanded_distance_m": round(
                self.final_approach_commanded_distance_m, 3
            ),
            "final_approach_measured_distance_m": round(
                self.final_approach_measured_distance_m, 3
            ),
            "initial_hello_status": self.initial_hello_status,
            "initial_hello_error": self.initial_hello_error,
            "match_stretch_status": self.match_stretch_status,
            "match_stretch_error": self.match_stretch_error,
            "arrival_hello_status": self.arrival_hello_status,
            "arrival_hello_error": self.arrival_hello_error,
            "arrival_rest_status": self.arrival_rest_status,
            "arrival_rest_error": self.arrival_rest_error,
            "arrival_pointing_status": self.arrival_pointing_status,
            "arrival_pointing_error": self.arrival_pointing_error,
            "arrival_pointing_target": self.arrival_pointing_target,
            "contact_status": self.contact_status,
            "search_progress_deg": round(math.degrees(self.search_progress_rad), 1),
            "home_pose": self.home_pose,
            "home_pose_validation": self.home_pose_validation,
            "return_backend": self.return_backend,
            "nav2_status": self.nav2_status,
            "return_home_status": self.return_home_status,
            "return_distance_m": None
            if self.return_distance_m is None
            else round(self.return_distance_m, 3),
            "return_heading_error_deg": None
            if self.return_heading_error_rad is None
            else round(math.degrees(self.return_heading_error_rad), 1),
            "return_progress_m": round(self.return_progress_m, 3),
            "return_turn_status": self.return_turn_status,
            "return_turn_recovery_count": self.return_turn_recovery_count,
        }
