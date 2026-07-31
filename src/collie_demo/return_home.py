"""Pure short-range odometry return planner for the stage demo.

The planner consumes a captured start pose and fresh local odometry.  It does
not own motors: runtime.py maps its four states onto the existing exclusive
direct-yaw and obstacle-avoidance motion leases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .heading import normalize_angle


@dataclass(frozen=True, slots=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.x_m, self.y_m, self.yaw_rad)
        ):
            raise ValueError("pose values must be finite")


@dataclass(frozen=True, slots=True)
class PoseWindowAssessment:
    pose: Pose2D
    sample_count: int
    maximum_position_span_m: float
    maximum_yaw_span_rad: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_count": self.sample_count,
            "maximum_position_span_m": round(
                self.maximum_position_span_m,
                4,
            ),
            "maximum_yaw_span_deg": round(
                math.degrees(self.maximum_yaw_span_rad),
                2,
            ),
        }


class ReturnMode(str, Enum):
    TURN_TO_HOME = "turn_to_home"
    DRIVE_TO_HOME = "drive_to_home"
    RESTORE_HEADING = "restore_heading"
    COMPLETE = "complete"


@dataclass(slots=True)
class ReturnTurnDirectionLatch:
    """Keep a near-180-degree turn from changing direction under yaw noise.

    The normalized shortest-path error is discontinuous at +/-pi. A target
    directly behind the robot can therefore alternate between positive and
    negative errors before the robot has made visible progress. This latch
    preserves the initial direction only while that ambiguity exists, then
    returns to ordinary shortest-path steering.
    """

    target_yaw_rad: float
    last_yaw_rad: float
    direction: float
    release_progress_rad: float
    active: bool
    directed_progress_rad: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        target_yaw_rad: float,
        current_yaw_rad: float,
        release_progress_rad: float,
        activation_margin_rad: float,
    ) -> "ReturnTurnDirectionLatch":
        values = (
            target_yaw_rad,
            current_yaw_rad,
            release_progress_rad,
            activation_margin_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("turn-direction latch values must be finite")
        if release_progress_rad <= 0.0:
            raise ValueError("turn-direction release progress must be positive")
        if not 0.0 < activation_margin_rad < math.pi:
            raise ValueError("turn-direction activation margin is invalid")
        shortest_error = normalize_angle(target_yaw_rad - current_yaw_rad)
        return cls(
            target_yaw_rad=target_yaw_rad,
            last_yaw_rad=current_yaw_rad,
            direction=1.0 if shortest_error >= 0.0 else -1.0,
            release_progress_rad=release_progress_rad,
            active=(
                abs(shortest_error)
                >= math.pi - activation_margin_rad
            ),
        )

    def update(self, current_yaw_rad: float) -> float:
        """Return the control error while updating measured turn progress."""

        if not math.isfinite(current_yaw_rad):
            raise ValueError("current yaw must be finite")
        yaw_step_rad = normalize_angle(current_yaw_rad - self.last_yaw_rad)
        self.last_yaw_rad = current_yaw_rad
        if self.active:
            self.directed_progress_rad = max(
                0.0,
                self.directed_progress_rad
                + self.direction * yaw_step_rad,
            )
            if self.directed_progress_rad >= self.release_progress_rad:
                self.active = False

        shortest_error = normalize_angle(
            self.target_yaw_rad - current_yaw_rad
        )
        if not self.active or shortest_error == 0.0:
            return shortest_error
        if self.direction > 0.0:
            return (
                self.target_yaw_rad - current_yaw_rad
            ) % (2.0 * math.pi)
        return -(
            (current_yaw_rad - self.target_yaw_rad)
            % (2.0 * math.pi)
        )


@dataclass(frozen=True, slots=True)
class ReturnStep:
    mode: ReturnMode
    distance_m: float
    target_yaw_rad: float
    heading_error_rad: float
    forward_mps: float
    yaw_rps: float


def assess_pose_window(samples: list[Pose2D]) -> PoseWindowAssessment:
    """Summarize a stationary odometry window and its averaged pose."""

    if len(samples) < 2:
        raise ValueError("at least two pose samples are required")
    maximum_position_span_m = 0.0
    maximum_yaw_span_rad = 0.0
    for index, first in enumerate(samples):
        for second in samples[index + 1 :]:
            maximum_position_span_m = max(
                maximum_position_span_m,
                math.hypot(
                    second.x_m - first.x_m,
                    second.y_m - first.y_m,
                ),
            )
            maximum_yaw_span_rad = max(
                maximum_yaw_span_rad,
                abs(normalize_angle(second.yaw_rad - first.yaw_rad)),
            )
    mean_x = sum(sample.x_m for sample in samples) / len(samples)
    mean_y = sum(sample.y_m for sample in samples) / len(samples)
    mean_yaw = math.atan2(
        sum(math.sin(sample.yaw_rad) for sample in samples),
        sum(math.cos(sample.yaw_rad) for sample in samples),
    )
    return PoseWindowAssessment(
        pose=Pose2D(mean_x, mean_y, mean_yaw),
        sample_count=len(samples),
        maximum_position_span_m=maximum_position_span_m,
        maximum_yaw_span_rad=maximum_yaw_span_rad,
    )


@dataclass(frozen=True, slots=True)
class ReturnPlannerConfig:
    arrival_tolerance_m: float
    heading_tolerance_rad: float
    heading_gate_rad: float
    maximum_forward_mps: float
    yaw_gain: float
    maximum_yaw_rps: float
    minimum_forward_mps: float = 0.06

    def __post_init__(self) -> None:
        finite = (
            self.arrival_tolerance_m,
            self.heading_tolerance_rad,
            self.heading_gate_rad,
            self.maximum_forward_mps,
            self.yaw_gain,
            self.maximum_yaw_rps,
            self.minimum_forward_mps,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("return planner configuration must be finite")
        if self.arrival_tolerance_m <= 0.0:
            raise ValueError("arrival tolerance must be positive")
        if not 0.0 < self.heading_tolerance_rad <= self.heading_gate_rad:
            raise ValueError("heading tolerances are inconsistent")
        if self.maximum_forward_mps <= 0.0 or self.maximum_yaw_rps <= 0.0:
            raise ValueError("return speed limits must be positive")
        if not 0.0 < self.minimum_forward_mps <= self.maximum_forward_mps:
            raise ValueError("minimum return speed is inconsistent")


def plan_return_step(
    *,
    home: Pose2D,
    current: Pose2D,
    config: ReturnPlannerConfig,
) -> ReturnStep:
    """Plan one closed-loop return step from current odometry."""

    dx = home.x_m - current.x_m
    dy = home.y_m - current.y_m
    distance = math.hypot(dx, dy)

    if distance <= config.arrival_tolerance_m:
        heading_error = normalize_angle(home.yaw_rad - current.yaw_rad)
        mode = (
            ReturnMode.COMPLETE
            if abs(heading_error) <= config.heading_tolerance_rad
            else ReturnMode.RESTORE_HEADING
        )
        return ReturnStep(
            mode=mode,
            distance_m=distance,
            target_yaw_rad=home.yaw_rad,
            heading_error_rad=heading_error,
            forward_mps=0.0,
            yaw_rps=_bounded_yaw(heading_error, config),
        )

    target_yaw = math.atan2(dy, dx)
    heading_error = normalize_angle(target_yaw - current.yaw_rad)
    if abs(heading_error) > config.heading_gate_rad:
        return ReturnStep(
            mode=ReturnMode.TURN_TO_HOME,
            distance_m=distance,
            target_yaw_rad=target_yaw,
            heading_error_rad=heading_error,
            forward_mps=0.0,
            yaw_rps=_bounded_yaw(heading_error, config),
        )

    remaining = max(0.0, distance - config.arrival_tolerance_m)
    forward = min(
        config.maximum_forward_mps,
        max(config.minimum_forward_mps, remaining * 0.8),
    ) * max(0.25, math.cos(heading_error))
    return ReturnStep(
        mode=ReturnMode.DRIVE_TO_HOME,
        distance_m=distance,
        target_yaw_rad=target_yaw,
        heading_error_rad=heading_error,
        forward_mps=forward,
        yaw_rps=_bounded_yaw(heading_error, config),
    )


def _bounded_yaw(
    heading_error_rad: float,
    config: ReturnPlannerConfig,
) -> float:
    requested = config.yaw_gain * heading_error_rad
    return max(
        -config.maximum_yaw_rps,
        min(config.maximum_yaw_rps, requested),
    )
