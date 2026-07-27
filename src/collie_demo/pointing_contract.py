"""Pure-Python contracts for a guarded Go2 pointing-policy adapter.

This module deliberately has no Unitree SDK or Torch dependency.  It contains
the ordering and observation math shared by the read-only shadow runner and a
future real-motor runner.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ISAAC_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

UNITREE_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

# For each Isaac-order value, the motor index in Unitree LowState/LowCmd.
ISAAC_TO_UNITREE_MOTOR_INDEX = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

# This is the actual default used in Isaac during training.  The calf values
# were clipped to the Isaac/MuJoCo joint limit; it intentionally differs from
# Woof's measured built-in Sit values below.
ISAAC_GROUNDED_SIT_RAD = (
    0.0716288675,
    1.2460068195,
    -2.70,
    -0.0683550522,
    1.2439890466,
    -2.70,
    0.4025715036,
    1.2751952262,
    -2.70,
    -0.3932350988,
    1.2788276997,
    -2.70,
)

WOOF_BUILTIN_STANDDOWN_RAD = (
    0.0716288675,
    1.2460068195,
    -2.7855600319,
    -0.0683550522,
    1.2439890466,
    -2.7789528589,
    0.4025715036,
    1.2751952262,
    -2.7940665855,
    -0.3932350988,
    1.2788276997,
    -2.7953930345,
)

# Compatibility alias for older reports written before the built-in action was
# identified correctly.  These angles are from SportClient.StandDown(), not
# SportClient.Sit().
WOOF_BUILTIN_SIT_RAD = WOOF_BUILTIN_STANDDOWN_RAD

# Limits from the Unitree Go2 model used for the contact calibration.  A real
# runner must still confirm these against the exact hardware/firmware model.
GO2_JOINT_LIMITS_RAD = (
    (-1.0472, 1.0472),
    (-1.5708, 3.4907),
    (-2.7227, -0.83776),
    (-1.0472, 1.0472),
    (-1.5708, 3.4907),
    (-2.7227, -0.83776),
    (-1.0472, 1.0472),
    (-0.5236, 4.5379),
    (-2.7227, -0.83776),
    (-1.0472, 1.0472),
    (-0.5236, 4.5379),
    (-2.7227, -0.83776),
)

# Maximum displacement from Woof's measured StandDown pose used by the initial
# reduced-amplitude hardware proof.  Front legs may point a little; rear
# support legs stay close to their captured pose.
REDUCED_PROOF_MIN_DELTA_RAD = (
    -0.14,
    -0.14,
    -0.02,
    -0.14,
    -0.14,
    -0.02,
    -0.05,
    -0.05,
    -0.02,
    -0.05,
    -0.05,
    -0.02,
)

REDUCED_PROOF_MAX_DELTA_RAD = (
    0.14,
    0.14,
    0.12,
    0.14,
    0.14,
    0.12,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
)

# Full-policy envelope relative to the measured StandDown pose.  It permits the
# actor's trained front-leg extension, while never asking a calf to fold deeper
# than the stock grounded pose.  The live runner still enforces rate, torque,
# velocity, tilt, perception, and communication guards.
FULL_POLICY_MIN_DELTA_RAD = (
    -0.60,
    -1.50,
    -0.02,
    -0.60,
    -1.50,
    -0.02,
    -0.60,
    -0.80,
    -0.02,
    -0.60,
    -0.80,
    -0.02,
)

FULL_POLICY_MAX_DELTA_RAD = (
    0.60,
    1.00,
    0.90,
    0.60,
    1.00,
    0.90,
    0.60,
    1.00,
    0.90,
    0.60,
    1.00,
    0.90,
)


def unitree_to_isaac(values: Sequence[float]) -> tuple[float, ...]:
    """Remap the first twelve Unitree motor values into Isaac actor order."""

    if len(values) < 12:
        raise ValueError(f"expected at least 12 Unitree motor values, got {len(values)}")
    return tuple(float(values[index]) for index in ISAAC_TO_UNITREE_MOTOR_INDEX)


def isaac_to_unitree(values: Sequence[float]) -> tuple[float, ...]:
    """Remap twelve Isaac-order values into Unitree motor order."""

    if len(values) != 12:
        raise ValueError(f"expected 12 Isaac joint values, got {len(values)}")
    result = [0.0] * 12
    for isaac_index, unitree_index in enumerate(ISAAC_TO_UNITREE_MOTOR_INDEX):
        result[unitree_index] = float(values[isaac_index])
    return tuple(result)


def projected_gravity_wxyz(quaternion_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """Rotate world gravity ``(0, 0, -1)`` into the Go2 body frame."""

    if len(quaternion_wxyz) != 4:
        raise ValueError("quaternion must contain (w, x, y, z)")
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = (value / norm for value in (w, x, y, z))

    # Third row of the world-from-body rotation matrix, negated.  This equals
    # R(q)^T @ (0, 0, -1).
    return (
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    )


def build_actor_observation(
    *,
    base_linear_velocity_body: Sequence[float],
    base_angular_velocity_body: Sequence[float],
    gravity_body: Sequence[float],
    joint_position_isaac: Sequence[float],
    joint_velocity_isaac: Sequence[float],
    previous_action: Sequence[float],
    bbox_xyxy_normalized: Sequence[float],
) -> tuple[float, ...]:
    """Build the exact 49-value observation used by the exported actor."""

    groups = (
        ("base_linear_velocity_body", base_linear_velocity_body, 3),
        ("base_angular_velocity_body", base_angular_velocity_body, 3),
        ("gravity_body", gravity_body, 3),
        ("joint_position_isaac", joint_position_isaac, 12),
        ("joint_velocity_isaac", joint_velocity_isaac, 12),
        ("previous_action", previous_action, 12),
        ("bbox_xyxy_normalized", bbox_xyxy_normalized, 4),
    )
    for name, values, expected in groups:
        if len(values) != expected:
            raise ValueError(f"{name} must contain {expected} values, got {len(values)}")

    joint_position_relative = tuple(
        float(position) - default
        for position, default in zip(
            joint_position_isaac, ISAAC_GROUNDED_SIT_RAD, strict=True
        )
    )
    observation = tuple(
        float(value)
        for values in (
            base_linear_velocity_body,
            base_angular_velocity_body,
            gravity_body,
            joint_position_relative,
            joint_velocity_isaac,
            previous_action,
            bbox_xyxy_normalized,
        )
        for value in values
    )
    if len(observation) != 49:
        raise AssertionError(f"actor observation has {len(observation)} values")
    if not all(math.isfinite(value) for value in observation):
        raise ValueError("actor observation contains non-finite values")
    return observation


def actor_action_to_joint_target(action: Sequence[float]) -> tuple[float, ...]:
    """Apply the training-time action scale without clamping or rate limiting."""

    if len(action) != 12:
        raise ValueError(f"actor action must contain 12 values, got {len(action)}")
    if not all(math.isfinite(float(value)) for value in action):
        raise ValueError("actor action contains non-finite values")
    return tuple(
        default + 0.5 * float(value)
        for default, value in zip(ISAAC_GROUNDED_SIT_RAD, action, strict=True)
    )


def guarded_policy_target(
    *,
    raw_target: Sequence[float],
    start_target: Sequence[float],
    previous_target: Sequence[float],
    action_gain: float,
    maximum_step_rad: float,
    minimum_delta_rad: Sequence[float] = REDUCED_PROOF_MIN_DELTA_RAD,
    maximum_delta_rad: Sequence[float] = REDUCED_PROOF_MAX_DELTA_RAD,
) -> tuple[float, ...]:
    """Scale, envelope-clamp, and slew-limit one Isaac-order policy target."""

    if not (
        len(raw_target)
        == len(start_target)
        == len(previous_target)
        == len(minimum_delta_rad)
        == len(maximum_delta_rad)
        == 12
    ):
        raise ValueError("policy, start, and previous targets must contain 12 values")
    result = []
    for raw, start, previous, minimum_delta, maximum_delta in zip(
        raw_target,
        start_target,
        previous_target,
        minimum_delta_rad,
        maximum_delta_rad,
        strict=True,
    ):
        desired = float(start) + action_gain * (float(raw) - float(start))
        desired = max(
            float(start) + minimum_delta,
            min(float(start) + maximum_delta, desired),
        )
        desired = max(
            float(previous) - maximum_step_rad,
            min(float(previous) + maximum_step_rad, desired),
        )
        result.append(desired)
    return tuple(result)


def joint_limit_violations(target_isaac: Sequence[float]) -> tuple[str, ...]:
    """Return human-readable target violations in Isaac order."""

    if len(target_isaac) != 12:
        raise ValueError(f"joint target must contain 12 values, got {len(target_isaac)}")
    violations = []
    for name, value, (minimum, maximum) in zip(
        ISAAC_JOINT_ORDER, target_isaac, GO2_JOINT_LIMITS_RAD, strict=True
    ):
        value = float(value)
        if value < minimum or value > maximum:
            violations.append(
                f"{name}={value:.4f} outside [{minimum:.4f}, {maximum:.4f}]"
            )
    return tuple(violations)


def selected_target_bbox_from_status(
    status: Mapping[str, Any],
    *,
    maximum_age_s: float = 0.5,
) -> dict[str, Any]:
    """Return the normalized box for Collie's explicitly selected target.

    The pointing policy must follow the user's target lock.  It must never
    silently substitute the highest-confidence detection from another class.
    """

    width = int(status.get("frame_width") or 0)
    height = int(status.get("frame_height") or 0)
    label = status.get("selected_target_name")
    target = status.get("selected_target")
    age_s = status.get("selected_target_age_s")
    if not label or not isinstance(target, Mapping):
        raise ValueError("Collie has no selected target")
    if width <= 0 or height <= 0:
        raise ValueError("Collie status has invalid frame dimensions")
    if age_s is None or not math.isfinite(float(age_s)) or float(age_s) > maximum_age_s:
        raise ValueError(f"selected target is stale: age={age_s!r}")

    values = target.get("bbox_xywh")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("selected target has no bbox_xywh")
    if len(values) != 4:
        raise ValueError("selected target bbox_xywh must contain four values")
    x, y, box_width, box_height = (float(value) for value in values)
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError("selected target box must have positive size")

    x1 = max(0.0, min(1.0, x / width))
    y1 = max(0.0, min(1.0, y / height))
    x2 = max(0.0, min(1.0, (x + box_width) / width))
    y2 = max(0.0, min(1.0, (y + box_height) / height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("selected target box is outside the camera frame")

    return {
        "bbox": (x1, y1, x2, y2),
        "label": str(label),
        "confidence": float(target.get("confidence") or 0.0),
        "frame_age_s": float(age_s),
        "lock_id": status.get("target_lock_id"),
    }


def standdown_error_rad(joint_position_isaac: Sequence[float]) -> tuple[float, float]:
    """Return RMS and maximum error from Woof's measured StandDown pose."""

    if len(joint_position_isaac) != 12:
        raise ValueError(f"joint position must contain 12 values, got {len(joint_position_isaac)}")
    errors = tuple(
        float(value) - reference
        for value, reference in zip(
            joint_position_isaac, WOOF_BUILTIN_STANDDOWN_RAD, strict=True
        )
    )
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    return rms, max(abs(error) for error in errors)


def sit_error_rad(joint_position_isaac: Sequence[float]) -> tuple[float, float]:
    """Deprecated compatibility alias for :func:`standdown_error_rad`."""

    return standdown_error_rad(joint_position_isaac)
