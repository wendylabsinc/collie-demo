"""Deployment geometry for the locked standing-point policy.

The balance actor controls the nine support joints.  The front-right leg is
excluded from the action space and follows the deterministic schedule below.
These values are copied from the policy's training branch so inference and
hardware use the same nominal stance and pointing trajectory.
"""

from __future__ import annotations

import math


GO2_THIGH_LENGTH_M = 0.213
GO2_CALF_LENGTH_M = 0.213
GO2_HIP_X_SEPARATION_M = 0.3762

NOMINAL_STAND_JOINT_POSITIONS_RAD = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
}


def foot_drop_from_hip_m(
    thigh_rad: float,
    calf_rad: float,
    hip_rad: float = 0.0,
) -> float:
    """Return the foot's vertical drop below the hip axis."""

    in_plane_drop = (
        GO2_THIGH_LENGTH_M * math.cos(thigh_rad)
        + GO2_CALF_LENGTH_M * math.cos(thigh_rad + calf_rad)
    )
    return in_plane_drop * math.cos(hip_rad)


def _drop(prefix: str) -> float:
    return foot_drop_from_hip_m(
        NOMINAL_STAND_JOINT_POSITIONS_RAD[f"{prefix}_thigh_joint"],
        NOMINAL_STAND_JOINT_POSITIONS_RAD[f"{prefix}_calf_joint"],
        NOMINAL_STAND_JOINT_POSITIONS_RAD[f"{prefix}_hip_joint"],
    )


_FRONT_DROP_M = _drop("FL")
_REAR_DROP_M = _drop("RL")
STANDING_TRUNK_HEIGHT_M = (_FRONT_DROP_M + _REAR_DROP_M) / 2.0
_PITCH_RAD = math.atan2(
    _FRONT_DROP_M - _REAR_DROP_M,
    GO2_HIP_X_SEPARATION_M,
)

GROUNDED_STAND_ROOT_POSITION_M = (0.0, 0.0, STANDING_TRUNK_HEIGHT_M)
GROUNDED_STAND_ROOT_QUAT_WXYZ = (
    math.cos(_PITCH_RAD / 2.0),
    0.0,
    -math.sin(_PITCH_RAD / 2.0),
    0.0,
)

GO2_FRONT_CALF_LIMIT_RAD = -0.83776
LOCKED_POINT_CALF_MARGIN_RAD = 0.11
FRONT_HIP_FORWARD_M = 0.19
MINIMUM_POINT_EXTENSION_FRACTION = 0.85

_POINT_CALF_RAD = GO2_FRONT_CALF_LIMIT_RAD - LOCKED_POINT_CALF_MARGIN_RAD
_POINT_THIGH_RAD = -math.pi / 2.0 - _POINT_CALF_RAD / 2.0
LOCKED_POINT_REACH_FROM_HIP_M = (
    2.0 * GO2_THIGH_LENGTH_M * math.cos(abs(_POINT_CALF_RAD) / 2.0)
)

LOCKED_POINT_FR_JOINTS_RAD = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": _POINT_THIGH_RAD,
    "FR_calf_joint": _POINT_CALF_RAD,
}

LOCKED_POINT_EXTENSION_FRACTION = LOCKED_POINT_REACH_FROM_HIP_M / (
    GO2_THIGH_LENGTH_M + GO2_CALF_LENGTH_M
)

BALANCE_ACTUATED_JOINTS = tuple(
    name
    for name in NOMINAL_STAND_JOINT_POSITIONS_RAD
    if not name.startswith("FR_")
)


def locked_point_joint_positions() -> dict[str, float]:
    """Return the nominal stance with the front-right leg fully pointing."""

    pose = dict(NOMINAL_STAND_JOINT_POSITIONS_RAD)
    pose.update(LOCKED_POINT_FR_JOINTS_RAD)
    return pose


def front_paw_in_base_frame(
    thigh_rad: float,
    calf_rad: float,
) -> tuple[float, float]:
    """Return the front paw's forward position and height in the base frame."""

    forward = -(
        GO2_THIGH_LENGTH_M * math.sin(thigh_rad)
        + GO2_CALF_LENGTH_M * math.sin(thigh_rad + calf_rad)
    )
    drop = (
        GO2_THIGH_LENGTH_M * math.cos(thigh_rad)
        + GO2_CALF_LENGTH_M * math.cos(thigh_rad + calf_rad)
    )
    return FRONT_HIP_FORWARD_M + forward, STANDING_TRUNK_HEIGHT_M - drop


LOCKED_POINT_SETUP_S = 1.0
LOCKED_POINT_RAMP_S = 1.0


def locked_point_phase(elapsed_s: float, setup_s: float, ramp_s: float) -> float:
    """Return extension progress in ``[0, 1]`` after setup and ramp."""

    if elapsed_s < setup_s:
        return 0.0
    if ramp_s <= 0.0:
        return 1.0
    return min(1.0, (elapsed_s - setup_s) / ramp_s)
