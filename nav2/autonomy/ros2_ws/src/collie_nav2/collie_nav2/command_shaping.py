"""Pure velocity shaping for the physical Go2 Nav2 handoff."""

from __future__ import annotations

import math


def apply_measured_motion_floors(
    forward_mps: float,
    yaw_rps: float,
    *,
    minimum_forward_mps: float,
    minimum_rotation_yaw_rps: float,
) -> tuple[float, float]:
    """Raise fresh non-zero commands to measured physical actuation floors."""

    forward = float(forward_mps)
    yaw = float(yaw_rps)
    if (
        abs(forward) <= 0.02
        and 0.001 < abs(yaw) < minimum_rotation_yaw_rps
    ):
        # Treat tiny translation paired with yaw as a rotation-only command.
        yaw = math.copysign(minimum_rotation_yaw_rps, yaw)
    elif 0.001 < forward < minimum_forward_mps:
        forward = minimum_forward_mps
    return forward, yaw
