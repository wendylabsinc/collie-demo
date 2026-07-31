import math
from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("scipy")
TOOLS = Path(__file__).parents[1] / "nav2" / "tools"
sys.path.insert(0, str(TOOLS))

from estimate_hesai_extrinsic import (  # noqa: E402
    estimate_transform,
    rotation_from_rpy,
)


def room_cloud() -> np.ndarray:
    axis = np.linspace(-2.5, 2.5, 48)
    x, y = np.meshgrid(axis, axis)
    floor = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -0.31)])

    horizontal = np.linspace(-2.5, 2.5, 72)
    vertical = np.linspace(-0.31, 1.6, 28)
    h, z = np.meshgrid(horizontal, vertical)
    wall_x = np.column_stack([np.full(h.size, 2.8), h.ravel(), z.ravel()])
    wall_y = np.column_stack([h.ravel(), np.full(h.size, -2.4), z.ravel()])

    obstacle_axis = np.linspace(-0.35, 0.35, 18)
    obstacle_z = np.linspace(-0.31, 0.75, 24)
    o, oz = np.meshgrid(obstacle_axis, obstacle_z)
    obstacle = np.column_stack(
        [np.full(o.size, 1.35), o.ravel() + 0.65, oz.ravel()]
    )
    return np.vstack([floor, wall_x, wall_y, obstacle])


def test_estimator_recovers_a_body_local_transform() -> None:
    rng = np.random.default_rng(71)
    base = room_cloud()
    expected_rotation = rotation_from_rpy(
        math.radians(1.5),
        math.radians(-1.2),
        math.radians(8.0),
    )
    expected_translation = np.array([0.13, -0.06, 0.19])
    hesai = (base - expected_translation) @ expected_rotation
    hesai += rng.normal(0.0, 0.004, hesai.shape)

    # Different sensor FOVs, sampling, and a few non-corresponding returns.
    base = base[rng.choice(base.shape[0], int(base.shape[0] * 0.78), replace=False)]
    hesai = hesai[
        rng.choice(hesai.shape[0], int(hesai.shape[0] * 0.82), replace=False)
    ]
    hesai = np.vstack(
        [hesai, rng.uniform([-3, -3, -0.4], [3, 3, 1.7], (120, 3))]
    )

    estimate = estimate_transform(
        hesai,
        base,
        voxel_m=0.11,
        maximum_range_m=6.0,
        yaw_search_degrees=(-30, 0, 30),
    )

    assert estimate.x_m == pytest.approx(expected_translation[0], abs=0.035)
    assert estimate.y_m == pytest.approx(expected_translation[1], abs=0.035)
    assert estimate.z_m == pytest.approx(expected_translation[2], abs=0.035)
    assert estimate.roll_deg == pytest.approx(1.5, abs=1.2)
    assert estimate.pitch_deg == pytest.approx(-1.2, abs=1.2)
    assert estimate.yaw_deg == pytest.approx(8.0, abs=1.5)
    assert estimate.median_residual_m < 0.06
    assert estimate.symmetric_inlier_ratio > 0.60
    assert estimate.body_nonfloor_coverage is not None
    assert estimate.body_nonfloor_coverage > 0.80
