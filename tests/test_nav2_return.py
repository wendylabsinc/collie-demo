import math

import pytest

from collie_demo.nav2_return import (
    MapHomeCapture,
    Nav2ReturnError,
    Nav2ReturnStatus,
)


def test_map_home_capture_requires_map_frame() -> None:
    payload = {
        "home": {
            "x_m": 1.25,
            "y_m": -0.5,
            "yaw_rad": 0.75,
            "frame_id": "map",
        },
        "validation": {
            "sample_count": 12,
            "maximum_position_span_m": 0.012,
            "maximum_yaw_span_deg": 1.5,
        },
    }

    capture = MapHomeCapture.from_payload(payload)

    assert capture.frame_id == "map"
    assert capture.maximum_yaw_span_rad == pytest.approx(math.radians(1.5))
    assert capture.validation_dict()["source"] == "nav2_map_localization"

    payload["home"]["frame_id"] = "odom"
    with pytest.raises(Nav2ReturnError, match="failed validation"):
        MapHomeCapture.from_payload(payload)


def test_nav2_status_preserves_goal_error_and_sensor_health() -> None:
    status = Nav2ReturnStatus.from_payload(
        {
            "navigation": {
                "state": "succeeded",
                "reason": "goal verified",
                "distance_remaining_m": 0.02,
                "position_error_m": 0.04,
                "heading_error_deg": -3.0,
                "navigation_time_s": 6.5,
                "recoveries": 1,
            },
            "health": {
                "localization_healthy": True,
                "map_healthy": True,
                "scan_healthy": True,
            },
            "motion": {"armed": False},
        }
    )

    assert status.terminal is True
    assert status.position_error_m == pytest.approx(0.04)
    assert status.heading_error_rad == pytest.approx(math.radians(-3.0))
    assert status.recoveries == 1
