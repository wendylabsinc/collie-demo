from __future__ import annotations

from pathlib import Path

import numpy as np

from collie_demo.max_vision_server import (
    MaxVisionRuntime,
    best_detection_per_label,
)
from collie_demo.shadow_server import RemoteDetection
from collie_demo.temporal_fusion import ReferenceTemporalFusion


def payload(
    frame_id: int,
    detections: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "camera_fps": 29.0,
        "frame_age_s": 0.02,
        "produce": {
            "frame_id": frame_id,
            "inference_ms": 91.0,
            "detections": detections,
        },
    }


def pear(x: int, confidence: float = 0.8) -> dict[str, object]:
    return {
        "label": "pear",
        "confidence": confidence,
        "bbox_xyxy": [x, 80, x + 60, 130],
    }


def test_best_detection_per_label_keeps_highest_confidence() -> None:
    detections = (
        RemoteDetection("pear", 0.4, (10, 20, 40, 60)),
        RemoteDetection("pear", 0.9, (100, 120, 160, 200)),
        RemoteDetection("apple", 0.7, (30, 40, 80, 90)),
    )

    best = best_detection_per_label(detections)

    assert best["pear"].confidence == 0.9
    assert best["apple"].confidence == 0.7


def test_runtime_fuses_fresh_measurement_and_bridges_bounded_gap() -> None:
    runtime = MaxVisionRuntime(
        source_url="http://unused",
        fusion=ReferenceTemporalFusion(),
        maximum_prediction_age_s=0.45,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    runtime.process_observation(frame, payload(1, [pear(100)]), now=1.0)
    runtime.process_observation(frame, payload(2, [pear(108)]), now=1.1)
    runtime.process_observation(frame, payload(3, []), now=1.2)
    status = runtime.status()

    tracks = status["max_vision"]["tracks"]
    assert len(tracks) == 1
    assert tracks[0]["label"] == "pear"
    assert tracks[0]["source"] == "bounded_max_prediction"
    assert status["max_vision"]["bridged_track_frames"] == 1
    assert status["control_authority"] == "none_read_only"
    assert status["motion_enabled"] is False


def test_runtime_expires_prediction_after_bound() -> None:
    runtime = MaxVisionRuntime(
        source_url="http://unused",
        fusion=ReferenceTemporalFusion(),
        maximum_prediction_age_s=0.45,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    runtime.process_observation(frame, payload(1, [pear(100)]), now=1.0)
    runtime.process_observation(frame, payload(2, []), now=1.6)
    status = runtime.status()

    assert status["max_vision"]["tracks"] == []
    assert status["max_vision"]["expired_tracks"] == 1


def test_ui_describes_max_accurately_and_has_no_motion_controls() -> None:
    html = (
        Path(__file__).parents[1]
        / "src"
        / "collie_demo"
        / "max_vision_ui.html"
    ).read_text(encoding="utf-8")

    assert "MAX + Mojo temporal fusion" in html
    assert "one fruit model" in html
    assert "never creates a class or confidence" in html
    assert "Read only" in html
    assert "/api/follow" not in html
    assert "/api/arm" not in html
