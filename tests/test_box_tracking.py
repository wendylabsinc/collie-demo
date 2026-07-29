from __future__ import annotations

import cv2
import numpy as np
import pytest

from collie_demo.box_tracking import (
    BoxTrackerInitializationError,
    KLTBoxTracker,
    KLTBoxTrackerConfig,
    produce_tracker_factory_from_mode,
)


def _textured_patch(width: int = 90, height: int = 70) -> np.ndarray:
    rng = np.random.default_rng(12)
    patch = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.rectangle(patch, (1, 1), (width - 2, height - 2), (255, 255, 255), 2)
    cv2.circle(patch, (width // 2, height // 2), 14, (0, 0, 0), 3)
    return patch


def _frame_with_patch(
    patch: np.ndarray,
    x: int,
    y: int,
    *,
    frame_width: int = 320,
    frame_height: int = 240,
) -> np.ndarray:
    frame = np.full((frame_height, frame_width, 3), 30, dtype=np.uint8)
    height, width = patch.shape[:2]
    frame[y : y + height, x : x + width] = patch
    return frame


def test_klt_tracker_follows_translation_and_reports_metrics() -> None:
    patch = _textured_patch()
    initial = _frame_with_patch(patch, 70, 80)
    tracker = KLTBoxTracker(initial, (70, 80, 90, 70))

    ok, bbox = tracker.update(_frame_with_patch(patch, 79, 86))

    assert ok is True
    x, y, width, height = bbox
    assert x == pytest.approx(79, abs=2.0)
    assert y == pytest.approx(86, abs=2.0)
    assert width == pytest.approx(90, abs=3.0)
    assert height == pytest.approx(70, abs=3.0)
    status = tracker.status()
    assert status["backend"] == "opencv_klt_affine"
    assert status["failed"] is False
    assert status["metrics"]["accepted_updates"] == 1
    assert status["metrics"]["points_in_use"] >= 6
    assert status["metrics"]["last_update_ms"] >= 0.0


def test_klt_tracker_follows_many_camera_rate_frames() -> None:
    patch = _textured_patch()
    tracker = KLTBoxTracker(
        _frame_with_patch(patch, 50, 70),
        (50, 70, 90, 70),
    )

    bbox = (50.0, 70.0, 90.0, 70.0)
    for step in range(1, 25):
        ok, bbox = tracker.update(
            _frame_with_patch(patch, 50 + 2 * step, 70 + step)
        )
        assert ok is True

    x, y, width, height = bbox
    assert x == pytest.approx(98, abs=3.0)
    assert y == pytest.approx(94, abs=3.0)
    assert width == pytest.approx(90, abs=4.0)
    assert height == pytest.approx(70, abs=4.0)
    metrics = tracker.status()["metrics"]
    assert metrics["accepted_updates"] == 24
    assert metrics["rejected_updates"] == 0


def test_klt_tracker_fails_closed_when_frame_shape_changes() -> None:
    patch = _textured_patch()
    tracker = KLTBoxTracker(
        _frame_with_patch(patch, 70, 80),
        (70, 80, 90, 70),
    )

    ok, _ = tracker.update(
        _frame_with_patch(
            patch,
            70,
            80,
            frame_width=400,
            frame_height=300,
        )
    )

    assert ok is False
    assert tracker.status()["failed"] is True
    assert tracker.status()["metrics"]["last_failure"] == "frame_shape_changed"


def test_klt_tracker_rejects_textureless_selection() -> None:
    frame = np.full((240, 320, 3), 30, dtype=np.uint8)

    with pytest.raises(
        BoxTrackerInitializationError, match="too little image texture"
    ):
        KLTBoxTracker(frame, (70, 80, 90, 70))


def test_klt_tracker_validates_bbox() -> None:
    patch = _textured_patch()
    frame = _frame_with_patch(patch, 70, 80)

    with pytest.raises(BoxTrackerInitializationError, match="too small"):
        KLTBoxTracker(frame, (70, 80, 2, 2))

    with pytest.raises(
        BoxTrackerInitializationError, match="outside the camera frame"
    ):
        KLTBoxTracker(frame, (400, 300, 20, 20))


def test_tracker_factory_is_disabled_by_default_and_opt_in() -> None:
    assert produce_tracker_factory_from_mode("") is None
    assert produce_tracker_factory_from_mode("off") is None
    assert produce_tracker_factory_from_mode("yolo") is None

    factory = produce_tracker_factory_from_mode("klt-affine")
    assert factory is not None
    patch = _textured_patch()
    tracker = factory(
        _frame_with_patch(patch, 70, 80),
        (70, 80, 90, 70),
    )
    assert tracker.status()["backend"] == "opencv_klt_affine"

    with pytest.raises(ValueError, match="must be off or klt_affine"):
        produce_tracker_factory_from_mode("mil")


def test_tracker_config_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="minimum_points"):
        KLTBoxTrackerConfig(minimum_points=2)
    with pytest.raises(ValueError, match="new_box_weight"):
        KLTBoxTrackerConfig(new_box_weight=0.0)
