import pytest

from tools.camera_fruit_soak import SoakConfig, SoakMonitor

GENERATION = "voice-process.1"


def camera_status(*, frame_id: int, generation: str = GENERATION) -> dict:
    return {
        "ready": True,
        "peer_state": "connected",
        "video_enabled": True,
        "generation": generation,
        "frame_id": frame_id,
        "last_frame_age_s": 0.04,
        "fps_5s": 14.0,
    }


def main_status(
    *,
    frame_id: int,
    generation: str = GENERATION,
    detections: list[dict] | None = None,
) -> dict:
    return {
        "health": {
            "camera_live": True,
            "produce_live": True,
            "gpu_ready": True,
        },
        "frame_count": frame_id,
        "frame_age_s": 0.06,
        "camera_fps": 10.0,
        "camera_stream_generation": generation,
        "camera_rpc": {
            "generation": generation,
            "last_source_age_s": 0.03,
        },
        "produce": {
            "frame_id": frame_id,
            "age_s": 0.08,
            "inference_ms": 72.0,
            "class_thresholds": {
                "apple": 0.70,
                "banana": 0.20,
                "pear": 0.35,
            },
            "detections": detections or [],
        },
    }


def observe(
    monitor: SoakMonitor,
    *,
    now_s: float,
    frame_id: int,
    generation: str = GENERATION,
    detections: list[dict] | None = None,
) -> dict[str, object]:
    return monitor.observe(
        now_s=now_s,
        main_status=main_status(
            frame_id=frame_id,
            generation=generation,
            detections=detections,
        ),
        camera_status=camera_status(
            frame_id=frame_id,
            generation=generation,
        ),
    )


def test_healthy_advancing_camera_and_detector_pass() -> None:
    monitor = SoakMonitor(SoakConfig())

    first = observe(monitor, now_s=10.0, frame_id=100)
    second = observe(monitor, now_s=10.5, frame_id=105)

    assert first["healthy"] is True
    assert second["healthy"] is True
    summary = monitor.summary()
    assert summary["passed"] is True
    assert summary["samples"] == 2
    assert summary["metrics"]["broker_fps"]["mean"] == 14.0


def test_unchanged_frame_ids_exceeding_stall_window_fail() -> None:
    monitor = SoakMonitor(SoakConfig(maximum_frame_stall_s=1.5))
    observe(monitor, now_s=10.0, frame_id=100)

    stalled = observe(monitor, now_s=11.6, frame_id=100)

    assert stalled["healthy"] is False
    assert stalled["errors"] == [
        "broker_frame_frozen",
        "detector_frame_frozen",
        "runtime_frame_frozen",
    ]
    assert monitor.summary()["passed"] is False


def test_generation_change_is_counted_and_requires_explicit_allowance() -> None:
    strict = SoakMonitor(SoakConfig())
    observe(strict, now_s=1.0, frame_id=10)
    failed = observe(
        strict,
        now_s=1.5,
        frame_id=1,
        generation="voice-process.2",
    )
    assert "too_many_generation_changes" in failed["errors"]

    reconnect_probe = SoakMonitor(SoakConfig(maximum_generation_changes=1))
    observe(reconnect_probe, now_s=1.0, frame_id=10)
    recovered = observe(
        reconnect_probe,
        now_s=1.5,
        frame_id=1,
        generation="voice-process.2",
    )
    assert recovered["healthy"] is True
    assert reconnect_probe.summary()["generation_changes"] == 1


def test_slow_status_request_is_recorded_as_a_failure() -> None:
    monitor = SoakMonitor(SoakConfig(maximum_status_request_s=0.5))

    sample = monitor.observe(
        now_s=1.0,
        main_status=main_status(frame_id=1),
        camera_status=camera_status(frame_id=1),
        request_metrics={
            "main_status_request_ms": 501.0,
            "camera_status_request_ms": 20.0,
        },
    )

    assert sample["healthy"] is False
    assert sample["errors"] == ["main_status_request_ms_slow"]
    assert monitor.summary()["metrics"]["main_status_request_ms"] == {
        "minimum": 501.0,
        "mean": 501.0,
        "maximum": 501.0,
    }


def test_expected_fruit_uses_configured_threshold_and_ratio() -> None:
    monitor = SoakMonitor(
        SoakConfig(
            expected_fruit="pear",
            minimum_recognition_ratio=0.75,
            recognition_warmup_s=0.0,
        )
    )
    detections = [
        {"label": "pear", "confidence": 0.80},
        {"label": "apple", "confidence": 0.90},
    ]
    observe(monitor, now_s=1.0, frame_id=1, detections=detections)
    observe(monitor, now_s=1.5, frame_id=2, detections=detections)
    observe(monitor, now_s=2.0, frame_id=3, detections=[])
    observe(monitor, now_s=2.5, frame_id=4, detections=detections)

    summary = monitor.summary()
    assert summary["passed"] is True
    assert summary["recognition_ratio"] == 0.75
    assert summary["recognition_confidence"] == {
        "minimum": 0.8,
        "mean": 0.8,
        "maximum": 0.8,
    }


def test_expected_fruit_below_class_threshold_is_a_miss() -> None:
    monitor = SoakMonitor(
        SoakConfig(
            expected_fruit="pear",
            minimum_recognition_ratio=1.0,
            recognition_warmup_s=0.0,
        )
    )

    observe(
        monitor,
        now_s=1.0,
        frame_id=1,
        detections=[{"label": "pear", "confidence": 0.34}],
    )

    assert monitor.summary()["passed"] is False


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="minimum_recognition_ratio"):
        SoakConfig(minimum_recognition_ratio=1.1)
