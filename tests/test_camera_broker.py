from __future__ import annotations

import inspect
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.modules.setdefault(
    "websocket",
    types.SimpleNamespace(create_connection=lambda *args, **kwargs: None),
)

from voice.camera_broker import CameraBrokerState, register_camera_routes
from voice import main as voice_main


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


EXPECTED_STATUS_KEYS = {
    "ready",
    "source",
    "peer_state",
    "video_enabled",
    "generation",
    "frame_id",
    "frames_total",
    "last_frame_age_s",
    "fps_5s",
    "width",
    "height",
    "freshness_limit_s",
    "reconnect_count",
    "stale_reconnect_count",
    "last_error",
}


def test_camera_state_exact_status_generation_and_freshness() -> None:
    clock = FakeClock(10.0)
    state = CameraBrokerState(process_id="process", clock=clock)

    generation = state.begin_session()
    assert generation == "process.1"
    before = state.status()
    assert set(before) == EXPECTED_STATUS_KEYS
    assert before["ready"] is False
    assert before["generation"] == generation
    assert before["frame_id"] is None
    assert before["freshness_limit_s"] == 0.75

    assert state.set_connected(generation, video_enabled=True) is True
    packet = state.publish(
        generation,
        b"jpeg-one",
        width=1280,
        height=720,
        captured_monotonic_s=10.1,
    )
    assert packet is not None and packet.frame_id == 1
    clock.value = 10.3
    fresh = state.status()
    assert fresh["ready"] is True
    assert fresh["last_frame_age_s"] == 0.2
    assert fresh["width"] == 1280
    assert fresh["height"] == 720

    clock.value = 10.85
    stale = state.status()
    assert stale["ready"] is False
    assert stale["last_frame_age_s"] == 0.75

    state.disconnect(generation, "silent stall", stale=True, now=10.85)
    disconnected = state.status(now=10.85)
    assert disconnected["ready"] is False
    assert disconnected["stale_reconnect_count"] == 1

    generation_two = state.begin_session()
    assert generation_two == "process.2"
    assert generation_two != generation
    assert state.status(now=11.0)["frame_id"] is None
    assert state.status(now=11.0)["reconnect_count"] == 1
    state.set_connected(generation_two, video_enabled=True)
    packet_two = state.publish(
        generation_two,
        b"jpeg-two",
        width=640,
        height=360,
        captured_monotonic_s=11.1,
    )
    assert packet_two is not None and packet_two.frame_id == 1
    assert state.status(now=11.1)["frames_total"] == 2


def test_camera_snapshot_api_is_broadcast_bounded_and_never_serves_stale() -> None:
    clock = FakeClock(20.0)
    state = CameraBrokerState(process_id="api", clock=clock)
    app = FastAPI()
    register_camera_routes(app, state)
    client = TestClient(app)

    unavailable = client.get("/api/camera/frame.jpg?wait_ms=0")
    assert unavailable.status_code == 503
    assert unavailable.headers["retry-after"] == "0.25"
    assert unavailable.headers["content-type"].startswith("application/json")
    assert unavailable.json()["camera"]["ready"] is False

    generation = state.begin_session()
    state.set_connected(generation, video_enabled=True)
    state.publish(
        generation,
        b"first-jpeg",
        width=1280,
        height=720,
        captured_monotonic_s=20.0,
    )

    first = client.get("/api/camera/frame.jpg")
    second_consumer = client.get("/api/camera/frame.jpg")
    assert first.status_code == second_consumer.status_code == 200
    assert first.content == second_consumer.content == b"first-jpeg"
    assert first.headers["x-woof-camera-generation"] == generation
    assert first.headers["x-woof-camera-frame-id"] == "1"
    assert first.headers["x-woof-camera-age-ms"] == "0"
    assert first.headers["x-woof-camera-width"] == "1280"
    assert first.headers["x-woof-camera-height"] == "720"
    assert first.headers["x-woof-camera-captured-monotonic-s"] == "20.000000"
    assert first.headers["cache-control"] == (
        "no-store, no-cache, must-revalidate"
    )

    unchanged = client.get(
        "/api/camera/frame.jpg",
        params={
            "after_generation": generation,
            "after_frame_id": 1,
            "wait_ms": 0,
        },
    )
    assert unchanged.status_code == 204
    assert unchanged.content == b""

    future_marker = client.get(
        "/api/camera/frame.jpg",
        params={
            "after_generation": generation,
            "after_frame_id": 999,
            "wait_ms": 0,
        },
    )
    assert future_marker.status_code == 204
    assert future_marker.content == b""

    clock.value = 20.1
    state.publish(
        generation,
        b"second-jpeg",
        width=1280,
        height=720,
        captured_monotonic_s=20.1,
    )
    newer = client.get(
        "/api/camera/frame.jpg",
        params={
            "after_generation": generation,
            "after_frame_id": 1,
            "wait_ms": 0,
        },
    )
    assert newer.status_code == 200
    assert newer.content == b"second-jpeg"
    assert newer.headers["x-woof-camera-frame-id"] == "2"

    clock.value = 20.85
    stale = client.get("/api/camera/frame.jpg?wait_ms=0")
    assert stale.status_code == 503
    assert stale.headers["content-type"].startswith("application/json")
    assert stale.content != b"second-jpeg"
    assert stale.json()["camera"]["ready"] is False


def test_camera_watchdog_pli_reconnect_backoff_and_route_audit() -> None:
    action = voice_main._camera_watchdog_action
    assert action(
        connected_elapsed_s=0.49,
        frame_age_s=None,
        pli_elapsed_s=None,
    ) == "wait"
    assert action(
        connected_elapsed_s=0.5,
        frame_age_s=None,
        pli_elapsed_s=None,
    ) == "pli"
    assert action(
        connected_elapsed_s=1.0,
        frame_age_s=None,
        pli_elapsed_s=0.5,
    ) == "wait"
    assert action(
        connected_elapsed_s=1.5,
        frame_age_s=None,
        pli_elapsed_s=1.0,
    ) == "pli"
    assert action(
        connected_elapsed_s=5.0,
        frame_age_s=None,
        pli_elapsed_s=0.0,
    ) == "reconnect"
    assert action(
        connected_elapsed_s=20.0,
        frame_age_s=0.4,
        pli_elapsed_s=None,
    ) == "pli"
    assert action(
        connected_elapsed_s=20.0,
        frame_age_s=0.75,
        pli_elapsed_s=0.2,
    ) == "reconnect"

    assert [voice_main._webrtc_backoff_s(i) for i in range(1, 7)] == [
        0.75,
        1.5,
        3.0,
        6.0,
        10.0,
        10.0,
    ]
    assert voice_main._next_webrtc_attempt(3, 9.99) == 4
    assert voice_main._next_webrtc_attempt(5, 10.0) == 1

    camera_routes = [
        route
        for route in voice_main.app.routes
        if getattr(route, "path", "").startswith("/api/camera")
    ]
    assert camera_routes
    assert all("POST" not in getattr(route, "methods", set()) for route in camera_routes)
    webrtc_source = inspect.getsource(voice_main._webrtc_once).lower()
    assert "video.add_track_callback" in webrtc_source
    assert "audio.add_track_callback" in webrtc_source
    assert webrtc_source.index("video.add_track_callback") < webrtc_source.index(
        "video.switchvideochannel"
    )
    assert webrtc_source.index("audio.add_track_callback") < webrtc_source.index(
        "audio.switchaudiochannel"
    )
    assert "sport" not in webrtc_source
