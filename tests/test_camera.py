from __future__ import annotations

import cv2
import numpy as np
import pytest

from collie_demo import camera as camera_module
from collie_demo.camera import BrokerCamera, CameraUnavailable, UnitreeCamera


class FakeVideoClient:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.returned_image = False

    def GetImageSample(self) -> tuple[int, bytes]:
        self.returned_image = True
        return 0, self.payload


class SequenceVideoClient:
    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self.responses = iter(responses)

    def GetImageSample(self) -> tuple[int, bytes]:
        return next(self.responses)


class FakeHttpResponse:
    def __init__(
        self,
        status: int,
        payload: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    def getcode(self) -> int:
        return self.status

    def read(self, maximum: int = -1) -> bytes:
        return self.payload if maximum < 0 else self.payload[:maximum]

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_camera_timestamp_is_taken_after_blocking_sdk_read(monkeypatch) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    client = FakeVideoClient(encoded.tobytes())

    def receipt_time() -> float:
        assert client.returned_image, "timestamp was taken before GetImageSample returned"
        return 123.5

    monkeypatch.setattr(camera_module.time, "monotonic", receipt_time)

    frame = UnitreeCamera(client).read()

    assert frame.frame_id == 1
    assert frame.captured_monotonic_s == 123.5
    assert frame.width == 16
    assert frame.height == 12
    assert frame._bgr is None
    assert frame.bgr.shape == image.shape
    assert frame.source_jpeg == encoded.tobytes()


def test_camera_reports_persistent_timeout_telemetry_and_recovers() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    camera = UnitreeCamera(
        SequenceVideoClient(
            [
                (3104, b""),
                (3104, b""),
                (0, encoded.tobytes()),
            ]
        ),
        timeout_s=3.0,
    )

    with pytest.raises(
        CameraUnavailable,
        match=r"3104 \(client API timeout; timeout_count=1, consecutive=1\)",
    ):
        camera.read()
    with pytest.raises(
        CameraUnavailable,
        match=r"timeout_count=2, consecutive=2",
    ):
        camera.read()

    failed = camera.telemetry()
    assert failed["request_count"] == 2
    assert failed["success_count"] == 0
    assert failed["error_count"] == 2
    assert failed["timeout_count"] == 2
    assert failed["consecutive_errors"] == 2
    assert failed["consecutive_timeouts"] == 2
    assert failed["last_error_code"] == 3104
    assert failed["last_error"] == "client API timeout"
    assert failed["last_error_age_s"] is not None

    frame = camera.read()
    recovered = camera.telemetry()

    assert frame.frame_id == 1
    assert recovered["request_count"] == 3
    assert recovered["success_count"] == 1
    assert recovered["error_count"] == 2
    assert recovered["timeout_count"] == 2
    assert recovered["consecutive_errors"] == 0
    assert recovered["consecutive_timeouts"] == 0
    assert recovered["last_response_code"] == 0
    assert recovered["last_error_code"] == 3104
    assert recovered["last_success_age_s"] is not None


def test_broker_camera_rejects_duplicates_and_keeps_local_ids_monotonic(
    monkeypatch,
) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok

    def headers(generation: str, frame_id: int, age_ms: int) -> dict[str, str]:
        return {
            "Content-Type": "image/jpeg",
            "X-Woof-Camera-Generation": generation,
            "X-Woof-Camera-Frame-Id": str(frame_id),
            "X-Woof-Camera-Age-Ms": str(age_ms),
            "X-Woof-Camera-Width": "16",
            "X-Woof-Camera-Height": "12",
            "X-Woof-Camera-Captured-Monotonic-S": "50.0",
        }

    responses = iter(
        [
            FakeHttpResponse(200, encoded.tobytes(), headers("peer-a", 7, 100)),
            FakeHttpResponse(200, encoded.tobytes(), headers("peer-a", 7, 100)),
            FakeHttpResponse(204),
            FakeHttpResponse(200, encoded.tobytes(), headers("peer-b", 1, 50)),
        ]
    )
    requested_urls: list[str] = []

    def fake_urlopen(request, **_kwargs):
        requested_urls.append(request.full_url)
        return next(responses)

    receipt_times = iter([100.0, 200.0, 201.0])
    monkeypatch.setattr(camera_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: next(receipt_times))
    camera = BrokerCamera("http://voice/api/camera/frame.jpg")

    first = camera.read()
    assert first.frame_id == 1
    assert first.stream_generation == "peer-a"
    assert first.captured_monotonic_s == pytest.approx(99.9)

    with pytest.raises(CameraUnavailable, match="duplicate frame tuple"):
        camera.read()
    with pytest.raises(CameraUnavailable, match="no newer frame"):
        camera.read()

    replacement = camera.read()
    assert replacement.frame_id == 2
    assert replacement.stream_generation == "peer-b"
    assert replacement.captured_monotonic_s == pytest.approx(199.95)
    assert "after_generation=peer-a" in requested_urls[1]
    assert "after_frame_id=7" in requested_urls[1]

    telemetry = camera.telemetry()
    assert telemetry["request_count"] == 4
    assert telemetry["success_count"] == 2
    assert telemetry["error_count"] == 2
    assert telemetry["consecutive_errors"] == 0
    assert telemetry["generation"] == "peer-b"
    assert telemetry["source_frame_id"] == 1


def test_broker_camera_rejects_stale_source_without_refreshing_frame_id(
    monkeypatch,
) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    response = FakeHttpResponse(
        200,
        encoded.tobytes(),
        {
            "Content-Type": "image/jpeg",
            "X-Woof-Camera-Generation": "stale-peer",
            "X-Woof-Camera-Frame-Id": "3",
            "X-Woof-Camera-Age-Ms": "750",
            "X-Woof-Camera-Width": "16",
            "X-Woof-Camera-Height": "12",
            "X-Woof-Camera-Captured-Monotonic-S": "50.0",
        },
    )
    monkeypatch.setattr(camera_module, "urlopen", lambda *_args, **_kwargs: response)
    camera = BrokerCamera("http://voice/api/camera/frame.jpg")

    with pytest.raises(CameraUnavailable, match="frame is stale"):
        camera.read()

    telemetry = camera.telemetry()
    assert telemetry["success_count"] == 0
    assert telemetry["source_frame_id"] is None
