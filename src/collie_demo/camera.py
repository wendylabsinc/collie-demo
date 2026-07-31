"""Brokered WebRTC and legacy VideoClient adapters for the Go2 camera."""

from __future__ import annotations

import json
import os
import time
from threading import Lock
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .types import CameraFrame


class VideoClientProtocol(Protocol):
    def GetImageSample(self) -> tuple[int, object]: ...


class CameraUnavailable(RuntimeError):
    pass


VIDEOCLIENT_API_TIMEOUT = 3104
DEFAULT_BROKER_FRAME_URL = "http://127.0.0.1:8098/api/camera/frame.jpg"


class UnitreeCamera:
    def __init__(
        self,
        client: VideoClientProtocol,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self._client = client
        self._frame_id = 0
        self._timeout_s = timeout_s
        self._telemetry_lock = Lock()
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._invalid_payload_count = 0
        self._consecutive_errors = 0
        self._consecutive_timeouts = 0
        self._last_response_code: int | None = None
        self._last_error_code: int | None = None
        self._last_error = ""
        self._last_success_at: float | None = None
        self._last_error_at: float | None = None
        self._last_request_duration_s: float | None = None
        self._maximum_request_duration_s = 0.0

    def read(self) -> CameraFrame:
        request_started = time.perf_counter()
        code, payload = self._client.GetImageSample()
        # GetImageSample is a blocking RPC.  Timestamping before it starts makes
        # a newly returned image appear older by the full camera/network wait,
        # which can consume most of the controller's freshness budget before
        # inference even begins.  The SDK payload has no source timestamp, so
        # receipt time is the honest local freshness boundary.
        received = time.monotonic()
        duration_s = time.perf_counter() - request_started
        response_code = int(code)
        with self._telemetry_lock:
            self._request_count += 1
            self._last_response_code = response_code
            self._last_request_duration_s = duration_s
            self._maximum_request_duration_s = max(
                self._maximum_request_duration_s,
                duration_s,
            )
        if response_code != 0:
            with self._telemetry_lock:
                self._error_count += 1
                self._consecutive_errors += 1
                self._last_error_code = response_code
                self._last_error_at = received
                if response_code == VIDEOCLIENT_API_TIMEOUT:
                    self._timeout_count += 1
                    self._consecutive_timeouts += 1
                    detail = "client API timeout"
                else:
                    self._consecutive_timeouts = 0
                    detail = "RPC error"
                self._last_error = detail
                timeout_count = self._timeout_count
                consecutive = self._consecutive_timeouts
            suffix = (
                f"; timeout_count={timeout_count}, consecutive={consecutive}"
                if response_code == VIDEOCLIENT_API_TIMEOUT
                else ""
            )
            raise CameraUnavailable(
                f"VideoClient returned {response_code} ({detail}{suffix})"
            )
        source_jpeg = bytes(payload)
        try:
            width, height = _jpeg_dimensions(source_jpeg)
        except CameraUnavailable as exc:
            with self._telemetry_lock:
                self._error_count += 1
                self._invalid_payload_count += 1
                self._consecutive_errors += 1
                self._consecutive_timeouts = 0
                self._last_error_code = None
                self._last_error = str(exc)
                self._last_error_at = received
            raise
        with self._telemetry_lock:
            self._success_count += 1
            self._consecutive_errors = 0
            self._consecutive_timeouts = 0
            self._last_success_at = received
        self._frame_id += 1
        return CameraFrame(
            self._frame_id,
            received,
            None,
            source_jpeg,
            width,
            height,
        )

    def telemetry(self) -> dict[str, object]:
        """Return persistent RPC health counters for the stage status API."""

        now = time.monotonic()
        with self._telemetry_lock:
            return {
                "source": "unitree_video_client_rpc",
                "timeout_s": self._timeout_s,
                "request_count": self._request_count,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "timeout_count": self._timeout_count,
                "invalid_payload_count": self._invalid_payload_count,
                "consecutive_errors": self._consecutive_errors,
                "consecutive_timeouts": self._consecutive_timeouts,
                "last_response_code": self._last_response_code,
                "last_error_code": self._last_error_code,
                "last_error": self._last_error,
                "last_success_age_s": _age_s(now, self._last_success_at),
                "last_error_age_s": _age_s(now, self._last_error_at),
                "last_request_duration_s": _rounded_duration(
                    self._last_request_duration_s
                ),
                "maximum_request_duration_s": _rounded_duration(
                    self._maximum_request_duration_s
                ),
            }


class BrokerCamera:
    """Bounded consumer of the voice service's latest WebRTC camera frame."""

    def __init__(
        self,
        frame_url: str,
        *,
        timeout_s: float = 0.75,
        maximum_source_age_s: float = 0.75,
        wait_ms: int = 400,
        maximum_jpeg_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if timeout_s <= 0.0 or maximum_source_age_s <= 0.0:
            raise ValueError("broker timeouts and source age must be positive")
        if maximum_jpeg_bytes < 1024:
            raise ValueError("maximum_jpeg_bytes is too small")
        self._frame_url = frame_url.strip()
        if not self._frame_url:
            raise ValueError("broker frame URL is required")
        self._timeout_s = float(timeout_s)
        self._maximum_source_age_s = float(maximum_source_age_s)
        self._wait_ms = max(0, min(500, int(wait_ms)))
        self._maximum_jpeg_bytes = int(maximum_jpeg_bytes)
        self._frame_id = 0
        self._source_generation: str | None = None
        self._source_frame_id: int | None = None
        self._telemetry_lock = Lock()
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._last_error = "waiting for camera broker"
        self._last_source_age_s: float | None = None
        self._last_remote_capture_s: float | None = None
        self._last_success_at: float | None = None

    def read(self) -> CameraFrame:
        params: dict[str, object] = {"wait_ms": self._wait_ms}
        if self._source_generation is not None and self._source_frame_id is not None:
            params.update(
                {
                    "after_generation": self._source_generation,
                    "after_frame_id": self._source_frame_id,
                }
            )
        separator = "&" if "?" in self._frame_url else "?"
        url = f"{self._frame_url}{separator}{urlencode(params)}"
        with self._telemetry_lock:
            self._request_count += 1
        try:
            request = Request(
                url,
                headers={"Cache-Control": "no-cache", "Accept": "image/jpeg"},
            )
            with urlopen(request, timeout=self._timeout_s) as response:
                status_value = getattr(response, "status", None)
                status = int(
                    response.getcode() if status_value is None else status_value
                )
                if status == 204:
                    raise CameraUnavailable("camera broker has no newer frame")
                if status != 200:
                    raise CameraUnavailable(f"camera broker returned HTTP {status}")
                content_type = str(response.headers.get("Content-Type", ""))
                if not content_type.lower().startswith("image/jpeg"):
                    raise CameraUnavailable(
                        f"camera broker returned {content_type or 'unknown content type'}"
                    )
                payload = response.read(self._maximum_jpeg_bytes + 1)
                if len(payload) > self._maximum_jpeg_bytes:
                    raise CameraUnavailable("camera broker JPEG exceeds size limit")
                headers = response.headers
        except CameraUnavailable as exc:
            self._record_error(str(exc))
            raise
        except HTTPError as exc:
            detail = _http_error_detail(exc, self._maximum_jpeg_bytes)
            error = f"camera broker HTTP {exc.code}: {detail}"
            self._record_error(error)
            raise CameraUnavailable(error) from exc
        except (URLError, OSError, TimeoutError) as exc:
            error = f"camera broker unavailable: {exc}"
            self._record_error(error)
            raise CameraUnavailable(error) from exc

        try:
            generation = _required_header(
                headers,
                "X-Woof-Camera-Generation",
            )
            source_frame_id = int(
                _required_header(headers, "X-Woof-Camera-Frame-Id")
            )
            source_age_s = int(
                _required_header(headers, "X-Woof-Camera-Age-Ms")
            ) / 1000.0
            width = int(_required_header(headers, "X-Woof-Camera-Width"))
            height = int(_required_header(headers, "X-Woof-Camera-Height"))
            remote_capture_s = float(
                _required_header(
                    headers,
                    "X-Woof-Camera-Captured-Monotonic-S",
                )
            )
        except (TypeError, ValueError) as exc:
            error = f"camera broker returned invalid frame metadata: {exc}"
            self._record_error(error)
            raise CameraUnavailable(error) from exc
        if not generation or len(generation) > 200 or source_frame_id < 1:
            error = "camera broker returned invalid generation/frame identity"
            self._record_error(error)
            raise CameraUnavailable(error)
        if source_age_s < 0.0 or source_age_s >= self._maximum_source_age_s:
            error = f"camera broker frame is stale ({source_age_s:.3f}s)"
            self._record_error(error)
            raise CameraUnavailable(error)
        if (
            generation == self._source_generation
            and self._source_frame_id is not None
            and source_frame_id == self._source_frame_id
        ):
            error = "camera broker returned a duplicate frame tuple"
            self._record_error(error)
            raise CameraUnavailable(error)
        if (
            generation == self._source_generation
            and self._source_frame_id is not None
            and source_frame_id < self._source_frame_id
        ):
            error = "camera broker returned an older frame tuple"
            self._record_error(error)
            raise CameraUnavailable(error)
        try:
            jpeg_width, jpeg_height = _jpeg_dimensions(payload)
        except CameraUnavailable as exc:
            self._record_error(str(exc))
            raise
        if (jpeg_width, jpeg_height) != (width, height):
            error = "camera broker JPEG dimensions do not match its metadata"
            self._record_error(error)
            raise CameraUnavailable(error)

        received = time.monotonic()
        captured = received - source_age_s
        self._frame_id += 1
        self._source_generation = generation
        self._source_frame_id = source_frame_id
        with self._telemetry_lock:
            self._success_count += 1
            self._consecutive_errors = 0
            self._last_error = ""
            self._last_source_age_s = source_age_s
            self._last_remote_capture_s = remote_capture_s
            self._last_success_at = received
        return CameraFrame(
            self._frame_id,
            captured,
            None,
            payload,
            width,
            height,
            stream_generation=generation,
        )

    def telemetry(self) -> dict[str, object]:
        now = time.monotonic()
        with self._telemetry_lock:
            return {
                "source": "voice_webrtc_camera_broker",
                "frame_url": self._frame_url,
                "request_timeout_s": self._timeout_s,
                "maximum_source_age_s": self._maximum_source_age_s,
                "request_count": self._request_count,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "consecutive_errors": self._consecutive_errors,
                "generation": self._source_generation,
                "source_frame_id": self._source_frame_id,
                "last_source_age_s": self._last_source_age_s,
                "remote_captured_monotonic_s": self._last_remote_capture_s,
                "last_success_age_s": _age_s(now, self._last_success_at),
                "last_error": self._last_error,
            }

    def _record_error(self, error: str) -> None:
        with self._telemetry_lock:
            self._error_count += 1
            self._consecutive_errors += 1
            self._last_error = str(error)[:500]


def _age_s(now: float, event_at: float | None) -> float | None:
    if event_at is None:
        return None
    return round(max(0.0, now - event_at), 3)


def _rounded_duration(duration_s: float | None) -> float | None:
    if duration_s is None:
        return None
    return round(max(0.0, duration_s), 3)


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions without decoding the 1920x1080 image."""

    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise CameraUnavailable("camera returned an invalid JPEG")
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    offset = 2
    while offset < len(payload):
        if payload[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker == 0xDA:  # Start of scan; SOF must already have appeared.
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue
        if offset + 2 > len(payload):
            break
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(payload):
            break
        if marker in sof_markers and segment_length >= 7:
            height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
            width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
            if width > 0 and height > 0:
                return width, height
            break
        offset += segment_length
    raise CameraUnavailable("camera returned a JPEG without dimensions")


def _required_header(headers: object, name: str) -> str:
    getter = getattr(headers, "get", None)
    value = None if not callable(getter) else getter(name)
    if value is None or not str(value).strip():
        raise ValueError(f"missing {name}")
    return str(value).strip()


def _http_error_detail(exc: HTTPError, maximum_bytes: int) -> str:
    try:
        payload = exc.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            return "response too large"
        decoded = json.loads(payload or b"{}")
        return str(decoded.get("detail") or exc.reason)
    except Exception:
        return str(exc.reason)


def create_camera(timeout_s: float = 3.0) -> UnitreeCamera | BrokerCamera:
    source = os.environ.get("COLLIE_CAMERA_SOURCE", "broker").strip().lower()
    if source in {"broker", "webrtc", "http"}:
        return BrokerCamera(
            os.environ.get(
                "COLLIE_CAMERA_BROKER_URL",
                DEFAULT_BROKER_FRAME_URL,
            ),
            timeout_s=float(
                os.environ.get("COLLIE_CAMERA_BROKER_TIMEOUT_S", "0.75")
            ),
            maximum_source_age_s=float(
                os.environ.get("COLLIE_CAMERA_MAX_SOURCE_AGE_S", "0.75")
            ),
            wait_ms=int(os.environ.get("COLLIE_CAMERA_BROKER_WAIT_MS", "400")),
        )
    if source != "unitree_rpc":
        raise CameraUnavailable(
            "COLLIE_CAMERA_SOURCE must be broker or unitree_rpc"
        )
    from unitree_sdk2py.go2.video.video_client import VideoClient

    client = VideoClient()
    client.SetTimeout(float(timeout_s))
    client.Init()
    return UnitreeCamera(client, timeout_s=float(timeout_s))
