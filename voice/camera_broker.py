"""Sensor-only Go2 WebRTC camera broker state and HTTP routes.

The voice service owns the robot's single native WebRTC peer.  This module
shares decoded video from that peer without importing or exposing any Sport or
motion API.  The immutable latest-frame slot is broadcast to every HTTP
consumer; reads never consume a frame or rejuvenate its source timestamp.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse


CAMERA_SOURCE = "go2_webrtc_front"


@dataclass(frozen=True, slots=True)
class CameraPacket:
    generation: str
    frame_id: int
    jpeg: bytes
    captured_monotonic_s: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class FrameWaitResult:
    outcome: str
    packet: CameraPacket | None
    status: dict[str, object]


class CameraBrokerState:
    """Thread-safe latest-frame store with explicit transport generations."""

    def __init__(
        self,
        *,
        freshness_limit_s: float = 0.75,
        process_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if freshness_limit_s <= 0.0:
            raise ValueError("freshness_limit_s must be positive")
        self.freshness_limit_s = float(freshness_limit_s)
        self._process_id = process_id or uuid.uuid4().hex
        self._clock = clock
        self._condition = threading.Condition(threading.Lock())
        self._generation_counter = 0
        self._generation: str | None = None
        self._peer_state = "starting"
        self._video_enabled = False
        self._packet: CameraPacket | None = None
        self._last_frame_id: int | None = None
        self._last_frame_at: float | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._frames_total = 0
        self._frame_times: deque[float] = deque()
        self._reconnect_count = 0
        self._stale_reconnect_count = 0
        self._last_error = "waiting for WebRTC camera"
        self._terminal_track_error = ""
        self._continuous_fresh_started_at: float | None = None
        self._last_session_fresh_duration_s = 0.0

    def begin_session(self) -> str:
        """Rotate generation and clear all bytes before a connection attempt."""

        with self._condition:
            self._generation_counter += 1
            if self._generation_counter > 1:
                self._reconnect_count += 1
            self._generation = (
                f"{self._process_id}.{self._generation_counter}"
            )
            self._peer_state = "connecting"
            self._video_enabled = False
            self._packet = None
            self._last_frame_id = None
            self._last_frame_at = None
            self._width = None
            self._height = None
            self._frame_times.clear()
            self._last_error = "waiting for first decoded video frame"
            self._terminal_track_error = ""
            self._continuous_fresh_started_at = None
            self._last_session_fresh_duration_s = 0.0
            self._condition.notify_all()
            return self._generation

    def set_connected(
        self,
        generation: str,
        *,
        video_enabled: bool,
    ) -> bool:
        with self._condition:
            if generation != self._generation:
                return False
            self._peer_state = "connected"
            self._video_enabled = bool(video_enabled)
            self._last_error = (
                "waiting for first decoded video frame"
                if self._video_enabled
                else "WebRTC video channel is disabled"
            )
            self._condition.notify_all()
            return True

    def set_peer_state(self, generation: str, peer_state: str) -> None:
        rendered = str(peer_state or "unknown")
        with self._condition:
            if generation == self._generation:
                self._peer_state = rendered
                self._condition.notify_all()

    def publish(
        self,
        generation: str,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        captured_monotonic_s: float | None = None,
    ) -> CameraPacket | None:
        if not jpeg or width <= 0 or height <= 0:
            raise ValueError("camera packet requires JPEG bytes and dimensions")
        captured = (
            self._clock()
            if captured_monotonic_s is None
            else float(captured_monotonic_s)
        )
        with self._condition:
            if generation != self._generation or not self._video_enabled:
                return None
            frame_id = 1 if self._last_frame_id is None else self._last_frame_id + 1
            packet = CameraPacket(
                generation=generation,
                frame_id=frame_id,
                jpeg=bytes(jpeg),
                captured_monotonic_s=captured,
                width=int(width),
                height=int(height),
            )
            self._packet = packet
            self._last_frame_id = frame_id
            self._last_frame_at = captured
            self._width = int(width)
            self._height = int(height)
            self._frames_total += 1
            self._frame_times.append(captured)
            self._prune_frame_times_locked(captured)
            self._last_error = ""
            self._terminal_track_error = ""
            if self._continuous_fresh_started_at is None:
                self._continuous_fresh_started_at = captured
            self._condition.notify_all()
            return packet

    def note_track_error(self, generation: str, error: str) -> None:
        with self._condition:
            if generation != self._generation:
                return
            self._terminal_track_error = str(error)[:500]
            self._last_error = self._terminal_track_error
            self._condition.notify_all()

    def terminal_track_error(self, generation: str) -> str:
        with self._condition:
            if generation != self._generation:
                return "camera generation superseded"
            return self._terminal_track_error

    def disconnect(
        self,
        generation: str | None,
        error: str,
        *,
        stale: bool = False,
        now: float | None = None,
    ) -> None:
        current = self._clock() if now is None else float(now)
        with self._condition:
            if generation is not None and generation != self._generation:
                return
            if stale:
                self._stale_reconnect_count += 1
            if self._continuous_fresh_started_at is not None:
                self._last_session_fresh_duration_s = max(
                    0.0,
                    current - self._continuous_fresh_started_at,
                )
            self._continuous_fresh_started_at = None
            self._peer_state = "disconnected"
            self._video_enabled = False
            self._packet = None
            self._last_error = str(error)[:500]
            self._condition.notify_all()

    def last_session_fresh_duration_s(self) -> float:
        with self._condition:
            return self._last_session_fresh_duration_s

    def frame_marker(self, generation: str) -> tuple[int | None, float | None]:
        with self._condition:
            if generation != self._generation:
                return None, None
            return self._last_frame_id, self._last_frame_at

    def status(self, *, now: float | None = None) -> dict[str, object]:
        current = self._clock() if now is None else float(now)
        with self._condition:
            return self._status_locked(current)

    def wait_for_frame(
        self,
        *,
        after_generation: str | None,
        after_frame_id: int | None,
        timeout_s: float,
    ) -> FrameWaitResult:
        timeout_s = max(0.0, min(0.5, float(timeout_s)))
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                now = self._clock()
                status = self._status_locked(now)
                packet = self._packet if bool(status["ready"]) else None
                if packet is not None and (
                    after_generation is None
                    or packet.generation != after_generation
                    or (
                        after_frame_id is not None
                        and packet.frame_id > after_frame_id
                    )
                ):
                    return FrameWaitResult("frame", packet, status)
                if not bool(status["ready"]):
                    return FrameWaitResult("unavailable", None, status)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return FrameWaitResult("no_new_frame", None, status)
                self._condition.wait(timeout=remaining)

    def _status_locked(self, now: float) -> dict[str, object]:
        self._prune_frame_times_locked(now)
        age_s = (
            None
            if self._last_frame_at is None
            else max(0.0, now - self._last_frame_at)
        )
        ready = bool(
            self._peer_state == "connected"
            and self._video_enabled
            and self._packet is not None
            and age_s is not None
            and age_s < self.freshness_limit_s
        )
        fps_5s: float | None = None
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0.0:
                fps_5s = (len(self._frame_times) - 1) / elapsed
        return {
            "ready": ready,
            "source": CAMERA_SOURCE,
            "peer_state": self._peer_state,
            "video_enabled": self._video_enabled,
            "generation": self._generation,
            "frame_id": self._last_frame_id,
            "frames_total": self._frames_total,
            "last_frame_age_s": None if age_s is None else round(age_s, 3),
            "fps_5s": None if fps_5s is None else round(fps_5s, 1),
            "width": self._width,
            "height": self._height,
            "freshness_limit_s": self.freshness_limit_s,
            "reconnect_count": self._reconnect_count,
            "stale_reconnect_count": self._stale_reconnect_count,
            "last_error": self._last_error,
        }

    def _prune_frame_times_locked(self, now: float) -> None:
        cutoff = now - 5.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()


def _packet_headers(
    packet: CameraPacket,
    *,
    now: float,
) -> dict[str, str]:
    age_s = max(0.0, now - packet.captured_monotonic_s)
    return {
        "X-Woof-Camera-Generation": packet.generation,
        "X-Woof-Camera-Frame-Id": str(packet.frame_id),
        "X-Woof-Camera-Age-Ms": str(int(round(age_s * 1000.0))),
        "X-Woof-Camera-Width": str(packet.width),
        "X-Woof-Camera-Height": str(packet.height),
        "X-Woof-Camera-Captured-Monotonic-S": (
            f"{packet.captured_monotonic_s:.6f}"
        ),
        "Cache-Control": "no-store, no-cache, must-revalidate",
    }


def register_camera_routes(app: FastAPI, broker: CameraBrokerState) -> None:
    @app.get("/api/camera/status")
    def camera_status() -> dict[str, object]:
        return broker.status()

    @app.get("/api/camera/frame.jpg")
    def camera_frame(
        after_generation: str | None = None,
        after_frame_id: int | None = None,
        wait_ms: int = 400,
    ) -> Response:
        if (after_generation is None) != (after_frame_id is None):
            raise HTTPException(
                status_code=422,
                detail="after_generation and after_frame_id must be provided together",
            )
        result = broker.wait_for_frame(
            after_generation=after_generation,
            after_frame_id=after_frame_id,
            timeout_s=max(0, min(500, int(wait_ms))) / 1000.0,
        )
        if result.outcome == "frame" and result.packet is not None:
            return Response(
                result.packet.jpeg,
                media_type="image/jpeg",
                headers=_packet_headers(
                    result.packet,
                    now=broker._clock(),
                ),
            )
        if result.outcome == "no_new_frame":
            return Response(
                status_code=204,
                headers={"Cache-Control": "no-store"},
            )
        return JSONResponse(
            {
                "detail": str(
                    result.status.get("last_error")
                    or "camera frame is unavailable"
                ),
                "camera": result.status,
            },
            status_code=503,
            headers={
                "Retry-After": "0.25",
                "Cache-Control": "no-store",
            },
        )

    def mjpeg_frames():
        generation: str | None = None
        frame_id: int | None = None
        while True:
            result = broker.wait_for_frame(
                after_generation=generation,
                after_frame_id=frame_id,
                timeout_s=0.5,
            )
            if result.outcome != "frame" or result.packet is None:
                if result.outcome == "unavailable":
                    time.sleep(0.05)
                continue
            packet = result.packet
            generation, frame_id = packet.generation, packet.frame_id
            headers = _packet_headers(packet, now=broker._clock())
            part_headers = "".join(
                f"{name}: {value}\r\n" for name, value in headers.items()
            ).encode("ascii")
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(packet.jpeg)}\r\n".encode("ascii")
                + part_headers
                + b"\r\n"
                + packet.jpeg
                + b"\r\n"
            )

    @app.get("/api/camera/stream.mjpg")
    def camera_stream() -> StreamingResponse:
        return StreamingResponse(
            mjpeg_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Accel-Buffering": "no",
            },
        )
