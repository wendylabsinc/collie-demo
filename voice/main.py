#!/usr/bin/env python3
"""Persistent Go2 microphone -> Scribe -> guarded Collie mission bridge."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import inspect
import json
import logging
import math
import os
import queue
import struct
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import wave

import cv2
from fastapi import FastAPI, HTTPException, WebSocket as FastAPIWebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import websocket

try:
    from .camera_broker import CameraBrokerState, register_camera_routes
    from .commands import parse_voice_command
except ImportError:  # Docker runs this file directly from /app.
    from camera_broker import CameraBrokerState, register_camera_routes
    from commands import parse_voice_command


GO2_IP = os.environ.get("GO2_IP", "192.168.123.161")
COLLIE_BASE_URL = os.environ.get(
    "COLLIE_BASE_URL", "http://127.0.0.1:8096"
).rstrip("/")
PORT = int(os.environ.get("COLLIE_VOICE_PORT", "8098"))
AUTO_START = os.environ.get("COLLIE_VOICE_AUTO_START", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
BARK_UUID = os.environ.get(
    "COLLIE_BARK_UUID", "161387de-21ab-4f0b-b4e9-97124b000d06"
).strip()
THERMAL_BEEP_NAME = "woof_thermal_warning_beep"
THERMAL_BEEP_PATH = "/tmp/woof_thermal_warning_beep.wav"
LOW_BATTERY_NAME = "woof_low_battery"
LOW_BATTERY_PATH = "/opt/collie-audio/woof_low_battery.wav"
STATE_ENV_PATH = os.environ.get(
    "COLLIE_VOICE_ENV_FILE", "/state/elevenlabs.env"
)
SCRIBE_MODEL = "scribe_v2_realtime"
SCRIBE_RATE = 16000
SCRIBE_CHUNK_MS = 100
SCRIBE_SAMPLE_BYTES = 2
SCRIBE_CHUNK_BYTES = (
    SCRIBE_RATE * SCRIBE_SAMPLE_BYTES * SCRIBE_CHUNK_MS // 1000
)
COMMAND_DEBOUNCE_S = 4.0
BARK_DURATION_S = 0.65
MISSION_POLL_S = float(os.environ.get("COLLIE_VOICE_MISSION_POLL_S", "0.10"))
MISSION_TIMEOUT_S = float(
    os.environ.get("COLLIE_VOICE_MISSION_TIMEOUT_S", "120")
)
STAGE_AUDIO_URL = os.environ.get("COLLIE_STAGE_AUDIO_URL", "").rstrip("/")
STAGE_MIC_FRESH_S = 0.75
CAMERA_FRESHNESS_LIMIT_S = 0.75
CAMERA_FIRST_PLI_S = 0.5
CAMERA_PLI_INTERVAL_S = 1.0
CAMERA_FIRST_FRAME_TIMEOUT_S = 5.0
CAMERA_MIDSTREAM_PLI_S = 0.4
CAMERA_WATCHDOG_PERIOD_S = 0.05
CAMERA_JPEG_QUALITY = int(os.environ.get("COLLIE_CAMERA_JPEG_QUALITY", "85"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("collie-voice")
logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)


class TypedCommandRequest(BaseModel):
    command: str


class _DropFrameSpam(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in {
            "Receiving audio frame",
            "Receiving video frame",
        }


logging.getLogger().addFilter(_DropFrameSpam())


class VoiceState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, object] = {
            "ok": False,
            "listening": AUTO_START,
            "webrtc_connected": False,
            "scribe_connected": False,
            "credential_configured": False,
            "mic_frames": 0,
            "mic_bytes": 0,
            "mic_last_at": None,
            "mic_age_s": None,
            "mic_source": "none",
            "go2_mic_frames": 0,
            "stage_mic_frames": 0,
            "stage_mic_last_at": None,
            "stage_mic_age_s": None,
            "stage_mic_clients": 0,
            "last_partial": "",
            "last_heard": "",
            "last_target": None,
            "last_event": "starting",
            "last_error": "",
            "mission_busy": False,
            "arrival_bark_status": "not_requested",
            "arrival_bark_error": "",
            "bark_ready": bool(BARK_UUID),
            "stage_speaker_connected": False,
            "stage_speaker_error": "",
            "last_spoken": "",
        }

    def update(self, **values: object) -> None:
        with self._lock:
            self._values.update(values)

    def increment_mic(self, byte_count: int, source: str) -> None:
        with self._lock:
            self._values["mic_frames"] = int(self._values["mic_frames"]) + 1
            self._values["mic_bytes"] = int(self._values["mic_bytes"]) + byte_count
            self._values["mic_last_at"] = time.monotonic()
            self._values["mic_source"] = source
            counter = (
                "stage_mic_frames" if source == "Thor USB" else "go2_mic_frames"
            )
            self._values[counter] = int(self._values[counter]) + 1
            if source == "Thor USB":
                self._values["stage_mic_last_at"] = time.monotonic()

    def stage_mic_fresh(self) -> bool:
        with self._lock:
            last_at = self._values["stage_mic_last_at"]
        return bool(
            last_at is not None
            and time.monotonic() - float(last_at) < STAGE_MIC_FRESH_S
        )

    def adjust_stage_clients(self, delta: int) -> None:
        with self._lock:
            current = int(self._values["stage_mic_clients"])
            self._values["stage_mic_clients"] = max(0, current + delta)

    def try_begin_mission(self, target: str) -> bool:
        with self._lock:
            if bool(self._values["mission_busy"]):
                return False
            self._values.update(
                {
                    "mission_busy": True,
                    "last_target": target,
                    "last_event": "validating_mission_preflight",
                    "last_error": "",
                    "arrival_bark_status": "pending",
                    "arrival_bark_error": "",
                }
            )
            return True

    def finish_mission(self, *, event: str, error: str = "") -> None:
        with self._lock:
            self._values.update(
                {
                    "mission_busy": False,
                    "last_event": event,
                    "last_error": error[-500:],
                }
            )

    def mission_busy(self) -> bool:
        with self._lock:
            return bool(self._values["mission_busy"])

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            result = dict(self._values)
        mic_last_at = result.pop("mic_last_at")
        stage_mic_last_at = result.pop("stage_mic_last_at")
        result["mic_age_s"] = (
            None
            if mic_last_at is None
            else round(max(0.0, time.monotonic() - float(mic_last_at)), 3)
        )
        result["stage_mic_age_s"] = (
            None
            if stage_mic_last_at is None
            else round(
                max(0.0, time.monotonic() - float(stage_mic_last_at)), 3
            )
        )
        result["input_ready"] = bool(
            result["webrtc_connected"]
            or (
                result["stage_mic_age_s"] is not None
                and float(result["stage_mic_age_s"]) < 1.0
            )
        )
        result["ok"] = bool(
            result["input_ready"]
            and result["credential_configured"]
            and (not result["listening"] or result["scribe_connected"])
        )
        result["model"] = SCRIBE_MODEL
        result["allowed_targets"] = ["apple", "banana", "pear"]
        result["command_pattern"] = "Say apple, banana, or pear"
        result["camera"] = camera_state.status()
        return result


state = VoiceState()
camera_state = CameraBrokerState(
    freshness_limit_s=CAMERA_FRESHNESS_LIMIT_S
)
mic_chunks: queue.Queue[bytes] = queue.Queue(maxsize=256)
shutdown_event = threading.Event()
listen_event = threading.Event()
if AUTO_START:
    listen_event.set()
command_lock = threading.Lock()
webrtc_loop_ref: asyncio.AbstractEventLoop | None = None
audiohub_ref: Any = None
thermal_beep_uuid: str | None = None
thermal_beep_lock = threading.Lock()
low_battery_uuid: str | None = None
low_battery_lock = threading.Lock()
active_scribe_ws: Any = None
active_scribe_ws_lock = threading.Lock()
last_command_key = ""
last_command_at = 0.0
mission_monitor_thread: threading.Thread | None = None


def _enqueue_pcm(pcm: bytes, source: str) -> None:
    if not pcm:
        return
    try:
        mic_chunks.put_nowait(pcm)
    except queue.Full:
        try:
            mic_chunks.get_nowait()
        except queue.Empty:
            pass
        mic_chunks.put_nowait(pcm)
    state.increment_mic(len(pcm), source)


def _load_api_key() -> str:
    direct = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if direct:
        return direct
    try:
        with open(STATE_ENV_PATH, encoding="utf-8") as handle:
            for line in handle:
                key, separator, value = line.strip().partition("=")
                if separator and key.strip() == "ELEVENLABS_API_KEY":
                    return value.strip().strip("\"'")
    except OSError as exc:
        state.update(last_error=f"credential file unavailable: {exc}")
    return ""


ELEVENLABS_API_KEY = _load_api_key()
state.update(credential_configured=bool(ELEVENLABS_API_KEY))


def _scribe_url() -> str:
    query = urlencode(
        {
            "model_id": SCRIBE_MODEL,
            "audio_format": "pcm_16000",
            "language_code": "en",
            "commit_strategy": "vad",
            "vad_threshold": "0.45",
            "vad_silence_threshold_secs": "0.45",
            "min_speech_duration_ms": "120",
            "min_silence_duration_ms": "300",
            "filter_background_audio": "true",
        }
    )
    return f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{query}"


def _collie_post(path: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        f"{COLLIE_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=8.0) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"Collie HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Collie unavailable: {exc.reason}") from exc
    return json.loads(body) if body else {}


def _collie_status() -> dict[str, object]:
    try:
        with urlopen(f"{COLLIE_BASE_URL}/api/status", timeout=4.0) as response:
            body = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Collie HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Collie unavailable: {exc.reason}") from exc
    return json.loads(body) if body else {}


def _require_stage_preflight() -> None:
    status = _collie_status()
    if not status.get("stage_ready"):
        failed = [
            str(name)
            for name, healthy in dict(status.get("health") or {}).items()
            if not healthy
        ]
        reason = ", ".join(failed) if failed else "unknown health gate"
        raise RuntimeError(f"stage is not ready: {reason}")
    if bool(dict(status.get("mission") or {}).get("active")):
        raise RuntimeError("a Collie mission is already active")
    if status.get("armed"):
        raise RuntimeError("Collie motion is already armed")


def _report_event(
    event: str,
    *,
    transcript: str = "",
    target: str | None = None,
    error: str = "",
) -> None:
    try:
        _collie_post(
            "/api/voice/event",
            {
                "event": event,
                "transcript": transcript,
                "target": target,
                "error": error,
            },
        )
    except Exception as exc:
        log.warning("Could not report voice event to Collie: %s", exc)


async def _play_bark_async() -> None:
    if audiohub_ref is None:
        raise RuntimeError("Go2 AudioHub is not connected")
    if not BARK_UUID:
        raise RuntimeError("real bark UUID is not configured")
    await audiohub_ref.play_by_uuid(BARK_UUID)


def _play_bark() -> None:
    if webrtc_loop_ref is None:
        raise RuntimeError("Go2 WebRTC loop is not connected")
    future = asyncio.run_coroutine_threadsafe(
        _play_bark_async(), webrtc_loop_ref
    )
    future.result(timeout=4.0)


def _audiohub_entries(response: Any) -> list[dict[str, Any]]:
    """Normalize AudioHub's nested JSON response into audio records."""

    if not isinstance(response, dict):
        return []
    payload: Any = response.get("data", response)
    if isinstance(payload, dict):
        payload = payload.get("data", payload)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    records = payload.get("audio_list", [])
    return [record for record in records if isinstance(record, dict)]


def _named_audio_id(
    records: list[dict[str, Any]], custom_name: str
) -> str | None:
    for record in records:
        if str(record.get("CUSTOM_NAME") or "") != custom_name:
            continue
        unique_id = str(record.get("UNIQUE_ID") or "").strip()
        if unique_id:
            return unique_id
    return None


def _thermal_beep_id(records: list[dict[str, Any]]) -> str | None:
    return _named_audio_id(records, THERMAL_BEEP_NAME)


def _write_thermal_beep(path: str = THERMAL_BEEP_PATH) -> None:
    """Create three short 880 Hz pulses without adding a binary asset."""

    sample_rate = 44_100
    amplitude = 13_000
    frames = bytearray()
    segments = ((0.16, True), (0.09, False)) * 2 + ((0.16, True),)
    phase = 0
    for duration_s, tone_on in segments:
        count = int(sample_rate * duration_s)
        for _ in range(count):
            value = (
                int(amplitude * math.sin(2.0 * math.pi * 880.0 * phase / sample_rate))
                if tone_on
                else 0
            )
            frames.extend(struct.pack("<h", value))
            phase += 1
    with wave.open(path, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(bytes(frames))


async def _ensure_thermal_beep_async() -> str:
    global thermal_beep_uuid

    if audiohub_ref is None:
        raise RuntimeError("Go2 AudioHub is not connected")
    if thermal_beep_uuid:
        return thermal_beep_uuid

    response = await audiohub_ref.get_audio_list()
    unique_id = _thermal_beep_id(_audiohub_entries(response))
    if unique_id is None:
        _write_thermal_beep(THERMAL_BEEP_PATH)
        await audiohub_ref.upload_audio_file(THERMAL_BEEP_PATH)
        for _ in range(5):
            response = await audiohub_ref.get_audio_list()
            unique_id = _thermal_beep_id(_audiohub_entries(response))
            if unique_id:
                break
            await asyncio.sleep(0.2)
    if unique_id is None:
        raise RuntimeError("thermal warning beep was not registered by AudioHub")
    thermal_beep_uuid = unique_id
    return unique_id


async def _play_thermal_beep_async() -> str:
    unique_id = await _ensure_thermal_beep_async()
    if audiohub_ref is None:
        raise RuntimeError("Go2 AudioHub disconnected before thermal beep")
    await audiohub_ref.play_by_uuid(unique_id)
    return unique_id


def _play_thermal_beep() -> str:
    if webrtc_loop_ref is None:
        raise RuntimeError("Go2 WebRTC loop is not connected")
    with thermal_beep_lock:
        future = asyncio.run_coroutine_threadsafe(
            _play_thermal_beep_async(), webrtc_loop_ref
        )
        return str(future.result(timeout=12.0))


async def _ensure_low_battery_audio_async() -> str:
    global low_battery_uuid

    if audiohub_ref is None:
        raise RuntimeError("Go2 AudioHub is not connected")
    if low_battery_uuid:
        return low_battery_uuid
    if not os.path.isfile(LOW_BATTERY_PATH):
        raise RuntimeError("low-battery announcement asset is unavailable")

    response = await audiohub_ref.get_audio_list()
    unique_id = _named_audio_id(_audiohub_entries(response), LOW_BATTERY_NAME)
    if unique_id is None:
        await audiohub_ref.upload_audio_file(LOW_BATTERY_PATH)
        for _ in range(5):
            response = await audiohub_ref.get_audio_list()
            unique_id = _named_audio_id(
                _audiohub_entries(response), LOW_BATTERY_NAME
            )
            if unique_id:
                break
            await asyncio.sleep(0.2)
    if unique_id is None:
        raise RuntimeError(
            "low-battery announcement was not registered by AudioHub"
        )
    low_battery_uuid = unique_id
    return unique_id


async def _play_low_battery_async() -> str:
    unique_id = await _ensure_low_battery_audio_async()
    if audiohub_ref is None:
        raise RuntimeError(
            "Go2 AudioHub disconnected before low-battery announcement"
        )
    await audiohub_ref.play_by_uuid(unique_id)
    return unique_id


def _play_low_battery() -> str:
    if webrtc_loop_ref is None:
        raise RuntimeError("Go2 WebRTC loop is not connected")
    with low_battery_lock:
        future = asyncio.run_coroutine_threadsafe(
            _play_low_battery_async(), webrtc_loop_ref
        )
        return str(future.result(timeout=12.0))


def _announce_on_stage(event: str, target: str | None = None) -> None:
    if not STAGE_AUDIO_URL:
        return
    request = Request(
        f"{STAGE_AUDIO_URL}/api/announce",
        data=json.dumps({"event": event, "target": target}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=4.0) as response:
            payload = json.loads(response.read() or b"{}")
        state.update(
            stage_speaker_connected=True,
            stage_speaker_error="",
            last_spoken=str(payload.get("phrase") or ""),
        )
    except Exception as exc:
        state.update(
            stage_speaker_connected=False,
            stage_speaker_error=str(exc)[-500:],
        )
        log.warning("Stage speaker announcement failed: %s", exc)


def _announce_on_stage_background(
    event: str, target: str | None = None
) -> None:
    threading.Thread(
        target=_announce_on_stage,
        args=(event, target),
        daemon=True,
        name=f"stage-announce-{event}",
    ).start()


def _stage_audio_supervisor() -> None:
    """Track Thor speaker health without delaying the UI status endpoint."""

    if not STAGE_AUDIO_URL:
        return
    while not shutdown_event.is_set():
        try:
            with urlopen(f"{STAGE_AUDIO_URL}/api/status", timeout=3.0) as response:
                payload = json.loads(response.read() or b"{}")
            state.update(
                stage_speaker_connected=bool(payload.get("speaker_ready")),
                stage_speaker_error=str(payload.get("last_error") or "")[-500:],
                last_spoken=str(payload.get("last_spoken") or ""),
            )
        except Exception as exc:
            state.update(
                stage_speaker_connected=False,
                stage_speaker_error=str(exc)[-500:],
            )
        shutdown_event.wait(2.0)


def _monitor_guarded_mission(round_id: str, target: str) -> None:
    """Keep voice input gated until the robot is home or safely aborted."""

    deadline = time.monotonic() + MISSION_TIMEOUT_S
    arrival_barked = False
    last_status_error = ""
    while not shutdown_event.is_set() and time.monotonic() < deadline:
        try:
            status = _collie_status()
            last_status_error = ""
        except Exception as exc:
            last_status_error = str(exc)
            state.update(
                last_event="voice_mission_status_reconnecting",
                last_error=last_status_error[-500:],
            )
            shutdown_event.wait(MISSION_POLL_S)
            continue

        current_round_id = str(status.get("round_id") or "")
        if round_id and current_round_id and current_round_id != round_id:
            state.finish_mission(
                event="voice_mission_superseded",
                error="the active Collie round changed",
            )
            return

        mission = dict(status.get("mission") or {})
        phase = str(mission.get("phase") or "")
        rest_status = str(mission.get("arrival_rest_status") or "")
        return_status = str(mission.get("return_home_status") or "")

        if rest_status == "holding" and not arrival_barked:
            arrival_barked = True
            state.update(
                last_event="barking_while_resting_at_target",
                arrival_bark_status="playing",
                arrival_bark_error="",
            )
            try:
                _play_bark()
            except Exception as exc:
                state.update(
                    arrival_bark_status="failed",
                    arrival_bark_error=str(exc)[-500:],
                )
                _report_event(
                    "arrival_bark_failed",
                    target=target,
                    error=str(exc),
                )
                log.warning("Arrival bark failed without aborting mission: %s", exc)
            else:
                state.update(
                    arrival_bark_status="complete",
                    arrival_bark_error="",
                )
                _report_event("arrival_bark_played", target=target)

        mission_active = bool(mission.get("active"))
        if (
            phase == "success"
            and return_status == "complete"
            and not mission_active
        ):
            _announce_on_stage_background("mission_complete", target)
            state.finish_mission(event="voice_mission_complete_ready")
            _report_event("voice_mission_complete_ready", target=target)
            return
        if phase == "aborted":
            reason = str(mission.get("reason") or "voice mission aborted")
            _announce_on_stage_background("mission_aborted", target)
            state.finish_mission(
                event="voice_mission_aborted_ready",
                error=reason,
            )
            _report_event(
                "voice_mission_aborted_ready",
                target=target,
                error=reason,
            )
            return

        shutdown_event.wait(MISSION_POLL_S)

    if shutdown_event.is_set():
        return
    try:
        _collie_post("/api/stop", {})
    except Exception as exc:
        last_status_error = str(exc)
    error = "voice mission monitor timed out"
    if last_status_error:
        error = f"{error}: {last_status_error}"
    state.finish_mission(event="voice_mission_monitor_timeout", error=error)
    _report_event(
        "voice_mission_monitor_timeout",
        target=target,
        error=error,
    )


def _handle_committed_transcript(transcript: str) -> str:
    global last_command_at, last_command_key, mission_monitor_thread
    cleaned = " ".join(transcript.strip().split())
    if not cleaned:
        return "speech_ignored_empty"
    state.update(
        last_heard=cleaned,
        last_partial="",
        last_event="committed_transcript",
    )
    command = parse_voice_command(cleaned)
    if command is None:
        state.update(last_event="speech_ignored_not_a_command")
        _report_event("speech_ignored_not_a_command", transcript=cleaned)
        return "speech_ignored_not_a_command"

    command_key = f"{command.kind}:{command.target or ''}"
    now = time.monotonic()
    if command_key == last_command_key and now - last_command_at < COMMAND_DEBOUNCE_S:
        state.update(last_event="duplicate_command_ignored")
        return "duplicate_command_ignored"
    last_command_key = command_key
    last_command_at = now

    if command.kind == "stop":
        try:
            _collie_post("/api/stop", {})
            state.update(last_event="voice_stop_applied", last_error="")
            _report_event("voice_stop_applied", transcript=cleaned)
            return "voice_stop_applied"
        except Exception as exc:
            state.update(last_event="voice_stop_failed", last_error=str(exc))
            _report_event(
                "voice_stop_failed", transcript=cleaned, error=str(exc)
            )
            return "voice_stop_failed"

    if not command_lock.acquire(blocking=False):
        state.update(last_event="command_ignored_mission_busy")
        return "command_ignored_mission_busy"
    target = command.target
    assert target is not None
    if not state.try_begin_mission(target):
        command_lock.release()
        state.update(last_event="command_ignored_mission_busy")
        return "command_ignored_mission_busy"
    _report_event("voice_command_accepted", transcript=cleaned, target=target)
    try:
        _require_stage_preflight()
        _announce_on_stage_background("command_heard", target)
        state.update(last_event="submitting_guarded_mission")
        mission_status = _collie_post(
            "/api/voice/mission",
            {
                "target": target,
                "transcript": cleaned,
                "confirmation": "VOICE COMMAND HEARD",
            },
        )
        round_id = str(mission_status.get("round_id") or "")
        state.update(last_event="guarded_mission_active", last_error="")
        _report_event(
            "guarded_mission_started", transcript=cleaned, target=target
        )
        mission_monitor_thread = threading.Thread(
            target=_monitor_guarded_mission,
            args=(round_id, target),
            daemon=True,
            name=f"mission-monitor-{target}",
        )
        mission_monitor_thread.start()
        return "guarded_mission_active"
    except Exception as exc:
        state.finish_mission(event="voice_mission_failed", error=str(exc))
        _report_event(
            "voice_mission_failed",
            transcript=cleaned,
            target=target,
            error=str(exc),
        )
        return "voice_mission_failed"
    finally:
        command_lock.release()


def _scribe_sender(
    ws: Any, stop_event: threading.Event
) -> None:
    pending = bytearray()
    while not stop_event.is_set() and listen_event.is_set():
        try:
            chunk = mic_chunks.get(timeout=0.25)
        except queue.Empty:
            continue
        pending.extend(chunk)
        while len(pending) >= SCRIBE_CHUNK_BYTES:
            payload = bytes(pending[:SCRIBE_CHUNK_BYTES])
            del pending[:SCRIBE_CHUNK_BYTES]
            try:
                ws.send(
                    json.dumps(
                        {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": base64.b64encode(payload).decode(
                                "ascii"
                            ),
                            "commit": False,
                            "sample_rate": SCRIBE_RATE,
                        }
                    )
                )
            except Exception as exc:
                log.warning("Scribe audio sender stopped: %s", exc)
                stop_event.set()
                return


def _scribe_session() -> str:
    global active_scribe_ws
    if not ELEVENLABS_API_KEY:
        return "ELEVENLABS_API_KEY is not configured"
    ws = None
    sender_stop = threading.Event()
    sender_thread: threading.Thread | None = None
    try:
        while True:
            try:
                mic_chunks.get_nowait()
            except queue.Empty:
                break
        ws = websocket.create_connection(
            _scribe_url(),
            header=[f"xi-api-key: {ELEVENLABS_API_KEY}"],
            timeout=6,
        )
        ws.settimeout(0.5)
        with active_scribe_ws_lock:
            active_scribe_ws = ws
        state.update(
            scribe_connected=True,
            last_event="scribe_connected",
            last_error="",
        )
        _report_event("scribe_connected")
        sender_thread = threading.Thread(
            target=_scribe_sender,
            args=(ws, sender_stop),
            daemon=True,
            name="scribe-audio-sender",
        )
        sender_thread.start()
        while (
            not shutdown_event.is_set()
            and listen_event.is_set()
            and not sender_stop.is_set()
        ):
            try:
                message = ws.recv()
            except Exception as exc:
                if exc.__class__.__name__ in {
                    "TimeoutError",
                    "WebSocketTimeoutException",
                }:
                    continue
                raise
            if not message:
                raise RuntimeError("Scribe websocket closed")
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                continue
            event_type = str(
                event.get("message_type") or event.get("type") or ""
            )
            if event_type == "session_started":
                state.update(last_event="scribe_session_started")
            elif event_type == "partial_transcript":
                partial = str(
                    event.get("text") or event.get("transcript") or ""
                ).strip()
                if partial:
                    state.update(last_partial=partial)
            elif event_type == "committed_transcript":
                transcript = str(
                    event.get("text") or event.get("transcript") or ""
                ).strip()
                if transcript:
                    _handle_committed_transcript(transcript)
            elif event_type in {
                "error",
                "auth_error",
                "quota_exceeded",
                "transcriber_error",
                "input_error",
                "rate_limited",
                "queue_overflow",
                "resource_exhausted",
                "session_time_limit_exceeded",
                "chunk_size_exceeded",
                "insufficient_audio_activity",
                "unaccepted_terms",
            }:
                raise RuntimeError(f"Scribe {event_type}: {json.dumps(event)[-500:]}")
        return ""
    except Exception as exc:
        return str(exc)
    finally:
        sender_stop.set()
        with active_scribe_ws_lock:
            if active_scribe_ws is ws:
                active_scribe_ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if sender_thread is not None:
            sender_thread.join(timeout=2)
        state.update(scribe_connected=False, last_partial="")


def _scribe_supervisor() -> None:
    attempt = 0
    while not shutdown_event.is_set():
        if not listen_event.wait(timeout=0.5):
            continue
        failure = _scribe_session()
        if shutdown_event.is_set():
            return
        if not listen_event.is_set():
            attempt = 0
            state.update(last_event="listening_stopped", last_error="")
            continue
        attempt += 1
        delay = min(10.0, 0.75 * (2 ** min(attempt - 1, 4)))
        state.update(
            last_event="scribe_reconnecting",
            last_error=failure[-500:],
        )
        log.warning("Scribe session ended; reconnecting in %.2fs: %s", delay, failure)
        _report_event("scribe_reconnecting", error=failure[-500:])
        shutdown_event.wait(delay)


async def _on_mic_frame(frame: Any) -> None:
    try:
        # Prefer the stage speakerphone while it is actively delivering audio.
        # The Go2 microphone remains connected so it takes over automatically
        # if the Thor stream disappears.
        if state.stage_mic_fresh():
            return
        from av.audio.resampler import AudioResampler

        resampler = getattr(_on_mic_frame, "_resampler", None)
        if resampler is None:
            resampler = AudioResampler(
                format="s16", layout="mono", rate=SCRIBE_RATE
            )
            setattr(_on_mic_frame, "_resampler", resampler)
        for resampled in resampler.resample(frame):
            pcm = bytes(resampled.planes[0])[: resampled.samples * 2]
            _enqueue_pcm(pcm, "Go2")
    except Exception as exc:
        state.update(last_error=f"microphone decode failed: {exc}")


def _encode_video_frame(frame: Any) -> tuple[bytes, int, int]:
    image = frame.to_ndarray(format="bgr24")
    height, width = int(image.shape[0]), int(image.shape[1])
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY],
    )
    if not encoded_ok:
        raise RuntimeError("could not encode decoded WebRTC camera frame")
    return encoded.tobytes(), width, height


async def _on_video_track(track: Any, generation: str) -> None:
    first_frame = True
    while not shutdown_event.is_set():
        try:
            frame = await track.recv()
            jpeg, width, height = await asyncio.to_thread(
                _encode_video_frame,
                frame,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"video track ended: {type(exc).__name__}: {exc}"
            camera_state.note_track_error(generation, error)
            log.warning(error)
            return
        packet = camera_state.publish(
            generation,
            jpeg,
            width=width,
            height=height,
        )
        if packet is None:
            return
        if first_frame:
            first_frame = False
            log.info(
                "First decoded video frame generation=%s size=%dx%d",
                generation,
                width,
                height,
            )


async def _request_video_pli(connection: Any) -> bool:
    requested = False
    for transceiver in connection.pc.getTransceivers():
        receiver = getattr(transceiver, "receiver", None)
        track = None if receiver is None else getattr(receiver, "track", None)
        if track is None or getattr(track, "kind", None) != "video":
            continue
        send_pli = getattr(receiver, "_send_rtcp_pli", None)
        ssrc = getattr(receiver, "_ssrc", None) or getattr(
            receiver,
            "_track_id",
            None,
        )
        if not callable(send_pli) or ssrc is None:
            continue
        outcome = send_pli(ssrc)
        if inspect.isawaitable(outcome):
            await outcome
        requested = True
    if requested:
        log.info("Requested Go2 H.264 keyframe (PLI)")
    else:
        log.warning("Could not find a Go2 video receiver for PLI")
    return requested


class _CameraSessionFailure(RuntimeError):
    def __init__(self, message: str, *, stale: bool = False) -> None:
        super().__init__(message)
        self.stale = bool(stale)


def _camera_watchdog_action(
    *,
    connected_elapsed_s: float,
    frame_age_s: float | None,
    pli_elapsed_s: float | None,
) -> str:
    """Return the bounded PLI/reconnect action for current media freshness."""

    if frame_age_s is None:
        if connected_elapsed_s >= CAMERA_FIRST_FRAME_TIMEOUT_S:
            return "reconnect"
        if connected_elapsed_s >= CAMERA_FIRST_PLI_S and (
            pli_elapsed_s is None
            or pli_elapsed_s >= CAMERA_PLI_INTERVAL_S
        ):
            return "pli"
        return "wait"
    if frame_age_s >= CAMERA_FRESHNESS_LIMIT_S:
        return "reconnect"
    if frame_age_s >= CAMERA_MIDSTREAM_PLI_S and pli_elapsed_s is None:
        return "pli"
    return "wait"


def _next_webrtc_attempt(attempt: int, fresh_duration_s: float) -> int:
    return 1 if fresh_duration_s >= 10.0 else max(0, int(attempt)) + 1


def _webrtc_backoff_s(attempt: int) -> float:
    return min(10.0, 0.75 * (2 ** max(0, int(attempt) - 1)))


async def _webrtc_once() -> None:
    global audiohub_ref, webrtc_loop_ref
    from unitree_webrtc_connect import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )
    from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub

    generation = camera_state.begin_session()
    webrtc_loop_ref = asyncio.get_running_loop()
    connection = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA, ip=GO2_IP
    )
    disconnect_error = "WebRTC session ended"
    stale_disconnect = False
    try:
        await connection.connect()
        connection.video.add_track_callback(
            lambda track: _on_video_track(track, generation)
        )
        connection.video.switchVideoChannel(True)
        # The peer connection can be healthy while the robot's microphone
        # channel remains disabled. Register before switching the channel so
        # the first emitted sample cannot race callback installation.
        connection.audio.add_track_callback(_on_mic_frame)
        connection.audio.switchAudioChannel(True)
        audiohub_ref = WebRTCAudioHub(connection)
        camera_state.set_connected(generation, video_enabled=True)
        state.update(
            webrtc_connected=True,
            last_event="go2_microphone_camera_connected",
            last_error="",
        )
        _report_event("go2_microphone_camera_connected")
        connected_at = time.monotonic()
        observed_frame_id: int | None = None
        last_pli_at: float | None = None
        while not shutdown_event.is_set():
            peer_state = str(getattr(connection.pc, "connectionState", "connected"))
            camera_state.set_peer_state(generation, peer_state)
            if peer_state in {"failed", "closed", "disconnected"}:
                raise _CameraSessionFailure(
                    f"Go2 WebRTC state is {peer_state}"
                )
            track_error = camera_state.terminal_track_error(generation)
            if track_error:
                raise _CameraSessionFailure(track_error)
            now = time.monotonic()
            frame_id, last_frame_at = camera_state.frame_marker(generation)
            frame_age_s = (
                None
                if frame_id is None or last_frame_at is None
                else max(0.0, now - last_frame_at)
            )
            action = _camera_watchdog_action(
                connected_elapsed_s=max(0.0, now - connected_at),
                frame_age_s=frame_age_s,
                pli_elapsed_s=(
                    None
                    if last_pli_at is None
                    else max(0.0, now - last_pli_at)
                ),
            )
            if frame_age_s is None:
                if action == "reconnect":
                    raise _CameraSessionFailure(
                        "no first decoded WebRTC video frame within 5 seconds",
                        stale=True,
                    )
                if action == "pli":
                    try:
                        await _request_video_pli(connection)
                    except Exception as exc:
                        log.warning("Initial video PLI failed: %s", exc)
                    last_pli_at = now
            else:
                if frame_id != observed_frame_id:
                    observed_frame_id = frame_id
                    last_pli_at = None
                    action = _camera_watchdog_action(
                        connected_elapsed_s=max(0.0, now - connected_at),
                        frame_age_s=frame_age_s,
                        pli_elapsed_s=None,
                    )
                if action == "reconnect":
                    raise _CameraSessionFailure(
                        f"decoded WebRTC video stalled for {frame_age_s:.3f}s",
                        stale=True,
                    )
                if action == "pli":
                    try:
                        await _request_video_pli(connection)
                    except Exception as exc:
                        log.warning("Midstream video PLI failed: %s", exc)
                    last_pli_at = now
            await asyncio.sleep(CAMERA_WATCHDOG_PERIOD_S)
    except _CameraSessionFailure as exc:
        disconnect_error = str(exc)
        stale_disconnect = exc.stale
        raise
    except Exception as exc:
        disconnect_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.update(webrtc_connected=False)
        audiohub_ref = None
        camera_state.disconnect(
            generation,
            disconnect_error if not shutdown_event.is_set() else "shutdown",
            stale=stale_disconnect,
        )
        try:
            await connection.disconnect()
        except Exception:
            log.exception("Go2 WebRTC disconnect failed")


def _webrtc_supervisor() -> None:
    attempt = 0
    while not shutdown_event.is_set():
        try:
            asyncio.run(_webrtc_once())
        except Exception as exc:
            attempt = _next_webrtc_attempt(
                attempt,
                camera_state.last_session_fresh_duration_s(),
            )
            delay = _webrtc_backoff_s(attempt)
            state.update(
                webrtc_connected=False,
                last_event="go2_webrtc_reconnecting",
                last_error=str(exc)[-500:],
            )
            _report_event(
                "go2_webrtc_reconnecting", error=str(exc)[-500:]
            )
            shutdown_event.wait(delay)


webrtc_thread: threading.Thread | None = None
scribe_thread: threading.Thread | None = None
stage_audio_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global webrtc_thread, scribe_thread, stage_audio_thread
    shutdown_event.clear()
    webrtc_thread = threading.Thread(
        target=_webrtc_supervisor, daemon=True, name="go2-webrtc"
    )
    scribe_thread = threading.Thread(
        target=_scribe_supervisor, daemon=True, name="scribe-supervisor"
    )
    stage_audio_thread = threading.Thread(
        target=_stage_audio_supervisor,
        daemon=True,
        name="stage-audio-supervisor",
    )
    webrtc_thread.start()
    scribe_thread.start()
    stage_audio_thread.start()
    try:
        yield
    finally:
        shutdown_event.set()
        listen_event.set()
        with active_scribe_ws_lock:
            if active_scribe_ws is not None:
                try:
                    active_scribe_ws.close()
                except Exception:
                    pass
        if webrtc_thread is not None:
            webrtc_thread.join(timeout=3)
        if scribe_thread is not None:
            scribe_thread.join(timeout=3)
        if stage_audio_thread is not None:
            stage_audio_thread.join(timeout=3)


app = FastAPI(title="Collie Go2 voice bridge", version="1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
register_camera_routes(app, camera_state)


@app.get("/api/status")
def api_status() -> dict[str, object]:
    return state.snapshot()


@app.post("/api/command")
def typed_command(request: TypedCommandRequest) -> dict[str, object]:
    """Run a typed fruit label through the same guarded path as speech."""

    command = parse_voice_command(request.command)
    if command is None or command.kind != "find":
        raise HTTPException(
            status_code=422,
            detail="Type apple, banana, or pear.",
        )
    outcome = _handle_committed_transcript(request.command)
    snapshot = state.snapshot()
    if outcome != "guarded_mission_active":
        detail = {
            "duplicate_command_ignored": "That command was just submitted.",
            "command_ignored_mission_busy": "A fruit mission is already active.",
            "voice_mission_failed": str(
                snapshot.get("last_error") or "The guarded mission could not start."
            ),
        }.get(outcome, "The typed fruit command was not accepted.")
        raise HTTPException(status_code=409, detail=detail)
    return snapshot


@app.websocket("/api/audio/ws")
async def stage_audio_stream(websocket: FastAPIWebSocket) -> None:
    """Accept trusted LAN PCM16/16 kHz/mono chunks from the stage mic."""

    await websocket.accept()
    state.adjust_stage_clients(1)
    try:
        while True:
            pcm = await websocket.receive_bytes()
            if not pcm or len(pcm) > 131_072 or len(pcm) % 2:
                continue
            _enqueue_pcm(pcm, "Thor USB")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("Stage microphone stream ended: %s", exc)
    finally:
        state.adjust_stage_clients(-1)


@app.post("/api/listen/start")
def start_listening() -> dict[str, object]:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=503, detail="Scribe credential is not configured"
        )
    listen_event.set()
    state.update(listening=True, last_event="listening_requested", last_error="")
    _report_event("listening_requested")
    return state.snapshot()


@app.post("/api/listen/stop")
def stop_listening() -> dict[str, object]:
    listen_event.clear()
    state.update(listening=False, last_event="listening_stopped")
    with active_scribe_ws_lock:
        if active_scribe_ws is not None:
            try:
                active_scribe_ws.close()
            except Exception:
                pass
    _report_event("listening_stopped")
    return state.snapshot()


@app.post("/api/bark")
def bark() -> JSONResponse:
    try:
        _play_bark()
    except Exception as exc:
        state.update(last_event="bark_failed", last_error=str(exc))
        return JSONResponse(
            {"ok": False, "error": str(exc)}, status_code=503
        )
    state.update(last_event="bark_played", last_error="")
    return JSONResponse({"ok": True, "uuid": BARK_UUID})


@app.post("/api/thermal/beep")
def thermal_beep() -> JSONResponse:
    """Play the audible over-temperature alarm without issuing motor commands."""

    try:
        unique_id = _play_thermal_beep()
    except Exception as exc:
        state.update(last_event="thermal_beep_failed", last_error=str(exc))
        return JSONResponse(
            {"ok": False, "error": str(exc)}, status_code=503
        )
    state.update(last_event="thermal_beep_played", last_error="")
    return JSONResponse({"ok": True, "uuid": unique_id})


@app.post("/api/battery/low")
def low_battery() -> JSONResponse:
    """Speak the low-battery warning without issuing motor commands."""

    try:
        unique_id = _play_low_battery()
    except Exception as exc:
        state.update(
            last_event="low_battery_announcement_failed",
            last_error=str(exc),
        )
        return JSONResponse(
            {"ok": False, "error": str(exc)}, status_code=503
        )
    state.update(
        last_event="low_battery_announced",
        last_error="",
        last_spoken="Low battery",
    )
    return JSONResponse({"ok": True, "uuid": unique_id})


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
    )
