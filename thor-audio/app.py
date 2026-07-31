#!/usr/bin/env python3
"""Oliver's-desk Thor stage microphone and allowlisted speaker service."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gi
import websocket

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


VOICE_WS_URL = os.environ.get(
    "COLLIE_VOICE_WS_URL", "ws://192.168.0.107:8098/api/audio/ws"
)
CAPTURE_DEVICE = os.environ.get("ALSA_CAPTURE_DEVICE", "hw:0,0")
PLAYBACK_DEVICE = os.environ.get("ALSA_PLAYBACK_DEVICE", "plughw:0,0")
PIPEWIRE_CAPTURE_NODE = os.environ.get(
    "PIPEWIRE_CAPTURE_NODE", "walker.echo.cancel.source"
)
PIPEWIRE_PLAYBACK_NODE = os.environ.get(
    "PIPEWIRE_PLAYBACK_NODE", "walker.echo.cancel.sink"
)
PCM_RATE = 16_000

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("collie-stage-audio")

Gst.init(None)
glib_loop = GLib.MainLoop()
threading.Thread(target=glib_loop.run, daemon=True, name="glib").start()


class StageAudio:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._speak_lock = threading.Lock()
        self._chunks: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._shutdown = threading.Event()
        self._pipeline: Any = None
        self._capture_thread: threading.Thread | None = None
        self._forward_thread: threading.Thread | None = None
        self._values: dict[str, object] = {
            "ok": False,
            "capture_device": CAPTURE_DEVICE,
            "playback_device": PLAYBACK_DEVICE,
            "capture_ready": False,
            "voice_bridge_connected": False,
            "samples": 0,
            "bytes": 0,
            "forwarded_samples": 0,
            "last_sample_at": None,
            "last_sample_age_s": None,
            "last_error": "",
            "last_spoken": "",
            "last_event": "starting",
        }

    def _update(self, **values: object) -> None:
        with self._lock:
            self._values.update(values)

    def status(self) -> dict[str, object]:
        with self._lock:
            result = dict(self._values)
        last_at = result.pop("last_sample_at")
        result["last_sample_age_s"] = (
            None
            if last_at is None
            else round(max(0.0, time.monotonic() - float(last_at)), 3)
        )
        result["speaker_ready"] = bool(PIPEWIRE_PLAYBACK_NODE)
        result["microphones"] = self._alsa_devices("arecord")
        result["speakers"] = self._alsa_devices("aplay")
        result["ok"] = bool(
            result["capture_ready"]
            and result["voice_bridge_connected"]
            and result["last_sample_age_s"] is not None
            and float(result["last_sample_age_s"]) < 1.0
        )
        result["voice_ws_url"] = VOICE_WS_URL
        return result

    @staticmethod
    def _alsa_devices(command: str) -> list[dict[str, str]]:
        try:
            output = subprocess.check_output(
                [command, "-l"],
                stderr=subprocess.STDOUT,
                timeout=2,
                text=True,
            )
        except Exception:
            return []
        devices: list[dict[str, str]] = []
        for line in output.splitlines():
            match = re.match(
                r"^card\s+(\d+):\s*([^\[]*).*device\s+(\d+):\s*([^\[]*)",
                line,
            )
            if match:
                card, card_name, device, device_name = match.groups()
                devices.append(
                    {
                        "id": f"hw:{card},{device}",
                        "name": f"{card_name.strip()} - {device_name.strip()}",
                    }
                )
        return devices

    def _on_sample(self, sink: Any) -> Any:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        mapped, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not mapped:
            return Gst.FlowReturn.OK
        pcm = bytes(mapinfo.data)
        buffer.unmap(mapinfo)
        try:
            self._chunks.put_nowait(pcm)
        except queue.Full:
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                pass
            self._chunks.put_nowait(pcm)
        with self._lock:
            self._values["samples"] = int(self._values["samples"]) + 1
            self._values["bytes"] = int(self._values["bytes"]) + len(pcm)
            self._values["last_sample_at"] = time.monotonic()
        return Gst.FlowReturn.OK

    def _start_capture(self) -> None:
        sink = (
            f"audio/x-raw,format=S16LE,channels=1,rate={PCM_RATE} ! "
            "appsink name=sink emit-signals=true max-buffers=8 drop=true sync=false"
        )
        # WendyOS owns the USB device through PipeWire, which permits safe
        # shared capture. Direct ALSA remains a fallback for simpler hosts.
        candidates = [
            (
                f"PipeWire {PIPEWIRE_CAPTURE_NODE}",
                f'pipewiresrc target-object="{PIPEWIRE_CAPTURE_NODE}" ! '
                f"audioconvert ! audioresample ! {sink}",
            ),
            (
                CAPTURE_DEVICE,
                f'alsasrc device="{CAPTURE_DEVICE}" ! audioconvert ! '
                f"audioresample ! {sink}",
            ),
        ]
        errors: list[str] = []
        for source, description in candidates:
            try:
                pipeline = Gst.parse_launch(description)
                app_sink = pipeline.get_by_name("sink")
                app_sink.connect("new-sample", self._on_sample)
                result = pipeline.set_state(Gst.State.PLAYING)
                if result == Gst.StateChangeReturn.FAILURE:
                    pipeline.set_state(Gst.State.NULL)
                    errors.append(f"{source}: state change failed")
                    continue
                self._pipeline = pipeline
                self._update(
                    capture_device=source,
                    capture_ready=True,
                    last_event="usb_speakerphone_capture_started",
                    last_error="",
                )
                log.info("Capturing USB speakerphone via %s", source)
                return
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        raise RuntimeError("could not start audio capture: " + "; ".join(errors))

    def _forward_loop(self) -> None:
        while not self._shutdown.is_set():
            client = None
            try:
                client = websocket.create_connection(
                    VOICE_WS_URL, timeout=5, enable_multithread=True
                )
                client.settimeout(5)
                self._update(
                    voice_bridge_connected=True,
                    last_event="woof_voice_bridge_connected",
                    last_error="",
                )
                while not self._shutdown.is_set():
                    try:
                        pcm = self._chunks.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    client.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)
                    with self._lock:
                        self._values["forwarded_samples"] = (
                            int(self._values["forwarded_samples"]) + 1
                        )
            except Exception as exc:
                self._update(
                    voice_bridge_connected=False,
                    last_event="woof_voice_bridge_reconnecting",
                    last_error=str(exc)[-300:],
                )
                self._shutdown.wait(1.0)
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

    def start(self) -> None:
        self._shutdown.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True,
            name="audio-capture",
        )
        self._capture_thread.start()
        self._forward_thread = threading.Thread(
            target=self._forward_loop,
            daemon=True,
            name="pcm-forwarder",
        )
        self._forward_thread.start()

    def _capture_worker(self) -> None:
        try:
            self._start_capture()
        except Exception as exc:
            self._update(
                capture_ready=False,
                last_event="capture_start_failed",
                last_error=str(exc)[-500:],
            )
            log.exception("Stage audio capture failed")

    def stop(self) -> None:
        self._shutdown.set()
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
        if self._forward_thread is not None:
            self._forward_thread.join(timeout=3)

    def announce(self, event: str, target: str | None) -> str:
        targets = {"apple", "banana", "pear"}
        if target is not None and target not in targets:
            raise ValueError("target must be apple, banana, or pear")
        phrases = {
            "ready": "Collie voice system ready.",
            "command_heard": f"I heard {target}. Searching now.",
            "target_found": f"I found the {target}.",
            "mission_complete": f"Mission complete. I found the {target}.",
            "mission_aborted": "Mission stopped.",
        }
        if event not in phrases or (event != "ready" and target is None):
            raise ValueError("unsupported stage announcement")
        phrase = phrases[event]
        with self._speak_lock, tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            subprocess.run(
                ["espeak-ng", "-v", "en-us", "-s", "155", "-w", wav.name, phrase],
                check=True,
                timeout=10,
            )
            subprocess.run(
                [
                    "gst-launch-1.0",
                    "-q",
                    "filesrc",
                    f"location={wav.name}",
                    "!",
                    "wavparse",
                    "!",
                    "audioconvert",
                    "!",
                    "audioresample",
                    "!",
                    "pipewiresink",
                    f"target-object={PIPEWIRE_PLAYBACK_NODE}",
                ],
                check=True,
                timeout=15,
            )
        self._update(
            last_spoken=phrase,
            last_event=f"speaker_{event}",
            last_error="",
        )
        return phrase


stage_audio = StageAudio()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        stage_audio.start()
    except Exception as exc:
        stage_audio._update(
            capture_ready=False,
            last_event="capture_start_failed",
            last_error=str(exc),
        )
        log.exception("Stage audio capture failed")
    try:
        yield
    finally:
        stage_audio.stop()


app = FastAPI(title="Collie stage audio", version="1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/status")
def api_status() -> dict[str, object]:
    return stage_audio.status()


@app.post("/api/announce")
def api_announce(payload: dict[str, object]) -> dict[str, object]:
    try:
        phrase = stage_audio.announce(
            str(payload.get("event") or ""),
            None if payload.get("target") is None else str(payload["target"]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        stage_audio._update(last_error=str(exc)[-300:])
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "phrase": phrase}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8099)
