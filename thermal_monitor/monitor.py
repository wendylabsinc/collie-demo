#!/usr/bin/env python3
"""Read-only thermal recorder and audible warning service for Woof."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from glob import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import struct
import threading
import time
from typing import Any, Iterable
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import wave


log = logging.getLogger("woof-thermal-monitor")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

UNITREE_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
THERMAL_BEEP_NAME = "woof_thermal_warning_beep"
THERMAL_BEEP_PATH = "/tmp/woof_thermal_warning_beep.wav"
LOW_BATTERY_NAME = "woof_low_battery"
LOW_BATTERY_PATH = "/opt/thermal-audio/woof_low_battery.wav"


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    port: int = 8102
    state_dir: str = "/state"
    sample_s: float = 2.0
    write_s: float = 30.0
    warning_c: float = 85.0
    critical_c: float = 95.0
    clear_c: float = 80.0
    warning_sustain_s: float = 10.0
    clear_sustain_s: float = 120.0
    beep_repeat_s: float = 60.0
    motor_warning_c: float = 70.0
    motor_critical_c: float = 80.0
    motor_clear_c: float = 65.0
    motor_rise_warning_c_per_min: float = 5.0
    motor_rise_min_delta_c: float = 8.0
    motor_rise_window_s: float = 180.0
    battery_warning_percent: int = 25
    battery_clear_percent: int = 30
    battery_retry_s: float = 60.0
    retention_days: int = 30
    beep_url: str = "http://127.0.0.1:8110/api/thermal/beep"
    battery_announce_url: str = "http://127.0.0.1:8098/api/battery/low"
    go2_interface: str = "enP8p1s0"
    go2_ip: str = "192.168.123.161"
    go2_reconnect_s: float = 2.0
    go2_sample_max_age_s: float = 2.0
    direct_audio_enabled: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            port=_env_int("WOOF_THERMAL_PORT", 8102),
            state_dir=os.environ.get("WOOF_THERMAL_STATE_DIR", "/state"),
            sample_s=_env_float("WOOF_THERMAL_SAMPLE_S", 2.0),
            write_s=_env_float("WOOF_THERMAL_WRITE_S", 30.0),
            warning_c=_env_float("WOOF_THERMAL_WARNING_C", 85.0),
            critical_c=_env_float("WOOF_THERMAL_CRITICAL_C", 95.0),
            clear_c=_env_float("WOOF_THERMAL_CLEAR_C", 80.0),
            warning_sustain_s=_env_float(
                "WOOF_THERMAL_WARNING_SUSTAIN_S", 10.0
            ),
            clear_sustain_s=_env_float(
                "WOOF_THERMAL_CLEAR_SUSTAIN_S", 120.0
            ),
            beep_repeat_s=_env_float("WOOF_THERMAL_BEEP_REPEAT_S", 60.0),
            motor_warning_c=_env_float("WOOF_MOTOR_WARNING_C", 70.0),
            motor_critical_c=_env_float("WOOF_MOTOR_CRITICAL_C", 80.0),
            motor_clear_c=_env_float("WOOF_MOTOR_CLEAR_C", 65.0),
            motor_rise_warning_c_per_min=_env_float(
                "WOOF_MOTOR_RISE_WARNING_C_PER_MIN", 5.0
            ),
            motor_rise_min_delta_c=_env_float(
                "WOOF_MOTOR_RISE_MIN_DELTA_C", 8.0
            ),
            motor_rise_window_s=_env_float(
                "WOOF_MOTOR_RISE_WINDOW_S", 180.0
            ),
            battery_warning_percent=_env_int(
                "WOOF_BATTERY_WARNING_PERCENT", 25
            ),
            battery_clear_percent=_env_int("WOOF_BATTERY_CLEAR_PERCENT", 30),
            battery_retry_s=_env_float("WOOF_BATTERY_RETRY_S", 60.0),
            retention_days=_env_int("WOOF_THERMAL_RETENTION_DAYS", 30),
            beep_url=os.environ.get(
                "WOOF_THERMAL_BEEP_URL",
                "http://127.0.0.1:8110/api/thermal/beep",
            ),
            battery_announce_url=os.environ.get(
                "WOOF_BATTERY_ANNOUNCE_URL",
                "http://127.0.0.1:8098/api/battery/low",
            ),
            go2_interface=os.environ.get(
                "GO2_NETWORK_INTERFACE", "enP8p1s0"
            ),
            go2_ip=os.environ.get("GO2_IP", "192.168.123.161"),
            go2_reconnect_s=_env_float("WOOF_GO2_RECONNECT_S", 2.0),
            go2_sample_max_age_s=_env_float(
                "WOOF_GO2_SAMPLE_MAX_AGE_S", 2.0
            ),
            direct_audio_enabled=_env_bool(
                "WOOF_DIRECT_AUDIO_ENABLED", False
            ),
        )

    def validate(self) -> None:
        if self.sample_s <= 0 or self.write_s <= 0:
            raise ValueError("sample and write intervals must be positive")
        if not self.clear_c < self.warning_c < self.critical_c:
            raise ValueError("expected clear < warning < critical thresholds")
        if not self.motor_clear_c < self.motor_warning_c < self.motor_critical_c:
            raise ValueError(
                "expected motor clear < warning < critical thresholds"
            )
        if (
            self.motor_rise_warning_c_per_min <= 0
            or self.motor_rise_min_delta_c <= 0
            or self.motor_rise_window_s <= 0
        ):
            raise ValueError("motor rise thresholds must be positive")
        if self.retention_days < 1:
            raise ValueError("retention must be at least one day")
        if not 0 <= self.battery_warning_percent < self.battery_clear_percent <= 100:
            raise ValueError(
                "expected battery warning < clear thresholds within 0..100"
            )
        if self.battery_retry_s <= 0:
            raise ValueError("battery retry interval must be positive")
        if not 0.1 <= self.go2_reconnect_s <= 60.0:
            raise ValueError("Go2 reconnect interval must be within 0.1..60 seconds")
        if not 0.1 <= self.go2_sample_max_age_s <= 60.0:
            raise ValueError(
                "Go2 sample freshness limit must be within 0.1..60 seconds"
            )


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def read_jetson_temperatures(
    pattern: str = "/sys/class/thermal/thermal_zone*",
) -> dict[str, float]:
    readings: dict[str, float] = {}
    for zone in sorted(glob(pattern)):
        try:
            name = _read_text(os.path.join(zone, "type"))
            millidegrees = int(_read_text(os.path.join(zone, "temp")))
        except (OSError, ValueError):
            continue
        readings[name] = round(millidegrees / 1000.0, 3)
    return readings


class Go2Telemetry:
    """Read-only subscriber for actuator, battery, NTC, and fan telemetry."""

    def __init__(
        self,
        interface: str,
        reconnect_s: float = 2.0,
        sample_max_age_s: float = 2.0,
    ) -> None:
        self.interface = interface
        self.reconnect_s = reconnect_s
        self.sample_max_age_s = sample_max_age_s
        self._lock = threading.Lock()
        self._sample: dict[str, Any] | None = None
        self._error = "waiting for rt/lowstate"
        self._subscriber: Any = None
        self._sample_event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._connected = False
        self._attempts = 0
        self._reconnect_count = 0
        self._reconnect_reason = ""
        self._last_attempt_monotonic_s: float | None = None
        self._factory_initialized = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop.clear()
            self._sample_event.clear()
        self._thread = threading.Thread(
            target=self._reconnect_until_ready,
            daemon=True,
            name="go2-telemetry-reconnect",
        )
        self._thread.start()

    def _connect_once(self) -> Any:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

        if not self._factory_initialized:
            ChannelFactoryInitialize(0, self.interface)
            self._factory_initialized = True
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        try:
            subscriber.Init(self._on_lowstate, 10)
        except Exception:
            try:
                subscriber.Close()
            except Exception:
                pass
            raise
        return subscriber

    def _close_subscriber(self, subscriber: Any) -> None:
        if subscriber is None:
            return
        try:
            subscriber.Close()
        except Exception:
            log.exception("Could not close Go2 telemetry subscriber")

    def _record_reconnect(self, reason: str, subscriber: Any) -> None:
        self._close_subscriber(subscriber)
        with self._lock:
            if self._subscriber is subscriber:
                self._subscriber = None
            self._sample = None
            self._connected = False
            self._error = reason
            self._reconnect_count += 1
            self._reconnect_reason = reason
        self._sample_event.clear()

    def _reconnect_until_ready(self) -> None:
        while not self._stop.is_set():
            subscriber: Any = None
            with self._lock:
                self._attempts += 1
                attempt = self._attempts
                self._last_attempt_monotonic_s = time.monotonic()
                self._sample = None
                self._connected = False
                self._error = "waiting for rt/lowstate after subscriber start"
            self._sample_event.clear()
            try:
                subscriber = self._connect_once()
                with self._lock:
                    self._subscriber = subscriber
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._record_reconnect(error, subscriber)
                log.warning(
                    "Go2 telemetry connection attempt %d failed on %s: %s; "
                    "retrying in %.1fs",
                    attempt,
                    self.interface,
                    error,
                    self.reconnect_s,
                )
                if self._stop.wait(self.reconnect_s):
                    return
                continue

            if not self._sample_event.wait(self.sample_max_age_s):
                reason = (
                    "first rt/lowstate sample did not arrive within "
                    f"{self.sample_max_age_s:.3f}s"
                )
                self._record_reconnect(reason, subscriber)
                log.warning(
                    "Go2 telemetry connection attempt %d produced no sample on %s; "
                    "retrying in %.1fs",
                    attempt,
                    self.interface,
                    self.reconnect_s,
                )
                if self._stop.wait(self.reconnect_s):
                    return
                continue
            if self._stop.is_set():
                return

            log.info(
                "Subscribed read-only to rt/lowstate on %s after %d attempt(s)",
                self.interface,
                attempt,
            )
            check_s = max(0.01, min(0.25, self.sample_max_age_s / 2.0))
            while not self._stop.wait(check_s):
                with self._lock:
                    sample = self._sample
                    received_at = (
                        None
                        if sample is None
                        else float(sample["received_monotonic_s"])
                    )
                age_s = (
                    float("inf")
                    if received_at is None
                    else max(0.0, time.monotonic() - received_at)
                )
                if age_s <= self.sample_max_age_s:
                    continue
                reason = (
                    f"stale rt/lowstate sample: {age_s:.3f}s exceeds "
                    f"{self.sample_max_age_s:.3f}s"
                )
                self._record_reconnect(reason, subscriber)
                log.warning(
                    "Go2 telemetry stream became stale on %s: %s; "
                    "retrying in %.1fs",
                    self.interface,
                    reason,
                    self.reconnect_s,
                )
                if self._stop.wait(self.reconnect_s):
                    return
                break
            else:
                return

    def stop(self) -> None:
        self._stop.set()
        self._sample_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.reconnect_s + 0.5))
        subscriber, self._subscriber = self._subscriber, None
        self._close_subscriber(subscriber)
        with self._lock:
            self._started = False
            self._connected = False

    def connection_status(self) -> dict[str, Any]:
        with self._lock:
            last_attempt_age_s = (
                None
                if self._last_attempt_monotonic_s is None
                else round(
                    max(0.0, time.monotonic() - self._last_attempt_monotonic_s),
                    3,
                )
            )
            return {
                "connected": self._connected,
                "attempts": self._attempts,
                "reconnect_count": self._reconnect_count,
                "reconnect_reason": self._reconnect_reason,
                "interface": self.interface,
                "reconnect_s": self.reconnect_s,
                "last_attempt_age_s": last_attempt_age_s,
                "error": self._error,
            }

    def _on_lowstate(self, message: Any) -> None:
        try:
            motor_c = {
                name: float(message.motor_state[index].temperature)
                for index, name in enumerate(UNITREE_JOINT_ORDER)
            }
            bms = message.bms_state
            sample = {
                "received_monotonic_s": time.monotonic(),
                "imu_c": float(message.imu_state.temperature),
                "motor_c": motor_c,
                "battery_bq_c": [float(value) for value in bms.bq_ntc],
                "battery_mcu_c": [float(value) for value in bms.mcu_ntc],
                "battery_soc_percent": int(bms.soc),
                "ntc_c": {
                    "ntc1": float(message.temperature_ntc1),
                    "ntc2": float(message.temperature_ntc2),
                },
                "power": {
                    "voltage_v": float(message.power_v),
                    "current_a": float(message.power_a),
                },
                "fan_hz": [int(value) for value in message.fan_frequency],
            }
        except Exception as exc:
            with self._lock:
                self._error = f"invalid rt/lowstate: {type(exc).__name__}: {exc}"
            return
        with self._lock:
            self._sample = sample
            self._error = ""
            self._connected = True
        self._sample_event.set()

    def snapshot(self) -> tuple[dict[str, Any] | None, str]:
        with self._lock:
            if self._sample is None:
                return None, self._error
            result = dict(self._sample)
            age_s = max(
                0.0, time.monotonic() - result.pop("received_monotonic_s")
            )
            if age_s > self.sample_max_age_s:
                self._connected = False
                self._error = (
                    f"stale rt/lowstate sample: {age_s:.3f}s exceeds "
                    f"{self.sample_max_age_s:.3f}s"
                )
                return None, self._error
            result["age_s"] = round(age_s, 3)
            return result, self._error


class AlertController:
    """Threshold state with sustained warning and cool-down hysteresis."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.level = "normal"
        self.warning_since: float | None = None
        self.clear_since: float | None = None
        self.last_beep_at: float | None = None

    def update(
        self,
        hottest_c: float,
        now: float,
        sensor_name: str = "monitored sensor",
    ) -> tuple[str, bool, str]:
        previous = self.level
        if hottest_c >= self.config.critical_c:
            self.level = "critical"
            if self.warning_since is None:
                self.warning_since = now
            self.clear_since = None
        elif hottest_c >= self.config.warning_c:
            if self.warning_since is None:
                self.warning_since = now
            self.clear_since = None
            if now - self.warning_since >= self.config.warning_sustain_s:
                self.level = "warning"
        else:
            self.warning_since = None
            if self.level != "normal" and hottest_c < self.config.clear_c:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= self.config.clear_sustain_s:
                    self.level = "normal"
                    self.clear_since = None
            elif self.level != "normal":
                self.clear_since = None

        transitioned_hot = (
            previous == "normal" and self.level in {"warning", "critical"}
        ) or (
            previous == "warning" and self.level == "critical"
        )
        repeat_due = self.level in {"warning", "critical"} and (
            self.last_beep_at is None
            or now - self.last_beep_at >= self.config.beep_repeat_s
        )
        should_beep = transitioned_hot or repeat_due
        if should_beep:
            self.last_beep_at = now
        if self.level == "normal":
            reason = "within configured thermal margin"
        else:
            reason = (
                f"{sensor_name} is {hottest_c:.1f} C "
                f"({self.level} threshold)"
            )
        return self.level, should_beep, reason


class BatteryAlertController:
    """Announce once per low-battery episode and retry failed audio delivery."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.low = False
        self.announced = False
        self.last_attempt_at: float | None = None

    def update(self, soc_percent: int, now: float) -> tuple[str, bool, str]:
        if self.low and soc_percent >= self.config.battery_clear_percent:
            self.low = False
            self.announced = False
            self.last_attempt_at = None
        elif not self.low and soc_percent <= self.config.battery_warning_percent:
            self.low = True

        retry_due = (
            self.last_attempt_at is None
            or now - self.last_attempt_at >= self.config.battery_retry_s
        )
        should_announce = self.low and not self.announced and retry_due
        if should_announce:
            self.last_attempt_at = now
        level = "low" if self.low else "normal"
        reason = (
            f"battery state of charge is {soc_percent}%"
            if self.low
            else "battery is above the configured warning boundary"
        )
        return level, should_announce, reason

    def record_result(self, success: bool) -> None:
        if success:
            self.announced = True


class MotorAlertController:
    """Detect absolute motor heat and unusually rapid temperature increases."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.level = "normal"
        self.warning_since: float | None = None
        self.clear_since: float | None = None
        self.last_beep_at: float | None = None
        self.history: dict[str, deque[tuple[float, float]]] = defaultdict(deque)

    def _rise(self, name: str, value: float, now: float) -> tuple[float, float]:
        samples = self.history[name]
        samples.append((now, value))
        cutoff = now - self.config.motor_rise_window_s
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.popleft()
        oldest_at, oldest_c = samples[0]
        elapsed_s = max(0.0, now - oldest_at)
        delta_c = value - oldest_c
        rate = 0.0 if elapsed_s <= 0 else delta_c * 60.0 / elapsed_s
        return round(rate, 3), round(delta_c, 3)

    def update(
        self, motor_c: dict[str, float], now: float
    ) -> dict[str, Any]:
        if not motor_c:
            return {
                "level": "unknown",
                "reason": "waiting for motor telemetry",
                "should_beep": False,
                "hottest_motor": "",
                "hottest_motor_c": None,
                "fastest_rising_motor": "",
                "fastest_rise_c_per_min": 0.0,
                "fastest_rise_delta_c": 0.0,
            }

        rises = {
            name: self._rise(name, float(value), now)
            for name, value in motor_c.items()
        }
        hottest_name, hottest_c = max(
            motor_c.items(), key=lambda item: float(item[1])
        )
        fastest_name, (fastest_rate, fastest_delta) = max(
            rises.items(), key=lambda item: item[1][0]
        )
        hottest_c = float(hottest_c)
        rapid = (
            fastest_rate >= self.config.motor_rise_warning_c_per_min
            and fastest_delta >= self.config.motor_rise_min_delta_c
        )
        absolute_warning = hottest_c >= self.config.motor_warning_c
        critical = hottest_c >= self.config.motor_critical_c
        previous = self.level

        if critical:
            self.level = "critical"
            if self.warning_since is None:
                self.warning_since = now
            self.clear_since = None
        elif absolute_warning or rapid:
            if self.warning_since is None:
                self.warning_since = now
            self.clear_since = None
            if now - self.warning_since >= self.config.warning_sustain_s:
                self.level = "warning"
        else:
            self.warning_since = None
            safe_to_clear = (
                hottest_c < self.config.motor_clear_c
                and fastest_rate
                < self.config.motor_rise_warning_c_per_min / 2.0
            )
            if self.level != "normal" and safe_to_clear:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= self.config.clear_sustain_s:
                    self.level = "normal"
                    self.clear_since = None
            elif self.level != "normal":
                self.clear_since = None

        transitioned_hot = (
            previous == "normal" and self.level in {"warning", "critical"}
        ) or (previous == "warning" and self.level == "critical")
        repeat_due = self.level in {"warning", "critical"} and (
            self.last_beep_at is None
            or now - self.last_beep_at >= self.config.beep_repeat_s
        )
        should_beep = transitioned_hot or repeat_due
        if should_beep:
            self.last_beep_at = now

        if critical:
            reason = f"motor.{hottest_name} is {hottest_c:.1f} C (critical)"
        elif absolute_warning:
            reason = f"motor.{hottest_name} is {hottest_c:.1f} C (warning)"
        elif rapid:
            reason = (
                f"motor.{fastest_name} rose {fastest_delta:.1f} C at "
                f"{fastest_rate:.1f} C/min"
            )
        elif self.level != "normal":
            reason = "motor temperatures are cooling toward the clear boundary"
        else:
            reason = "motor temperatures and rise rates are within limits"

        return {
            "level": self.level,
            "reason": reason,
            "should_beep": should_beep,
            "hottest_motor": hottest_name,
            "hottest_motor_c": hottest_c,
            "fastest_rising_motor": fastest_name,
            "fastest_rise_c_per_min": fastest_rate,
            "fastest_rise_delta_c": fastest_delta,
        }


def _audiohub_entries(response: Any) -> list[dict[str, Any]]:
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


def _write_thermal_beep(path: str = THERMAL_BEEP_PATH) -> None:
    """Create three short 880 Hz pulses without storing a binary asset."""

    sample_rate = 44_100
    amplitude = 13_000
    frames = bytearray()
    segments = ((0.16, True), (0.09, False)) * 2 + ((0.16, True),)
    phase = 0
    for duration_s, tone_on in segments:
        count = int(sample_rate * duration_s)
        for _ in range(count):
            value = (
                int(
                    amplitude
                    * math.sin(2.0 * math.pi * 880.0 * phase / sample_rate)
                )
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


def _audio_playback_hold_s(path: str) -> float:
    """Allow AudioHub playback to start and finish before WebRTC disconnects."""

    try:
        with wave.open(path, "rb") as stream:
            frame_rate = stream.getframerate()
            duration_s = stream.getnframes() / frame_rate if frame_rate else 0.0
    except (OSError, wave.Error):
        duration_s = 1.0
    return max(1.0, duration_s + 1.0)


class DirectGo2Audio:
    """Open a short-lived, audio-only WebRTC peer for standalone alarms."""

    def __init__(self, robot_ip: str) -> None:
        self.robot_ip = robot_ip
        self._lock = threading.Lock()
        self._audio_ids: dict[str, str] = {}

    async def _play_once(self, custom_name: str, path: str) -> str:
        from unitree_webrtc_connect import (
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
        from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub

        connection = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA, ip=self.robot_ip
        )
        try:
            await connection.connect()
            audiohub = WebRTCAudioHub(connection)
            unique_id = self._audio_ids.get(custom_name)
            if not unique_id:
                response = await audiohub.get_audio_list()
                unique_id = _named_audio_id(
                    _audiohub_entries(response), custom_name
                )
            if not unique_id:
                await audiohub.upload_audio_file(path)
                for _ in range(10):
                    response = await audiohub.get_audio_list()
                    unique_id = _named_audio_id(
                        _audiohub_entries(response), custom_name
                    )
                    if unique_id:
                        break
                    await asyncio.sleep(0.2)
            if not unique_id:
                raise RuntimeError(
                    f"{custom_name} was not registered by Go2 AudioHub"
                )
            await audiohub.play_by_uuid(unique_id)
            await asyncio.sleep(_audio_playback_hold_s(path))
            self._audio_ids[custom_name] = unique_id
            return unique_id
        finally:
            try:
                await connection.disconnect()
            except Exception:
                log.exception("Direct Go2 audio disconnect failed")

    async def _play_async(self, custom_name: str, path: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return await self._play_once(custom_name, path)
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    break
                delay_s = float(attempt * 2)
                log.warning(
                    "Direct Go2 audio attempt %d failed: %s; retrying in %.0fs",
                    attempt,
                    exc,
                    delay_s,
                )
                await asyncio.sleep(delay_s)
        assert last_error is not None
        raise last_error

    def play(self, custom_name: str, path: str) -> str:
        if not os.path.isfile(path):
            raise RuntimeError(f"audio asset is unavailable: {path}")
        with self._lock:
            return str(
                asyncio.run(
                    asyncio.wait_for(
                        self._play_async(custom_name, path), timeout=40.0
                    )
                )
            )


def _numeric_series(sample: dict[str, Any]) -> dict[str, float]:
    result = {
        f"jetson.{name}": float(value)
        for name, value in sample.get("jetson_c", {}).items()
    }
    go2 = sample.get("go2") or {}
    if go2.get("imu_c") is not None:
        result["go2.imu"] = float(go2["imu_c"])
    for name, value in go2.get("motor_c", {}).items():
        result[f"motor.{name}"] = float(value)
    for index, value in enumerate(go2.get("battery_bq_c", [])):
        result[f"battery.bq{index}"] = float(value)
    for index, value in enumerate(go2.get("battery_mcu_c", [])):
        result[f"battery.mcu{index}"] = float(value)
    for name, value in go2.get("ntc_c", {}).items():
        result[f"go2.{name}"] = float(value)
    return result


def select_alert_temperature(
    jetson_c: dict[str, float], go2: dict[str, Any] | None
) -> tuple[str, float]:
    """Return the hottest sensor covered by the audible alarm.

    Motor, battery, and NTC values remain recorded for inspection, but their
    hardware-specific limits have not been qualified. The Go2 IMU uses the
    same configurable operational boundary as the Jetson so that this
    high-value sensor cannot silently overheat.
    """
    monitored = {
        f"jetson.{name}": float(value) for name, value in jetson_c.items()
    }
    if go2 and go2.get("imu_c") is not None:
        monitored["go2.imu"] = float(go2["imu_c"])
    return max(monitored.items(), key=lambda item: item[1])


def aggregate_window(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    count = 0
    latest: dict[str, Any] = {}
    for sample in samples:
        count += 1
        latest = sample
        for name, value in _numeric_series(sample).items():
            values[name].append(value)
    aggregates = {
        name: {
            "min_c": round(min(series), 3),
            "avg_c": round(sum(series) / len(series), 3),
            "max_c": round(max(series), 3),
        }
        for name, series in sorted(values.items())
    }
    return {"sample_count": count, "temperatures": aggregates, "latest": latest}


class Store:
    def __init__(self, path: Path, retention_days: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.retention_days = retention_days
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    hottest_jetson_c REAL,
                    alert_level TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS samples_time ON samples(captured_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def write(self, payload: dict[str, Any]) -> None:
        captured_at = datetime.now(timezone.utc).isoformat()
        latest = payload.get("latest") or {}
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO samples (
                    captured_at, hottest_jetson_c, alert_level, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    captured_at,
                    latest.get("hottest_jetson_c"),
                    latest.get("alert", {}).get("level", "unknown"),
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                ),
            )

    def prune(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM samples WHERE captured_at < ?", (cutoff,)
            )
            return int(cursor.rowcount)

    def history(self, limit: int) -> list[dict[str, Any]]:
        limit = max(1, min(2880, int(limit)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT captured_at, payload_json FROM samples
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {"captured_at": captured_at, **json.loads(payload)}
            for captured_at, payload in rows
        ]


class ThermalMonitor:
    def __init__(
        self,
        config: Config,
        store: Store,
        go2: Go2Telemetry,
        direct_audio: DirectGo2Audio | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.go2 = go2
        self.alert = AlertController(config)
        self.motor_alert = MotorAlertController(config)
        self.battery_alert = BatteryAlertController(config)
        self.direct_audio = direct_audio or DirectGo2Audio(config.go2_ip)
        self._lock = threading.Lock()
        self._current: dict[str, Any] = {
            "ok": False,
            "error": "waiting for first thermal sample",
        }
        self._window: list[dict[str, Any]] = []
        self._last_write_at = time.monotonic()
        self._last_prune_at = 0.0
        self._stop = threading.Event()

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._current)
        result["config"] = {
            "sample_s": self.config.sample_s,
            "write_s": self.config.write_s,
            "warning_c": self.config.warning_c,
            "critical_c": self.config.critical_c,
            "clear_c": self.config.clear_c,
            "retention_days": self.config.retention_days,
            "alert_sources": ["jetson", "go2.imu", "go2.motors"],
            "motor_warning_c": self.config.motor_warning_c,
            "motor_critical_c": self.config.motor_critical_c,
            "motor_clear_c": self.config.motor_clear_c,
            "motor_rise_warning_c_per_min": (
                self.config.motor_rise_warning_c_per_min
            ),
            "motor_rise_min_delta_c": self.config.motor_rise_min_delta_c,
            "motor_rise_window_s": self.config.motor_rise_window_s,
            "battery_warning_percent": self.config.battery_warning_percent,
            "battery_clear_percent": self.config.battery_clear_percent,
            "direct_audio_enabled": self.config.direct_audio_enabled,
            "go2_reconnect_s": self.config.go2_reconnect_s,
            "go2_sample_max_age_s": self.config.go2_sample_max_age_s,
        }
        return result

    def stop(self) -> None:
        self._stop.set()

    def _beep(self) -> tuple[bool, str]:
        endpoint_error = "voice endpoint is disabled"
        if self.config.beep_url:
            request = Request(self.config.beep_url, data=b"", method="POST")
            try:
                with urlopen(request, timeout=15.0) as response:
                    payload = json.loads(response.read() or b"{}")
                if payload.get("ok"):
                    return True, ""
                endpoint_error = str(
                    payload.get("error") or "beep request failed"
                )
            except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
                endpoint_error = f"{type(exc).__name__}: {exc}"
        if not self.config.direct_audio_enabled:
            return False, endpoint_error
        try:
            _write_thermal_beep()
            self.direct_audio.play(THERMAL_BEEP_NAME, THERMAL_BEEP_PATH)
            return True, ""
        except Exception as exc:
            return False, (
                f"voice endpoint: {endpoint_error}; direct Go2 audio: "
                f"{type(exc).__name__}: {exc}"
            )

    def _announce_low_battery(self) -> tuple[bool, str]:
        endpoint_error = "voice endpoint is disabled"
        if self.config.battery_announce_url:
            request = Request(
                self.config.battery_announce_url, data=b"", method="POST"
            )
            try:
                with urlopen(request, timeout=15.0) as response:
                    payload = json.loads(response.read() or b"{}")
                if payload.get("ok"):
                    return True, ""
                endpoint_error = str(
                    payload.get("error")
                    or "low-battery announcement failed"
                )
            except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
                endpoint_error = f"{type(exc).__name__}: {exc}"
        if not self.config.direct_audio_enabled:
            return False, endpoint_error
        try:
            self.direct_audio.play(LOW_BATTERY_NAME, LOW_BATTERY_PATH)
            return True, ""
        except Exception as exc:
            return False, (
                f"voice endpoint: {endpoint_error}; direct Go2 audio: "
                f"{type(exc).__name__}: {exc}"
            )

    def sample_once(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        jetson = read_jetson_temperatures()
        if not jetson:
            sample = {
                "ok": False,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "error": "no readable Jetson thermal zones",
                "alert": {"level": "sensor_error", "beep": False},
            }
            with self._lock:
                self._current = sample
            return sample

        hottest_jetson_name, hottest_jetson_c = max(
            jetson.items(), key=lambda item: item[1]
        )
        go2, go2_error = self.go2.snapshot()
        connection_status = getattr(self.go2, "connection_status", None)
        go2_connection = (
            connection_status() if callable(connection_status) else None
        )
        battery_status: dict[str, Any] = {
            "level": "unknown",
            "reason": "waiting for battery telemetry",
            "announcement_requested": False,
            "announcement_ok": None,
            "announcement_error": "",
        }
        if go2 and go2.get("battery_soc_percent") is not None:
            soc_percent = int(go2["battery_soc_percent"])
            battery_level, should_announce, battery_reason = (
                self.battery_alert.update(soc_percent, now)
            )
            announcement_ok: bool | None = None
            announcement_error = ""
            if should_announce:
                announcement_ok, announcement_error = self._announce_low_battery()
                self.battery_alert.record_result(bool(announcement_ok))
                if announcement_ok:
                    log.warning(
                        "LOW BATTERY: %d%%; audible announcement played",
                        soc_percent,
                    )
                else:
                    log.error(
                        "LOW BATTERY: %d%%; announcement failed: %s",
                        soc_percent,
                        announcement_error,
                    )
            battery_status = {
                "level": battery_level,
                "reason": battery_reason,
                "announcement_requested": should_announce,
                "announcement_ok": announcement_ok,
                "announcement_error": announcement_error,
            }
        hottest_name, hottest_c = select_alert_temperature(jetson, go2)
        thermal_level, thermal_should_beep, thermal_reason = self.alert.update(
            hottest_c, now, sensor_name=hottest_name
        )
        motor_status = self.motor_alert.update(
            (go2 or {}).get("motor_c", {}), now
        )
        priorities = {"unknown": -1, "normal": 0, "warning": 1, "critical": 2}
        motor_level = str(motor_status["level"])
        if priorities.get(motor_level, -1) > priorities.get(thermal_level, -1):
            level = motor_level
            reason = str(motor_status["reason"])
        else:
            level = thermal_level
            reason = thermal_reason
        should_beep = thermal_should_beep or bool(motor_status["should_beep"])
        beep_ok: bool | None = None
        beep_error = ""
        if should_beep:
            beep_ok, beep_error = self._beep()
            if beep_ok:
                log.error("THERMAL %s: %s; audible alarm played", level, reason)
            else:
                log.error("THERMAL %s: %s; beep failed: %s", level, reason, beep_error)

        motor_should_beep = bool(motor_status.pop("should_beep"))
        sample = {
            "ok": True,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "jetson_c": jetson,
            "hottest_jetson_zone": hottest_jetson_name,
            "hottest_jetson_c": hottest_jetson_c,
            "hottest_monitored_sensor": hottest_name,
            "hottest_monitored_c": hottest_c,
            "go2": go2,
            "go2_error": go2_error,
            "go2_connection": go2_connection,
            "motor_alert": {
                **motor_status,
                "beep_requested": motor_should_beep,
            },
            "battery_alert": battery_status,
            "alert": {
                "level": level,
                "reason": reason,
                "beep_requested": should_beep,
                "beep_ok": beep_ok,
                "beep_error": beep_error,
            },
            "error": "",
        }
        with self._lock:
            self._current = sample
            self._window.append(sample)
        return sample

    def run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.sample_once(started)
                with self._lock:
                    write_due = started - self._last_write_at >= self.config.write_s
                    window = list(self._window) if write_due else []
                    if write_due:
                        self._window.clear()
                        self._last_write_at = started
                if window:
                    payload = aggregate_window(window)
                    self.store.write(payload)
                    log.info(
                        "Recorded %d samples; hottest monitored sensor %.1f C; alert=%s",
                        payload["sample_count"],
                        float(payload["latest"]["hottest_monitored_c"]),
                        payload["latest"]["alert"]["level"],
                    )
                if started - self._last_prune_at >= 86_400:
                    removed = self.store.prune()
                    self._last_prune_at = started
                    if removed:
                        log.info("Pruned %d expired thermal records", removed)
            except Exception:
                log.exception("Thermal sampling loop failed")
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.1, self.config.sample_s - elapsed))


def _handler(monitor: ThermalMonitor, store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: Any) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/api/status"}:
                self._json(200, monitor.status())
                return
            if parsed.path == "/healthz":
                status = monitor.status()
                self._json(200 if status.get("ok") else 503, status)
                return
            if parsed.path == "/api/history":
                query = parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["120"])[0])
                except ValueError:
                    self._json(400, {"ok": False, "error": "limit must be an integer"})
                    return
                self._json(200, {"ok": True, "samples": store.history(limit)})
                return
            self._json(404, {"ok": False, "error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            log.debug(format, *args)

    return Handler


def main() -> None:
    config = Config.from_env()
    config.validate()
    store = Store(
        Path(config.state_dir) / "thermal.sqlite3",
        retention_days=config.retention_days,
    )
    go2 = Go2Telemetry(
        config.go2_interface,
        config.go2_reconnect_s,
        config.go2_sample_max_age_s,
    )
    go2.start()
    monitor = ThermalMonitor(config, store, go2)
    thread = threading.Thread(target=monitor.run, daemon=True, name="thermal-sampler")
    thread.start()
    server = ThreadingHTTPServer(("0.0.0.0", config.port), _handler(monitor, store))
    log.info(
        "Woof thermal monitor listening on :%d; sample=%.1fs write=%.1fs warning=%.1fC critical=%.1fC",
        config.port,
        config.sample_s,
        config.write_s,
        config.warning_c,
        config.critical_c,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        monitor.stop()
        thread.join(timeout=5.0)
        go2.stop()


if __name__ == "__main__":
    main()
