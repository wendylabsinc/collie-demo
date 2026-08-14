from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor  # noqa: E402


def _config(**overrides: object) -> monitor.Config:
    values = {
        "warning_c": 85.0,
        "critical_c": 95.0,
        "clear_c": 80.0,
        "warning_sustain_s": 10.0,
        "clear_sustain_s": 120.0,
        "beep_repeat_s": 60.0,
    }
    values.update(overrides)
    return monitor.Config(**values)


def test_alert_requires_sustained_warning_but_critical_is_immediate() -> None:
    alert = monitor.AlertController(_config())

    assert alert.update(86.0, 0.0)[:2] == ("normal", False)
    assert alert.update(86.0, 9.9)[:2] == ("normal", False)
    assert alert.update(86.0, 10.0)[:2] == ("warning", True)
    assert alert.update(96.0, 11.0)[:2] == ("critical", True)


def test_alert_repeats_and_requires_sustained_cooling_to_clear() -> None:
    alert = monitor.AlertController(_config())

    alert.update(96.0, 0.0)
    assert alert.update(90.0, 59.0)[:2] == ("warning", False)
    assert alert.update(90.0, 60.0)[:2] == ("warning", True)
    assert alert.update(79.0, 61.0)[0] == "warning"
    assert alert.update(79.0, 180.9)[0] == "warning"
    assert alert.update(79.0, 181.0)[0] == "normal"


def test_battery_alert_announces_once_and_rearms_after_charging() -> None:
    alert = monitor.BatteryAlertController(
        _config(
            battery_warning_percent=25,
            battery_clear_percent=30,
            battery_retry_s=60.0,
        )
    )

    assert alert.update(26, 0.0)[:2] == ("normal", False)
    assert alert.update(25, 1.0)[:2] == ("low", True)
    alert.record_result(True)
    assert alert.update(24, 120.0)[:2] == ("low", False)
    assert alert.update(30, 121.0)[:2] == ("normal", False)
    assert alert.update(25, 122.0)[:2] == ("low", True)


def test_battery_alert_retries_failed_announcement() -> None:
    alert = monitor.BatteryAlertController(
        _config(battery_retry_s=60.0)
    )

    assert alert.update(25, 0.0)[:2] == ("low", True)
    alert.record_result(False)
    assert alert.update(24, 59.9)[:2] == ("low", False)
    assert alert.update(24, 60.0)[:2] == ("low", True)


def test_go2_telemetry_retries_failed_dds_initialization_without_restart(
    monkeypatch,
) -> None:
    telemetry = monitor.Go2Telemetry("enP8p1s0", reconnect_s=0.01)
    connected = threading.Event()
    attempts = 0

    class FakeReader:
        def Close(self) -> None:
            return None

    def connect_once() -> FakeReader:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("interface is not ready")
        with telemetry._lock:
            telemetry._sample = {
                "received_monotonic_s": time.monotonic(),
                "imu_c": 79.0,
                "motor_c": {},
            }
            telemetry._connected = True
            telemetry._error = ""
        telemetry._sample_event.set()
        connected.set()
        return FakeReader()

    monkeypatch.setattr(telemetry, "_connect_once", connect_once)

    telemetry.start()
    try:
        assert connected.wait(0.5)
        assert attempts == 3
        assert telemetry.connection_status()["connected"] is True
        assert telemetry.connection_status()["attempts"] == 3
        assert telemetry.connection_status()["reconnect_count"] == 2
        assert "interface is not ready" in telemetry.connection_status()[
            "reconnect_reason"
        ]
    finally:
        telemetry.stop()


def test_go2_snapshot_rejects_stale_data_as_disconnected() -> None:
    telemetry = monitor.Go2Telemetry(
        "enP8p1s0",
        reconnect_s=0.01,
        sample_max_age_s=0.02,
    )
    telemetry._sample = {
        "received_monotonic_s": time.monotonic() - 1.0,
        "imu_c": 79.0,
        "motor_c": {"FR_thigh_joint": 41.0},
    }
    telemetry._connected = True

    sample, error = telemetry.snapshot()
    status = telemetry.connection_status()

    assert sample is None
    assert "stale" in error
    assert status["connected"] is False


def test_go2_telemetry_retries_when_first_sample_never_arrives(
    monkeypatch,
) -> None:
    telemetry = monitor.Go2Telemetry(
        "enP8p1s0",
        reconnect_s=0.01,
        sample_max_age_s=0.03,
    )
    readers: list[FakeReader] = []

    class FakeReader:
        open_count = 0
        maximum_open_count = 0

        def __init__(self) -> None:
            self.closed = False
            self.close_calls = 0
            type(self).open_count += 1
            type(self).maximum_open_count = max(
                type(self).maximum_open_count,
                type(self).open_count,
            )

        def Close(self) -> None:
            self.close_calls += 1
            if self.closed:
                return
            self.closed = True
            type(self).open_count -= 1

    def connect_once() -> FakeReader:
        reader = FakeReader()
        readers.append(reader)
        return reader

    monkeypatch.setattr(telemetry, "_connect_once", connect_once)

    telemetry.start()
    deadline = time.monotonic() + 0.5
    while len(readers) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    telemetry.stop()

    assert len(readers) >= 2
    assert readers[0].closed is True
    assert FakeReader.maximum_open_count == 1
    assert FakeReader.open_count == 0
    assert all(reader.close_calls == 1 for reader in readers)
    status = telemetry.connection_status()
    assert status["reconnect_count"] >= 1
    assert "first rt/lowstate sample" in status["reconnect_reason"]


def test_go2_telemetry_reconnects_after_live_stream_becomes_stale(
    monkeypatch,
) -> None:
    telemetry = monitor.Go2Telemetry(
        "enP8p1s0",
        reconnect_s=0.01,
        sample_max_age_s=0.03,
    )
    readers: list[FakeReader] = []

    class FakeReader:
        open_count = 0
        maximum_open_count = 0

        def __init__(self) -> None:
            self.close_calls = 0
            type(self).open_count += 1
            type(self).maximum_open_count = max(
                type(self).maximum_open_count,
                type(self).open_count,
            )

        def Close(self) -> None:
            self.close_calls += 1
            type(self).open_count -= 1

    def connect_once() -> FakeReader:
        reader = FakeReader()
        readers.append(reader)
        with telemetry._lock:
            telemetry._sample = {
                "received_monotonic_s": time.monotonic(),
                "imu_c": 79.0,
                "motor_c": {"FR_thigh_joint": 41.0},
            }
            telemetry._connected = True
            telemetry._error = ""
        telemetry._sample_event.set()
        return reader

    monkeypatch.setattr(telemetry, "_connect_once", connect_once)

    telemetry.start()
    deadline = time.monotonic() + 0.5
    while len(readers) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    status = telemetry.connection_status()
    sample, error = telemetry.snapshot()
    telemetry.stop()

    assert len(readers) >= 2
    assert readers[0].close_calls == 1
    assert FakeReader.maximum_open_count == 1
    assert FakeReader.open_count == 0
    assert status["reconnect_count"] >= 1
    assert "stale rt/lowstate sample" in status["reconnect_reason"]
    assert sample is not None
    assert error == ""


def test_config_reads_and_validates_go2_sample_freshness(monkeypatch) -> None:
    monkeypatch.setenv("WOOF_GO2_SAMPLE_MAX_AGE_S", "3.5")

    config = monitor.Config.from_env()

    assert config.go2_sample_max_age_s == 3.5
    config.validate()
    with pytest.raises(ValueError, match="sample freshness"):
        monitor.Config(go2_sample_max_age_s=0.09).validate()
    with pytest.raises(ValueError, match="sample freshness"):
        monitor.Config(go2_sample_max_age_s=60.1).validate()


def test_deployed_config_requires_fresh_go2_data_and_disables_direct_audio() -> None:
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "wendy.json").read_text()
    )

    assert manifest["env"]["WOOF_GO2_SAMPLE_MAX_AGE_S"] == "2"
    assert manifest["env"]["WOOF_DIRECT_AUDIO_ENABLED"] == "0"
    assert manifest["env"]["WOOF_THERMAL_BEEP_URL"] == (
        "http://127.0.0.1:8110/api/thermal/beep"
    )


def test_motor_alert_warns_on_sustained_rapid_rise() -> None:
    alert = monitor.MotorAlertController(_config())

    first = alert.update({"FR_thigh_joint": 50.0}, 0.0)
    rising = alert.update({"FR_thigh_joint": 61.0}, 120.0)
    warned = alert.update({"FR_thigh_joint": 62.0}, 130.0)

    assert first["level"] == "normal"
    assert rising["level"] == "normal"
    assert rising["fastest_rise_c_per_min"] == 5.5
    assert warned["level"] == "warning"
    assert warned["should_beep"] is True
    assert "rose 12.0 C" in warned["reason"]


def test_motor_alert_warns_at_absolute_limit_and_critical_is_immediate() -> None:
    alert = monitor.MotorAlertController(_config())

    assert alert.update({"RR_hip_joint": 70.0}, 0.0)["level"] == "normal"
    warning = alert.update({"RR_hip_joint": 71.0}, 10.0)
    critical = alert.update({"RR_hip_joint": 80.0}, 11.0)

    assert warning["level"] == "warning"
    assert warning["should_beep"] is True
    assert critical["level"] == "critical"
    assert critical["should_beep"] is True
    assert critical["hottest_motor"] == "RR_hip_joint"


def test_alarm_falls_back_to_direct_go2_audio(monkeypatch) -> None:
    class FakeGo2:
        @staticmethod
        def snapshot():
            return None, ""

    class FakeDirectAudio:
        calls: list[tuple[str, str]] = []

        def play(self, name: str, path: str) -> str:
            self.calls.append((name, path))
            return "audio-id"

    def unavailable(*args, **kwargs):
        raise monitor.URLError("voice service stopped")

    monkeypatch.setattr(monitor, "urlopen", unavailable)
    direct_audio = FakeDirectAudio()
    service = monitor.ThermalMonitor(
        _config(direct_audio_enabled=True),
        store=object(),
        go2=FakeGo2(),
        direct_audio=direct_audio,
    )

    assert service._beep() == (True, "")
    assert direct_audio.calls == [
        (monitor.THERMAL_BEEP_NAME, monitor.THERMAL_BEEP_PATH)
    ]


def test_low_battery_falls_back_to_direct_go2_audio(monkeypatch) -> None:
    class FakeGo2:
        @staticmethod
        def snapshot():
            return None, ""

    class FakeDirectAudio:
        calls: list[tuple[str, str]] = []

        def play(self, name: str, path: str) -> str:
            self.calls.append((name, path))
            return "audio-id"

    def unavailable(*args, **kwargs):
        raise monitor.URLError("voice service stopped")

    monkeypatch.setattr(monitor, "urlopen", unavailable)
    direct_audio = FakeDirectAudio()
    service = monitor.ThermalMonitor(
        _config(direct_audio_enabled=True),
        store=object(),
        go2=FakeGo2(),
        direct_audio=direct_audio,
    )

    assert service._announce_low_battery() == (True, "")
    assert direct_audio.calls == [
        (monitor.LOW_BATTERY_NAME, monitor.LOW_BATTERY_PATH)
    ]


def test_direct_audio_retries_transient_webrtc_failure(monkeypatch) -> None:
    direct_audio = monitor.DirectGo2Audio("192.0.2.1")
    attempts = 0

    async def flaky(name: str, path: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("rate limited")
        return "audio-id"

    async def no_delay(seconds: float) -> None:
        return None

    monkeypatch.setattr(direct_audio, "_play_once", flaky)
    monkeypatch.setattr(monitor.asyncio, "sleep", no_delay)

    result = asyncio.run(direct_audio._play_async("alarm", "/tmp/alarm.wav"))

    assert result == "audio-id"
    assert attempts == 3


def test_direct_audio_holds_connection_for_wave_duration(tmp_path: Path) -> None:
    audio_path = tmp_path / "alarm.wav"
    with monitor.wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(b"\x00\x00" * 2_000)

    hold_s = monitor._audio_playback_hold_s(str(audio_path))

    assert abs(hold_s - 1.25) < 0.001


def test_read_jetson_temperatures_reads_all_zones(tmp_path: Path) -> None:
    for index, (name, value) in enumerate(
        (("cpu-thermal", "61250"), ("gpu-thermal", "59875"))
    ):
        zone = tmp_path / f"thermal_zone{index}"
        zone.mkdir()
        (zone / "type").write_text(name)
        (zone / "temp").write_text(value)

    assert monitor.read_jetson_temperatures(str(tmp_path / "thermal_zone*")) == {
        "cpu-thermal": 61.25,
        "gpu-thermal": 59.875,
    }


def test_aggregate_window_keeps_every_sensor() -> None:
    payload = monitor.aggregate_window(
        [
            {
                "jetson_c": {"cpu-thermal": 60.0},
                "go2": {
                    "imu_c": 70.0,
                    "motor_c": {"FR_hip_joint": 40.0},
                    "battery_bq_c": [30.0],
                    "battery_mcu_c": [32.0],
                    "ntc_c": {"ntc1": 48.0},
                },
            },
            {
                "jetson_c": {"cpu-thermal": 64.0},
                "go2": {
                    "imu_c": 74.0,
                    "motor_c": {"FR_hip_joint": 44.0},
                    "battery_bq_c": [32.0],
                    "battery_mcu_c": [34.0],
                    "ntc_c": {"ntc1": 50.0},
                },
            },
        ]
    )

    assert payload["sample_count"] == 2
    assert payload["temperatures"]["jetson.cpu-thermal"] == {
        "min_c": 60.0,
        "avg_c": 62.0,
        "max_c": 64.0,
    }
    assert payload["temperatures"]["motor.FR_hip_joint"]["avg_c"] == 42.0
    assert payload["temperatures"]["battery.bq0"]["max_c"] == 32.0
    assert payload["temperatures"]["go2.imu"]["avg_c"] == 72.0


def test_alert_temperature_includes_imu_but_not_unqualified_go2_sensors() -> None:
    name, value = monitor.select_alert_temperature(
        {"cpu-thermal": 60.0},
        {
            "imu_c": 79.0,
            "motor_c": {"RR_thigh_joint": 90.0},
            "battery_bq_c": [91.0],
            "ntc_c": {"ntc1": 92.0},
        },
    )

    assert (name, value) == ("go2.imu", 79.0)


def test_alert_temperature_falls_back_to_hottest_jetson_zone() -> None:
    assert monitor.select_alert_temperature(
        {"cpu-thermal": 60.0, "soc1-thermal": 62.0}, None
    ) == ("jetson.soc1-thermal", 62.0)


def test_critical_imu_temperature_drives_sample_alert(monkeypatch) -> None:
    class FakeGo2:
        @staticmethod
        def snapshot():
            return {"imu_c": 96.0}, ""

    monkeypatch.setattr(
        monitor,
        "read_jetson_temperatures",
        lambda: {"cpu-thermal": 60.0},
    )
    service = monitor.ThermalMonitor(
        _config(beep_url=""),
        store=object(),
        go2=FakeGo2(),
    )

    sample = service.sample_once(now=0.0)

    assert sample["hottest_jetson_c"] == 60.0
    assert sample["hottest_monitored_sensor"] == "go2.imu"
    assert sample["hottest_monitored_c"] == 96.0
    assert sample["alert"]["level"] == "critical"
    assert "go2.imu is 96.0 C" in sample["alert"]["reason"]


def test_store_persists_payload_and_history(tmp_path: Path) -> None:
    path = tmp_path / "thermal.sqlite3"
    store = monitor.Store(path, retention_days=30)
    payload = {
        "sample_count": 1,
        "temperatures": {},
        "latest": {
            "hottest_jetson_c": 61.0,
            "alert": {"level": "normal"},
        },
    }

    store.write(payload)

    assert store.history(1)[0]["latest"]["hottest_jetson_c"] == 61.0
    with sqlite3.connect(path) as connection:
        stored = connection.execute("SELECT payload_json FROM samples").fetchone()[0]
    assert json.loads(stored) == payload
