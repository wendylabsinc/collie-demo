from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import sqlite3
import sys
import threading
from urllib.request import urlopen


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


def test_dashboard_uses_wendy_brand_and_is_explicitly_read_only() -> None:
    dashboard = monitor.DASHBOARD.read_text(encoding="utf-8")

    assert "WENDY" in dashboard
    assert "#f1eee7" in dashboard
    assert "Read-only monitor" in dashboard
    assert "does not move, stop, or throttle Woof" in dashboard
    assert "http://127.0.0.1:8088/" in dashboard


def test_demo_advertises_read_only_monitor() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "wendy.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "wendy-demo.json").read_text(encoding="utf-8"))

    assert {"type": "http", "port": 8102} in manifest["entitlements"]
    assert metadata["safety"] == "view"
    assert metadata["links"][0] == {
        "label": "Open monitor",
        "port": 8102,
        "path": "/",
        "kind": "ui",
    }


def test_dashboard_and_status_api_are_served_on_separate_routes() -> None:
    class FakeMonitor:
        @staticmethod
        def status() -> dict[str, object]:
            return {"ok": True, "alert": {"level": "normal"}}

    class FakeStore:
        @staticmethod
        def history(_limit: int) -> list[object]:
            return []

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), monitor._handler(FakeMonitor(), FakeStore())
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base_url}/", timeout=2) as response:
            assert response.headers.get_content_type() == "text/html"
            assert b"Woof Thermal Monitor" in response.read()
        with urlopen(f"{base_url}/api/status", timeout=2) as response:
            assert json.load(response)["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


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
