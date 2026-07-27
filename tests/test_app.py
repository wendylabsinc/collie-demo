from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from collie_demo.app import create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.selected: str | None = None
        self.center: tuple[int, int] | None = None

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def status(self) -> dict[str, object]:
        return {"selected_target_name": self.selected}

    async def jpeg(self) -> bytes:
        return b"annotated-jpeg"

    async def raw_jpeg(self) -> bytes:
        return b"raw-jpeg"

    async def wait_for_stream_frame(
        self, last_frame_id: int
    ) -> tuple[int, bytes] | None:
        if last_frame_id < 1:
            return 1, b"stream-jpeg"
        return None

    async def select_target(
        self, target: str, center: tuple[int, int] | None = None
    ) -> dict[str, object]:
        self.selected = target
        self.center = center
        return await self.status()

    async def start_follow(self, confirmation: str) -> dict[str, object]:
        return {
            "selected_target_name": self.selected,
            "confirmation": confirmation,
            "follow_active": True,
        }

    async def remember_target(
        self,
        target: str,
        center: tuple[int, int] | None = None,
        expected_round_id: str | None = None,
    ) -> dict[str, object]:
        self.selected = target
        self.center = center
        return {
            "memory": {"id": "round-1", "label": target},
            "expected_round_id": expected_round_id,
        }

    async def reset_round(self) -> dict[str, object]:
        self.selected = None
        return {"round_id": "round-2", "memory": None}

    async def clear_memory(self) -> dict[str, object]:
        self.selected = None
        return {"memory": None}

    async def memory_reference_jpeg(self) -> bytes:
        return b"reference-jpeg"

    async def start_demo(self, confirmation: str) -> dict[str, object]:
        return {"mission": {"active": True}, "confirmation": confirmation}

    async def stop_demo(self) -> dict[str, object]:
        return {"mission": {"active": False}}

    async def prepare_pointing(self, confirmation: str) -> dict[str, object]:
        return {
            "pointing": {"prepared": True, "phase": "prepared"},
            "confirmation": confirmation,
        }

    async def start_pointing(self, confirmation: str) -> dict[str, object]:
        return {
            "pointing": {"active": True, "phase": "starting"},
            "confirmation": confirmation,
        }

    async def stop_pointing(self) -> dict[str, object]:
        return {"pointing": {"active": False, "phase": "aborted_safely"}}

    async def set_final_approach_distance(
        self, distance_m: float
    ) -> dict[str, object]:
        if not 0.02 <= distance_m <= 0.30:
            from collie_demo.runtime import RuntimeCommandError

            raise RuntimeCommandError(
                "final approach distance must be between 0.02 and 0.30 m"
            )
        return {"mission": {"config": {"final_approach_distance_m": distance_m}}}

    async def approve_demo_go(self, confirmation: str) -> dict[str, object]:
        return {
            "mission": {"active": True, "phase": "confirming"},
            "confirmation": confirmation,
        }

    async def record_voice_event(
        self,
        *,
        event: str,
        transcript: str = "",
        target: str | None = None,
        error: str = "",
    ) -> dict[str, object]:
        return {
            "voice": {
                "last_event": event,
                "last_heard": transcript,
                "last_target": target,
                "error": error,
            }
        }

    async def start_voice_mission(
        self,
        target: str,
        transcript: str,
        confirmation: str,
    ) -> dict[str, object]:
        return {
            "memory": {"label": target},
            "voice": {
                "last_heard": transcript,
                "last_target": target,
                "mission_active": True,
            },
            "confirmation": confirmation,
        }

    async def navigation_status(self) -> dict[str, object]:
        return {"available": True, "armed": False}

    async def navigation_arm(self, confirmation: str) -> dict[str, object]:
        return {"available": True, "armed": True, "confirmation": confirmation}

    async def navigation_command(
        self, forward_mps: float, yaw_rps: float
    ) -> dict[str, object]:
        return {
            "available": True,
            "armed": True,
            "command": {"forward_mps": forward_mps, "yaw_rps": yaw_rps},
        }

    async def stop(self, reason: str = "user_stop") -> dict[str, object]:
        return {"armed": False, "reason": reason}


def test_target_endpoint_selects_a_specific_detection(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/target", json={"target": "apple", "center": [1010, 240]}
        )

        assert response.status_code == 200
        assert response.json()["selected_target_name"] == "apple"
        assert runtime.center == (1010, 240)
        assert client.post("/api/target", json={"center": [1, 2]}).status_code == 422
        assert client.post(
            "/api/target", json={"target": "banana", "center": [1]}
        ).status_code == 422

        follow = client.post(
            "/api/follow", json={"confirmation": "TARGET AND PATH CLEAR"}
        )
        assert follow.status_code == 200
        assert follow.json()["follow_active"] is True
        assert client.get("/camera.jpg").content == b"annotated-jpeg"
        assert client.get("/camera-raw.jpg").content == b"raw-jpeg"
        with client.stream("GET", "/camera-stream.mjpg") as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith(
                "multipart/x-mixed-replace; boundary=frame"
            )
            payload = b"".join(stream.iter_bytes())
        assert b"Content-Type: image/jpeg" in payload
        assert b"stream-jpeg" in payload


def test_navigation_motion_endpoints_are_separate_from_fruit_follow(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        assert client.get("/api/navigation/status").json()["available"] is True
        armed = client.post(
            "/api/navigation/arm", json={"confirmation": "MAP AND PATH CLEAR"}
        )
        assert armed.status_code == 200
        assert armed.json()["armed"] is True
        command = client.post(
            "/api/navigation/cmd",
            json={"forward_mps": 0.3, "yaw_rps": -0.2},
        )
        assert command.status_code == 200
        assert command.json()["command"] == {
            "forward_mps": 0.3,
            "yaw_rps": -0.2,
        }
        assert client.post("/api/navigation/stop").json()["armed"] is False


def test_memory_and_demo_endpoints_are_local_and_explicit(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        saved = client.post(
            "/api/memory/capture",
            json={"target": "pear", "center": [300, 400], "round_id": "round-1"},
        )
        assert saved.status_code == 200
        assert saved.json()["memory"]["label"] == "pear"
        assert saved.json()["expected_round_id"] == "round-1"
        assert client.get("/api/memory/reference.jpg").content == b"reference-jpeg"
        started = client.post(
            "/api/demo/start",
            json={"confirmation": "TARGET SAVED AND AREA CLEAR"},
        )
        assert started.status_code == 200
        assert started.json()["mission"]["active"] is True
        go = client.post(
            "/api/demo/go",
            json={"confirmation": "CLASS LOCKED AND PATH CLEAR"},
        )
        assert go.status_code == 200
        assert go.json()["mission"]["phase"] == "confirming"
        assert client.post("/api/demo/stop").json()["mission"]["active"] is False
        reset = client.post("/api/round/reset")
        assert reset.status_code == 200
        assert reset.json()["round_id"] == "round-2"
        assert client.delete("/api/memory").json()["memory"] is None


def test_final_approach_calibration_is_bounded(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        calibrated = client.post(
            "/api/calibration/final-approach",
            json={"distance_m": 0.12},
        )
        assert calibrated.status_code == 200
        assert (
            calibrated.json()["mission"]["config"]["final_approach_distance_m"]
            == 0.12
        )
        assert client.post(
            "/api/calibration/final-approach",
            json={"distance_m": 0.31},
        ).status_code == 409


def test_pointing_policy_endpoints_are_explicit_and_stoppable(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        prepared = client.post(
            "/api/pointing/prepare",
            json={"confirmation": "WOOF IS CLEAR TO LIE DOWN"},
        )
        assert prepared.status_code == 200
        assert prepared.json()["pointing"]["prepared"] is True

        started = client.post(
            "/api/pointing/run",
            json={
                "confirmation": "WOOF IS LYING DOWN AND TARGET AREA IS CLEAR"
            },
        )
        assert started.status_code == 200
        assert started.json()["pointing"]["active"] is True

        stopped = client.post("/api/pointing/stop")
        assert stopped.status_code == 200
        assert stopped.json()["pointing"]["active"] is False


def test_voice_service_can_report_and_start_an_allowlisted_mission(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        event = client.post(
            "/api/voice/event",
            json={
                "event": "committed_transcript",
                "transcript": "Find the apple",
                "target": "apple",
            },
        )
        assert event.status_code == 200
        assert event.json()["voice"]["last_heard"] == "Find the apple"

        mission = client.post(
            "/api/voice/mission",
            json={
                "target": "apple",
                "transcript": "Find the apple",
                "confirmation": "VOICE COMMAND HEARD",
            },
        )
        assert mission.status_code == 200
        assert mission.json()["memory"]["label"] == "apple"
        assert mission.json()["voice"]["mission_active"] is True

        assert client.post(
            "/api/voice/mission",
            json={"target": "apple", "transcript": "Find the apple"},
        ).status_code == 422
