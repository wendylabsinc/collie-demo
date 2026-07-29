from __future__ import annotations

import sys
import threading
import types

sys.modules.setdefault(
    "websocket",
    types.SimpleNamespace(create_connection=lambda *args, **kwargs: None),
)
from voice import main


def _fresh_voice_state(monkeypatch) -> main.VoiceState:
    state = main.VoiceState()
    monkeypatch.setattr(main, "state", state)
    monkeypatch.setattr(main, "shutdown_event", threading.Event())
    monkeypatch.setattr(main, "MISSION_POLL_S", 0.001)
    monkeypatch.setattr(main, "MISSION_TIMEOUT_S", 0.1)
    return state


def test_mission_monitor_barks_during_hold_and_rearms_after_home(
    monkeypatch,
) -> None:
    state = _fresh_voice_state(monkeypatch)
    statuses = iter(
        [
            {
                "round_id": "round-1",
                "mission": {
                    "phase": "celebrating",
                    "active": True,
                    "arrival_rest_status": "holding",
                    "return_home_status": "pending",
                },
            },
            {
                "round_id": "round-1",
                "mission": {
                    "phase": "success",
                    "active": False,
                    "arrival_rest_status": "complete",
                    "return_home_status": "complete",
                },
            },
        ]
    )
    barks: list[str] = []
    announcements: list[tuple[str, str | None]] = []
    events: list[str] = []
    monkeypatch.setattr(main, "_collie_status", lambda: next(statuses))
    monkeypatch.setattr(main, "_play_bark", lambda: barks.append("bark"))
    monkeypatch.setattr(
        main,
        "_announce_on_stage_background",
        lambda event, target=None: announcements.append((event, target)),
    )
    monkeypatch.setattr(
        main,
        "_report_event",
        lambda event, **_: events.append(event),
    )

    assert state.try_begin_mission("pear") is True
    main._monitor_guarded_mission("round-1", "pear")

    snapshot = state.snapshot()
    assert barks == ["bark"]
    assert snapshot["arrival_bark_status"] == "complete"
    assert snapshot["mission_busy"] is False
    assert snapshot["last_event"] == "voice_mission_complete_ready"
    assert announcements == [("mission_complete", "pear")]
    assert events == ["arrival_bark_played", "voice_mission_complete_ready"]


def test_mission_monitor_rearms_after_safe_abort(monkeypatch) -> None:
    state = _fresh_voice_state(monkeypatch)
    monkeypatch.setattr(
        main,
        "_collie_status",
        lambda: {
            "round_id": "round-2",
            "mission": {
                "phase": "aborted",
                "active": False,
                "arrival_rest_status": "not_requested",
                "return_home_status": "failed",
                "reason": "operator_stop",
            },
        },
    )
    monkeypatch.setattr(main, "_announce_on_stage_background", lambda *args: None)
    monkeypatch.setattr(main, "_report_event", lambda *args, **kwargs: None)

    assert state.try_begin_mission("apple") is True
    assert state.try_begin_mission("banana") is False
    main._monitor_guarded_mission("round-2", "apple")

    snapshot = state.snapshot()
    assert snapshot["mission_busy"] is False
    assert snapshot["last_event"] == "voice_mission_aborted_ready"
    assert snapshot["last_error"] == "operator_stop"
    assert state.try_begin_mission("banana") is True


def test_committed_command_stays_busy_until_monitor_finishes(monkeypatch) -> None:
    state = _fresh_voice_state(monkeypatch)
    monkeypatch.setattr(main, "command_lock", threading.Lock())
    monkeypatch.setattr(main, "last_command_key", "")
    monkeypatch.setattr(main, "last_command_at", 0.0)
    monkeypatch.setattr(main, "BARK_DURATION_S", 0.0)
    monkeypatch.setattr(main, "_require_stage_preflight", lambda: None)
    monkeypatch.setattr(main, "_play_bark", lambda: None)
    monkeypatch.setattr(main, "_announce_on_stage_background", lambda *args: None)
    monkeypatch.setattr(main, "_report_event", lambda *args, **kwargs: None)
    submitted: list[str] = []

    def collie_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        submitted.append(str(payload.get("target")))
        return {"round_id": "round-3"}

    monkeypatch.setattr(main, "_collie_post", collie_post)
    started_threads: list[tuple[object, ...]] = []

    class FakeThread:
        def __init__(self, *, args=(), **kwargs) -> None:
            self.args = args

        def start(self) -> None:
            started_threads.append(self.args)

    monkeypatch.setattr(main.threading, "Thread", FakeThread)

    main._handle_committed_transcript("pear")
    assert state.snapshot()["mission_busy"] is True
    assert state.snapshot()["last_event"] == "guarded_mission_active"
    assert submitted == ["pear"]
    assert started_threads == [("round-3", "pear")]

    main._handle_committed_transcript("banana")
    assert state.snapshot()["mission_busy"] is True
    assert state.snapshot()["last_event"] == "command_ignored_mission_busy"
    assert submitted == ["pear"]
