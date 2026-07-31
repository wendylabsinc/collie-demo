import sys

import pytest

from collie_demo.supervisor import (
    pointing_warmup_command,
    process_failure_is_fatal,
    process_probe_ok,
)


def test_process_probe_accepts_responsive_but_sensor_degraded_app() -> None:
    assert process_probe_ok(
        200,
        {
            "ok": True,
            "stage_ready": False,
            "health": {"produce_live": False},
            "armed": False,
        },
    )


def test_process_probe_rejects_bad_or_malformed_api_response() -> None:
    assert process_probe_ok(503, {"ok": True}) is False
    assert process_probe_ok(200, {"ok": False}) is False
    assert process_probe_ok(200, "not-json-status") is False


def test_supervisor_ignores_cold_cuda_warmup_failures() -> None:
    assert process_failure_is_fatal(
        failures=12,
        failure_limit=12,
        elapsed_s=20.0,
        startup_grace_s=60.0,
    ) is False
    assert process_failure_is_fatal(
        failures=12,
        failure_limit=12,
        elapsed_s=60.0,
        startup_grace_s=60.0,
    ) is True


def test_pointing_warmup_is_disabled_unless_explicitly_enabled() -> None:
    assert pointing_warmup_command({}) is None
    assert (
        pointing_warmup_command({"COLLIE_POINTING_ENABLED": "false"})
        is None
    )


def test_pointing_warmup_loads_and_executes_the_configured_policy() -> None:
    command = pointing_warmup_command(
        {
            "COLLIE_POINTING_ENABLED": "1",
            "COLLIE_POINTING_POLICY": "/models/locked-point.jit",
        }
    )
    assert command is not None
    assert command[:2] == [sys.executable, "-c"]
    assert "torch.jit.load" in command[2]
    assert "LOCKED_POINT_OBSERVATION_SIZE" in command[2]
    assert command[-1] == "/models/locked-point.jit"


def test_pointing_warmup_requires_a_policy_path() -> None:
    with pytest.raises(RuntimeError, match="COLLIE_POINTING_POLICY"):
        pointing_warmup_command({"COLLIE_POINTING_ENABLED": "true"})
