import asyncio
from pathlib import Path

import pytest

from collie_demo.pointing import (
    PointingPolicyConfig,
    PointingPolicyError,
    PointingPolicyManager,
)
from collie_demo.pointing_shadow import EXPECTED_POLICY_SHA256


ROOT = Path(__file__).parents[1]
POLICY = ROOT / "models" / "pointing" / "policy_actor_42500.jit"


def test_stage_policy_is_hash_pinned_and_has_no_guard_bypass() -> None:
    manager = PointingPolicyManager(
        PointingPolicyConfig(enabled=True, policy_path=POLICY)
    )

    status = manager.status()
    command = manager._build_command(  # noqa: SLF001 - command is the contract
        target_label="pear",
        minimum_confidence=0.7,
        output_path=Path("/tmp/test-pointing.json"),
    )

    assert status["available"] is True
    assert status["policy_sha256"] == EXPECTED_POLICY_SHA256
    assert status["duration_s"] == 1.0
    assert status["safety"]["bypass_available"] is False
    assert "--full-power" in command
    assert "--execute" in command
    assert "--bypass-roll-guard" not in command
    assert command[command.index("--target-label") + 1] == "pear"
    assert command[command.index("--duration") + 1] == "1.000"


def test_stage_policy_cannot_be_configured_past_the_visible_safe_window() -> None:
    with pytest.raises(ValueError, match="between 0.1 and 1.0"):
        PointingPolicyConfig(
            enabled=True,
            policy_path=POLICY,
            duration_s=1.1,
        )


def test_wait_requires_a_started_policy_run() -> None:
    async def scenario() -> None:
        manager = PointingPolicyManager(
            PointingPolicyConfig(enabled=True, policy_path=POLICY)
        )
        with pytest.raises(PointingPolicyError, match="has not been started"):
            await manager.wait(timeout_s=1.0)

    asyncio.run(scenario())


def test_report_summary_keeps_recovery_and_peak_guard_telemetry() -> None:
    report = {
        "outcome": "completed_full_policy",
        "error": None,
        "selected_label": "apple",
        "policy_ticks": 50,
        "requested_duration_s": 1.0,
        "controller_restored": True,
        "final_standdown_rms_error_rad": 0.01,
        "records": [
            {
                "rpy": [0.1, -0.1, 0.0],
                "max_joint_speed_rad_s": 0.5,
                "max_estimated_torque": 3.0,
                "confidence": 0.75,
            },
            {
                "rpy": [-0.2, -0.1, 0.0],
                "max_joint_speed_rad_s": 0.8,
                "max_estimated_torque": 4.0,
                "confidence": 0.9,
            },
        ],
    }

    summary = PointingPolicyManager._summarize_report(report)  # noqa: SLF001

    assert summary["controller_restored"] is True
    assert summary["maximum_roll_rad"] == 0.2
    assert summary["maximum_joint_speed_rad_s"] == 0.8
    assert summary["maximum_estimated_torque"] == 4.0
    assert summary["confidence_range"] == [0.75, 0.9]
