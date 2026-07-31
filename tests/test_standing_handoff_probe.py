from dataclasses import dataclass

import pytest

from collie_demo.pointing_runner import ContinuousJointHold
from collie_demo.standing_handoff_probe import (
    standing_reference_error_rad,
    validate_direct_handoff_start,
)
from collie_demo.pointing_contract import ISAAC_GROUNDED_STAND_RAD


@dataclass
class FakePublisher:
    writes: int = 0

    def write(self, target_unitree, *, kp, kd) -> None:
        assert len(target_unitree) == 12
        assert kp == 60.0
        assert kd == 5.0
        self.writes += 1


def test_continuous_hold_publishes_before_release_boundary() -> None:
    publisher = FakePublisher()
    hold = ContinuousJointHold(
        publisher=publisher,  # type: ignore[arg-type]
        target_isaac=ISAAC_GROUNDED_STAND_RAD,
        kp=60.0,
        kd=5.0,
    )

    hold.start()
    hold.stop()

    assert hold.ticks >= 1
    assert publisher.writes == hold.ticks


def test_policy_standing_pose_has_zero_reference_error() -> None:
    rms, maximum = standing_reference_error_rad(
        ISAAC_GROUNDED_STAND_RAD
    )

    assert rms == pytest.approx(0.0)
    assert maximum == pytest.approx(0.0)


def test_direct_handoff_rejects_stale_state() -> None:
    with pytest.raises(RuntimeError, match="stale"):
        validate_direct_handoff_start(
            {
                "received_at": 0.0,
                "q": [0.0] * 12,
                "dq": [0.0] * 12,
                "tau_est": [0.0] * 12,
                "rpy": [0.0, 0.0, 0.0],
            }
        )
