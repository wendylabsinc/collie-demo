import math

import pytest

from collie_demo.return_home import (
    Pose2D,
    ReturnMode,
    ReturnPlannerConfig,
    ReturnTurnDirectionLatch,
    assess_pose_window,
    plan_return_step,
)


CONFIG = ReturnPlannerConfig(
    arrival_tolerance_m=0.10,
    heading_tolerance_rad=math.radians(10),
    heading_gate_rad=math.radians(30),
    maximum_forward_mps=0.30,
    yaw_gain=1.2,
    maximum_yaw_rps=0.8,
)


def test_pose_window_averages_wrap_safe_yaw_and_reports_drift() -> None:
    assessment = assess_pose_window(
        [
            Pose2D(1.00, -2.00, math.radians(179)),
            Pose2D(1.02, -2.01, math.radians(-179)),
            Pose2D(1.01, -2.00, math.radians(180)),
        ]
    )

    assert assessment.sample_count == 3
    assert assessment.pose.x_m == pytest.approx(1.01)
    assert assessment.pose.y_m == pytest.approx(-2.003333333)
    assert abs(abs(assessment.pose.yaw_rad) - math.pi) < math.radians(0.1)
    assert assessment.maximum_position_span_m == pytest.approx(
        math.hypot(0.02, 0.01)
    )
    assert assessment.maximum_yaw_span_rad == pytest.approx(math.radians(2))


def test_pose_window_requires_multiple_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        assess_pose_window([Pose2D(0.0, 0.0, 0.0)])


def test_return_turns_toward_saved_home_before_translating() -> None:
    step = plan_return_step(
        home=Pose2D(0.0, 0.0, 0.0),
        current=Pose2D(1.0, 0.0, 0.0),
        config=CONFIG,
    )

    assert step.mode == ReturnMode.TURN_TO_HOME
    assert step.forward_mps == 0.0
    assert abs(abs(step.heading_error_rad) - math.pi) < 1.0e-6


def test_return_turn_direction_stays_latched_across_pi_noise() -> None:
    latch = ReturnTurnDirectionLatch.create(
        target_yaw_rad=math.radians(179.0),
        current_yaw_rad=0.0,
        release_progress_rad=math.radians(10.0),
        activation_margin_rad=math.radians(20.0),
    )

    assert latch.active is True
    assert latch.update(math.radians(-1.5)) > math.pi
    assert latch.update(math.radians(1.0)) > 0.0
    assert latch.active is True


def test_return_turn_direction_releases_after_clear_progress() -> None:
    latch = ReturnTurnDirectionLatch.create(
        target_yaw_rad=math.radians(179.0),
        current_yaw_rad=0.0,
        release_progress_rad=math.radians(10.0),
        activation_margin_rad=math.radians(20.0),
    )

    error = latch.update(math.radians(12.0))

    assert latch.active is False
    assert error == pytest.approx(math.radians(167.0))


def test_return_turn_direction_does_not_latch_an_unambiguous_turn() -> None:
    latch = ReturnTurnDirectionLatch.create(
        target_yaw_rad=math.radians(90.0),
        current_yaw_rad=0.0,
        release_progress_rad=math.radians(10.0),
        activation_margin_rad=math.radians(20.0),
    )

    assert latch.active is False
    assert latch.update(math.radians(2.0)) == pytest.approx(
        math.radians(88.0)
    )


def test_return_drives_with_closed_loop_yaw_inside_heading_gate() -> None:
    step = plan_return_step(
        home=Pose2D(0.0, 0.0, 0.0),
        current=Pose2D(1.0, 0.0, math.radians(170)),
        config=CONFIG,
    )

    assert step.mode == ReturnMode.DRIVE_TO_HOME
    assert 0.0 < step.forward_mps <= CONFIG.maximum_forward_mps
    assert 0.0 < step.yaw_rps <= CONFIG.maximum_yaw_rps


def test_return_restores_original_heading_only_after_position_arrival() -> None:
    step = plan_return_step(
        home=Pose2D(0.0, 0.0, math.radians(90)),
        current=Pose2D(0.05, 0.0, 0.0),
        config=CONFIG,
    )

    assert step.mode == ReturnMode.RESTORE_HEADING
    assert step.forward_mps == 0.0
    assert step.target_yaw_rad == pytest.approx(math.radians(90))


def test_return_completes_only_when_position_and_heading_match() -> None:
    step = plan_return_step(
        home=Pose2D(0.0, 0.0, math.radians(90)),
        current=Pose2D(0.05, 0.0, math.radians(85)),
        config=CONFIG,
    )

    assert step.mode == ReturnMode.COMPLETE
    assert step.forward_mps == 0.0


def test_return_speed_tapers_near_home() -> None:
    far = plan_return_step(
        home=Pose2D(0.0, 0.0, 0.0),
        current=Pose2D(1.0, 0.0, math.pi),
        config=CONFIG,
    )
    near = plan_return_step(
        home=Pose2D(0.0, 0.0, 0.0),
        current=Pose2D(0.15, 0.0, math.pi),
        config=CONFIG,
    )

    assert far.mode == near.mode == ReturnMode.DRIVE_TO_HOME
    assert far.forward_mps > near.forward_mps
