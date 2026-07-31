from pathlib import Path

import pytest

from collie_demo.pointing_contract import (
    ISAAC_GROUNDED_STAND_RAD,
    ISAAC_JOINT_ORDER,
    LOCKED_POINT_ACTION_JOINTS,
    LOCKED_POINT_ACTION_SIZE,
    LOCKED_POINT_OBSERVATION_SIZE,
    build_locked_point_observation,
    locked_point_joint_targets,
    selected_target_bbox_from_status,
)
from collie_demo.pointing_shadow import EXPECTED_POLICY_SHA256, sha256_file
from collie_demo.standing_pose import (
    LOCKED_POINT_FR_JOINTS_RAD,
    NOMINAL_STAND_JOINT_POSITIONS_RAD,
)


ZERO9 = [0.0] * 9
ZERO12 = [0.0] * 12
BBOX = [0.40, 0.35, 0.60, 0.65]


def test_locked_point_actor_artifact_is_hash_pinned() -> None:
    policy = (
        Path(__file__).parents[1]
        / "models"
        / "pointing"
        / "locked_point_actor.jit"
    )

    assert sha256_file(policy) == EXPECTED_POLICY_SHA256


def test_locked_point_actor_accepts_47_values_and_returns_9_actions() -> None:
    torch = pytest.importorskip("torch")
    policy = (
        Path(__file__).parents[1]
        / "models"
        / "pointing"
        / "locked_point_actor.jit"
    )
    actor = torch.jit.load(str(policy), map_location="cpu").eval()

    with torch.inference_mode():
        output = actor(torch.zeros((1, LOCKED_POINT_OBSERVATION_SIZE)))

    assert tuple(output.shape) == (1, LOCKED_POINT_ACTION_SIZE)
    assert bool(torch.isfinite(output).all())


def test_locked_point_observation_contract_is_47_values() -> None:
    observation = build_locked_point_observation(
        base_linear_velocity_body=[0.0] * 3,
        base_angular_velocity_body=[0.0] * 3,
        gravity_body=[0.0, 0.0, -1.0],
        joint_position_isaac=ISAAC_GROUNDED_STAND_RAD,
        joint_velocity_isaac=ZERO12,
        previous_action=ZERO9,
        point_phase=0.5,
        bbox_xyxy_normalized=BBOX,
    )

    assert LOCKED_POINT_OBSERVATION_SIZE == 47
    assert len(observation) == 47
    assert observation[9:21] == pytest.approx(ZERO12)
    assert observation[33:42] == pytest.approx(ZERO9)
    assert observation[42] == pytest.approx(0.5)
    assert observation[43:47] == pytest.approx(BBOX)


def test_locked_point_actor_controls_only_nine_support_joints() -> None:
    assert LOCKED_POINT_ACTION_SIZE == 9
    assert len(LOCKED_POINT_ACTION_JOINTS) == 9
    assert not any(name.startswith("FR_") for name in LOCKED_POINT_ACTION_JOINTS)


def test_front_right_point_is_imposed_by_phase_not_actor_action() -> None:
    quiet = locked_point_joint_targets(ZERO9, point_phase=1.0)
    loud = locked_point_joint_targets(
        [5.0, -5.0, 4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0],
        point_phase=1.0,
    )

    for name, quiet_value, loud_value in zip(
        ISAAC_JOINT_ORDER,
        quiet,
        loud,
        strict=True,
    ):
        if name.startswith("FR_"):
            assert quiet_value == pytest.approx(
                LOCKED_POINT_FR_JOINTS_RAD[name]
            )
            assert loud_value == pytest.approx(quiet_value)


def test_zero_action_starts_from_nominal_standing_pose() -> None:
    target = locked_point_joint_targets(ZERO9, point_phase=0.0)
    expected = tuple(
        NOMINAL_STAND_JOINT_POSITIONS_RAD[name]
        for name in ISAAC_JOINT_ORDER
    )

    assert target == pytest.approx(expected)


def test_stale_tracker_reacquires_only_fresh_selected_class() -> None:
    status = {
        "frame_width": 1920,
        "frame_height": 1080,
        "selected_target_name": "pear",
        "target_lock_id": 7,
        "selected_target_age_s": 0.83,
        "selected_target": {
            "center": [1360, 808],
            "bbox_xywh": [1335, 780, 51, 57],
            "confidence": 0.88,
        },
        "mission": {"target_policy": "same_class"},
        "produce": {
            "age_s": 0.12,
            "confidence_threshold": 0.2,
            "class_thresholds": {"pear": 0.35},
            "detections": [
                {
                    "label": "banana",
                    "confidence": 0.99,
                    "bbox_xyxy": [600, 760, 700, 840],
                    "center": [650, 800],
                },
                {
                    "label": "pear",
                    "confidence": 0.71,
                    "bbox_xyxy": [1082, 750, 1130, 801],
                    "center": [1106, 775],
                },
            ],
        },
    }

    selected = selected_target_bbox_from_status(status, maximum_age_s=0.8)

    assert selected["label"] == "pear"
    assert selected["lock_id"] == 7
    assert selected["confidence"] == pytest.approx(0.71)
    assert selected["source"] == "same_class_reacquired"
    assert selected["frame_age_s"] == pytest.approx(0.12)
    assert selected["bbox"] == pytest.approx(
        [1082 / 1920, 750 / 1080, 1130 / 1920, 801 / 1080]
    )


def test_stale_tracker_does_not_reacquire_without_same_class_policy() -> None:
    status = {
        "frame_width": 1920,
        "frame_height": 1080,
        "selected_target_name": "pear",
        "target_lock_id": 7,
        "selected_target_age_s": 0.83,
        "selected_target": {
            "center": [1360, 808],
            "bbox_xywh": [1335, 780, 51, 57],
            "confidence": 0.88,
        },
        "mission": {"target_policy": "exact_instance"},
        "produce": {
            "age_s": 0.12,
            "class_thresholds": {"pear": 0.35},
            "detections": [
                {
                    "label": "pear",
                    "confidence": 0.71,
                    "bbox_xyxy": [1082, 750, 1130, 801],
                    "center": [1106, 775],
                }
            ],
        },
    }

    with pytest.raises(ValueError, match="selected target is stale"):
        selected_target_bbox_from_status(status, maximum_age_s=0.8)
