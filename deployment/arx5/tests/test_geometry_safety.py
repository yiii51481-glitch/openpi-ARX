from __future__ import annotations

import numpy as np

from arx5_deploy.robot import Arx5SingleArm, ease_in_out_quad, estimate_move_duration
from arx5_deploy.geometry import (
    rotation_distance,
    rotation_matrix_to_rot6d,
    rotation_matrix_to_rpy,
    rot6d_to_rotation_matrix,
    rpy_to_rotation_matrix,
)
from arx5_deploy.safety import Limits, validate_action


def test_rotation_round_trip() -> None:
    rotation = rpy_to_rotation_matrix(0.2, -0.3, 0.4)
    decoded = rot6d_to_rotation_matrix(rotation_matrix_to_rot6d(rotation))
    assert rotation_distance(rotation, decoded) < 1e-7
    reconstructed = rpy_to_rotation_matrix(*rotation_matrix_to_rpy(rotation))
    assert rotation_distance(rotation, reconstructed) < 1e-7


def test_limits_reject_large_step_without_clipping() -> None:
    rotation = rotation_matrix_to_rot6d(np.eye(3))
    current = np.concatenate([[0.3, 0.0, 0.2], rotation, [0.05]])
    action = current.copy()
    action[0] += 0.1
    limits = Limits(
        workspace_min=np.array([0.2, -0.2, 0.0]),
        workspace_max=np.array([0.5, 0.2, 0.4]),
        max_position_step=0.025,
        max_rotation_step=0.08,
        max_gripper_step=0.01,
        max_joint_step=0.2,
        z_safety_margin=0.005,
    )
    safe, reasons, metrics = validate_action(action, current, limits, gripper_max=0.085)
    assert safe is None
    assert any("position step" in reason for reason in reasons)
    assert metrics["position_step_m"] > limits.max_position_step


def test_replay_interpolation_and_duration() -> None:
    assert ease_in_out_quad(0.0) == 0.0
    assert ease_in_out_quad(0.5) == 0.5
    assert ease_in_out_quad(1.0) == 1.0
    assert ease_in_out_quad(0.25) == 0.125
    assert ease_in_out_quad(0.75) == 0.875
    duration = estimate_move_duration(
        np.zeros(6),
        0.04,
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        0.06,
        joint_speed=0.1,
        gripper_speed=0.01,
        minimum_duration=2.0,
    )
    assert duration == 10.0


def test_return_targets_saved_prelaunch_pose_directly() -> None:
    class FakeArm:
        startup_joints = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
        startup_gripper = 0.04

        def __init__(self) -> None:
            self.measured_joints = np.ones(6)
            self.measured_gripper = 0.06
            self.move_target = None
            self.hold_calls = 0

        def hold_for(self, *, seconds: float, refresh_hz: float):
            self.hold_calls += 1
            return self.measured_joints.copy(), self.measured_gripper

        def move_joint_target(self, target_joints, target_gripper, **kwargs):
            assert not kwargs["stop_event"].is_set()
            self.move_target = (np.asarray(target_joints).copy(), target_gripper)
            self.measured_joints = np.asarray(target_joints).copy()
            self.measured_gripper = target_gripper

        def read_joint_state(self):
            return self.measured_joints.copy(), self.measured_gripper

    arm = FakeArm()
    report = Arx5SingleArm.return_to_prelaunch_pose(
        arm,
        hold_before_seconds=1.0,
        hold_after_seconds=1.0,
        hold_refresh_hz=20.0,
        joint_speed=0.1,
        gripper_speed=0.01,
        minimum_duration=2.0,
        max_tracking_error=0.15,
        final_joint_tolerance=0.05,
        final_gripper_tolerance=0.01,
    )
    assert arm.hold_calls == 2
    assert np.array_equal(arm.move_target[0], arm.startup_joints)
    assert arm.move_target[1] == arm.startup_gripper
    assert report["return_joint_error"] == 0.0
