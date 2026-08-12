from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arx5_deploy.geometry import (
    rotation_distance,
    rotation_matrix_to_rot6d,
    rot6d_to_rotation_matrix,
)

ACTION_DIM = 10


@dataclass(frozen=True)
class Limits:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_position_step: float
    max_rotation_step: float
    max_gripper_step: float
    max_joint_step: float
    z_safety_margin: float


def validate_action(
    action: np.ndarray,
    current: np.ndarray,
    limits: Limits,
    *,
    gripper_max: float,
) -> tuple[np.ndarray | None, list[str], dict[str, float]]:
    action = np.asarray(action, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    metrics: dict[str, float] = {}
    if action.shape != (ACTION_DIM,):
        return None, [f"unexpected action shape {action.shape}"], metrics
    if current.shape != (ACTION_DIM,) or not np.isfinite(current).all():
        return None, [f"invalid measured state shape={current.shape}"], metrics
    if not np.isfinite(action).all():
        return None, ["action contains NaN or Inf"], metrics

    try:
        action_rotation = rot6d_to_rotation_matrix(action[3:9])
        current_rotation = rot6d_to_rotation_matrix(current[3:9])
    except ValueError as exc:
        return None, [str(exc)], metrics

    sanitized = action.copy()
    sanitized[3:9] = rotation_matrix_to_rot6d(action_rotation)
    metrics["position_step_m"] = float(np.linalg.norm(sanitized[:3] - current[:3]))
    metrics["rotation_step_rad"] = rotation_distance(current_rotation, action_rotation)
    metrics["gripper_step_m"] = float(abs(sanitized[9] - current[9]))

    reasons: list[str] = []
    if np.any(current[:3] < limits.workspace_min) or np.any(current[:3] > limits.workspace_max):
        reasons.append(f"measured xyz {current[:3].round(5).tolist()} is outside workspace")
    if np.any(sanitized[:3] < limits.workspace_min) or np.any(sanitized[:3] > limits.workspace_max):
        reasons.append(f"target xyz {sanitized[:3].round(5).tolist()} is outside workspace")
    elif sanitized[2] < limits.workspace_min[2] + limits.z_safety_margin:
        guarded_z = limits.workspace_min[2] + limits.z_safety_margin
        reasons.append(f"target z {sanitized[2]:.5f} is below guarded minimum {guarded_z:.5f}")
    if metrics["position_step_m"] > limits.max_position_step:
        reasons.append(
            f"position step {metrics['position_step_m']:.5f} > {limits.max_position_step:.5f} m"
        )
    if metrics["rotation_step_rad"] > limits.max_rotation_step:
        reasons.append(
            f"rotation step {metrics['rotation_step_rad']:.5f} > {limits.max_rotation_step:.5f} rad"
        )
    if metrics["gripper_step_m"] > limits.max_gripper_step:
        reasons.append(
            f"gripper step {metrics['gripper_step_m']:.5f} > {limits.max_gripper_step:.5f} m"
        )
    if not 0.0 <= sanitized[9] <= gripper_max:
        reasons.append(f"gripper {sanitized[9]:.5f} is outside [0, {gripper_max:.5f}] m")
    return (None if reasons else sanitized), reasons, metrics

