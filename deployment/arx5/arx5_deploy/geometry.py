from __future__ import annotations

import math

import numpy as np


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rotation_matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([rotation[:, 0], rotation[:, 1]])


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape != (6,) or not np.isfinite(rot6d).all():
        raise ValueError("rot6d must contain six finite values")
    first = rot6d[:3]
    second = rot6d[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-6:
        raise ValueError("first rot6d axis is degenerate")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-6:
        raise ValueError("second rot6d axis is degenerate")
    second = second / second_norm
    return np.column_stack([first, second, np.cross(first, second)])


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def encode_eef_state(eef_state: object) -> np.ndarray:
    pose = np.asarray(eef_state.pose_6d().copy(), dtype=np.float64)  # type: ignore[attr-defined]
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError(f"SDK returned invalid EEF pose: {pose}")
    rotation = rpy_to_rotation_matrix(*pose[3:])
    return np.concatenate(
        [pose[:3], rotation_matrix_to_rot6d(rotation), [float(eef_state.gripper_pos)]]  # type: ignore[attr-defined]
    ).astype(np.float32)

