from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

from arx5_deploy.geometry import (
    encode_eef_state,
    rotation_matrix_to_rpy,
    rot6d_to_rotation_matrix,
)

logger = logging.getLogger(__name__)


def ease_in_out_quad(alpha: float) -> float:
    """Match the smooth interpolation used by replay_arx5_episode.py."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha < 0.5:
        return 2.0 * alpha * alpha
    return 1.0 - (-2.0 * alpha + 2.0) ** 2 / 2.0


def estimate_move_duration(
    start_joints: np.ndarray,
    start_gripper: float,
    target_joints: np.ndarray,
    target_gripper: float,
    *,
    joint_speed: float,
    gripper_speed: float,
    minimum_duration: float,
) -> float:
    if joint_speed <= 0 or gripper_speed <= 0 or minimum_duration <= 0:
        raise ValueError("motion speeds and minimum_duration must be positive")
    return max(
        float(np.max(np.abs(np.asarray(target_joints) - np.asarray(start_joints)))) / joint_speed,
        abs(float(target_gripper) - float(start_gripper)) / gripper_speed,
        minimum_duration,
    )


class Arx5SingleArm:
    """One ARX5 joint controller with explicit position-hold semantics."""

    def __init__(self, model: str, interface: str, gripper_open_readout: float) -> None:
        try:
            import arx5_interface as arx5
        except ImportError as exc:
            raise RuntimeError(
                "arx5_interface is unavailable; run this command in an environment with the ARX5 SDK installed"
            ) from exc

        self.arx5 = arx5
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
        robot_config.gripper_open_readout = float(gripper_open_readout)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        controller_config.background_send_recv = True
        controller_config.gravity_compensation = True
        controller_config.over_current_cnt_max = 1000
        # This SDK forces damping/passive during controller destruction. The caller
        # therefore completes the explicit hold-and-return sequence before releasing
        # the controller; on a return failure it keeps this object alive in hold_forever.
        controller_config.shutdown_to_passive = True
        logger.info(
            "Connecting ARX5 model=%s interface=%s gripper_open_readout=%.5f",
            model,
            interface,
            gripper_open_readout,
        )
        self.controller = arx5.Arx5JointController(robot_config, controller_config, interface)
        self.robot_config = self.controller.get_robot_config()
        self.controller_config = self.controller.get_controller_config()
        self.solver = arx5.Arx5Solver(
            self.robot_config.urdf_path,
            self.robot_config.joint_dof,
            self.robot_config.joint_pos_min,
            self.robot_config.joint_pos_max,
            self.robot_config.base_link_name,
            self.robot_config.eef_link_name,
            self.robot_config.gravity_vector,
        )
        self._hold_lock = threading.Lock()
        self._hold_joints: np.ndarray | None = None
        self._hold_gripper: float | None = None
        time.sleep(0.5)
        joints, gripper = self.read_joint_state()
        # Snapshot BEFORE reset_to_home, collector start, or policy initialization.
        # This is the user's real pre-launch pose, not either policy start pose.
        self.startup_joints = joints.copy()
        self.startup_gripper = float(np.clip(gripper, 0.0, self.gripper_width))
        logger.info("Connected at joints=%s gripper=%.5f", np.array2string(joints, precision=5), gripper)
        logger.info(
            "Saved pre-launch pose: joints=%s gripper=%.5f",
            np.array2string(self.startup_joints, precision=5),
            self.startup_gripper,
        )

    @property
    def gripper_width(self) -> float:
        return float(self.robot_config.gripper_width)

    def read_joint_state(self) -> tuple[np.ndarray, float]:
        state = self.controller.get_joint_state()
        joints = np.asarray(state.pos().copy(), dtype=np.float64)
        gripper = float(state.gripper_pos)
        if joints.shape != (6,) or not np.isfinite(joints).all() or not math.isfinite(gripper):
            raise RuntimeError(f"SDK returned invalid joint state: joints={joints}, gripper={gripper}")
        return joints, gripper

    def read_eef_state(self) -> np.ndarray:
        return encode_eef_state(self.controller.get_eef_state())

    def _send_joint_target(
        self,
        joints: np.ndarray,
        gripper: float,
        *,
        duration: float | None = None,
        latch: bool = True,
    ) -> None:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError(f"invalid target joints: {joints}")
        lower = np.asarray(self.robot_config.joint_pos_min, dtype=np.float64)
        upper = np.asarray(self.robot_config.joint_pos_max, dtype=np.float64)
        if np.any(joints < lower) or np.any(joints > upper):
            raise RuntimeError("target joints exceed SDK limits")
        gripper = float(np.clip(gripper, 0.0, self.gripper_width))
        command = self.arx5.JointState(self.robot_config.joint_dof)
        command.pos()[:] = joints
        command.gripper_pos = gripper
        if duration is not None:
            command.timestamp = float(self.controller.get_timestamp() + duration)
        self.controller.set_joint_cmd(command)
        if latch:
            with self._hold_lock:
                self._hold_joints = joints.copy()
                self._hold_gripper = gripper

    def latch_measured_hold(self) -> tuple[np.ndarray, float]:
        joints, gripper = self.read_joint_state()
        gripper = float(np.clip(gripper, 0.0, self.gripper_width))
        self._send_joint_target(joints, gripper)
        return joints, gripper

    def resend_hold(self) -> None:
        with self._hold_lock:
            joints = None if self._hold_joints is None else self._hold_joints.copy()
            gripper = self._hold_gripper
        if joints is None or gripper is None:
            self.latch_measured_hold()
        else:
            self._send_joint_target(joints, gripper, latch=False)

    def restore_gains_holding_current(self, transition_time: float) -> None:
        joints, measured_gripper = self.latch_measured_hold()
        gripper = float(np.clip(measured_gripper, 0.0, self.gripper_width))
        dt = float(self.controller_config.controller_dt)
        start_gain = self.controller.get_gain()
        start_kp = np.asarray(start_gain.kp().copy(), dtype=np.float64)
        start_kd = np.asarray(start_gain.kd().copy(), dtype=np.float64)
        start_gripper_kp = float(start_gain.gripper_kp)
        start_gripper_kd = float(start_gain.gripper_kd)
        target_kp = np.asarray(self.controller_config.default_kp, dtype=np.float64)
        target_kd = np.asarray(self.controller_config.default_kd, dtype=np.float64)
        steps = max(1, math.ceil(transition_time / dt))
        logger.info("Restoring control gains over %.2f s while holding", transition_time)
        for step in range(1, steps + 1):
            alpha = step / steps
            gain = self.controller.get_gain()
            gain.kp()[:] = start_kp * (1.0 - alpha) + target_kp * alpha
            gain.kd()[:] = start_kd * (1.0 - alpha) + target_kd * alpha
            gain.gripper_kp = start_gripper_kp * (1.0 - alpha) + 2.0 * alpha
            gain.gripper_kd = start_gripper_kd * (1.0 - alpha) + 0.2 * alpha
            self.controller.set_gain(gain)
            self._send_joint_target(joints, gripper)
            time.sleep(dt)

    def move_joint_target(
        self,
        target_joints: np.ndarray,
        target_gripper: float,
        *,
        joint_speed: float,
        gripper_speed: float,
        minimum_duration: float,
        max_tracking_error: float,
        stop_event: threading.Event,
    ) -> None:
        target_joints = np.asarray(target_joints, dtype=np.float64)
        start_joints, measured_gripper = self.read_joint_state()
        start_gripper = float(np.clip(measured_gripper, 0.0, self.gripper_width))
        duration = estimate_move_duration(
            start_joints,
            start_gripper,
            target_joints,
            target_gripper,
            joint_speed=joint_speed,
            gripper_speed=gripper_speed,
            minimum_duration=minimum_duration,
        )
        dt = float(self.controller_config.controller_dt)
        samples = max(1, math.ceil(duration / dt))
        logger.info("Moving to joint target over %.2f s: %s", duration, target_joints)
        next_tick = time.monotonic()
        for sample in range(1, samples + 1):
            if stop_event.is_set():
                raise InterruptedError("motion interrupted")
            ratio = sample / samples
            blend = ease_in_out_quad(ratio)
            commanded_joints = start_joints * (1.0 - blend) + target_joints * blend
            commanded_gripper = start_gripper * (1.0 - blend) + target_gripper * blend
            self._send_joint_target(commanded_joints, commanded_gripper)
            if sample % max(1, round(0.1 / dt)) == 0:
                measured_joints, _ = self.read_joint_state()
                error = float(np.max(np.abs(measured_joints - commanded_joints)))
                if error > max_tracking_error:
                    raise RuntimeError(f"joint motion tracking error {error:.4f} rad")
            next_tick += dt
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

    def initialize(
        self,
        *,
        collector_start_joints: np.ndarray,
        collector_start_gripper: float,
        policy_start_joints: np.ndarray,
        policy_start_gripper: float,
        joint_speed: float,
        gripper_speed: float,
        minimum_duration: float,
        gain_transition_time: float,
        max_tracking_error: float,
        stop_event: threading.Event,
    ) -> None:
        logger.info("Initialization: SDK home -> collector start -> recorded policy start")
        self.controller.reset_to_home()
        self.restore_gains_holding_current(gain_transition_time)
        common = dict(
            joint_speed=joint_speed,
            gripper_speed=gripper_speed,
            minimum_duration=minimum_duration,
            max_tracking_error=max_tracking_error,
            stop_event=stop_event,
        )
        self.move_joint_target(collector_start_joints, collector_start_gripper, **common)
        self.move_joint_target(policy_start_joints, policy_start_gripper, **common)
        measured, _ = self.read_joint_state()
        error = float(np.max(np.abs(measured - policy_start_joints)))
        if error > 0.05:
            raise RuntimeError(f"initialization final error {error:.4f} rad")
        self.latch_measured_hold()
        logger.info("Initialization complete; final error %.4f rad", error)

    def send_absolute_eef(
        self,
        action: np.ndarray,
        *,
        duration: float,
        max_joint_step: float,
    ) -> tuple[np.ndarray, float]:
        rotation = rot6d_to_rotation_matrix(action[3:9])
        pose_6d = np.concatenate([action[:3], rotation_matrix_to_rpy(rotation)])
        current_joints, _ = self.read_joint_state()
        status, target_joints = self.solver.multi_trial_ik(
            pose_6d.astype(np.float64), current_joints, 5
        )
        if int(status) != 0:
            name = self.solver.get_ik_status_name(int(status))
            raise RuntimeError(f"IK failed: {name} ({status})")
        target_joints = np.asarray(target_joints, dtype=np.float64)
        joint_step = float(np.max(np.abs(target_joints - current_joints)))
        if joint_step > max_joint_step:
            raise RuntimeError(f"IK joint step {joint_step:.4f} > {max_joint_step:.4f} rad")
        target_gripper = float(action[9])
        self._send_joint_target(target_joints, target_gripper, duration=duration)
        return target_joints, target_gripper

    def tracking_error(self, target_joints: np.ndarray) -> float:
        measured, _ = self.read_joint_state()
        return float(np.max(np.abs(measured - target_joints)))

    def hold_for(self, *, seconds: float, refresh_hz: float) -> tuple[np.ndarray, float]:
        """Latch the measured pose and actively hold it for a bounded interval."""
        if seconds < 0 or refresh_hz <= 0:
            raise ValueError("hold seconds must be non-negative and refresh_hz must be positive")
        joints, gripper = self.latch_measured_hold()
        logger.info(
            "Holding measured pose for %.2f s: joints=%s gripper=%.5f",
            seconds,
            np.array2string(joints, precision=5),
            gripper,
        )
        deadline = time.monotonic() + seconds
        period = 1.0 / refresh_hz
        while time.monotonic() < deadline:
            started = time.monotonic()
            self.resend_hold()
            remaining = min(period - (time.monotonic() - started), deadline - time.monotonic())
            if remaining > 0:
                time.sleep(remaining)
        return joints, gripper

    def return_to_prelaunch_pose(
        self,
        *,
        hold_before_seconds: float,
        hold_after_seconds: float,
        hold_refresh_hz: float,
        joint_speed: float,
        gripper_speed: float,
        minimum_duration: float,
        max_tracking_error: float,
        final_joint_tolerance: float,
        final_gripper_tolerance: float,
    ) -> dict[str, float | list[float]]:
        """Hold the final pose, replay-interpolate to pre-launch pose, then release."""
        final_joints, final_gripper = self.hold_for(
            seconds=hold_before_seconds,
            refresh_hz=hold_refresh_hz,
        )
        logger.warning(
            "Returning to saved pre-launch pose (not collector/policy start): %s",
            np.array2string(self.startup_joints, precision=5),
        )
        # Exit signals have already stopped inference. A fresh unset event makes
        # the safety return atomic: repeated Ctrl+C cannot interrupt it halfway.
        self.move_joint_target(
            self.startup_joints,
            self.startup_gripper,
            joint_speed=joint_speed,
            gripper_speed=gripper_speed,
            minimum_duration=minimum_duration,
            max_tracking_error=max_tracking_error,
            stop_event=threading.Event(),
        )
        measured_joints, measured_gripper = self.read_joint_state()
        joint_error = float(np.max(np.abs(measured_joints - self.startup_joints)))
        gripper_error = abs(float(measured_gripper) - self.startup_gripper)
        if joint_error > final_joint_tolerance or gripper_error > final_gripper_tolerance:
            raise RuntimeError(
                "pre-launch return did not converge: "
                f"joint_error={joint_error:.4f} rad gripper_error={gripper_error:.4f} m"
            )
        self.hold_for(seconds=hold_after_seconds, refresh_hz=hold_refresh_hz)
        logger.info(
            "Returned to pre-launch pose: joint_error=%.4f rad gripper_error=%.4f m",
            joint_error,
            gripper_error,
        )
        return {
            "final_joints": final_joints.tolist(),
            "final_gripper": final_gripper,
            "startup_joints": self.startup_joints.tolist(),
            "startup_gripper": self.startup_gripper,
            "return_joint_error": joint_error,
            "return_gripper_error": gripper_error,
        }

    def release_after_return(self) -> None:
        """Gracefully make joints movable after reaching the saved startup pose."""
        logger.warning("Pre-launch pose is stable; entering damping and closing controller")
        self.controller.set_to_damping()

    def hold_forever(self, *, refresh_hz: float, release_event: threading.Event) -> None:
        joints, gripper = self.latch_measured_hold()
        logger.warning(
            "SAFETY HOLD active at joints=%s gripper=%.5f. Keep this process running.",
            np.array2string(joints, precision=5),
            gripper,
        )
        period = 1.0 / refresh_hz
        failures = 0
        while not release_event.is_set():
            started = time.monotonic()
            try:
                self.resend_hold()
                failures = 0
            except Exception as exc:
                failures += 1
                logger.error("Hold refresh failed (%d): %s", failures, exc)
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        logger.warning("Explicit release requested; entering damping mode")
        self.controller.set_to_damping()
