#!/usr/bin/env python3
"""
Replay an ARX5 bimanual episode recorded by the collector.

Usage examples:
  python replay_arx5_episode.py ~/DataCollectionSystemV2/collector/data/.../episode_0000
  python replay_arx5_episode.py ~/episode_0000 --speed 2.0
  python replay_arx5_episode.py ~/episode_0000 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("replay_arx5_episode")

DOF_PER_ARM = 6
JOINT_COLUMNS = [
    "timestamp_ms",
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
]

_arx5 = None


@dataclass
class Frame:
    timestamp_ms: int
    left_joint: np.ndarray
    left_gripper: float
    right_joint: np.ndarray
    right_gripper: float


@dataclass
class ReplaySpec:
    csv_path: Path
    frames: list[Frame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay ARX5 episode actions")
    parser.add_argument("episode_dir", type=Path, help="Episode directory, e.g. ~/episode_0000")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--left-model", default="X5", help="Left arm model")
    parser.add_argument("--left-interface", default="can0", help="Left arm CAN interface")
    parser.add_argument("--right-model", default="X5", help="Right arm model")
    parser.add_argument("--right-interface", default="can1", help="Right arm CAN interface")
    parser.add_argument(
        "--left-gripper-open-readout",
        type=float,
        default=-3.33,
        help="Left gripper full-open motor readout; use None only if SDK default is calibrated",
    )
    parser.add_argument(
        "--right-gripper-open-readout",
        type=float,
        default=-3.33,
        help="Right gripper full-open motor readout; use None only if SDK default is calibrated",
    )
    parser.add_argument(
        "--sdk-root",
        type=Path,
        default=Path.home() / "code" / "auto" / "arx5-sdk",
        help="ARX5 SDK root directory",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Replay from frame index")
    parser.add_argument("--max-frames", type=int, default=0, help="Replay at most N frames, 0 means all")
    parser.add_argument("--initial-move-duration", type=float, default=2.0, help="Seconds to move to first frame")
    parser.add_argument("--settle-time", type=float, default=0.5, help="Seconds to wait after first-frame move")
    parser.add_argument("--dry-run", action="store_true", help="Only parse and print episode info")
    parser.add_argument("--keep-enabled", action="store_true", help="Do not reset home at the end")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    if args.start_frame < 0:
        parser.error("--start-frame must be >= 0")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.initial_move_duration < 0:
        parser.error("--initial-move-duration must be >= 0")
    if args.settle_time < 0:
        parser.error("--settle-time must be >= 0")
    if args.left_interface == args.right_interface:
        parser.error("left and right interfaces must be different")
    return args


def _bootstrap_arx5_sdk(sdk_root: Path) -> Any:
    global _arx5
    if _arx5 is not None:
        return _arx5

    sdk_root = sdk_root.expanduser().resolve()
    sdk_python = sdk_root / "python"
    if sdk_python.exists() and str(sdk_python) not in sys.path:
        sys.path.insert(0, str(sdk_python))

    lib_dirs = [
        sdk_root / "runtime_prefix" / "lib",
        sdk_root / "lib" / "x86_64",
        sdk_root / "build",
    ]
    preload_names = [
        "libfmt.so.12",
        "libspdlog.so.1.17",
        "libboost_container.so",
        "libhardware.so",
        "libsolver.so",
        "libArxJointController.so",
        "libArxCartesianController.so",
    ]
    for lib_dir in lib_dirs:
        if not lib_dir.exists():
            continue
        for name in preload_names:
            lib_path = lib_dir / name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    logger.warning("Failed to preload %s: %s", lib_path, exc)

    conda_prefix = Path(sys.executable).resolve().parent.parent
    conda_lib = conda_prefix / "lib"
    for name in ["liborocos-kdl.so", "libkdl_parser.so", "libsoem.so"]:
        lib_path = conda_lib / name
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                logger.warning("Failed to preload %s: %s", lib_path, exc)

    import arx5_interface as arx5

    _arx5 = arx5
    return arx5


def _load_joint_frames(csv_path: Path) -> list[Frame]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        if fieldnames != JOINT_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns in {csv_path}:\n"
                f"  expected={JOINT_COLUMNS}\n"
                f"  actual={fieldnames}"
            )

        frames: list[Frame] = []
        for row in reader:
            frames.append(
                Frame(
                    timestamp_ms=int(float(row["timestamp_ms"])),
                    left_joint=np.array([float(row[f"left_j{i}"]) for i in range(1, DOF_PER_ARM + 1)], dtype=np.float64),
                    left_gripper=float(row["left_gripper"]),
                    right_joint=np.array([float(row[f"right_j{i}"]) for i in range(1, DOF_PER_ARM + 1)], dtype=np.float64),
                    right_gripper=float(row["right_gripper"]),
                )
            )
    return frames


def load_episode_frames(episode_dir: Path) -> ReplaySpec:
    csv_path = episode_dir / "actions.joint_position" / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No replayable action CSV found: {csv_path}")
    frames = _load_joint_frames(csv_path)
    if not frames:
        raise ValueError(f"No action frames found in {csv_path}")
    return ReplaySpec(csv_path=csv_path, frames=frames)


class ArxBimanualReplayRobot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.arx5 = None
        self.left = None
        self.right = None
        self.robot_config = None
        self.controller_dt = None

    def connect(self) -> None:
        self.arx5 = _bootstrap_arx5_sdk(self.args.sdk_root)
        self.left = self._init_controller(
            self.args.left_model,
            self.args.left_interface,
            self.args.left_gripper_open_readout,
        )
        self.right = self._init_controller(
            self.args.right_model,
            self.args.right_interface,
            self.args.right_gripper_open_readout,
        )
        self.robot_config = self.left.get_robot_config()
        self.controller_dt = float(self.left.get_controller_config().controller_dt)
        self.left.reset_to_home()
        self.right.reset_to_home()

    def _init_controller(self, model: str, interface: str, gripper_open_readout: float | None):
        arx5 = self.arx5
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
        if gripper_open_readout is not None:
            robot_config.gripper_open_readout = float(gripper_open_readout)
            logger.info("Override %s gripper_open_readout=%s", interface, robot_config.gripper_open_readout)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        controller_config.background_send_recv = True
        return arx5.Arx5JointController(robot_config, controller_config, interface)

    def get_current_frame(self) -> Frame:
        left_state = self.left.get_joint_state()
        right_state = self.right.get_joint_state()
        return Frame(
            timestamp_ms=0,
            left_joint=np.asarray(left_state.pos().copy(), dtype=np.float64),
            left_gripper=float(left_state.gripper_pos),
            right_joint=np.asarray(right_state.pos().copy(), dtype=np.float64),
            right_gripper=float(right_state.gripper_pos),
        )

    def send_frame(self, frame: Frame) -> None:
        left_cmd = self.arx5.JointState(self.robot_config.joint_dof)
        left_cmd.pos()[:] = frame.left_joint
        left_cmd.gripper_pos = float(frame.left_gripper)
        self.left.set_joint_cmd(left_cmd)

        right_cmd = self.arx5.JointState(self.robot_config.joint_dof)
        right_cmd.pos()[:] = frame.right_joint
        right_cmd.gripper_pos = float(frame.right_gripper)
        self.right.set_joint_cmd(right_cmd)

    def move_to_frame(self, target: Frame, duration: float) -> None:
        if duration <= 0:
            self.send_frame(target)
            return

        start = self.get_current_frame()
        steps = max(int(round(duration / self.controller_dt)), 1)
        for i in range(1, steps + 1):
            alpha = i / steps
            if alpha < 0.5:
                t = 2.0 * alpha * alpha
            else:
                t = 1.0 - (-2.0 * alpha + 2.0) ** 2 / 2.0
            frame = Frame(
                timestamp_ms=0,
                left_joint=start.left_joint * (1.0 - t) + target.left_joint * t,
                left_gripper=float(start.left_gripper * (1.0 - t) + target.left_gripper * t),
                right_joint=start.right_joint * (1.0 - t) + target.right_joint * t,
                right_gripper=float(start.right_gripper * (1.0 - t) + target.right_gripper * t),
            )
            self.send_frame(frame)
            time.sleep(self.controller_dt)

    def disconnect(self, keep_enabled: bool = False) -> None:
        if self.left is None or self.right is None:
            return
        if not keep_enabled:
            self.left.reset_to_home()
            self.right.reset_to_home()


def replay_frames(robot: ArxBimanualReplayRobot, frames: list[Frame], speed: float) -> None:
    replay_start = time.monotonic()
    t0_ms = frames[0].timestamp_ms
    total = len(frames)

    for idx, frame in enumerate(frames):
        target_elapsed = (frame.timestamp_ms - t0_ms) / 1000.0 / speed
        now_elapsed = time.monotonic() - replay_start
        sleep_s = target_elapsed - now_elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

        robot.send_frame(frame)

        if idx == 0 or idx == total - 1 or idx % 100 == 0:
            logger.info(
                "frame %d/%d  t=%.3fs  left_gripper=%.4f  right_gripper=%.4f",
                idx + 1,
                total,
                (frame.timestamp_ms - t0_ms) / 1000.0,
                frame.left_gripper,
                frame.right_gripper,
            )


def main() -> int:
    args = parse_args()
    episode_dir = args.episode_dir.expanduser().resolve()
    spec = load_episode_frames(episode_dir)

    start = args.start_frame
    end = None if args.max_frames == 0 else start + args.max_frames
    frames = spec.frames[start:end]
    if not frames:
        raise ValueError("Selected frame range is empty")

    duration_s = (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 1000.0 if len(frames) > 1 else 0.0
    logger.info(
        "Loaded %d frames from %s, duration=%.3fs, replay_speed=%.2fx, left=%s@%s, right=%s@%s",
        len(frames),
        spec.csv_path,
        duration_s,
        args.speed,
        args.left_model,
        args.left_interface,
        args.right_model,
        args.right_interface,
    )

    if args.dry_run:
        return 0

    robot = ArxBimanualReplayRobot(args)
    keep_enabled = False
    try:
        logger.info("Connecting ARX5 arms")
        robot.connect()
        logger.info("Moving to first replay frame")
        robot.move_to_frame(frames[0], args.initial_move_duration)
        if args.settle_time > 0:
            time.sleep(args.settle_time)
        logger.info("Starting replay")
        replay_frames(robot, frames, args.speed)
        logger.info("Replay finished")
        keep_enabled = args.keep_enabled
        return 0
    finally:
        try:
            robot.disconnect(keep_enabled=keep_enabled)
        except Exception as exc:
            logger.warning("Failed to disconnect ARX5 cleanly: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
