#!/usr/bin/env python3
"""Run pi0.5 on ARX5 and safely return to the pre-launch pose on exit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from arx5_deploy.camera import RealSenseColorCamera, list_realsense_cameras
from arx5_deploy.robot import Arx5SingleArm
from arx5_deploy.safety import ACTION_DIM, Limits, validate_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("arx5_pi05_real")

_stop_event = threading.Event()
_return_active = threading.Event()
_fallback_hold_active = threading.Event()


def _signal_handler(signum: int, _frame: Any) -> None:
    if _return_active.is_set():
        logger.warning("Signal %s received during safety return; return motion will not be interrupted", signum)
    elif _fallback_hold_active.is_set():
        logger.warning("Signal %s received during fallback hold; position hold remains active", signum)
    else:
        logger.warning(
            "Signal %s received; stopping inference, holding final pose, then returning to pre-launch pose",
            signum,
        )
        _stop_event.set()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deployment_config.json"))
    parser.add_argument("--host", help="Override policy server host")
    parser.add_argument("--port", type=int, help="Override policy server port")
    parser.add_argument("--task", help="Task profile name from deployment_config.json")
    parser.add_argument("--prompt", help="Override language prompt")
    parser.add_argument("--execute-steps", type=int, help="Override number of leading chunk actions to execute")
    parser.add_argument("--action-hz", type=float, help="Override action execution frequency")
    parser.add_argument("--max-queries", type=int, help="Override maximum policy queries")
    parser.add_argument(
        "--max-consecutive-rejections",
        type=int,
        help="Override consecutive rejected chunks before inference stops",
    )
    parser.add_argument("--execute", action="store_true", help="Allow robot motion after typed confirmation")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Move to the selected task start pose, then run the normal safety return without policy actions",
    )
    parser.add_argument(
        "--initialize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run home -> collector start -> recorded policy start before execution",
    )
    parser.add_argument("--preview", action="store_true", help="Show base and wrist RGB streams")
    parser.add_argument("--record-images", action="store_true", help="Save query images")
    parser.add_argument("--list-cameras", action="store_true", help="List RealSense cameras and exit")
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    path = args.config.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if args.initialize_only and not args.execute:
        raise ValueError("--initialize-only requires --execute")
    tasks = config.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("configuration must define at least one task profile")
    if args.prompt is not None and args.task is None:
        raise ValueError("--prompt requires an explicit --task so initialization cannot use the wrong task pose")
    task_name = args.task or config.get("default_task")
    if task_name not in tasks:
        raise ValueError(f"unknown task {task_name!r}; choose one of {sorted(tasks)}")
    task = tasks[task_name]
    required_task_keys = {
        "prompt",
        "policy_start_joints",
        "policy_start_gripper",
        "policy_start_state",
        "policy_start_source_episode",
    }
    missing_task_keys = sorted(required_task_keys - task.keys())
    if missing_task_keys:
        raise ValueError(f"task {task_name!r} is missing: {missing_task_keys}")
    config.update({key: task[key] for key in required_task_keys})
    config["task"] = task_name
    overrides = {
        "server_host": args.host,
        "server_port": args.port,
        "prompt": args.prompt,
        "execute_steps": args.execute_steps,
        "action_hz": args.action_hz,
        "max_queries": args.max_queries,
        "max_consecutive_rejections": args.max_consecutive_rejections,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    required = {
        "server_host",
        "server_port",
        "prompt",
        "task",
        "expected_server_config",
        "expected_server_checkpoint_step",
        "model",
        "interface",
        "gripper_open_readout",
        "base_camera_serial",
        "wrist_camera_serial",
        "policy_action_horizon",
        "execute_steps",
        "action_hz",
        "max_queries",
        "max_consecutive_rejections",
        "workspace_min",
        "workspace_max",
        "return_hold_before_seconds",
        "return_hold_after_seconds",
        "return_joint_speed",
        "return_gripper_speed",
        "return_min_duration",
        "return_max_tracking_error",
        "return_final_joint_tolerance",
        "return_final_gripper_tolerance",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration is missing: {missing}")
    horizon = int(config["policy_action_horizon"])
    execute_steps = int(config["execute_steps"])
    if horizon != 50:
        raise ValueError(f"trained pi0.5 horizon must remain 50, got {horizon}")
    if not 1 <= execute_steps <= horizon:
        raise ValueError(f"execute_steps must be in [1, {horizon}]")
    if float(config["action_hz"]) <= 0 or int(config["max_queries"]) <= 0:
        raise ValueError("action_hz and max_queries must be positive")
    if int(config["max_consecutive_rejections"]) <= 0:
        raise ValueError("max_consecutive_rejections must be positive")
    positive_return_keys = (
        "return_joint_speed",
        "return_gripper_speed",
        "return_min_duration",
        "return_max_tracking_error",
        "return_final_joint_tolerance",
        "return_final_gripper_tolerance",
    )
    if any(float(config[key]) <= 0 for key in positive_return_keys):
        raise ValueError("return speeds, duration, tracking error, and tolerances must be positive")
    if float(config["return_hold_before_seconds"]) < 0 or float(config["return_hold_after_seconds"]) < 0:
        raise ValueError("return hold durations must be non-negative")
    policy_start_joints = np.asarray(config["policy_start_joints"], dtype=np.float64)
    policy_start_state = np.asarray(config["policy_start_state"], dtype=np.float64)
    if policy_start_joints.shape != (6,) or not np.isfinite(policy_start_joints).all():
        raise ValueError(f"task {config['task']!r} must define six finite policy_start_joints")
    if policy_start_state.shape != (10,) or not np.isfinite(policy_start_state).all():
        raise ValueError(f"task {config['task']!r} must define a finite 10-D policy_start_state")
    workspace_min = np.asarray(config["workspace_min"], dtype=np.float64)
    workspace_max = np.asarray(config["workspace_max"], dtype=np.float64)
    if workspace_min.shape != (3,) or workspace_max.shape != (3,) or np.any(workspace_min >= workspace_max):
        raise ValueError("workspace_min/workspace_max must be valid xyz vectors")
    return config


def _limits(config: dict[str, Any]) -> Limits:
    return Limits(
        workspace_min=np.asarray(config["workspace_min"], dtype=np.float64),
        workspace_max=np.asarray(config["workspace_max"], dtype=np.float64),
        max_position_step=float(config["max_position_step"]),
        max_rotation_step=float(config["max_rotation_step"]),
        max_gripper_step=float(config["max_gripper_step"]),
        max_joint_step=float(config["max_joint_step"]),
        z_safety_margin=float(config["z_safety_margin"]),
    )


def _confirm(config: dict[str, Any], initialize: bool) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("--execute requires an interactive terminal")
    print("\nWARNING: real robot commands will be sent.")
    print(f"interface: {config['interface']} (training data logical side: left)")
    print(f"task: {config['task']} ({config['prompt']})")
    print(f"task initialization source: {config['policy_start_source_episode']}")
    print(f"wrist camera: {config['wrist_camera_serial']} (training left_wrist_view)")
    print(f"initialization enabled: {initialize}")
    print(f"initialization-only test: {config.get('initialize_only', False)}")
    print(
        f"chunk: horizon={config['policy_action_horizon']}, "
        f"execute first {config['execute_steps']} at {config['action_hz']} Hz"
    )
    print("Clear the swept workspace and keep the physical emergency stop ready.")
    if input("Type EXECUTE CAN1 to continue: ").strip() != "EXECUTE CAN1":
        raise RuntimeError("execution confirmation was not accepted")


def _record_dir(config: dict[str, Any]) -> Path:
    root = Path(config["record_root"]).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    target = root / time.strftime("%Y%m%d-%H%M%S")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _serialize_config(config: dict[str, Any], args: argparse.Namespace, target: Path) -> None:
    record = dict(config)
    record["execute"] = args.execute
    record["initialize"] = args.initialize
    record["preview"] = args.preview
    record["record_images"] = args.record_images
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sleep_until(deadline: float) -> None:
    while not _stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def main() -> int:
    args = _parse_args()
    config = _load_config(args)
    config["initialize_only"] = bool(args.initialize_only)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
    signal.signal(signal.SIGQUIT, _signal_handler)

    if args.list_cameras:
        list_realsense_cameras()
        return 0
    if args.execute:
        _confirm(config, args.initialize)

    limits = _limits(config)
    record_dir = _record_dir(config)
    _serialize_config(config, args, record_dir / "config.json")
    image_dir = record_dir / "images"
    if args.record_images:
        image_dir.mkdir()
    logger.info("Run logs: %s", record_dir)
    logger.warning(
        "Deployment mapping: trained logical LEFT policy/camera -> requested physical interface %s",
        config["interface"],
    )
    logger.info(
        "Selected task=%s prompt=%r initialization_source=%s",
        config["task"],
        config["prompt"],
        config["policy_start_source_episode"],
    )

    base_camera: RealSenseColorCamera | None = None
    wrist_camera: RealSenseColorCamera | None = None
    arm: Arx5SingleArm | None = None
    error: BaseException | None = None
    completed_queries = 0
    try:
        base_camera = RealSenseColorCamera(
            str(config["base_camera_serial"]),
            "base",
            width=int(config["camera_width"]),
            height=int(config["camera_height"]),
            fps=int(config["camera_fps"]),
        )
        wrist_camera = RealSenseColorCamera(
            str(config["wrist_camera_serial"]),
            "wrist",
            width=int(config["camera_width"]),
            height=int(config["camera_height"]),
            fps=int(config["camera_fps"]),
        )
        client = websocket_client_policy.WebsocketClientPolicy(
            host=str(config["server_host"]),
            port=int(config["server_port"]),
            should_stop=_stop_event.is_set,
        )
        server_metadata = client.get_server_metadata()
        logger.info("Policy server metadata: %s", server_metadata)
        expected_config = str(config["expected_server_config"])
        expected_step = int(config["expected_server_checkpoint_step"])
        actual_config = server_metadata.get("training_config")
        actual_step = server_metadata.get("checkpoint_step")
        if actual_config != expected_config or actual_step != expected_step:
            raise RuntimeError(
                "wrong policy server: "
                f"expected config={expected_config} checkpoint_step={expected_step}, "
                f"got config={actual_config!r} checkpoint_step={actual_step!r}"
            )
        arm = Arx5SingleArm(
            str(config["model"]),
            str(config["interface"]),
            float(config["gripper_open_readout"]),
        )
        if args.execute:
            if _stop_event.is_set():
                logger.warning("Stop requested during connection; skipping initialization and returning safely")
            else:
                arm.restore_gains_holding_current(float(config["gain_transition_time"]))
                if args.initialize:
                    arm.initialize(
                        collector_start_joints=np.asarray(config["collector_start_joints"], dtype=np.float64),
                        collector_start_gripper=float(config["collector_start_gripper"]),
                        policy_start_joints=np.asarray(config["policy_start_joints"], dtype=np.float64),
                        policy_start_gripper=float(config["policy_start_gripper"]),
                        joint_speed=float(config["initial_joint_speed"]),
                        gripper_speed=float(config["initial_gripper_speed"]),
                        minimum_duration=float(config["initial_min_phase_duration"]),
                        gain_transition_time=float(config["gain_transition_time"]),
                        max_tracking_error=float(config["initial_max_tracking_error"]),
                        stop_event=_stop_event,
                    )
        else:
            logger.warning("INFERENCE-ONLY: policy actions will not be sent")

        consecutive_rejections = 0
        action_period = 1.0 / float(config["action_hz"])
        image_size = int(config["policy_image_size"])
        with (record_dir / "queries.jsonl").open("a", encoding="utf-8") as log_file:
            query_count = 0 if args.initialize_only else int(config["max_queries"])
            if args.initialize_only:
                logger.warning("Initialization-only test complete; skipping all policy actions")
            for query_index in range(query_count):
                if _stop_event.is_set():
                    break
                base_rgb = base_camera.read_rgb()
                wrist_rgb = wrist_camera.read_rgb()
                state = arm.read_eef_state()
                observation = {
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(base_rgb, image_size, image_size)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_rgb, image_size, image_size)
                    ),
                    "observation/state": state,
                    "prompt": str(config["prompt"]),
                }
                inference_started = time.monotonic()
                result = client.infer(observation)
                inference_ms = 1000.0 * (time.monotonic() - inference_started)
                actions = np.asarray(result["actions"], dtype=np.float64)
                expected_shape = (int(config["policy_action_horizon"]), ACTION_DIM)
                if actions.shape != expected_shape or not np.isfinite(actions).all():
                    raise RuntimeError(f"invalid policy actions: shape={actions.shape}, finite={np.isfinite(actions).all()}")

                executions: list[dict[str, Any]] = []
                query_rejected = False
                validation_state = state.astype(np.float64)
                next_action_time = time.monotonic()
                for action_index in range(int(config["execute_steps"])):
                    if _stop_event.is_set():
                        break
                    current = arm.read_eef_state().astype(np.float64) if args.execute else validation_state
                    safe_action, reasons, metrics = validate_action(
                        actions[action_index], current, limits, gripper_max=arm.gripper_width
                    )
                    executed = False
                    tracking_error = None
                    if safe_action is not None:
                        validation_state = safe_action
                        if args.execute:
                            try:
                                target_joints, _ = arm.send_absolute_eef(
                                    safe_action,
                                    duration=action_period,
                                    max_joint_step=limits.max_joint_step,
                                )
                                executed = True
                                next_action_time += action_period
                                _sleep_until(next_action_time)
                                tracking_error = arm.tracking_error(target_joints)
                                if tracking_error > float(config["max_joint_tracking_error"]):
                                    reasons.append(
                                        f"joint tracking error {tracking_error:.4f} > "
                                        f"{float(config['max_joint_tracking_error']):.4f} rad"
                                    )
                            except Exception as exc:
                                reasons.append(str(exc))
                    if reasons:
                        logger.error(
                            "query=%d action=%d rejected: %s",
                            query_index,
                            action_index,
                            "; ".join(reasons),
                        )
                        query_rejected = True
                        if args.execute:
                            arm.latch_measured_hold()
                    executions.append(
                        {
                            "action_index": action_index,
                            "current_state": current.tolist(),
                            "raw_action": actions[action_index].tolist(),
                            "safe_action": None if safe_action is None else safe_action.tolist(),
                            "metrics": metrics,
                            "reasons": reasons,
                            "executed": executed,
                            "joint_tracking_error": tracking_error,
                        }
                    )
                    if query_rejected:
                        logger.warning("Discarding remaining chunk and immediately replanning")
                        break

                if args.execute:
                    consecutive_rejections = consecutive_rejections + 1 if query_rejected else 0
                completed_queries = query_index + 1
                log_file.write(
                    json.dumps(
                        {
                            "query": query_index,
                            "time": time.time(),
                            "inference_ms": inference_ms,
                            "state": state.tolist(),
                            "action_shape": list(actions.shape),
                            "executions": executions,
                            "query_rejected": query_rejected,
                            "consecutive_rejections": consecutive_rejections,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_file.flush()
                logger.info(
                    "query=%d inference=%.1f ms executed=%d rejected=%s consecutive=%d",
                    query_index,
                    inference_ms,
                    sum(bool(item["executed"]) for item in executions),
                    query_rejected,
                    consecutive_rejections,
                )
                if args.record_images:
                    cv2.imwrite(str(image_dir / f"{query_index:06d}_base.jpg"), cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(image_dir / f"{query_index:06d}_wrist.jpg"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
                if args.preview:
                    preview = np.concatenate(
                        [cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)],
                        axis=1,
                    )
                    cv2.imshow("ARX5 base | wrist (q stops inference and holds)", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        _stop_event.set()
                if consecutive_rejections >= int(config["max_consecutive_rejections"]):
                    logger.error("Stopping inference after %d consecutive rejected chunks", consecutive_rejections)
                    _stop_event.set()
                delay = float(config.get("replan_delay_s", 0.0))
                if delay > 0:
                    _sleep_until(time.monotonic() + delay)
        logger.info("Inference loop ended after %d queries", completed_queries)
    except BaseException as exc:
        error = exc
        logger.exception("Deployment stopped by error; transitioning to safety return")
    finally:
        if base_camera is not None:
            base_camera.close()
        if wrist_camera is not None:
            wrist_camera.close()
        if args.preview:
            cv2.destroyAllWindows()
        if args.execute and arm is not None:
            _return_active.set()
            try:
                report = arm.return_to_prelaunch_pose(
                    hold_before_seconds=float(config["return_hold_before_seconds"]),
                    hold_after_seconds=float(config["return_hold_after_seconds"]),
                    hold_refresh_hz=float(config["hold_refresh_hz"]),
                    joint_speed=float(config["return_joint_speed"]),
                    gripper_speed=float(config["return_gripper_speed"]),
                    minimum_duration=float(config["return_min_duration"]),
                    max_tracking_error=float(config["return_max_tracking_error"]),
                    final_joint_tolerance=float(config["return_final_joint_tolerance"]),
                    final_gripper_tolerance=float(config["return_final_gripper_tolerance"]),
                )
                (record_dir / "shutdown.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                arm.release_after_return()
            except BaseException as return_exc:
                if error is None:
                    error = return_exc
                logger.exception(
                    "Safety return failed; keeping the current measured pose instead of releasing"
                )
                _fallback_hold_active.set()
                arm.hold_forever(
                    refresh_hz=float(config["hold_refresh_hz"]),
                    release_event=threading.Event(),
                )

    return 1 if error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
