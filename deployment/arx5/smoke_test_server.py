#!/usr/bin/env python3
"""Query the policy once with live cameras and a recorded initial state; no CAN access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from arx5_deploy.camera import RealSenseColorCamera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deployment_config.json"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--task", help="Task profile name from deployment_config.json")
    parser.add_argument("--prompt", help="Override task prompt")
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    host = args.host or config["server_host"]
    port = args.port or config["server_port"]
    tasks = config.get("tasks", {})
    task_name = args.task or config.get("default_task")
    if task_name not in tasks:
        raise ValueError(f"unknown task {task_name!r}; choose one of {sorted(tasks)}")
    task = tasks[task_name]
    prompt = args.prompt or task["prompt"]
    recorded_policy_start_state = np.asarray(task["policy_start_state"], dtype=np.float32)
    if recorded_policy_start_state.shape != (10,):
        raise ValueError(f"task {task_name!r} has invalid policy_start_state")
    base = RealSenseColorCamera(
        config["base_camera_serial"],
        "base",
        width=config["camera_width"],
        height=config["camera_height"],
        fps=config["camera_fps"],
    )
    wrist = RealSenseColorCamera(
        config["wrist_camera_serial"],
        "wrist",
        width=config["camera_width"],
        height=config["camera_height"],
        fps=config["camera_fps"],
    )
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        metadata = client.get_server_metadata()
        expected_config = str(config["expected_server_config"])
        expected_step = int(config["expected_server_checkpoint_step"])
        if metadata.get("training_config") != expected_config or metadata.get("checkpoint_step") != expected_step:
            raise RuntimeError(
                "wrong policy server: "
                f"expected config={expected_config} checkpoint_step={expected_step}, got {metadata}"
            )
        size = int(config["policy_image_size"])
        result = client.infer(
            {
                "observation/image": image_tools.resize_with_pad(base.read_rgb(), size, size),
                "observation/wrist_image": image_tools.resize_with_pad(wrist.read_rgb(), size, size),
                "observation/state": recorded_policy_start_state,
                "prompt": prompt,
            }
        )
        actions = np.asarray(result["actions"])
        expected = (int(config["policy_action_horizon"]), 10)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid actions shape={actions.shape} finite={np.isfinite(actions).all()}")
        print("Server smoke test passed (NO CAN / NO ROBOT COMMANDS).")
        print(f"task: {task_name} prompt={prompt!r}")
        print("metadata:", metadata)
        print("actions shape:", actions.shape)
        print("first action:", np.array2string(actions[0], precision=6))
        print("first-10 xyz range:", actions[:10, :3].min(axis=0), actions[:10, :3].max(axis=0))
        return 0
    finally:
        base.close()
        wrist.close()


if __name__ == "__main__":
    raise SystemExit(main())
