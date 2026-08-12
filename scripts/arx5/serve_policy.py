#!/usr/bin/env python3
"""Serve trained ARX5 pi0.5 LoRA checkpoints over a loopback WebSocket."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


@dataclasses.dataclass(frozen=True)
class ServerProfile:
    config_name: str
    checkpoint: Path
    expected_step: int
    supported_prompts: tuple[str, ...]


PROFILES: dict[str, ServerProfile] = {
    "pick_bread": ServerProfile(
        config_name="pi05_arx5_pick_bread_lora",
        checkpoint=Path("checkpoints/pi05_arx5_pick_bread_lora/pick_bread_lora/19999"),
        expected_step=19999,
        supported_prompts=("pick bread",),
    ),
    "six_tasks": ServerProfile(
        config_name="pi05_arx5_pick_six_tasks_lora",
        checkpoint=Path("checkpoints/pi05_arx5_pick_six_tasks_lora/pick_six_tasks_lora/49999"),
        expected_step=49999,
        supported_prompts=(
            "pick bread",
            "pick mango",
            "pick bottle",
            "pick cup",
            "pick carrot",
            "pick pen",
        ),
    ),
}


def serve_profile(
    profile_name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    config_name: str | None = None,
    checkpoint: Path | None = None,
    default_prompt: str | None = None,
) -> None:
    profile = PROFILES[profile_name]
    resolved_config_name = config_name or profile.config_name
    resolved_checkpoint = (checkpoint or profile.checkpoint).expanduser().resolve()
    resolved_prompt = default_prompt or profile.supported_prompts[0]

    if resolved_prompt not in profile.supported_prompts:
        raise ValueError(f"unsupported prompt {resolved_prompt!r}; choose one of {profile.supported_prompts}")
    if not resolved_checkpoint.is_dir():
        raise FileNotFoundError(resolved_checkpoint)

    config = training_config.get_config(resolved_config_name)
    if config.model.action_horizon != 50:
        raise RuntimeError(f"expected action horizon 50, got {config.model.action_horizon}")
    if resolved_checkpoint.name != str(profile.expected_step):
        raise RuntimeError(f"expected checkpoint step {profile.expected_step}, got {resolved_checkpoint.name}")

    logging.info("Loading profile=%s config=%s checkpoint=%s", profile_name, resolved_config_name, resolved_checkpoint)
    policy = policy_config.create_trained_policy(
        config,
        resolved_checkpoint,
        default_prompt=resolved_prompt,
    )
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "training_config": resolved_config_name,
            "checkpoint": str(resolved_checkpoint),
            "checkpoint_step": int(resolved_checkpoint.name),
            "supported_prompts": list(profile.supported_prompts),
            "action_horizon": int(config.model.action_horizon),
            "action_dim": 10,
        }
    )
    logging.info("Starting policy server on %s:%d", host, port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=metadata,
    )
    server.serve_forever()


def main(default_profile: str = "pick_bread") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", choices=sorted(PROFILES), default=default_profile)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--default-prompt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    serve_profile(
        args.profile,
        host=args.host,
        port=args.port,
        config_name=args.config,
        checkpoint=args.checkpoint,
        default_prompt=args.default_prompt,
    )


if __name__ == "__main__":
    main()
