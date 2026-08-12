#!/usr/bin/env python3
"""Serve the trained ARX5 pi0.5 checkpoint on a loopback WebSocket."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config

EXPECTED_CHECKPOINT_STEP = 19999


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="pi05_arx5_pick_bread_lora")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/pi05_arx5_pick_bread_lora/pick_bread_lora/19999"),
    )
    parser.add_argument("--default-prompt", default="pick bread")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    config = training_config.get_config(args.config)
    if config.model.action_horizon != 50:
        raise RuntimeError(f"expected action horizon 50, got {config.model.action_horizon}")
    if checkpoint.name != str(EXPECTED_CHECKPOINT_STEP):
        raise RuntimeError(
            f"expected checkpoint step {EXPECTED_CHECKPOINT_STEP}, got {checkpoint.name}"
        )
    logging.info("Loading config=%s checkpoint=%s", args.config, checkpoint)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        default_prompt=args.default_prompt,
    )
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "training_config": args.config,
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(checkpoint.name),
            "supported_prompts": ["pick bread"],
            "action_horizon": int(config.model.action_horizon),
            "action_dim": 10,
        }
    )
    logging.info("Starting policy server on %s:%d", args.host, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
