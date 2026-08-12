"""Compute exact OpenPI normalization statistics from a local LeRobot dataset's Parquet data.

Unlike ``compute_norm_stats.py``, this tool does not decode camera videos. It is
valid for datasets whose state/action transforms do not alter numeric values,
including the local ARX5 absolute-EEF-pose configurations. Action chunks are
constructed exactly as LeRobot does: future actions are queried over the model
action horizon and episode tails are padded by repeating the final action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as training_config


def _episode_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet episodes found in {root / 'data'}")
    return paths


def _flush(
    states: list[np.ndarray], actions: list[np.ndarray], state_stats: normalize.RunningStats, action_stats: normalize.RunningStats
) -> None:
    state_stats.update(np.stack(states, axis=0))
    action_stats.update(np.stack(actions, axis=0))
    states.clear()
    actions.clear()


def main(config_name: str) -> None:
    config = training_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("The selected config has no LeRobot repo_id.")

    lerobot_home = Path(__import__("os").environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
    root = lerobot_home / data_config.repo_id
    episodes = _episode_paths(root)
    metadata = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    if len(episodes) != len(metadata):
        raise ValueError(f"Episode Parquet/metadata mismatch: {len(episodes)} vs {len(metadata)}")

    batch_size = config.batch_size
    horizon = config.model.action_horizon
    total_frames = sum(episode["length"] for episode in metadata)
    used_frames = total_frames // batch_size * batch_size
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    state_batch: list[np.ndarray] = []
    action_batch: list[np.ndarray] = []
    seen = 0

    for episode_path, episode_meta in zip(episodes, metadata, strict=True):
        data = pq.read_table(episode_path, columns=["state", "actions"]).to_pydict()
        states = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if len(states) != episode_meta["length"] or len(actions) != len(states):
            raise ValueError(f"Unexpected length in {episode_path}")

        for index in range(len(states)):
            if seen == used_frames:
                break
            state_batch.append(states[index])
            chunk_indices = np.minimum(np.arange(index, index + horizon), len(actions) - 1)
            action_batch.append(actions[chunk_indices])
            seen += 1
            if len(state_batch) == batch_size:
                _flush(state_batch, action_batch, state_stats, action_stats)
        if seen == used_frames:
            break

    if seen != used_frames or state_batch or action_batch:
        raise RuntimeError(f"Expected {used_frames} complete batch frames, processed {seen}")

    output_path = config.assets_dirs / data_config.repo_id
    norm_stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    print(f"Writing stats for {seen} training frames to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
