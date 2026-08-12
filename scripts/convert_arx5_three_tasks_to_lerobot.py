"""Convert ARX5 pick-bread, pick-mango and pick-bottle recordings into one LeRobot dataset.

The output is intended for pi0.5 fine-tuning. It contains only the left arm:

* state: absolute left EEF pose, 10-D ``[x, y, z, rot6d_0..5, gripper]``;
* actions: absolute left EEF pose in the same 10-D layout;
* cameras: third-person ``image`` and left-wrist ``wrist_image`` only;
* frame rate: source 30 Hz recordings are downsampled to 15 Hz.

The dataset includes the first 100 lexicographically ordered bread episodes, followed
by all mango and bottle episodes. The right arm and right-wrist camera are never read.

Example:
    .tools/uv/uv run scripts/convert_arx5_three_tasks_to_lerobot.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from convert_arx5_pick_bread_to_lerobot import H264LeRobotDataset
from convert_arx5_pick_bread_to_lerobot import POSE_NAMES
from convert_arx5_pick_bread_to_lerobot import TARGET_FPS
from convert_arx5_pick_bread_to_lerobot import add_episode


OUTPUT_DIR = Path("data/pick_bread_mango_bottle_arx5_left_lerobot")
REPO_ID = "local/pick_bread_mango_bottle_arx5_left_eef"
TASK_SOURCES: tuple[tuple[str, Path, int | None], ...] = (
    ("pick_bread", Path("data/pick_bread"), 100),
    ("pick_mango", Path("data/pick_mango"), None),
    ("pick_bottle", Path("data/pick_bottle"), None),
)


def selected_episodes(source_dir: Path, limit: int | None) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    episodes = sorted(path for path in source_dir.glob("episode_*") if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"No episode directories found in {source_dir}")
    if limit is not None:
        if len(episodes) < limit:
            raise ValueError(f"{source_dir} contains {len(episodes)} episodes, fewer than requested {limit}")
        episodes = episodes[:limit]
    return episodes


def main(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Refusing to overwrite an existing dataset."
        )

    sources = [(task, selected_episodes(source_dir, limit)) for task, source_dir, limit in TASK_SOURCES]
    total_episodes = sum(len(episodes) for _, episodes in sources)
    logging.info("Preparing %d episodes: %s", total_episodes, {task: len(episodes) for task, episodes in sources})

    dataset = H264LeRobotDataset.create(
        repo_id=REPO_ID,
        root=output_dir,
        robot_type="arx5_left",
        fps=TARGET_FPS,
        features={
            "image": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (10,), "names": POSE_NAMES},
            "actions": {"dtype": "float32", "shape": (10,), "names": POSE_NAMES},
        },
    )

    converted = 0
    total_frames = 0
    for task, episodes in sources:
        for episode_dir in episodes:
            total_frames += add_episode(dataset, episode_dir)
            converted += 1
            logging.info("Converted %d/%d [%s]: %s", converted, total_episodes, task, episode_dir.name)

    logging.info(
        "Completed %d episodes and %d frames at %d Hz in %s", converted, total_frames, TARGET_FPS, output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(arguments.output_dir)
