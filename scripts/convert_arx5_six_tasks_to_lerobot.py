"""Convert six ARX5 pick tasks to one 15 Hz left-arm LeRobot dataset.

The dataset contains the first 100 pick-bread episodes and every episode from
pick-mango, pick-bottle, pick-cup, pick-carrot, and pick-pen. Only the left arm
is retained. State and action both use the absolute 10-D EEF-pose layout
``[x, y, z, rot6d_0, ..., rot6d_5, gripper]``. Third-person and left-wrist
videos are downsampled from 30 Hz to 15 Hz; right-arm/right-wrist data is never
read.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from convert_arx5_pick_bread_to_lerobot import H264LeRobotDataset, POSE_NAMES, TARGET_FPS, add_episode


DEFAULT_OUTPUT_DIR = Path("data/pick_six_tasks_arx5_left_lerobot")
REPO_ID = "local/pick_six_tasks_arx5_left_eef"
SOURCES: tuple[tuple[str, Path, int | None], ...] = (
    ("pick_bread", Path("data/pick_bread"), 100),
    ("pick_mango", Path("data/pick_mango"), None),
    ("pick_bottle", Path("data/pick_bottle"), None),
    ("pick_cup", Path("data/pick_cup"), None),
    ("pick_carrot", Path("data/pick_carrot"), None),
    ("pick_pen", Path("data/pick_pen"), None),
)


def source_episodes(source_name: str, source_dir: Path, limit: int | None) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"{source_name}: source directory does not exist: {source_dir}")
    episodes = sorted(path for path in source_dir.glob("episode_*") if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"{source_name}: no episode directories in {source_dir}")
    return episodes if limit is None else episodes[:limit]


def create_dataset(output_dir: Path) -> LeRobotDataset:
    return H264LeRobotDataset.create(
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


def main(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; refusing to overwrite it.")

    sources = [(name, source_episodes(name, directory, limit)) for name, directory, limit in SOURCES]
    total_episodes = sum(len(episodes) for _, episodes in sources)
    dataset = create_dataset(output_dir)

    total_frames = 0
    converted = 0
    for source_name, episodes in sources:
        for episode_dir in episodes:
            frames = add_episode(dataset, episode_dir)
            converted += 1
            total_frames += frames
            logging.info("Converted %d/%d [%s]: %s (%d frames)", converted, total_episodes, source_name, episode_dir.name, frames)

    logging.info(
        "Completed %d episodes and %d frames at %d Hz in %s",
        converted,
        total_frames,
        TARGET_FPS,
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(arguments.output_dir)
