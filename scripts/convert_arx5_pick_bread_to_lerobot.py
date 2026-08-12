"""Compatibility wrapper for the ARX5 pick-bread LeRobot converter."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.arx5.convert_to_lerobot import ACTION_POSE_COLUMNS
from scripts.arx5.convert_to_lerobot import FRAME_STRIDE
from scripts.arx5.convert_to_lerobot import H264LeRobotDataset
from scripts.arx5.convert_to_lerobot import OBSERVATION_POSE_COLUMNS
from scripts.arx5.convert_to_lerobot import POSE_NAMES
from scripts.arx5.convert_to_lerobot import PROFILES
from scripts.arx5.convert_to_lerobot import SOURCE_FPS
from scripts.arx5.convert_to_lerobot import TARGET_FPS
from scripts.arx5.convert_to_lerobot import add_episode
from scripts.arx5.convert_to_lerobot import compute_stats_without_videos
from scripts.arx5.convert_to_lerobot import convert_profile
from scripts.arx5.convert_to_lerobot import create_dataset
from scripts.arx5.convert_to_lerobot import read_metadata
from scripts.arx5.convert_to_lerobot import read_pose_csv
from scripts.arx5.convert_to_lerobot import source_episodes
from scripts.arx5.convert_to_lerobot import transcode_video

SOURCE_DIR = PROFILES["pick_bread"].sources[0].directory
OUTPUT_DIR = PROFILES["pick_bread"].output_dir
REPO_ID = PROFILES["pick_bread"].repo_id


def main(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, max_episodes=None) -> None:
    convert_profile("pick_bread", source_dir=source_dir, output_dir=output_dir, max_episodes=max_episodes)


if __name__ == "__main__":
    from scripts.arx5.convert_to_lerobot import main as _main

    _main("pick_bread")
