"""Compatibility wrapper for the three-task ARX5 LeRobot converter."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.arx5.convert_to_lerobot import PROFILES
from scripts.arx5.convert_to_lerobot import convert_profile

DEFAULT_OUTPUT_DIR = PROFILES["three_tasks"].output_dir
REPO_ID = PROFILES["three_tasks"].repo_id
SOURCES = tuple((source.name, source.directory, source.limit) for source in PROFILES["three_tasks"].sources)


def main(output_dir=DEFAULT_OUTPUT_DIR) -> None:
    convert_profile("three_tasks", output_dir=output_dir)


if __name__ == "__main__":
    from scripts.arx5.convert_to_lerobot import main as _main

    _main("three_tasks")
