"""Convert the recorded left-arm ARX5 pick-bread data to a local LeRobot dataset.

The source recorder stores 30 Hz bimanual data. This converter deliberately keeps
only the left arm and writes a 15 Hz dataset compatible with OpenPI's LeRobot data
loader. Both ``state`` and ``actions`` have the following absolute 10-D layout:

    [tcp_x, tcp_y, tcp_z, rot_6d_0, ..., rot_6d_5, gripper]

The third-person camera is stored as ``image`` and the left wrist camera as
``wrist_image``. Right-arm data and the right-wrist camera are never read.

Example:
    .tools/uv/uv run scripts/convert_arx5_pick_bread_to_lerobot.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import subprocess

from lerobot.common.datasets import lerobot_dataset as _lerobot_dataset
from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames
import numpy as np

SOURCE_FPS = 30
TARGET_FPS = 15
FRAME_STRIDE = SOURCE_FPS // TARGET_FPS
SOURCE_DIR = Path("data/pick_bread")
OUTPUT_DIR = Path("data/pick_bread_arx5_left_lerobot")
REPO_ID = "local/pick_bread_arx5_left_eef"

OBSERVATION_POSE_COLUMNS = [
    "left_x",
    "left_y",
    "left_z",
    "left_r1",
    "left_r2",
    "left_r3",
    "left_r4",
    "left_r5",
    "left_r6",
    "left_gripper",
]
ACTION_POSE_COLUMNS = [
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "left_gripper.pos",
]
POSE_NAMES = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]


def compute_stats_without_videos(episode_data: dict, features: dict) -> dict:
    """Avoid decoding staged PNGs because direct-transcoded videos already exist."""
    nonvisual_features = {key: feature for key, feature in features.items() if feature["dtype"] != "video"}
    nonvisual_data = {key: episode_data[key] for key in nonvisual_features}
    return compute_episode_stats(nonvisual_data, nonvisual_features)


# LeRobot's recorder saves every input frame as a PNG before computing visual statistics. This offline
# converter writes the final 15 Hz MP4s directly, so statistics are computed for state/action fields only.
_lerobot_dataset.compute_episode_stats = compute_stats_without_videos


class H264LeRobotDataset(LeRobotDataset):
    """Use LeRobot's standard video layout with faster H.264 encoding."""

    def encode_episode_videos(self, episode_index: int) -> dict:
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue
            image_dir = self._get_image_file_path(episode_index, key, frame_index=0).parent
            encode_video_frames(image_dir, video_path, self.fps, vcodec="h264", crf=23, overwrite=True)
        return video_paths


def read_pose_csv(path: Path, columns: list[str]) -> np.ndarray:
    """Read one recorder pose stream with the fixed left-arm 10-D layout."""
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        missing = set(columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = [[float(row[column]) for column in columns] for row in reader]
    return np.asarray(rows, dtype=np.float32)


def read_metadata(episode_dir: Path) -> dict:
    with (episode_dir / "metadata.json").open() as file:
        return json.load(file)


def transcode_video(source: Path, destination: Path, frame_count: int) -> None:
    """Downsample a source video to aligned 15 Hz H.264 without staging PNG frames."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"select=not(mod(n\\,{FRAME_STRIDE})),setpts=N/({TARGET_FPS}*TB)",
        "-frames:v",
        str(frame_count),
        "-r",
        str(TARGET_FPS),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-g",
        "2",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    subprocess.run(command, check=True)


def add_episode(dataset: LeRobotDataset, episode_dir: Path) -> int:
    metadata = read_metadata(episode_dir)
    state = read_pose_csv(
        episode_dir / "observation.state.eef_pose" / "data.csv", OBSERVATION_POSE_COLUMNS
    )
    actions = read_pose_csv(episode_dir / "actions.eef_pose" / "data.csv", ACTION_POSE_COLUMNS)
    if len(state) != len(actions):
        raise ValueError(f"{episode_dir.name}: state/action row mismatch: {len(state)} != {len(actions)}")
    source_frames = len(state)
    recorded_frames = int(metadata["total_frames"])
    if source_frames != recorded_frames:
        logging.warning(
            "%s: metadata reports %d frames, but the aligned pose streams contain %d; using pose streams.",
            episode_dir.name,
            recorded_frames,
            source_frames,
        )

    task = metadata.get("language_instruction", "").strip() or metadata.get("task_title", "pick bread").strip()
    sampled_indices = np.arange(0, source_frames, FRAME_STRIDE)
    episode_index = dataset.meta.total_episodes
    for feature_key, source_camera in (
        ("image", "observation.image.third_view"),
        ("wrist_image", "observation.image.left_wrist_view"),
    ):
        destination = dataset.root / dataset.meta.get_video_file_path(episode_index, feature_key)
        transcode_video(episode_dir / source_camera / "video.mp4", destination, len(sampled_indices))

    # LeRobot stores visual features only in MP4 files, not in the episode Parquet. Video placeholders are
    # sufficient here because the matching video paths already exist; `save_episode` verifies and records them.
    buffer = dataset.create_episode_buffer()
    buffer["size"] = len(sampled_indices)
    buffer["task"] = [task] * len(sampled_indices)
    buffer["frame_index"] = list(range(len(sampled_indices)))
    buffer["timestamp"] = [index / TARGET_FPS for index in range(len(sampled_indices))]
    buffer["state"] = list(state[sampled_indices])
    buffer["actions"] = list(actions[sampled_indices])
    buffer["image"] = [None] * len(sampled_indices)
    buffer["wrist_image"] = [None] * len(sampled_indices)
    dataset.episode_buffer = buffer
    dataset.save_episode()
    return len(sampled_indices)


def main(source_dir: Path, output_dir: Path, max_episodes: int | None = None) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Refusing to overwrite a dataset; choose another path."
        )

    episodes = sorted(path for path in source_dir.glob("episode_*") if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"No episode directories found in {source_dir}")
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    dataset = H264LeRobotDataset.create(
        repo_id=REPO_ID,
        root=output_dir,
        robot_type="arx5_left",
        fps=TARGET_FPS,
        features={
            "image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (10,), "names": POSE_NAMES},
            "actions": {"dtype": "float32", "shape": (10,), "names": POSE_NAMES},
        },
    )

    total_frames = 0
    for index, episode_dir in enumerate(episodes, start=1):
        frames = add_episode(dataset, episode_dir)
        total_frames += frames
        logging.info("Converted %d/%d: %s (%d frames)", index, len(episodes), episode_dir.name, frames)

    logging.info(
        "Completed %d episodes and %d frames at %d Hz in %s",
        len(episodes),
        total_frames,
        TARGET_FPS,
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-episodes", type=int, default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(arguments.source_dir, arguments.output_dir, arguments.max_episodes)
