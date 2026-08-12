# OpenPI π0.5 Fangzhou ARX5

This repository builds on [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi) and provides a single-arm training and deployment stack for Fangzhou ARX5 robots, including data conversion, π0.5 LoRA fine-tuning, policy serving, and real-robot execution.

The current implementation covers single-arm 10-D absolute EEF pose control. Bimanual support is planned as a future extension. Multitask training and deployment reuse the same data format, model I/O mapping, and safety-oriented execution framework.

> **Real-robot safety warning**: first-time deployment must proceed in this order: camera check -> server smoke test -> inference-only run -> low-risk execution. Do not use `--execute` until the emergency stop, workspace, CAN interface, target arm, and camera viewpoints have all been verified.

## What This Repository Adds

- Fangzhou ARX5 raw-data conversion to LeRobot / OpenPI format.
- Single-arm 10-D absolute EEF pose mapping and π0.5 action slicing.
- LoRA training configs.
- ARX5 single-arm policy-server profiles.
- Real-robot client code, camera checks, inference-only validation, safe execution, and shutdown handling.

## Repository Layout

```text
openpi-ARX/
├── src/openpi/              # Models, policies, training code, and transforms
├── packages/openpi-client/  # WebSocket inference client library
├── scripts/                 # Training, statistics, and generic policy-serving entry points
├── scripts/arx5/            # ARX5 data-conversion and policy-serving entry points
├── deployment/arx5/         # ARX5 real-robot client, safety controls, config, and tests
├── examples/                # Upstream OpenPI examples
├── docs/                    # Additional documentation
├── assets/                  # Local assets such as normalization stats; ignored by Git
├── data/                    # Local datasets; ignored by Git
└── checkpoints/             # Local checkpoints; ignored by Git
```

See [docs/project_structure.md](docs/project_structure.md) for more detail.

## Environment

Training environment:

- Ubuntu 22.04
- NVIDIA GPU with CUDA 12 driver
- Python 3.11
- `uv`
- `ffmpeg`

Key verified dependencies are pinned by [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock), including:

```text
jax[cuda12]==0.5.3
torch==2.7.1
transformers==4.53.2
```

Install:

```bash
git clone --recurse-submodules <repo-url>
cd openpi-ARX

uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Set local cache directories before data processing, normalization-stat computation, or training:

```bash
export HF_LEROBOT_HOME=$PWD/data
export HF_DATASETS_CACHE=$PWD/.cache/hf-datasets
export UV_CACHE_DIR=$PWD/.cache/uv
```

## Data Format

Place raw data under `data/`. Each episode must contain at least:

```text
metadata.json
observation.state.eef_pose/data.csv
actions.eef_pose/data.csv
observation.image.third_view/video.mp4
observation.image.left_wrist_view/video.mp4
```

Convert the pick-bread dataset:

```bash
uv run scripts/arx5/convert_to_lerobot.py pick_bread
```

Output directory:

```text
data/pick_bread_arx5_left_lerobot/
```

Create the local dataset link used by OpenPI:

```bash
mkdir -p data/local
ln -s ../pick_bread_arx5_left_lerobot data/local/pick_bread_arx5_left_eef
```

Converter conventions:

- The current single-arm implementation uses the raw recorder's left arm and left wrist camera fields; the right arm and right wrist camera are not used for training.
- Videos are downsampled from 30 Hz to 15 Hz and written as H.264 MP4 files.
- State and actions are stored as 10-D absolute EEF poses.

The 10-D layout is:

```text
[x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]
```

π0.5 pads the 10-D state/action internally to the model-required dimension. The LeRobot dataset, robot interface, and deployment output remain 10-D. This workflow uses absolute pose actions, not delta actions.

## Data Mapping and Training Config

[src/openpi/policies/arx5_policy.py](src/openpi/policies/arx5_policy.py) defines the ARX5 single-arm adapter. The current single-arm input comes from the recorder's left arm and left wrist camera fields; `left` in dataset IDs denotes the data source, not a project-level limitation to the left arm.

- `image` maps to the third-person view.
- `wrist_image` maps to the single-arm wrist camera input and currently comes from `left_wrist_view`.
- The missing opposite-side wrist view is filled by the adapter with a zero image and mask.
- `state` is a single-arm 10-D absolute EEF pose.
- The model output is sliced to the first 10 action dimensions before deployment.

ARX5 data configs live in [src/openpi/training/config.py](src/openpi/training/config.py). The core pick-bread LoRA config is:

```python
name="pi05_arx5_pick_bread_lora"
repo_id="local/pick_bread_arx5_left_eef"
num_train_steps=20_000
action_horizon=50
warmup_steps=1_000
peak_lr=5e-5
wandb_enabled=False
```

Recompute normalization statistics whenever the dataset, state/action dimensions, or action horizon changes.

## Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_arx5_pick_bread_lora
```

Output:

```text
assets/pi05_arx5_pick_bread_lora/
└── local/pick_bread_arx5_left_eef/norm_stats.json
```

Training and inference both depend on this file. Do not reuse old statistics after changing the dataset or action space.

## Training

Start a new run:

```bash
uv run scripts/train.py \
  pi05_arx5_pick_bread_lora \
  --exp-name=pick_bread_lora
```

Resume a run:

```bash
uv run scripts/train.py \
  pi05_arx5_pick_bread_lora \
  --exp-name=pick_bread_lora \
  --resume
```

Checkpoint directory:

```text
checkpoints/pi05_arx5_pick_bread_lora/pick_bread_lora/<step>/
```

Each checkpoint contains:

```text
assets/       # Normalization statistics
params/       # Model parameters for inference
train_state/  # Training state for resume
```

By default, checkpoints are saved every 1000 steps; every 5000-step checkpoint and the final checkpoint are retained.

## Policy Server

Start the pick-bread policy server:

```bash
uv run scripts/arx5/serve_policy.py pick_bread
```

This loads:

```text
config:     pi05_arx5_pick_bread_lora
checkpoint: checkpoints/pi05_arx5_pick_bread_lora/pick_bread_lora/19999
prompt:     pick bread
host:       127.0.0.1
port:       8000
```

[scripts/arx5/serve_policy.py](scripts/arx5/serve_policy.py) checks the action horizon and checkpoint step, then publishes the training config, checkpoint step, action horizon, and action dimension in WebSocket metadata. The real-robot client reads this metadata to avoid connecting to the wrong policy server.

Legacy entry points [scripts/start_server.sh](scripts/start_server.sh) and [scripts/serve_arx5_pick_bread.py](scripts/serve_arx5_pick_bread.py) are kept for compatibility.

## ARX5 Real-Robot Deployment

Real-robot code lives in [deployment/arx5](deployment/arx5). The robot runtime environment must provide the ARX5 SDK, RealSense dependencies, OpenCV, and NumPy. Configure `CONDA_SH` and `CONDA_ENV` for [deployment/arx5/run_client.sh](deployment/arx5/run_client.sh).

Safety thresholds, task start poses, camera serial numbers, CAN interface, and log root are configured in:

```text
deployment/arx5/deployment_config.json
```

Before deployment, verify:

- `interface` points to the target CAN interface.
- `base_camera_serial` and `wrist_camera_serial` match the third-person view and single-arm wrist view used during training.
- `tasks.<task>.policy_start_*` matches the task start pose.
- `expected_server_config` and `expected_server_checkpoint_step` match the running policy server.
- `workspace_min`, `workspace_max`, and step limits cover the task motion range and have been checked against inference-only logs.

### Camera and Server Smoke Test

```bash
cd deployment/arx5
./run_client.sh --list-cameras
./smoke_test_server.sh --task bread
```

`smoke_test_server.sh` reads live cameras and requests one policy inference. It does not open CAN and does not send robot commands. The expected action chunk shape is `(50, 10)`.

### Inference-Only Run

```bash
./run_client.sh --task bread --max-queries 10 --preview --record-images
```

Without `--execute`, the client does not send model actions. It reads the real EEF state, runs workspace, step, IK, and joint-step checks, and records results to `runs/<timestamp>/queries.jsonl`.

### Low-Risk Execution

First validate initialization and safe return:

```bash
./run_client.sh --task bread --execute --initialize --initialize-only
```

After confirming cameras, initialization, emergency stop, workspace, and return path, execute a small number of low-frequency actions:

```bash
./run_client.sh \
  --task bread \
  --execute \
  --initialize \
  --max-queries 20 \
  --execute-steps 10 \
  --action-hz 5 \
  --preview \
  --record-images
```

Before execution starts, the interactive terminal must receive the exact confirmation phrase `EXECUTE CAN1`. After motion is stable, `--execute-steps` and `--action-hz` can be adjusted gradually, but `--action-hz` should not exceed the 15 Hz training-data frequency.

### Safe Shutdown

- Every action is checked against workspace, EEF position/rotation/gripper step limits, IK, and joint-step limits.
- A rejected action discards the rest of the current chunk and immediately requests a new inference; execution stops once consecutive rejections reach the configured threshold.
- Normal completion, consecutive rejections, `q`, Ctrl+C, SIGTERM, and Python exceptions all latch the current pose first, then smoothly return to the measured pre-launch joint pose.
- If return interpolation, communication, or final tolerance checks fail, the program does not release the robot. It enters safety-hold and keeps the measured pose at the configured refresh rate.
- Software cannot protect against forced process termination, host crashes, power loss, CAN disconnection, or driver hardware faults. Those cases must be handled by mechanical support, emergency stop, and hardware braking.

Run records:

```text
runs/<timestamp>/config.json
runs/<timestamp>/queries.jsonl
runs/<timestamp>/images/
runs/<timestamp>/shutdown.json
```

## Multitask

The repository also includes ARX5 multitask conversion, training, and serving profiles:

- `three_tasks` and `six_tasks` profiles in [scripts/arx5/convert_to_lerobot.py](scripts/arx5/convert_to_lerobot.py)
- `pi05_arx5_pick_bread_mango_bottle_lora` in [src/openpi/training/config.py](src/openpi/training/config.py)
- `pi05_arx5_pick_six_tasks_lora` in [src/openpi/training/config.py](src/openpi/training/config.py)
- `six_tasks` profile in [scripts/arx5/serve_policy.py](scripts/arx5/serve_policy.py)

The multitask workflow is the same as pick bread; replace the dataset repo ID, training config, normalization statistics, checkpoint, and serving profile.
