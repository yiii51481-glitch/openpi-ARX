# OpenPI π0.5 ARX5：训练与真机部署

本仓库基于 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)，实现 ARX5 数据转换、π0.5 LoRA 微调和真机部署。

本文只以 **pick bread 单任务** 为例，完整说明从原始数据到 ARX5 真机执行的流程。其他任务或多任务训练可复用相同的数据格式、输入输出适配和部署流程。

> **真机安全警告**：首次部署时只能按“相机检查 → 服务冒烟测试 → 只读推理 → 低风险执行”的顺序操作。确认急停、工作空间、CAN 接口、目标机械臂和相机视角无误前，禁止使用 `--execute`。

## 1. Pick Bread 训练设置

| 项目 | 当前值 |
|---|---|
| 基座模型 | π0.5（`pi05_base`） |
| 微调方式 | LoRA |
| 训练配置 | `pi05_arx5_pick_bread_lora`（替换为你的任务配置） |
| 数据集 | `local/pick_bread_arx5_left_eef`（替换为你的数据集） |
| Episode 数 | 150 |
| 训练帧数 | 30,863 |
| 训练步数 | 20,000 |
| 全局 batch size | 32 |
| action horizon | 50 |
| 视频频率 | 15 Hz（原始 30 Hz 下采样） |
| 输入/输出 | 左臂 10 维 absolute EEF pose |
| W&B | 关闭 |

state 和 action 的 10 维排列：

~~~
[x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]
~~~

π0.5 内部会把 10 维 state/action 补齐到模型所需的 32 维；机器人接口和 LeRobot 数据集实际保存的仍是 10 维。此流程始终使用 absolute pose，不使用 delta action。

## 2. 目录结构


完整的目录职责和重复文件分析见 (docs/project_structure.md)。

## 3. 训练服务器环境

### 3.1 系统要求

- Ubuntu 22.04
- NVIDIA GPU 与 CUDA 12 驱动
- Python 3.11
- `uv`
- `ffmpeg`（视频转码）
- LoRA 单卡通常需要至少约 42GB 显存

当前项目已验证的关键依赖：

~~~
jax[cuda12]==0.5.3
torch==2.7.1
transformers==4.53.2
~~~

### 3.2 安装

~~~bash
git clone --recurse-submodules <github.com/yiii51481-glitch/openpi-ARX >
cd openpi-ARX


.tools/uv/uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 .tools/uv/uv sync
GIT_LFS_SKIP_SMUDGE=1 .tools/uv/uv pip install -e .
~~~

每次处理数据或训练前设置：

~~~bash
export HF_LEROBOT_HOME=$PWD/data
export HF_DATASETS_CACHE=$PWD/.cache/hf-datasets
export UV_CACHE_DIR=$PWD/.cache/uv
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
~~~

GPU 查看：

~~~bash
nvidia-smi
watch -n 1 nvidia-smi
~~~

## 4. Pick Bread 数据处理

### 4.1 原始数据

将 `pick_bread` 放在 `data/` 后执行：

每条 episode 至少应有：

~~~
metadata.json
observation.state.eef_pose/data.csv
actions.eef_pose/data.csv
observation.image.third_view/video.mp4
observation.image.left_wrist_view/video.mp4
~~~

### 4.2 转换为 LeRobot / OpenPI 格式

运行转换器：

~~~bash
UV_CACHE_DIR=$PWD/.cache/uv .tools/uv/uv run scripts/convert_arx5_pick_bread_to_lerobot.py
~~~

转换器执行以下操作：

- 仅读取左臂 state 和 action。
- 完全忽略右臂和右腕相机。
- 第三视角写为 `image`，左腕相机写为 `wrist_image`。
- state/action 都保存为 10 维 absolute EEF pose。
- 视频从 30 Hz 下采样至 15 Hz，写为 H.264 MP4。
- 原始 metadata 与 pose 帧数相差 1 帧时，以 state/action pose 的对齐帧数为准。

输出目录：

~~~
data/pick_bread_arx5_left_lerobot/
~~~

创建供 OpenPI 使用的本地数据集链接：

~~~bash
ln -s ../pick_bread_arx5_left_lerobot   data/local/pick_bread_arx5_left_eef
~~~

检查结果：

~~~bash
find data/pick_bread_arx5_left_lerobot/data -name '*.parquet' | wc -l
find data/pick_bread_arx5_left_lerobot/videos -name '*.mp4' | wc -l
~~~

当前数据预期为 150 个 Parquet 和 300 个 MP4。

## 5. 数据映射、配置与归一化

### 5.1 ARX5 数据映射

[src/openpi/policies/arx5_policy.py](src/openpi/policies/arx5_policy.py) 将数据转换为 π0.5 的输入输出：

- `image` → 第三视角 `base_0_rgb`
- `wrist_image` → 左腕 `left_wrist_0_rgb`
- 单臂操作时右腕相机不存在，使用零图像填充并通过 mask 标记
- state 为左臂 10 维 absolute EEF pose
- 部署时模型输出仅取前 10 维 action

不要对这组数据使用 `DeltaActions`。rot6d 不能像欧拉角那样直接相减；训练与部署都必须保持 absolute EEF pose 表示。

### 5.2 LoRA 配置

配置位于 [src/openpi/training/config.py](src/openpi/training/config.py)：

~~~python
name="pi05_arx5_pick_bread_lora"
repo_id="local/pick_bread_arx5_left_eef"
batch_size=32
num_train_steps=20_000
action_horizon=50
warmup_steps=1_000
peak_lr=5e-5
wandb_enabled=False
~~~

调整 batch size、训练步数、学习率、保存间隔等参数，均在此 `TrainConfig` 条目修改。若变更数据、state/action 维度或 action horizon，必须重新计算归一化统计。

### 5.3 计算归一化统计

~~~bash
HF_LEROBOT_HOME=$PWD/data HF_DATASETS_CACHE=$PWD/.cache/hf-datasets UV_CACHE_DIR=$PWD/.cache/uv .tools/uv/uv run scripts/compute_norm_stats.py   --config-name pi05_arx5_pick_bread_lora
~~~

输出文件：

~~~
assets/pi05_arx5_pick_bread_lora/
└── local/pick_bread_arx5_left_eef/norm_stats.json
~~~

确认文件存在后才启动训练。

## 6. π0.5 LoRA 微调

### 6.1 新训练

建议使用 tmux：

~~~bash
tmux new-session -s pi05_arx5_pick_bread_lora
~~~

例如 4 卡训练：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_LEROBOT_HOME=$PWD/data HF_DATASETS_CACHE=$PWD/.cache/hf-datasets UV_CACHE_DIR=$PWD/.cache/uv XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .tools/uv/uv run scripts/train.py   pi05_arx5_pick_bread_lora   --exp-name=pick_bread_lora
~~~

查看日志：

~~~bash
tmux attach -t pi05_arx5_pick_bread_lora
tail -f logs/pi05_arx5_pick_bread_lora_train.log
~~~

在 tmux 中按 `Ctrl+b`、再按 `d` 可安全分离会话。

### 6.2 断点恢复

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_LEROBOT_HOME=$PWD/data HF_DATASETS_CACHE=$PWD/.cache/hf-datasets UV_CACHE_DIR=$PWD/.cache/uv XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .tools/uv/uv run scripts/train.py   pi05_arx5_pick_bread_lora   --exp-name=pick_bread_lora   --resume
~~~

不要同时使用 `--overwrite`，因为它会删除已有 checkpoint。

`CUDA_VISIBLE_DEVICES` 仅在进程启动时生效。运行中的 JAX 训练不会在其他 GPU 空闲后自动扩容；若要改变 GPU 数量，请先保存 checkpoint、停止训练，再用 `--resume` 重启。

### 6.3 Checkpoint

保存目录：

~~~
checkpoints/pi05_arx5_pick_bread_lora/pick_bread_lora/<step>/
~~~

每个 checkpoint 包含：

~~~
assets/       # 归一化统计
params/       # 推理使用的模型参数
train_state/  # 继续训练所需状态（优化器、step 等）
~~~

默认每 1000 step 保存；每 5000 step 及最终 step 会保留。


## 7. ARX5 真机部署

真机端客户端代码位于 [deployment/arx5](deployment/arx5)。该目录与训练和服务端代码属于同一个项目；真机运行时从项目根目录进入该目录即可。

详细说明：

| 项目 | 当前值 |
|---|---|
| Conda 环境 | `需要有ARX5 SDK  RealSense 相关依赖 OpenCV 和NumPy` |
| CAN 接口 | `你的can口设置` |
| 第三视角 RealSense | `你的相机` |
| 左腕 RealSense | `你的相机` |
| 执行步数 | 默认每 chunk 前 10 / 50 步 |
| 初始动作频率 | 5 Hz |

安全阈值、起始姿态和相机参数在：

~~~
deployment/arx5/deployment_config.json
~~~
## 调参

所有安全/频率参数集中在 [deployment_config.json](deployment/arx5/deployment_config.json)。常用项：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `execute_steps` | 10 | 每个 50 步 chunk 执行的前 N 步 |
| `action_hz` | 5 | 动作发送频率；验证稳定后可逐步提高，训练数据为 15 Hz |
| `max_consecutive_rejections` | 3 | 连续超限停止推理的阈值 |
| `max_position_step` | 0.025 m | 单步末端平移上限 |
| `max_rotation_step` | 0.08 rad | 单步末端旋转上限 |
| `max_gripper_step` | 0.015 m | 单步夹爪开度上限 |
| `max_joint_step` | 0.20 rad | IK 解相对实测关节的单步上限 |
| `hold_refresh_hz` | 20 Hz | 最终姿态/返回到位后的保持刷新率 |
| `return_hold_before_seconds` | 1 s | 返回前保持最终姿态的时间 |
| `return_joint_speed` | 0.10 rad/s | 返回启动前关节位置的最大命令速度 |
| `return_gripper_speed` | 0.01 m/s | 返回启动前夹爪位置的最大命令速度 |
| `return_final_joint_tolerance` | 0.05 rad | 允许正常释放的返回到位误差 |

默认 step 上限覆盖当前 15 Hz 训练数据的实际单步范围，但仍应根据第一轮只读日志
逐步收紧。命令行的 `--execute-steps`、`--action-hz`、`--max-queries` 和
`--max-consecutive-rejections` 会覆盖 JSON；其余参数直接修改 JSON。
**重要**：确认左腕相机物理视角与训练时一致。




### 7.1 相机与服务冒烟测试

从项目根目录进入真机部署目录：

~~~bash
cd deployment/arx5
~~~

先检查相机：

~~~bash
./run_client.sh --list-cameras
~~~

然后仅请求一次推理，不打开 CAN：

~~~bash
./smoke_test_server.sh --task bread
~~~

预期得到有限的 `(50, 10)` action chunk。

### 7.2 只读推理测试

先关闭数采程序和所有可能占用 `can口` 的控制器：

~~~bash
./run_client.sh --task bread --max-queries 10 --preview --record-images
~~~

不加 `--execute` 时不会发送模型动作；客户端会读取真实 EEF state、执行 workspace/步长/IK 检查，并记录到 `runs/<timestamp>/queries.jsonl`。

### 7.3 低风险真机执行

确认急停、人员清空、工作空间和返回路径后，先只验证初始化与安全返回：

~~~bash
./run_client.sh --task bread --execute --initialize --initialize-only
~~~

确认正常后，才以低频、少步数执行：

~~~bash
./run_client.sh   --task bread   --execute   --initialize   --max-queries 20   --execute-steps 10   --action-hz 5   --preview   --record-images
~~~

必须输入精确确认词 `EXECUTE CAN1` 才会开始发送机械臂命令。确认动作稳定后，可逐步提高 `--action-hz`，但不应超过训练频率 15 Hz。

### 7.4 安全退出

- 每个动作都会检查 workspace、末端位置/旋转/夹爪步长、IK 和关节步长。
- 单次超限会丢弃当前 chunk 的剩余动作并请求新推理；默认连续 3 个 chunk 超限才停止。
- 正常完成、连续超限、`q`、Ctrl+C、SIGTERM 或 Python 异常均会先锁存当前姿态，再平滑返回启动前实测关节位置。
- 如果返回插值、通信或到位检查失败，程序不会释放机械臂，而会回退到永久
  safety-hold，以 20 Hz 保持当时实测位置；此时应处理故障，不能强制杀进程。
- 软件无法防护 `kill -9`、进程/主机崩溃、断电、CAN 拔线或驱动器硬件故障；这些
  情况必须依靠机械支撑、急停和硬件制动方案。



运行日志：

~~~
runs/<timestamp>/config.json
runs/<timestamp>/queries.jsonl
runs/<timestamp>/images/
runs/<timestamp>/shutdown.json
~~~

## 8. 常见问题

### W&B API key 错误

若出现 `wandb: ERROR api_key not configured (no-tty)`，确认对应 ARX5 训练配置中：

~~~python
wandb_enabled=False
~~~

### 找不到归一化统计

确认：

~~~bash
export HF_LEROBOT_HOME=$PWD/data
~~~

然后重新执行第 5.3 节命令。

### batch size 与 GPU 数量不匹配

`batch_size=32` 必须能被当前可见 GPU 数量整除。设置 `CUDA_VISIBLE_DEVICES` 后，在相同 shell 中启动训练。

### GPU 显存被占用

~~~bash
nvidia-smi
ps -fp <PID>
~~~

关闭终端不一定会停止 tmux/nohup 启动的模型服务。确认不再需要后才停止自己的进程；不要终止其他用户的进程。


## 9. 扩展到多任务

本仓库还包含 pick mango、bottle、cup、carrot、pen 等任务的转换与任务 LoRA 配置：

- [六任务转换器](scripts/convert_arx5_six_tasks_to_lerobot.py)
- [六任务配置](src/openpi/training/config.py) 中的 `pi05_arx5_pick_six_tasks_lora`
- [六任务策略服务](scripts/serve_arx5_multitask.py)

多任务流程与本文完全相同，只是更换数据集 repo ID、训练配置、归一化统计和 checkpoint。
