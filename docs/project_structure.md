# 项目结构

项目根目录为 `openpi-ARX/`，训练、数据转换、策略服务和真机部署属于同一个项目：

```text
openpi-ARX/
├── src/openpi/              # 模型、策略、训练和数据变换库
├── packages/openpi-client/  # WebSocket 推理客户端库
├── scripts/                 # 数据转换、训练、统计和策略服务入口
├── deployment/arx5/         # ARX5 真机客户端、安全控制、配置和测试
├── examples/                # 其他机器人或数据集示例
├── docs/                    # 项目文档
├── assets/                  # 归一化统计等本地资产（Git 忽略）
├── data/                    # 本地数据集（Git 忽略）
├── checkpoints/             # 本地 checkpoint（Git 忽略）
└── logs/                    # 运行日志（Git 忽略）
```

真机部署目录通过相对项目根目录定位 `packages/openpi-client/src`。策略服务器统一使用
`scripts/serve_arx5_pick_bread.py`，启动入口为 `scripts/start_server.sh`。

## 可进一步整理的重复文件

- `convert_arx5_three_tasks_to_lerobot.py` 与 `convert_arx5_multitask_to_lerobot.py`
  功能基本相同，可以保留后者。
- 多个 `convert_arx5_*_to_lerobot.py` 可以重构为一个配置驱动的通用转换器。
- `serve_arx5_pick_bread.py` 与 `serve_arx5_multitask.py` 可以合并为带预设参数的通用服务入口。
- `scripts/*.before_*` 是人工备份，确认无需回退后可以删除或移至仓库外归档。

不建议合并 JAX/PyTorch 训练入口、两种归一化统计工具，以及真机执行与无 CAN
冒烟测试程序；它们的运行语义或安全边界不同。
