# Project Structure

The repository root is `openpi-ARX/`. Training, data conversion, policy serving, and real-robot deployment are kept in one project:

```text
openpi-ARX/
├── src/openpi/              # Models, policies, training code, and transforms
├── packages/openpi-client/  # WebSocket inference client library
├── scripts/                 # Training, statistics, and generic policy-serving entry points
├── scripts/arx5/            # ARX5 data-conversion and policy-serving entry points
├── deployment/arx5/         # ARX5 real-robot client, safety controls, config, and tests
├── examples/                # Other robot and dataset examples from upstream OpenPI
├── docs/                    # Project documentation
├── assets/                  # Local assets such as normalization stats; ignored by Git
├── data/                    # Local datasets; ignored by Git
├── checkpoints/             # Local checkpoints; ignored by Git
└── logs/                    # Runtime logs; ignored by Git
```

The real-robot deployment code resolves `packages/openpi-client/src` relative to the repository root. ARX5 data conversion is centralized in profile-based entry points under `scripts/arx5/convert_to_lerobot.py`; ARX5 policy serving is centralized in profile-based entry points under `scripts/arx5/serve_policy.py`.

Legacy `scripts/convert_arx5_*.py`, `scripts/serve_arx5_*.py`, and `scripts/start_server.sh` entry points are retained as compatibility wrappers.

## Compatibility Wrappers

- `convert_arx5_three_tasks_to_lerobot.py` and `convert_arx5_multitask_to_lerobot.py` are compatibility wrappers. New workflows should use `scripts/arx5/convert_to_lerobot.py three_tasks`.
- `convert_arx5_*_to_lerobot.py` wrappers forward to the profile-driven converter.
- `serve_arx5_pick_bread.py` and `serve_arx5_multitask.py` wrappers forward to `scripts/arx5/serve_policy.py`.

The JAX and PyTorch training entry points, the two normalization-statistics tools, and the real-robot execution and no-CAN smoke-test programs intentionally remain separate because their runtime semantics and safety boundaries differ.
