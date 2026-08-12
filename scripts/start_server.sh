#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.50}"

cd "$OPENPI_ROOT"
exec .tools/uv/uv run scripts/serve_arx5_pick_bread.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --config pi05_arx5_pick_bread_lora \
  --checkpoint checkpoints/pi05_arx5_pick_bread_lora/pick_bread_lora/19999 \
  --default-prompt "pick bread"
