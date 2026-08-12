#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PORT="${PORT:-8000}"
UV_BIN="${UV_BIN:-uv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.50}"

cd "$OPENPI_ROOT"
exec "$UV_BIN" run scripts/arx5/serve_policy.py pick_bread \
  --host 127.0.0.1 \
  --port "$PORT" \
  --config pi05_arx5_pick_bread_lora \
  --checkpoint checkpoints/pi05_arx5_pick_bread_lora/pick_bread_lora/19999 \
  --default-prompt "pick bread"
