#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
CONDA_SH="${CONDA_SH:-你的conda环境路径.sh}"
CONDA_ENV="${CONDA_ENV:-dc}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-$PROJECT_ROOT/packages/openpi-client/src}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$ROOT_DIR:$OPENPI_CLIENT_SRC${PYTHONPATH:+:$PYTHONPATH}"
# websockets>=15 auto-discovers HTTP_PROXY. The policy endpoint is the local
# SSH tunnel and must never be routed through the desktop proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

LIST_CAMERAS_ONLY=false
for arg in "$@"; do
  if [[ "$arg" == "--list-cameras" ]]; then
    LIST_CAMERAS_ONLY=true
    break
  fi
done

if [[ "$LIST_CAMERAS_ONLY" == false ]] && pgrep -af 'collector.main.*--robot arx5' >/dev/null 2>&1; then
  echo "[ERROR] ARX5 collector appears to be running. Stop it before opening another controller."
  pgrep -af 'collector.main.*--robot arx5'
  exit 2
fi

exec python "$ROOT_DIR/run_real.py" "$@"
