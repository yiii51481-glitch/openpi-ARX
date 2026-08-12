#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
: "${CONDA_SH:?Set CONDA_SH to the path of conda.sh for the ARX5 runtime environment}"
CONDA_ENV="${CONDA_ENV:-dc}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$ROOT_DIR:${OPENPI_CLIENT_SRC:-$PROJECT_ROOT/packages/openpi-client/src}${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
exec python "$ROOT_DIR/smoke_test_server.py" "$@"
