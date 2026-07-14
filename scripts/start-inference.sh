#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/vla_training"

if [[ -n "${VLA_PYTHON:-}" ]]; then
  PYTHON_BIN="$VLA_PYTHON"
elif [[ -x "$TRAINING_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$TRAINING_DIR/.venv/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "/opt/anaconda3/bin/python" ]]; then
  PYTHON_BIN="/opt/anaconda3/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

cd "$TRAINING_DIR"
exec "$PYTHON_BIN" inference_server.py "$@"
