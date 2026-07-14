#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ffmpeg git tmux
fi

if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

python - <<'PY'
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Pod")
PY

echo "RunPod environment is ready. Add HF_TOKEN to $ROOT_DIR/.env before training."

