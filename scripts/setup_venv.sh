#!/usr/bin/env bash
# Create .venv and install CUDA 12.1 PyTorch + project deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Project root: $ROOT"
echo "==> Using: $PYTHON ($(${PYTHON} --version 2>&1))"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 for the reference environment"'

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating venv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
echo "==> Installing CUDA 12.1 PyTorch, then the pinned project stack"
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check

mkdir -p cache/models cache/huggingface outputs checkpoints

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run:"
echo "  bash scripts/env_check.sh"
