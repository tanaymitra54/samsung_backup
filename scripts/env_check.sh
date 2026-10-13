#!/usr/bin/env bash
# Sanity-check Python, CUDA, free VRAM, and key paths before a long run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "======== PRISM env check ========"
echo "cwd: $ROOT"
echo "user: $(whoami)"
echo "date: $(date -Is)"
echo ""

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "venv: active (.venv)"
else
  echo "venv: NOT found (run bash scripts/setup_venv.sh)"
fi

echo "python: $(command -v python3) ($(python3 --version 2>&1))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu \
    --format=csv
else
  echo "WARNING: nvidia-smi not found"
fi

echo ""
python3 - <<'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        print(f"  cuda:{i} {name} free={free/2**30:.1f}GiB total={total/2**30:.1f}GiB")
except Exception as e:
    print("torch check FAILED:", e)
    sys.exit(1)

import yaml
from pathlib import Path
cfg_path = Path("config/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
hw = cfg.get("hardware") or {}
print("config model:", cfg.get("model", {}).get("name"))
print("hardware:", hw)
print("eval batch_size:", cfg.get("evaluation", {}).get("batch_size"))
for p in ["cache/models", "outputs", "checkpoints", "data/budget40"]:
    path = Path(p)
    print(f"path {p}: {'OK' if path.exists() else 'MISSING'}")
PY

echo ""
echo "======== env check done ========"
