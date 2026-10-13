#!/usr/bin/env bash
# Run configured benchmarks (auto multi-GPU when hardware.multi_gpu and >1 GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

CONFIG="${CONFIG:-config/config.yaml}"
OUT="${OUT:-outputs/bench_a6000}"
SEED="${SEED:-42}"
SUBSET="${SUBSET:-200}"

mkdir -p "$OUT"
echo "==> Benchmarks config=$CONFIG subset=$SUBSET out=$OUT"
echo "==> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all visible>}"

python3 scripts/run_all_benchmarks.py \
  --config "$CONFIG" \
  --subset-size "$SUBSET" \
  --output-dir "$OUT" \
  --seed "$SEED" \
  "$@"

echo "==> Benchmarks complete. See $OUT/"
ls -lt "$OUT" | head -15
