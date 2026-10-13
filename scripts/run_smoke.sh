#!/usr/bin/env bash
# Tiny end-to-end smoke: 3 GSM8K questions, single or multi-GPU as configured.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

CONFIG="${CONFIG:-config/config.yaml}"
OUT="${OUT:-outputs/smoke}"
SEED="${SEED:-42}"

mkdir -p "$OUT"
echo "==> Smoke run config=$CONFIG out=$OUT"
python3 scripts/run_all_benchmarks.py \
  --config "$CONFIG" \
  --benchmarks gsm8k \
  --subset-size 3 \
  --output-dir "$OUT" \
  --seed "$SEED" \
  "$@"

echo "==> Smoke complete. See $OUT/"
ls -lt "$OUT" | head -10
