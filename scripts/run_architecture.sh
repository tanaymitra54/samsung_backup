#!/usr/bin/env bash
# Architecture loop: matched budget → QLoRA → re-eval.
# Eval/trace collection can multi-GPU; QLoRA stays on one GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

CONFIG="${CONFIG:-config/config.yaml}"
ROUNDS="${ROUNDS:-1}"
SEED="${SEED:-42}"

if [[ ! -f data/budget40/eval_manifest.jsonl ]] || [[ ! -f data/budget40/train_manifest.jsonl ]]; then
  echo "==> Budget40 manifests missing — running curate_budget40.py"
  python3 scripts/curate_budget40.py
fi

echo "==> Architecture pipeline config=$CONFIG rounds=$ROUNDS"
python3 scripts/run_architecture_pipeline.py \
  --config "$CONFIG" \
  --rounds "$ROUNDS" \
  --seed "$SEED" \
  "$@"

echo "==> Architecture complete."
echo "    outputs/architecture_40q/budget_table.md"
echo "    outputs/architecture_40q/budget_table_post_qlora.md"
