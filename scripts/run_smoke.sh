#!/usr/bin/env bash
# Real LLM trace-generation smoke; no invented weights or accuracy claim.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
python scripts/evaluate_gsm8k.py --config "${CONFIG:-config/config.yaml}" \
  --role fit --collect-only --questions 2 --seeds 42 \
  --no-multi-gpu --output-dir "${OUT:-outputs/generation_smoke}" "$@"
