#!/usr/bin/env bash
# Canonical fit/tune/calibrate/evaluate workflow, using the configured GPU mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
python scripts/run_diagnostics.py --config "${CONFIG:-config/config.yaml}" \
  --output-dir "${OUT:-outputs/a6000_run}" "$@"
