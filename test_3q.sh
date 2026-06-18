#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "====== QUICK 3Q TEST - Chat Template + Better Prompts ======"
echo "Started: $(date '+%H:%M:%S')"

python3 scripts/run_all_benchmarks.py \
    --subset-size 3 \
    --benchmarks gsm8k \
    --device cuda:1 \
    --output-dir outputs/test_3q_fixed

echo ""
echo "====== TEST COMPLETE ======"
echo "Check outputs/test_3q_fixed/ for results"
