#!/bin/bash

# Quick benchmark test with 5 questions to verify fixes
# This script shows REAL-TIME progress

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

echo "=========================================="
echo "QUICK BENCHMARK TEST - 5 QUESTIONS"
echo "=========================================="
echo ""
echo "Configuration:"
grep "model:\|name:" config/config.yaml | head -2
echo "Device: cuda:1 (GPU 1 - H100)"
echo "Sampling tokens: 256"
echo "Traces per question: 4×4 = 16"
echo ""
echo "Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Run with subset_size=5 for quick test
python3 scripts/run_all_benchmarks.py \
    --subset-size 5 \
    --benchmarks gsm8k \
    --device cuda:1 \
    --output-dir outputs/quick_test_5q

echo ""
echo "=========================================="
echo "Test completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved to: outputs/quick_test_5q/"
echo "=========================================="
