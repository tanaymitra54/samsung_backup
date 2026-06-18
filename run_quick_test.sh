#!/bin/bash
# Quick test with 5 samples, verbose progress

cd /root/24BCE5561_samsung/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning
source .venv/bin/activate

echo "========================================================================"
echo "  QUICK TEST - 5 Samples with Full Progress Feedback"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Model: Qwen/Qwen3.5-4B"
echo "  - Samples: 2x2=4 traces per question (reduced for speed)"
echo "  - Max tokens: 64 for sampling, 128 for final answer"
echo "  - Benchmarks: GSM8K only"
echo "  - Questions: 5"
echo ""
echo "Starting in 2 seconds..."
sleep 2

python scripts/run_all_benchmarks.py \
  --device cuda:1 \
  --subset-size 5 \
  --benchmarks gsm8k \
  --seed 42 \
  --no-batch \
  --output-dir outputs/quick_test_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee outputs/quick_test.log

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST COMPLETED SUCCESSFULLY"
else
    echo "✗ TEST FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"
echo ""
echo "Log saved to: outputs/quick_test.log"
echo ""

exit $EXIT_CODE
