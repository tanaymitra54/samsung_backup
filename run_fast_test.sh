#!/bin/bash
# Ultra-fast test configuration: 2 traces, 32 tokens, 10 questions
# Expected: ~10s per question = 100s total

cd /root/24BCE5561_samsung/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning
source .venv/bin/activate

echo "========================================================================"
echo "  FAST TEST - 10 Questions (~2 minutes total)"
echo "========================================================================"
echo "  Configuration: 2 traces × 32 tokens = FAST"
echo "  Benchmarks: GSM8K only"
echo "  Questions: 10"
echo "========================================================================"
echo ""

# Use fast config
cp config/config_fast.yaml config/config.yaml.bak
cp config/config_fast.yaml config/config.yaml

python scripts/run_all_benchmarks.py \
  --device cuda:1 \
  --subset-size 10 \
  --benchmarks gsm8k \
  --seed 42 \
  --no-batch \
  --output-dir outputs/fast_test_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee outputs/fast_test.log

EXIT_CODE=$?

# Restore original config if backup exists
if [ -f config/config.yaml.bak ]; then
    mv config/config.yaml.bak config/config.yaml
fi

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ FAST TEST COMPLETED"
    echo ""
    # Show results
    LATEST_CSV=$(ls -t outputs/fast_test_*/all_benchmarks_*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST_CSV" ]; then
        echo "Results:"
        python3 << EOF
import pandas as pd
df = pd.read_csv('$LATEST_CSV')
print(f"  Questions: {len(df)}")
print(f"  Greedy accuracy: {df['correct_greedy'].mean()*100:.1f}%")
print(f"  CoT accuracy: {df['correct_cot'].mean()*100:.1f}%")
print(f"  QUBO accuracy: {df['correct_qubo'].mean()*100:.1f}%")
print(f"  Avg time per question: {df[['runtime_greedy_s', 'runtime_cot_s', 'runtime_qubo_s']].sum(axis=1).mean():.1f}s")
EOF
    fi
else
    echo "✗ TEST FAILED"
fi
echo "========================================================================"

exit $EXIT_CODE
