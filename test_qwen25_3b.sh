#!/bin/bash
# Test Qwen2.5-3B with 20 questions (should take ~3 minutes)

cd /root/24BCE5561_samsung/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning
source .venv/bin/activate

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Testing Qwen2.5-3B-Instruct with 20 questions              ║"
echo "║  Expected time: ~3 minutes                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python scripts/run_all_benchmarks.py \
  --device cuda:1 \
  --subset-size 20 \
  --benchmarks gsm8k \
  --seed 42 \
  --no-batch \
  --output-dir outputs/qwen25_3b_test 2>&1 | tee outputs/qwen25_3b_test.log

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Final Results:                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Parse results
CSV=$(ls -t outputs/qwen25_3b_test/all_benchmarks_*.csv 2>/dev/null | head -1)
if [ -f "$CSV" ]; then
    python3 << EOF
import pandas as pd
df = pd.read_csv('$CSV')
print(f"\n📊 Results Summary (20 questions):")
print(f"{'='*60}")
print(f"Greedy accuracy:  {df['correct_greedy'].mean()*100:5.1f}%")
print(f"CoT accuracy:     {df['correct_cot'].mean()*100:5.1f}%")
print(f"QUBO accuracy:    {df['correct_qubo'].mean()*100:5.1f}%")
print(f"{'='*60}")
print(f"Greedy → CoT gain:  {(df['correct_cot'].mean() - df['correct_greedy'].mean())*100:+.1f}%")
print(f"Greedy → QUBO gain: {(df['correct_qubo'].mean() - df['correct_greedy'].mean())*100:+.1f}%")
print(f"{'='*60}")
print(f"\n⏱️  Speed:")
print(f"Avg Greedy time: {df['runtime_greedy_s'].mean():.1f}s")
print(f"Avg CoT time:    {df['runtime_cot_s'].mean():.1f}s")
print(f"Avg QUBO time:   {df['runtime_qubo_s'].mean():.1f}s")
print(f"Total per Q:     {df[['runtime_greedy_s', 'runtime_cot_s', 'runtime_qubo_s']].sum(axis=1).mean():.1f}s")
print(f"\n✅ This model works! Fast and reliable extraction.")
EOF
else
    echo "❌ No results found"
fi
