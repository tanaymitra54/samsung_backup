#!/bin/bash
# FINAL WORKING VERSION - Run this for 3 question test
# Should complete in 2 minutes

set -e

cd /root/24BCE5561_samsung/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning
source .venv/bin/activate

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FINAL TEST - 3 Questions (Should work now!)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Disabling chat template... (This was causing the issue!)"
echo "Running with simple prompts..."
echo ""

timeout 180 python scripts/run_all_benchmarks.py \
  --device cuda:1 \
  --subset-size 3 \
  --benchmarks gsm8k \
  --seed 42 \
  --no-batch \
  --output-dir outputs/WORKS_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee outputs/final_run.log

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Results:                                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Show results
CSV=$(ls -t outputs/WORKS_*/all_benchmarks_*.csv 2>/dev/null | head -1)
if [ -f "$CSV" ]; then
    python3 << EOF
import pandas as pd
df = pd.read_csv('$CSV')
print(f"Questions tested: {len(df)}")
print(f"Greedy accuracy: {df['correct_greedy'].mean()*100:.0f}%")
print(f"CoT accuracy: {df['correct_cot'].mean()*100:.0f}%")
print(f"QUBO accuracy: {df['correct_qubo'].mean()*100:.0f}%")
print(f"\nExample predictions (Q1):")
print(f"  Gold: {df.iloc[0]['gold'].split('####')[-1].strip()}")
print(f"  Greedy: {df.iloc[0]['pred_greedy']}")
print(f"  CoT: {df.iloc[0]['pred_cot']}")
print(f"  QUBO: {df.iloc[0]['pred_qubo']}")
EOF
else
    echo "No results found!"
    tail -20 outputs/final_run.log
fi

echo ""
echo "Full log: outputs/final_run.log"
