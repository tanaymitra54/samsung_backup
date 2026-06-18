#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "====== MMLU 3Q TEST - Fixed MCQ Extraction ======"
echo "Started: $(date '+%H:%M:%S')"

python3 scripts/run_all_benchmarks.py \
    --subset-size 3 \
    --benchmarks mmlu \
    --device cuda:1 \
    --batch-size 4 \
    --output-dir outputs/mmlu_3q_fixed

echo ""
echo "====== MMLU 3Q TEST COMPLETE ======"
find outputs/mmlu_3q_fixed -name "*.csv" | xargs python3 -c "
import csv, sys
with open(sys.argv[1]) as f:
    r = csv.DictReader(f)
    rows = list(r)
    g = sum(1 for x in rows if x['correct_greedy']=='1')
    c = sum(1 for x in rows if x['correct_cot']=='1')
    q = sum(1 for x in rows if x['correct_qubo']=='1')
    print(f'Total: {len(rows)}')
    print(f'Greedy: {g}/{len(rows)} ({100*g/len(rows):.1f}%)')
    print(f'CoT:    {c}/{len(rows)} ({100*c/len(rows):.1f}%)')
    print(f'QUBO:   {q}/{len(rows)} ({100*q/len(rows):.1f}%)')
"
