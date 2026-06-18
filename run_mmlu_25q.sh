#!/bin/bash

# MMLU Benchmark - 25 questions
# GPU 1 (H100)

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

echo "=========================================="
echo "MMLU BENCHMARK - 25 QUESTIONS"
echo "=========================================="
echo ""
echo "Model: Qwen/Qwen2.5-3B-Instruct"
echo "Device: cuda:1 (GPU 1 - H100)"
echo "Batch size: 4"
echo "Traces per question: 4×4 = 16"
echo ""
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

python3 scripts/run_all_benchmarks.py \
    --subset-size 25 \
    --benchmarks mmlu \
    --device cuda:1 \
    --batch-size 4 \
    --output-dir outputs/mmlu_25q

echo ""
echo "=========================================="
echo "MMLU test completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results location: outputs/mmlu_25q/"
echo "=========================================="

# Show quick summary
echo ""
echo "Quick Summary:"
LATEST_CSV=$(find outputs/mmlu_25q -name "*.csv" -type f | head -1)
if [ -f "$LATEST_CSV" ]; then
    python3 << 'EOF'
import csv
import sys

csv_file = sys.argv[1] if len(sys.argv) > 1 else None
if not csv_file:
    sys.exit(1)

with open(csv_file) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    
total = len(rows)
greedy_correct = sum(1 for r in rows if r['correct_greedy'] == '1')
cot_correct = sum(1 for r in rows if r['correct_cot'] == '1')
qubo_correct = sum(1 for r in rows if r['correct_qubo'] == '1')

print(f"Total questions: {total}")
print(f"Greedy accuracy: {greedy_correct}/{total} ({100*greedy_correct/total:.1f}%)")
print(f"CoT accuracy:    {cot_correct}/{total} ({100*cot_correct/total:.1f}%)")
print(f"QUBO accuracy:   {qubo_correct}/{total} ({100*qubo_correct/total:.1f}%)")
print(f"QUBO gain over Greedy: {100*(qubo_correct-greedy_correct)/total:+.1f}%")
EOF
    python3 -c "$(cat)" "$LATEST_CSV"
fi
