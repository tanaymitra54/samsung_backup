#!/bin/bash
#
# Comprehensive Benchmark Script for Qwen3.5-4B QUBO Pipeline
# This script runs all benchmarks and generates comparative analysis
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  QUBO Pipeline Benchmark Suite${NC}"
echo -e "${BLUE}  Model: Qwen/Qwen3.5-4B${NC}"
echo -e "${BLUE}  Device: GPU 1 (H100)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate virtual environment
echo -e "${YELLOW}[1/5]${NC} Activating virtual environment..."
source .venv/bin/activate || {
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    exit 1
}

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/qwen35_4b_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓${NC} Output directory: $OUTPUT_DIR"
echo ""

# Check GPU availability
echo -e "${YELLOW}[2/5]${NC} Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader | while IFS=, read -r idx name free total; do
    echo "  GPU $idx: $name"
    echo "    Free memory: $free / $total"
done
echo ""

# Run verification
echo -e "${YELLOW}[3/5]${NC} Running pre-flight checks..."
python verify_setup.py > "$OUTPUT_DIR/verification_log.txt" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All checks passed"
else
    echo -e "${RED}✗${NC} Verification failed. Check $OUTPUT_DIR/verification_log.txt"
    exit 1
fi
echo ""

# Benchmark execution modes
MODE="${1:-full}"  # Default to full if no argument

case "$MODE" in
    "quick")
        SUBSET_SIZE=10
        BENCHMARKS="gsm8k mmlu"
        echo -e "${YELLOW}[4/5]${NC} Running QUICK test (10 samples, 2 benchmarks)..."
        ;;
    "medium")
        SUBSET_SIZE=50
        BENCHMARKS="gsm8k mmlu arc_challenge bbh"
        echo -e "${YELLOW}[4/5]${NC} Running MEDIUM test (50 samples, 4 benchmarks)..."
        ;;
    "full")
        SUBSET_SIZE=200
        BENCHMARKS="gsm8k mmlu arc_challenge bbh"
        echo -e "${YELLOW}[4/5]${NC} Running FULL benchmark (200 samples, 4 benchmarks)..."
        ;;
    *)
        echo -e "${RED}Error: Unknown mode '$MODE'${NC}"
        echo "Usage: $0 [quick|medium|full]"
        exit 1
        ;;
esac

echo "  Subset size: $SUBSET_SIZE"
echo "  Benchmarks: $BENCHMARKS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Start timing
START_TIME=$(date +%s)

# Run benchmarks
python scripts/run_all_benchmarks.py \
    --device cuda:1 \
    --subset-size $SUBSET_SIZE \
    --benchmarks $BENCHMARKS \
    --seed 42 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/benchmark_run.log"

BENCHMARK_EXIT=$?

# End timing
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${YELLOW}[5/5]${NC} Post-processing results..."

if [ $BENCHMARK_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Benchmark completed successfully"
    
    # Find the most recent output files in the output directory
    CSV_FILE=$(ls -t "$OUTPUT_DIR"/*.csv 2>/dev/null | head -1)
    JSON_FILE=$(ls -t "$OUTPUT_DIR"/*.json 2>/dev/null | head -1)
    MD_FILE=$(ls -t "$OUTPUT_DIR"/*.md 2>/dev/null | head -1)
    
    # Create summary
    cat > "$OUTPUT_DIR/SUMMARY.txt" << EOF
========================================
QUBO Pipeline Benchmark Summary
========================================

Run Configuration:
------------------
Model: Qwen/Qwen3.5-4B
Device: GPU 1 (H100 PCIe)
Mode: $MODE
Subset Size: $SUBSET_SIZE
Benchmarks: $BENCHMARKS
Seed: 42

Timing:
-------
Start: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')
End: $(date -d @$END_TIME '+%Y-%m-%d %H:%M:%S')
Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s

Output Files:
-------------
EOF
    
    if [ -n "$CSV_FILE" ]; then
        echo "CSV (per-question): $(basename "$CSV_FILE")" >> "$OUTPUT_DIR/SUMMARY.txt"
    fi
    if [ -n "$JSON_FILE" ]; then
        echo "JSON (summary): $(basename "$JSON_FILE")" >> "$OUTPUT_DIR/SUMMARY.txt"
    fi
    if [ -n "$MD_FILE" ]; then
        echo "Markdown (report): $(basename "$MD_FILE")" >> "$OUTPUT_DIR/SUMMARY.txt"
    fi
    
    echo "" >> "$OUTPUT_DIR/SUMMARY.txt"
    
    # Extract accuracy from JSON if available
    if [ -n "$JSON_FILE" ]; then
        echo "Results Preview:" >> "$OUTPUT_DIR/SUMMARY.txt"
        echo "----------------" >> "$OUTPUT_DIR/SUMMARY.txt"
        python -c "
import json
with open('$JSON_FILE') as f:
    data = json.load(f)
for bench, results in data.get('benchmarks', {}).items():
    if 'accuracy' in results:
        acc = results['accuracy']
        print(f'{bench}:')
        print(f'  Greedy: {acc.get(\"greedy\", 0):.2%}')
        print(f'  CoT:    {acc.get(\"cot\", 0):.2%}')
        print(f'  QUBO:   {acc.get(\"qubo\", 0):.2%}')
        print(f'  Gain:   {results.get(\"abs_gain_vs_greedy\", 0):+.2%}')
        print()
" >> "$OUTPUT_DIR/SUMMARY.txt" 2>/dev/null || echo "  (Could not parse JSON)" >> "$OUTPUT_DIR/SUMMARY.txt"
    fi
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Benchmark Complete!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "Results saved to:"
    echo "  $OUTPUT_DIR/"
    echo ""
    
    if [ -n "$MD_FILE" ]; then
        echo "Quick summary (from markdown report):"
        echo "--------------------------------------"
        cat "$MD_FILE"
    fi
    
else
    echo -e "${RED}✗${NC} Benchmark failed with exit code $BENCHMARK_EXIT"
    echo "Check the log: $OUTPUT_DIR/benchmark_run.log"
    exit $BENCHMARK_EXIT
fi

echo ""
echo -e "${GREEN}All done!${NC}"
echo ""
