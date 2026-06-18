#!/bin/bash
# Real-time benchmark monitor

OUTPUT_DIR="outputs/qwen35_4b_20260617_025215"
LOG_FILE="$OUTPUT_DIR/benchmark_run.log"
CSV_FILE="$OUTPUT_DIR/all_benchmarks_20260617_025231.csv"

clear
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║               BENCHMARK PROGRESS MONITOR                               ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if benchmark is running
BENCH_PID=$(ps aux | grep "[p]ython scripts/run_all_benchmarks.py" | awk '{print $2}')

if [ -z "$BENCH_PID" ]; then
    echo "❌ No benchmark process found!"
    echo ""
    echo "Recent completions:"
    ls -lth outputs/ | head -5
    exit 0
fi

# Get runtime
RUNTIME=$(ps -p $BENCH_PID -o etime --no-headers | tr -d ' ')

echo "✅ Benchmark is RUNNING"
echo "   PID: $BENCH_PID"
echo "   Runtime: $RUNTIME"
echo ""

# GPU status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | while IFS=, read -r idx name util mem_used mem_total; do
    echo "GPU $idx ($name)"
    echo "  Utilization: $util"
    echo "  Memory: $mem_used / $mem_total"
done
echo ""

# Progress from CSV
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Progress:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$CSV_FILE" ]; then
    TOTAL_LINES=$(wc -l < "$CSV_FILE")
    COMPLETED=$((TOTAL_LINES - 1))  # Subtract header
    
    if [ $COMPLETED -gt 0 ]; then
        echo "✅ Questions completed: $COMPLETED / 20"
        echo ""
        
        # Show current accuracy if we have data
        if [ $COMPLETED -ge 2 ]; then
            echo "Running accuracy (not final):"
            tail -n +2 "$CSV_FILE" | awk -F',' '
                BEGIN {g=0; c=0; q=0; t=0}
                {
                    if ($8 == "1") g++
                    if ($9 == "1") c++
                    if ($10 == "1") q++
                    t++
                }
                END {
                    if (t > 0) {
                        printf "  Greedy: %.1f%%  CoT: %.1f%%  QUBO: %.1f%%\n", 
                               (g/t)*100, (c/t)*100, (q/t)*100
                    }
                }
            '
        fi
    else
        echo "⏳ Model loading / initialization phase..."
        echo "   This can take 1-2 minutes for first question"
    fi
else
    echo "⏳ Waiting for CSV file to be created..."
    echo "   Model is being loaded..."
fi

echo ""

# Recent log entries
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Recent log (last 5 lines):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$LOG_FILE" ]; then
    tail -5 "$LOG_FILE" | sed 's/^/  /'
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "To view full log: tail -f $LOG_FILE"
echo "To stop monitoring: Ctrl+C"
echo "To stop benchmark: kill $BENCH_PID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
