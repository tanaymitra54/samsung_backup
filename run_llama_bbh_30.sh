#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python scripts/run_all_benchmarks.py \
    --benchmarks bbh \
    --subset-size 30 \
    --batch-size 8 \
    --seed 777 \
    --output-dir outputs/llama_tests
