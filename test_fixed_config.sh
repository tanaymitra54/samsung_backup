#!/bin/bash
# Test the fixed configuration with 10 questions

cd /root/24BCE5561_samsung/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning

export CUDA_VISIBLE_DEVICES=1

echo "Testing Qwen2.5-3B with Tanay's fixes applied..."
echo "Config: 2 answers × 3 reasons = 6 traces, 256 tokens, diversity_bonus=0.5"
echo ""

python scripts/run_all_benchmarks.py \
  --benchmarks gsm8k \
  --subset-size 10 \
  --device cuda:1 \
  --output-dir outputs/fixed_config_test \
  --seed 42

echo ""
echo "Results saved to outputs/fixed_config_test/"
