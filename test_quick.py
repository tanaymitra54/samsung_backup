#!/usr/bin/env python3
"""
Quick test script with verbose progress - tests 1 question only
"""
import os
import sys

# Set GPU before torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("="*70)
print("QUICK TEST - 1 Question")
print("="*70)
print()

print("[1/8] Importing dependencies...")
sys.stdout.flush()
import torch
import yaml
from pathlib import Path

print("[2/8] Loading config...")
sys.stdout.flush()
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

print(f"  Model: {config['model']['name']}")
print(f"  Device: cuda:0 (GPU 1)")
print()

print("[3/8] Importing pipeline modules...")
sys.stdout.flush()
from pipeline.inference import InferencePipeline
from pipeline.sampling import DiverseSampler
from pipeline.verifier import ReasonVerifier
from pipeline.qubo_builder import QUBOBuilder
from pipeline.solver import SimulatedAnnealingSolver

print("[4/8] Loading model...")
sys.stdout.flush()
inference = InferencePipeline(device="cuda:0")
print(f"  ✓ Model loaded on {inference.device}")
print()

print("[5/8] Loading sampler (sharing model)...")
sys.stdout.flush()
sampler = DiverseSampler(
    device="cuda:0",
    shared_model=inference.model,
    shared_tokenizer=inference.tokenizer,
)
print("  ✓ Sampler ready")

print("[6/8] Loading verifier, QUBO builder, solver...")
sys.stdout.flush()
verifier = ReasonVerifier(device="cuda:0")
qubo_builder = QUBOBuilder(device="cuda:0")
solver = SimulatedAnnealingSolver(device="cuda:0")
print("  ✓ All components loaded")
print()

# Test question
question = "If a store has 15 apples and sells 6, how many apples are left?"
gold = "9"

print(f"[7/8] Testing with question: '{question}'")
print()

# Greedy
print("  → Testing Greedy baseline...", end='', flush=True)
import time
t0 = time.time()
prompt = f"Question: {question}\nAnswer:"
pred_greedy = inference.generate_answer(prompt)
t1 = time.time()
print(f" done ({t1-t0:.1f}s)")
print(f"    Answer: {pred_greedy[:100]}")
print()

# CoT
print("  → Testing CoT baseline...", end='', flush=True)
t0 = time.time()
prompt = f"Let's think step by step.\nQuestion: {question}\nAnswer:"
pred_cot = inference.generate_answer(prompt)
t1 = time.time()
print(f" done ({t1-t0:.1f}s)")
print(f"    Answer: {pred_cot[:100]}")
print()

# QUBO
print("  → Testing QUBO pipeline...")
t0 = time.time()

print("    [1/5] Sampling...")
sys.stdout.flush()
samples = sampler.sample(question)
print(f"       ✓ Generated {len(samples)} samples")

print("    [2/5] Verifying...")
sys.stdout.flush()
samples = verifier.score_batch(samples, task_type="math", gold=gold)
avg_score = sum(s['correctness_score'] for s in samples) / len(samples)
print(f"       ✓ Average correctness: {avg_score:.2f}")

print("    [3/5] Building QUBO...")
sys.stdout.flush()
Q, qubo_var_indices = qubo_builder.build_qubo(samples)
print(f"       ✓ QUBO matrix: {Q.shape[0]}x{Q.shape[1]}")

print("    [4/5] Solving...")
sys.stdout.flush()
state, energy = solver.solve(Q)
print(f"       ✓ Energy: {energy:.2f}")

selected_indices = [qubo_var_indices[i] for i in range(len(state)) if state[i] == 1]
if not selected_indices:
    selected_indices = list(range(min(6, len(samples))))
print(f"       ✓ Selected {len(selected_indices)} traces")

print("    [5/5] Final inference...")
sys.stdout.flush()
pred_qubo = inference.run(question, selected_indices, samples)

t1 = time.time()
print(f"  ✓ QUBO done ({t1-t0:.1f}s)")
print(f"    Answer: {pred_qubo[:100]}")
print()

print("[8/8] Summary:")
print(f"  Greedy: {pred_greedy[:50]}...")
print(f"  CoT:    {pred_cot[:50]}...")
print(f"  QUBO:   {pred_qubo[:50]}...")
print()
print("="*70)
print("✓ TEST PASSED - All components working!")
print("="*70)
