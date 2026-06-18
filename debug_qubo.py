#!/usr/bin/env python3
"""Debug script to understand why QUBO is getting 0% accuracy."""

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.inference import InferencePipeline
from pipeline.sampling import DiverseSampler
from pipeline.verifier import ReasonVerifier
from pipeline.qubo_builder import QUBOBuilder
from pipeline.solver import SimulatedAnnealingSolver
from evaluation import BenchmarkRunner
from evaluation.answer_utils import extract_predicted_answer, extract_gsm8k_gold

print("="*70)
print("QUBO DEBUG ANALYSIS")
print("="*70)

# Load one GSM8K question
runner = BenchmarkRunner()
questions, golds = runner.load_benchmark("gsm8k")
question = questions[0]  # Janet's ducks
gold = golds[0]
gold_answer = extract_gsm8k_gold(gold)

print(f"\nQuestion: {question}")
print(f"Gold answer: {gold_answer}")
print(f"\n{'='*70}")

# Initialize pipeline
print("Loading models...")
inference = InferencePipeline(device="cuda:0")
sampler = DiverseSampler(device="cuda:0", shared_model=inference.model, shared_tokenizer=inference.tokenizer)
verifier = ReasonVerifier(device="cuda:0")
qubo_builder = QUBOBuilder(device="cuda:0")
solver = SimulatedAnnealingSolver(device="cuda:0")

print("\n" + "="*70)
print("STEP 1: BASELINE PREDICTIONS")
print("="*70)

# Test baselines
prompt_greedy = f"{question}\n\nThe answer is:"
pred_greedy_raw = inference.generate_answer(prompt_greedy)
pred_greedy = extract_predicted_answer(pred_greedy_raw)
print(f"\nGreedy Prompt: {prompt_greedy[:100]}...")
print(f"Greedy Raw Output: {repr(pred_greedy_raw[:200])}")
print(f"Greedy Extracted: {pred_greedy}")
print(f"Greedy Correct: {pred_greedy == gold_answer}")

prompt_cot = f"{question}\n\nLet's solve this step by step. The answer is:"
pred_cot_raw = inference.generate_answer(prompt_cot)
pred_cot = extract_predicted_answer(pred_cot_raw)
print(f"\nCoT Prompt: {prompt_cot[:100]}...")
print(f"CoT Raw Output: {repr(pred_cot_raw[:200])}")
print(f"CoT Extracted: {pred_cot}")
print(f"CoT Correct: {pred_cot == gold_answer}")

print("\n" + "="*70)
print("STEP 2: TRACE SAMPLING")
print("="*70)

# Sample traces
samples = sampler.sample(question)
print(f"\nGenerated {len(samples)} traces")
for i, s in enumerate(samples):
    print(f"\nTrace {i+1}:")
    print(f"  Prompt: {s['prompt_template'][:80]}...")
    print(f"  Reason (first 150 chars): {s['reason'][:150]}")
    print(f"  Answer: {s['answer'][:100] if s['answer'] else '(empty)'}")
    # Try to extract number from reason
    extracted = extract_predicted_answer(s['reason'])
    print(f"  Extracted from reason: {extracted}")

print("\n" + "="*70)
print("STEP 3: VERIFICATION SCORES")
print("="*70)

# Score traces
samples_scored = verifier.score_batch(samples, task_type="math", gold=gold)
for i, s in enumerate(samples_scored):
    print(f"\nTrace {i+1} correctness_score: {s.get('correctness_score', 0.0):.3f}")
    print(f"  Reason: {s['reason'][:100]}...")
    # Check what numbers are in the reason
    extracted_num = extract_predicted_answer(s['reason'])
    print(f"  Extracted number: {extracted_num} (gold is {gold_answer})")

print("\n" + "="*70)
print("STEP 4: QUBO SELECTION")
print("="*70)

# Build QUBO and solve
Q, qubo_var_indices = qubo_builder.build_qubo(samples_scored)
print(f"\nQUBO matrix shape: {Q.shape}")
print(f"QUBO variables mapped to trace indices: {qubo_var_indices}")

state, energy = solver.solve(Q)
print(f"\nQUBO solution state: {state}")
print(f"QUBO solution energy: {energy}")

selected_indices = [qubo_var_indices[i] for i in range(len(state)) if state[i] == 1]
print(f"\nSelected trace indices: {selected_indices}")

if not selected_indices:
    print("WARNING: No traces selected by QUBO! Using fallback...")
    selected_indices = list(range(min(inference.subset_size, len(samples_scored))))
    print(f"Fallback selected indices: {selected_indices}")

print("\nSelected traces:")
for idx in selected_indices:
    print(f"\n  Trace {idx}:")
    print(f"    correctness_score: {samples_scored[idx].get('correctness_score', 0.0):.3f}")
    print(f"    Reason: {samples_scored[idx]['reason'][:150]}...")
    print(f"    Answer: {samples_scored[idx]['answer'][:100] if samples_scored[idx]['answer'] else '(empty)'}")
    extracted_num = extract_predicted_answer(samples_scored[idx]['reason'])
    print(f"    Extracted from reason: {extracted_num}")

print("\n" + "="*70)
print("STEP 5: FINAL ANSWER GENERATION")
print("="*70)

# Generate final answer
selected_reasons = [samples_scored[i]["reason"] for i in selected_indices]
final_prompt = inference.build_final_prompt(question, selected_reasons)
print(f"\nFinal prompt (first 500 chars):")
print(final_prompt[:500])
print("...")
print(f"\n(Full prompt length: {len(final_prompt)} chars)")

pred_qubo_raw = inference.generate_answer(final_prompt)
pred_qubo = extract_predicted_answer(pred_qubo_raw)

print(f"\nQUBO Raw Output: {repr(pred_qubo_raw[:200])}")
print(f"QUBO Extracted: {pred_qubo}")
print(f"QUBO Correct: {pred_qubo == gold_answer}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Gold: {gold_answer}")
print(f"Greedy: {pred_greedy} {'✓' if pred_greedy == gold_answer else '✗'}")
print(f"CoT: {pred_cot} {'✓' if pred_cot == gold_answer else '✗'}")
print(f"QUBO: {pred_qubo} {'✓' if pred_qubo == gold_answer else '✗'}")
print()
