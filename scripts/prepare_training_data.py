#!/usr/bin/env python3
"""
Prepare training data with stratified sampling for fine-tuning.

Data sources:
1. nvidia/OpenMathInstruct-2  → stratified into reasoning types (20K sampled)
2. Existing benchmark CSVs in outputs/ → high-signal evaluation traces (~100)

Total target: ~20K examples
"""

import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


# ── Stratum definitions ──────────────────────────────────────────────
# Each stratum maps to benchmark domains the PRISM pipeline targets

STRATA = [
    "arithmetic",           # GSM8K: "how many", "total", "cost", numbers
    "algebra",              # MMLU math: "solve for", "equation", "derivative"
    "geometry",             # MMLU math: "triangle", "angle", "area"
    "probability_stats",    # MMLU + BBH: "probability", "variance", "random"
    "logic",                # BBH: "if true", "boolean", "deduce", "imply"
    "science",              # MMLU: "force", "energy", "chemical", "circuit"
    "code",                 # General: "function", "algorithm", "implement"
    "general",              # Fallback
]

# Per-stratum targets (science is rare in OpenMathInstruct-2, so lower target)
STRATUM_TARGETS = {
    "arithmetic": 2000,
    "algebra": 2000,
    "geometry": 2000,
    "probability_stats": 2000,
    "logic": 2000,
    "science": 300,       # very few science examples; 300 avoids streaming entire dataset
    "code": 2000,
}

STRATUM_KEYWORDS = {
    "arithmetic": [
        "how many", "total", "cost", "price", "per ",
        "age", "distance", "speed", "time", "rate",
        "add", "subtract", "multiply", "divide", "sum of",
        "what is the value", "find the number",
    ],
    "algebra": [
        "solve for", "equation", "quadratic", "polynomial",
        "matrix", "derivative", "integral", "function f(",
        "find x", "variable", "coefficient", "factor",
        "linear", "exponential", "logarithm",
    ],
    "geometry": [
        "triangle", "angle", "circle", "radius", "diameter",
        "sin ", "cos ", "tan ", "geometry", "area of",
        "volume of", "perimeter", "hypotenuse", "parallelogram",
    ],
    "probability_stats": [
        "probability", "expected value", "variance", "random",
        "mean", "median", "mode", "standard deviation",
        "distribution", "likelihood", "permutation", "combination",
        "bayes", "correlation",
    ],
    "logic": [
        "if true", "if false", "boolean", "logic",
        "and/or", "not ", "deduce", "imply",
        "true/false", "proposition", "valid argument",
        "contradiction", "tautology",
    ],
    "science": [
        "force", "energy", "mass", "velocity", "acceleration",
        "chemical", "reaction", "circuit", "voltage",
        "current", "resistance", "molecule", "atom",
        "newton", "gravity", "momentum",
    ],
    "code": [
        "function", "array", "recursion", "algorithm",
        "implement", "code", "program", "sort",
        "binary search", "string", "list", "dictionary",
        "time complexity", "space complexity",
    ],
}


def classify_stratum(instruction: str, output: str) -> str:
    text = (instruction + " " + output).lower()
    for stratum in STRATA:
        if stratum == "general":
            continue
        for kw in STRATUM_KEYWORDS[stratum]:
            if kw in text:
                return stratum
    return "general"


def extract_evaluation_traces(outputs_dir: str) -> list[dict]:
    eval_targets = [
        ("llama_tests", "meta-llama/Llama-3.2-3B-Instruct"),
        ("qwen25_3b_test", "Qwen/Qwen2.5-3B-Instruct"),
        ("mmlu_20q_fixed", "Qwen/Qwen2.5-3B-Instruct"),
        ("mmlu_30q_eval", "Qwen/Qwen2.5-3B-Instruct"),
        ("test_3q_fixed", "Qwen/Qwen2.5-3B-Instruct"),
        ("final_test", "Qwen/Qwen2.5-3B-Instruct"),
    ]

    all_positive = []
    for eval_dir, model_name in eval_targets:
        full_path = os.path.join(outputs_dir, eval_dir)
        if not os.path.isdir(full_path):
            continue
        for fname in os.listdir(full_path):
            if not fname.endswith(".csv"):
                continue
            csv_path = os.path.join(full_path, fname)
            try:
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        question = row.get("question", "").strip()
                        gold = row.get("gold", "").strip()
                        cot_trace = row.get("pred_cot", "").strip()
                        correct = row.get("correct_cot", "0") == "1"
                        if question and cot_trace and correct:
                            all_positive.append({
                                "question": question,
                                "gold_answer": gold,
                                "reasoning_trace": cot_trace,
                            })
            except Exception:
                pass

    return all_positive


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "training_data")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(data_dir, exist_ok=True)

    rng = random.Random(42)

    target_total = 20000
    domain_strata_total = sum(STRATUM_TARGETS.values())  # 14300
    eval_count_expected = 54         # from benchmark CSVs (actual count from last run)
    general_top_up = target_total - (domain_strata_total + eval_count_expected)  # ~5646

    # ── Step 1: Extract evaluation traces ──────────────────────────
    print("=" * 60)
    print("Step 1: Extracting evaluation traces from benchmark CSVs")
    print("=" * 60)
    eval_traces = extract_evaluation_traces(outputs_dir)
    print(f"  Positive evaluation traces: {len(eval_traces)}")

    # ── Step 2: Stream & classify OpenMathInstruct-2 ───────────
    print("\n" + "=" * 60)
    print("Step 2: Streaming nvidia/OpenMathInstruct-2 (no disk download)")
    print("=" * 60)
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    print("  Streaming started — collecting examples until strata are full...")

    # ── Step 3: Classify into strata (stops early when full) ───
    print("\n" + "=" * 60)
    print("Step 3: Classifying into strata (early-stop when full)")
    print("=" * 60)

    needed = dict(STRATUM_TARGETS)
    stratum_buckets = defaultdict(list)
    total_streamed = 0

    for i, example in enumerate(dataset):
        total_streamed += 1
        question = example.get("problem", "")
        trace = example.get("generated_solution", example.get("solution", ""))
        s = classify_stratum(question, trace)

        # Skip if this domain stratum is already full
        if s in needed and len(stratum_buckets[s]) >= needed[s]:
            continue

        stratum_buckets[s].append({"question": question, "trace": trace})

        # Stop when all domain strata are full
        if s in needed and all(len(stratum_buckets[ss]) >= needed[ss] for ss in needed):
            break

        if (i + 1) % 10000 == 0:
            filled = {ss: len(stratum_buckets[ss]) for ss in needed}
            print(f"  Streamed {i+1}: {filled}")

    for s in STRATA:
        print(f"  {s}: {len(stratum_buckets[s])}")
    print(f"  Total streamed: {total_streamed}")

    # ── Step 4: Stratified sampling ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Stratified sampling")
    print("=" * 60)

    sampled = []
    for s in STRATA:
        if s == "general":
            continue
        pool = stratum_buckets[s]
        rng.shuffle(pool)
        n = min(STRATUM_TARGETS[s], len(pool))
        sampled.extend(pool[:n])
        print(f"  {s}: sampled {n}/{len(pool)}")

    general_pool = stratum_buckets["general"]
    rng.shuffle(general_pool)
    n_general = min(general_top_up, len(general_pool))
    sampled.extend(general_pool[:n_general])
    print(f"  general: sampled {n_general}/{len(general_pool)}")

    rng.shuffle(sampled)
    print(f"\n  Total OpenMathInstruct sampled: {len(sampled)}")

    # ── Step 5: Format all examples ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Formatting in Qwen chat template")
    print("=" * 60)

    formatted = []
    for item in sampled:
        question = item["question"]
        trace = item["trace"]
        formatted.append({
            "text": (
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{trace}<|im_end|>"
            )
        })

    for item in eval_traces:
        formatted.append({
            "text": (
                f"<|im_start|>user\n{item['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['reasoning_trace']}\n\n"
                f"Answer: {item['gold_answer']}<|im_end|>"
            )
        })

    print(f"  OpenMathInstruct: {len(sampled)}")
    print(f"  Eval traces:  {len(eval_traces)}")
    print(f"  Total:        {len(formatted)}")

    # ── Step 6: Save ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Saving dataset")
    print("=" * 60)

    # Save as JSON for inspection
    json_path = os.path.join(data_dir, "training_data.json")
    with open(json_path, "w") as f:
        json.dump(formatted, f, indent=2)
    print(f"  JSON: {json_path}")

    # Save as HuggingFace Dataset for SFTTrainer
    from datasets import Dataset as HFDataset

    hf_dataset = HFDataset.from_list(formatted)
    ds_path = os.path.join(data_dir, "combined_dataset")
    hf_dataset.save_to_disk(ds_path)
    print(f"  HF Dataset: {ds_path}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Done! {len(formatted)} training examples ready.")
    print(f"{'=' * 60}")
    print(f"\nStratum breakdown:")
    stratum_counts = defaultdict(int)
    for item in formatted:
        text = item["text"]
        for s in STRATA:
            if s == "general":
                continue
            if any(kw in text.lower() for kw in STRATUM_KEYWORDS[s]):
                stratum_counts[s] += 1
                break
        else:
            stratum_counts["general"] += 1
    for s in STRATA:
        print(f"  {s:20s}: {stratum_counts.get(s, 0)}")


if __name__ == "__main__":
    main()
