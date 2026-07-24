"""
scripts/generate_training_data.py
==================================
Generate a QUBO-curated fine-tuning dataset from multiple reasoning sources.

For each question in datasets 1-4 (GSM8K, StrategyQA, ARC-Challenge, LogiQA),
runs the full QUBO pipeline and saves the best selected reasoning chains in
Llama chat format. Dataset 5 (OpenOrca) is included directly as regularization
without QUBO processing.

Output:
    data/finetune_train.jsonl        -- 90% training split
    data/finetune_val.jsonl          -- 10% validation split
    data/generation_stats.json       -- per-dataset statistics

Hardware target: H100 80GB -- runs model in bfloat16, no 4-bit quantization.
Expected runtime: ~4-6 hours for all 6000 examples.

Usage:
    python scripts/generate_training_data.py [--config CONFIG] [--qubo-params PARAMS]
                                              [--output-dir DIR] [--resume] [--device DEVICE]
"""

import argparse
import gc
import json
import math
import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_WARNING"] = "1"
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Make project root importable ───────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.sampling import DiverseSampler
from pipeline.verifier import ReasonVerifier
from pipeline.qubo_builder import QUBOBuilder
from pipeline.solver import SimulatedAnnealingSolver

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path": "gsm8k",
        "hf_name": "main",
        "split": "train",
        "n": 2000,
        "task_type": "math",
        "target_frac": 0.33,
    },
    "strategyqa": {
        "hf_path": "wics/strategy-qa",
        "hf_name": None,
        "split": "train",
        "n": 800,
        "task_type": "commonsense",
        "target_frac": 0.13,
    },
    "arc": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split": "train",
        "n": 800,
        "task_type": "commonsense",
        "target_frac": 0.13,
    },
    "logiqa": {
        "hf_path": "lucasmccabe/logiqa",
        "hf_name": None,
        "split": "train",
        "n": 600,
        "task_type": "commonsense",
        "target_frac": 0.10,
    },
}

OPENORCA_CONFIG = {
    "hf_path": "Open-Orca/OpenOrca",
    "split": "train",
    "n": 1800,
    "target_frac": 0.30,
}

TOTAL_TARGET = 6000
TRAIN_FRAC = 0.90
SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config(config_path: str, qubo_params_path: str) -> dict:
    """Load config.yaml and override QUBO fields with best_qubo_params in-memory."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(qubo_params_path, encoding="utf-8") as f:
        best = yaml.safe_load(f)

    # Override QUBO section with tuned hyperparameters
    qubo_overrides = {
        "penalty_weight":    best.get("penalty_weight",    config["qubo"]["penalty_weight"]),
        "diversity_bonus":   best.get("diversity_bonus",   config["qubo"]["diversity_bonus"]),
        "cardinality_penalty": best.get("cardinality_penalty", config["qubo"]["cardinality_penalty"]),
        "answer_agree_weight": best.get("answer_agree_weight", config["qubo"]["answer_agree_weight"]),
        "answer_sim_weight":   best.get("answer_sim_weight",   config["qubo"]["answer_sim_weight"]),
    }
    config["qubo"].update(qubo_overrides)

    # H100 settings: bfloat16, no 4-bit
    config["model"]["load_in_4bit"] = False
    # Increase token budget for richer chains on H100
    config["pipeline"]["sampling_max_new_tokens"] = min(
        512, config["pipeline"].get("max_new_tokens", 512)
    )

    print("\n[Config] QUBO overrides applied:")
    for k, v in qubo_overrides.items():
        print(f"  qubo.{k} = {v}")
    print(f"  model.load_in_4bit = False (bfloat16 on H100)")

    return config


def load_model_bfloat16(config: dict, device: str) -> tuple:
    """Load model in bfloat16 on a single CUDA device."""
    model_cfg = config["model"]
    model_name = model_cfg["name"]

    print(f"\n[Model] Loading {model_name} in bfloat16 ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_cfg.get("cache_dir"),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = device.startswith("cuda")
    if use_cuda:
        device_index = 0 if device == "cuda" else int(device.split(":")[1])
        torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats()
        target_device = f"cuda:{device_index}"
    else:
        target_device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_cfg.get("cache_dir"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if use_cuda:
        model = model.to(target_device)
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9

        param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(
            f"[Model] Loaded on {target_device}. "
            f"allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, "
            f"peak={peak:.2f} GB, params={param_gb:.2f} GB"
        )
    else:
        print("[Model] Loaded on CPU.")

    model.eval()
    return model, tokenizer


def save_temp_config(config: dict, tmp_path: str) -> str:
    """Write in-memory config to a temp file so pipeline classes can read it."""
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return tmp_path


# ── Dataset loaders ─────────────────────────────────────────────────────────

def load_gsm8k(n: int) -> list:
    from datasets import load_dataset
    try:
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        rows = []
        for row in ds:
            rows.append(row)
            if len(rows) >= max(n * 5, 1000):
                break
    except Exception as e:
        print(f"[GSM8K] Streaming failed ({e}), trying standard load...")
        ds = load_dataset("gsm8k", "main", split="train")
        rows = list(ds)

    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:n]
    out = []
    for row in rows:
        gold_raw = row["answer"]
        # Gold is after "####" in the answer field
        if "####" in gold_raw:
            gold = gold_raw.split("####")[-1].strip()
        else:
            gold = gold_raw.strip()
        out.append({
            "question": row["question"],
            "gold": gold,
            "task_type": "math",
            "source": "gsm8k",
            "prompt_question": row["question"],
        })
    print(f"[GSM8K] Loaded {len(out)} questions")
    return out


def load_strategyqa(n: int) -> list:
    from datasets import load_dataset
    # voidful/StrategyQA is a parquet-backed mirror. Using streaming=True avoids
    # disk-space pre-checks (download_and_prepare) on restricted container mounts.
    try:
        ds = load_dataset("voidful/StrategyQA", split="train", streaming=True)
        rows = []
        for row in ds:
            rows.append(row)
            if len(rows) >= max(n * 5, 500):
                break
    except Exception as e:
        print(f"[StrategyQA] Streaming load failed ({e}), trying standard load...")
        ds = load_dataset("voidful/StrategyQA", split="train")
        rows = list(ds)

    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:n]
    out = []
    for row in rows:
        gold_bool = row["answer"]  # True/False
        gold = "yes" if gold_bool else "no"
        # Include facts (or decomposition steps) as context in the prompt
        facts = row.get("facts") or row.get("decomposition") or []
        if facts:
            facts_str = "\n".join(f"- {f}" for f in facts)
            prompt_q = f"{row['question']}\n\nContext:\n{facts_str}"
        else:
            prompt_q = row["question"]
        out.append({
            "question": row["question"],
            "gold": gold,
            "task_type": "commonsense",
            "source": "strategyqa",
            "prompt_question": prompt_q,
        })
    print(f"[StrategyQA] Loaded {len(out)} questions")
    return out


def load_arc(n: int) -> list:
    from datasets import load_dataset
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True)
        rows = []
        for row in ds:
            rows.append(row)
            if len(rows) >= max(n * 5, 500):
                break
    except Exception as e:
        print(f"[ARC-Challenge] Streaming failed ({e}), trying standard load...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        rows = list(ds)

    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:n]
    out = []
    for row in rows:
        gold = row["answerKey"]  # A/B/C/D
        choices = row["choices"]
        options_str = "  ".join(
            f"{label}) {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        prompt_q = f"{row['question']}\n\nOptions: {options_str}"
        out.append({
            "question": row["question"],
            "gold": gold,
            "task_type": "commonsense",
            "source": "arc",
            "prompt_question": prompt_q,
            "options_str": options_str,
        })
    print(f"[ARC-Challenge] Loaded {len(out)} questions")
    return out


def load_logiqa(n: int) -> list:
    from datasets import load_dataset
    try:
        ds = load_dataset("lucasmccabe/logiqa", split="train", streaming=True, trust_remote_code=True)
        rows = []
        for row in ds:
            rows.append(row)
            if len(rows) >= max(n * 5, 500):
                break
    except Exception as e:
        print(f"[LogiQA] Streaming failed ({e}), trying standard load...")
        ds = load_dataset("lucasmccabe/logiqa", split="train", trust_remote_code=True)
        rows = list(ds)

    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:n]
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    out = []
    for row in rows:
        correct_idx = row["correct_option"]  # 0-indexed int
        gold = idx_to_letter.get(correct_idx, "A")
        options = row.get("options", [])
        if options:
            options_str = "  ".join(
                f"{idx_to_letter[i]}) {opt}"
                for i, opt in enumerate(options)
                if i < 4
            )
        else:
            options_str = ""
        context = row.get("context", "")
        base_q = row["query"] if "query" in row else row.get("question", "")
        if context:
            full_q = f"Context: {context}\n\n{base_q}"
        else:
            full_q = base_q
        prompt_q = f"{full_q}\n\nOptions: {options_str}" if options_str else full_q
        out.append({
            "question": base_q,
            "gold": gold,
            "task_type": "commonsense",
            "source": "logiqa",
            "prompt_question": prompt_q,
            "options_str": options_str,
        })
    print(f"[LogiQA] Loaded {len(out)} questions")
    return out


def load_openorca(n: int) -> list:
    """Load OpenOrca rows where system_prompt mentions 'reasoning' or 'step'."""
    from datasets import load_dataset
    print("[OpenOrca] Streaming dataset (filtering for reasoning prompts)...")
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    rows = []
    for row in ds:
        sp = (row.get("system_prompt") or "").lower()
        if "reasoning" in sp or "step" in sp:
            rows.append(row)
        if len(rows) >= n * 5:  # collect a buffer to allow shuffling
            break
    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:n]
    out = []
    for row in rows:
        out.append({
            "system_prompt": row.get("system_prompt", ""),
            "question": row.get("question", ""),
            "response": row.get("response", ""),
            "source": "openorca",
        })
    print(f"[OpenOrca] Loaded {len(out)} reasoning examples")
    return out


# ── Prompt formatters ────────────────────────────────────────────────────────

def make_finetuning_example(item: dict, chain: dict) -> dict:
    """Format a QUBO-selected chain as a Llama chat fine-tuning example."""
    source = item["source"]
    reason = chain.get("reason", "").strip()
    answer = chain.get("answer", "").strip()
    correctness = chain.get("correctness_score", 0.0)
    consensus = chain.get("consensus_score", 0.0)

    if source == "gsm8k":
        user_content = f"Solve the following problem step by step.\n\nQuestion: {item['question']}"
    else:
        prompt_q = item.get("prompt_question", item["question"])
        user_content = f"Answer the following question with step-by-step reasoning.\n\nQuestion: {prompt_q}"

    assistant_content = f"{reason}\n\nAnswer: {answer}" if reason else f"Answer: {answer}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": source,
            "gold": item.get("gold", ""),
            "correctness_score": round(correctness, 4),
            "consensus_score": round(consensus, 4),
            "qubo_selected": True,
        },
    }


def make_openorca_example(item: dict) -> dict:
    """Format an OpenOrca row as a Llama chat fine-tuning example."""
    messages = []
    if item.get("system_prompt"):
        messages.append({"role": "system", "content": item["system_prompt"]})
    messages.append({"role": "user", "content": item["question"]})
    messages.append({"role": "assistant", "content": item["response"]})
    return {
        "messages": messages,
        "metadata": {
            "source": "openorca",
            "qubo_selected": False,
        },
    }


# ── Gold-match check for math filter ────────────────────────────────────────

def _extract_last_number(text: str):
    """Extract the last number from text (for math quality filter)."""
    import re
    if not text:
        return None
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _gold_match(pred, gold) -> bool:
    """Hybrid tolerance match for math answers."""
    if pred is None or gold is None:
        return False
    tol = max(0.01, 0.01 * abs(gold))
    return abs(pred - gold) <= tol


# ── Core QUBO pipeline per question ─────────────────────────────────────────

def run_qubo_pipeline(
    item: dict,
    sampler: DiverseSampler,
    verifier: ReasonVerifier,
    qubo_builder: QUBOBuilder,
    solver: SimulatedAnnealingSolver,
) -> tuple:
    """
    Run the full QUBO pipeline for a single question.

    Returns (selected_chains, stats_dict) where selected_chains is empty
    if the quality filter rejects the example.
    """
    question = item["prompt_question"]
    task_type = item["task_type"]
    gold = item.get("gold", "")

    stats = {
        "n_chains_generated": 0,
        "n_chains_selected": 0,
        "filtered": False,
        "filter_reason": "",
        "top_score": 0.0,
    }

    # 1. Sample chains using DiverseSampler (respects pipeline.num_answers from config.yaml)
    try:
        chains = sampler.sample(question, task_type=task_type)
    except Exception as e:
        stats["filtered"] = True
        stats["filter_reason"] = f"sampling_error: {e}"
        return [], stats

    stats["n_chains_generated"] = len(chains)
    if not chains:
        stats["filtered"] = True
        stats["filter_reason"] = "no_chains_generated"
        return [], stats

    # 2. Score chains -- verifier computes correctness_score and consensus_score
    #    score_batch() already blends consensus in via (1-gamma)*base + gamma*consensus
    try:
        chains = verifier.score_batch(chains, task_type=task_type, gold=gold, question=question)
    except Exception as e:
        stats["filtered"] = True
        stats["filter_reason"] = f"scoring_error: {e}"
        return [], stats

    # 3. Build QUBO matrix
    try:
        Q, selected_indices = qubo_builder.build_qubo(chains)
    except Exception as e:
        stats["filtered"] = True
        stats["filter_reason"] = f"qubo_error: {e}"
        return [], stats

    if Q is None or len(selected_indices) == 0:
        stats["filtered"] = True
        stats["filter_reason"] = "empty_qubo"
        return [], stats

    # 4. Solve QUBO with Simulated Annealing
    try:
        state, _ = solver.solve(Q)
    except Exception as e:
        stats["filtered"] = True
        stats["filter_reason"] = f"solver_error: {e}"
        return [], stats

    # 5. Map active state bits to chain indices
    active_local = [i for i, bit in enumerate(state) if bit == 1]
    if not active_local:
        stats["filtered"] = True
        stats["filter_reason"] = "no_bits_active"
        return [], stats

    final_indices = [selected_indices[i] for i in active_local if i < len(selected_indices)]
    selected_chains = [chains[idx] for idx in final_indices if idx < len(chains)]

    if not selected_chains:
        stats["filtered"] = True
        stats["filter_reason"] = "index_out_of_range"
        return [], stats

    # 6. Rank by correctness_score descending, take top 3
    selected_chains.sort(key=lambda c: c.get("correctness_score", 0.0), reverse=True)
    selected_chains = selected_chains[:3]

    top_score = selected_chains[0].get("correctness_score", 0.0)
    stats["n_chains_selected"] = len(selected_chains)
    stats["top_score"] = top_score

    # 7. Quality filter: score threshold
    if top_score < 0.4:
        stats["filtered"] = True
        stats["filter_reason"] = f"low_score:{top_score:.3f}"
        return [], stats

    # 7b. Math tasks: top chain answer must match gold
    if task_type == "math":
        top_chain = selected_chains[0]
        pred_num = _extract_last_number(top_chain.get("answer", "") or top_chain.get("reason", ""))
        gold_num = _extract_last_number(gold)
        if not _gold_match(pred_num, gold_num):
            stats["filtered"] = True
            stats["filter_reason"] = f"math_mismatch:pred={pred_num},gold={gold_num}"
            return [], stats

    return selected_chains, stats


# ── Dataset processing ───────────────────────────────────────────────────────

def process_dataset(
    dataset_name: str,
    items: list,
    sampler: DiverseSampler,
    verifier: ReasonVerifier,
    qubo_builder: QUBOBuilder,
    solver: SimulatedAnnealingSolver,
    cache_path: Path,
    resume: bool,
) -> tuple:
    """
    Process one QUBO dataset.
    Returns (examples_list, aggregate_stats).
    Saves intermediate results to cache_path after each question.
    """
    # Resume: load existing cache
    examples = []
    processed_count = 0
    if resume and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        processed_count = len(examples)
        print(f"[{dataset_name}] Resuming from {processed_count} cached examples")

    # Skip items already processed (cache is in order)
    items_to_process = items[processed_count:]

    agg_stats = {
        "generated": processed_count,
        "filtered": 0,
        "scores": [ex["metadata"]["correctness_score"] for ex in examples],
        "filter_reasons": {},
    }

    if not items_to_process:
        print(f"[{dataset_name}] All {processed_count} questions already cached.")
        return examples, agg_stats

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_f = open(cache_path, "a", encoding="utf-8")

    desc = f"{dataset_name} ({len(items_to_process)} remaining)"
    try:
        for item in tqdm(items_to_process, desc=desc, unit="q"):
            selected_chains, stats = run_qubo_pipeline(
                item, sampler, verifier, qubo_builder, solver
            )
            agg_stats["generated"] += 1

            if stats["filtered"] or not selected_chains:
                agg_stats["filtered"] += 1
                reason = stats.get("filter_reason", "unknown")
                agg_stats["filter_reasons"][reason] = agg_stats["filter_reasons"].get(reason, 0) + 1
                continue

            # Use the best selected chain (already ranked by correctness_score)
            ex = make_finetuning_example(item, selected_chains[0])
            examples.append(ex)
            agg_stats["scores"].append(stats["top_score"])

            cache_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            cache_f.flush()
    finally:
        cache_f.close()

    return examples, agg_stats


# ── Stratified sampling ──────────────────────────────────────────────────────

def stratified_sample(
    all_examples: dict,
    target_total: int,
    target_fracs: dict,
) -> list:
    """
    Sample proportionally from each source to reach target_total.
    If a source has fewer examples than its target, redistributes to others.
    """
    random.seed(SEED)

    targets = {k: int(round(target_total * frac)) for k, frac in target_fracs.items()}
    available = {k: len(v) for k, v in all_examples.items()}
    shortfall = 0
    actual_targets = {}

    for k, t in targets.items():
        if available.get(k, 0) < t:
            actual_targets[k] = available.get(k, 0)
            shortfall += t - actual_targets[k]
        else:
            actual_targets[k] = t

    # Redistribute shortfall to surplus sources
    surplus_sources = [k for k in targets if available.get(k, 0) > actual_targets.get(k, 0)]
    if shortfall > 0 and surplus_sources:
        surplus_total = sum(available[k] - actual_targets[k] for k in surplus_sources)
        if surplus_total > 0:
            for k in surplus_sources:
                extra = int(round(shortfall * (available[k] - actual_targets[k]) / surplus_total))
                actual_targets[k] = min(available[k], actual_targets[k] + extra)

    final = []
    for k, t in actual_targets.items():
        pool = all_examples.get(k, [])
        sampled = random.sample(pool, min(t, len(pool)))
        final.extend(sampled)
        print(f"  {k:<15}: target={targets[k]}, available={available.get(k,0)}, sampled={len(sampled)}")

    random.shuffle(final)
    return final


def stratified_split(examples: list, train_frac: float = 0.90) -> tuple:
    """90/10 train/val split preserving source proportions."""
    by_source = {}
    for ex in examples:
        src = ex.get("metadata", {}).get("source", "unknown")
        by_source.setdefault(src, []).append(ex)

    train, val = [], []
    random.seed(SEED + 1)
    for src, pool in by_source.items():
        random.shuffle(pool)
        n_train = max(1, int(len(pool) * train_frac))
        train.extend(pool[:n_train])
        val.extend(pool[n_train:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


# ── Output helpers ────────────────────────────────────────────────────────────

def write_jsonl(path: Path, examples: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples)} examples -> {path}")


def print_summary_table(all_stats: dict, final_train: list, final_val: list):
    print("\n" + "=" * 65)
    print(f"{'Dataset':<15} | {'Generated':>9} | {'Filtered':>8} | {'Final':>6} | {'Avg Score':>9}")
    print("-" * 65)

    total_gen = 0
    total_filt = 0
    scores_all = []

    for name, s in all_stats.items():
        gen = s.get("generated", 0)
        filt = s.get("filtered", 0)
        final_n = gen - filt
        scores = s.get("scores", [])
        avg_score = f"{sum(scores)/len(scores):.3f}" if scores else "N/A"
        print(f"{name:<15} | {gen:>9} | {filt:>8} | {final_n:>6} | {avg_score:>9}")
        total_gen += gen
        total_filt += filt
        scores_all.extend(scores)

    avg_all = f"{sum(scores_all)/len(scores_all):.3f}" if scores_all else "N/A"
    print("-" * 65)
    print(f"{'TOTAL':<15} | {total_gen:>9} | {total_filt:>8} | {'':>6} | {avg_all:>9}")
    print("=" * 65)
    print(f"\nFinal training set : {len(final_train)} examples")
    print(f"Final validation set: {len(final_val)} examples")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate QUBO-curated fine-tuning dataset")
    parser.add_argument("--config",      default="config/config.yaml",           help="Path to config.yaml")
    parser.add_argument("--qubo-params", default="config/best_qubo_params.yaml", help="Path to best_qubo_params.yaml")
    parser.add_argument("--output-dir",  default="data",                          help="Output directory")
    parser.add_argument("--resume",      action="store_true",                      help="Resume from cached intermediate files")
    parser.add_argument("--device",      default=None,                             help="Device override (cuda/cpu)")
    # ── Fast-run size overrides ───────────────────────────────────────────────
    parser.add_argument("--quick",       action="store_true",
                        help="Use reduced dataset sizes for a fast run "
                             "(gsm8k=100, arc=60, logiqa=40, strategyqa=40, openorca=200). "
                             "Individual --n-* flags override this.")
    parser.add_argument("--n-gsm8k",       type=int, default=None, help="# GSM8K questions  (default: 2000)")
    parser.add_argument("--n-arc",         type=int, default=None, help="# ARC-Challenge questions (default: 800)")
    parser.add_argument("--n-logiqa",      type=int, default=None, help="# LogiQA questions  (default: 600)")
    parser.add_argument("--n-strategyqa",  type=int, default=None, help="# StrategyQA questions (default: 800)")
    parser.add_argument("--n-openorca",    type=int, default=None, help="# OpenOrca examples  (default: 1800)")
    args = parser.parse_args()

    # ── Apply --quick defaults, then let explicit --n-* override ─────────────
    QUICK_DEFAULTS = {
        "gsm8k": 100, "arc": 60, "logiqa": 40, "strategyqa": 40, "openorca": 200
    }
    if args.quick:
        for ds, val in QUICK_DEFAULTS.items():
            if getattr(args, f"n_{ds}", None) is None:
                setattr(args, f"n_{ds}", val)
        print("[Quick] Using reduced dataset sizes for fast run:")
        for ds in QUICK_DEFAULTS:
            print(f"  {ds}: {getattr(args, f'n_{ds}')}")

    # Apply overrides to DATASET_CONFIGS and OPENORCA_CONFIG in-place
    _ds_map = {"gsm8k": "gsm8k", "arc": "arc", "logiqa": "logiqa", "strategyqa": "strategyqa"}
    for arg_name, cfg_key in _ds_map.items():
        override_n = getattr(args, f"n_{arg_name}", None)
        if override_n is not None:
            DATASET_CONFIGS[cfg_key]["n"] = override_n
    if args.n_openorca is not None:
        OPENORCA_CONFIG["n"] = args.n_openorca

    # Recompute TOTAL_TARGET from actual dataset sizes
    global TOTAL_TARGET
    TOTAL_TARGET = sum(cfg["n"] for cfg in DATASET_CONFIGS.values()) + OPENORCA_CONFIG["n"]
    print(f"[Dataset] Target total examples: {TOTAL_TARGET}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and patch config in-memory (no disk writes to config.yaml) ─────
    config = load_config(args.config, args.qubo_params)
    device = args.device or config.get("evaluation", {}).get("device", "cuda:0")

    # Write patched config to temp file so pipeline classes can read it
    tmp_config = str(cache_dir / "_tmp_config.yaml")
    save_temp_config(config, tmp_config)

    # ── Load model once and share across pipeline components ─────────────────
    shared_model, shared_tokenizer = load_model_bfloat16(config, device)

    # ── Initialise pipeline components ───────────────────────────────────────
    print("\n[Init] Initialising pipeline components ...")
    sampler = DiverseSampler(
        config_path=tmp_config,
        device=device,
        shared_model=shared_model,
        shared_tokenizer=shared_tokenizer,
    )

    verifier     = ReasonVerifier(config_path=tmp_config, device=device)
    qubo_builder = QUBOBuilder(config_path=tmp_config, device=device)
    solver       = SimulatedAnnealingSolver(config_path=tmp_config)
    print("[Init] All pipeline components ready.")

    # ── Process QUBO datasets (1-4) ──────────────────────────────────────────
    all_examples = {}
    all_stats = {}

    dataset_loaders = {
        "gsm8k":      lambda: load_gsm8k(DATASET_CONFIGS["gsm8k"]["n"]),
        "strategyqa": lambda: load_strategyqa(DATASET_CONFIGS["strategyqa"]["n"]),
        "arc":        lambda: load_arc(DATASET_CONFIGS["arc"]["n"]),
        "logiqa":     lambda: load_logiqa(DATASET_CONFIGS["logiqa"]["n"]),
    }

    for ds_name, loader in dataset_loaders.items():
        cache_path = cache_dir / f"{ds_name}_raw.jsonl"

        # If fully cached and --resume, skip generation
        if args.resume and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                cached = [json.loads(l) for l in f if l.strip()]
            target_n = DATASET_CONFIGS[ds_name]["n"]
            # Consider "fully cached" if we have at least 70% of target
            if len(cached) >= target_n * 0.7:
                print(f"[{ds_name}] Fully cached ({len(cached)} examples). Skipping generation.")
                all_examples[ds_name] = cached
                all_stats[ds_name] = {
                    "generated": len(cached),
                    "filtered": 0,
                    "scores": [ex["metadata"]["correctness_score"] for ex in cached],
                    "filter_reasons": {},
                }
                continue

        print(f"\n{'='*55}")
        print(f" Processing: {ds_name.upper()}")
        print(f"{'='*55}")
        items = loader()

        examples, stats = process_dataset(
            ds_name, items, sampler, verifier, qubo_builder, solver,
            cache_path=cache_path,
            resume=args.resume,
        )
        all_examples[ds_name] = examples
        all_stats[ds_name] = stats

        n_final = stats["generated"] - stats["filtered"]
        avg_score = (sum(stats["scores"]) / len(stats["scores"])) if stats["scores"] else 0.0
        print(f"[{ds_name}] Done: {n_final}/{stats['generated']} passed filter. Avg score: {avg_score:.3f}")
        if stats.get("filter_reasons"):
            print(f"  Filter breakdown: {stats['filter_reasons']}")

    # ── Process OpenOrca (dataset 5, no QUBO) ────────────────────────────────
    print(f"\n{'='*55}")
    print(" Processing: OPENORCA (regularization, no QUBO)")
    print(f"{'='*55}")
    orca_cache = cache_dir / "openorca_raw.jsonl"

    if args.resume and orca_cache.exists():
        with open(orca_cache, encoding="utf-8") as f:
            orca_examples = [json.loads(l) for l in f if l.strip()]
        print(f"[OpenOrca] Loaded {len(orca_examples)} examples from cache")
    else:
        orca_items = load_openorca(OPENORCA_CONFIG["n"])
        orca_examples = [make_openorca_example(item) for item in orca_items]
        with open(orca_cache, "w", encoding="utf-8") as f:
            for ex in orca_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[OpenOrca] Saved {len(orca_examples)} examples to cache")

    all_examples["openorca"] = orca_examples
    all_stats["openorca"] = {
        "generated": len(orca_examples),
        "filtered": 0,
        "scores": [],
        "filter_reasons": {},
    }

    # ── Stratified sampling ───────────────────────────────────────────────────
    print(f"\n[Sampling] Applying stratified sampling (target: {TOTAL_TARGET} total) ...")
    target_fracs = {
        "gsm8k":      DATASET_CONFIGS["gsm8k"]["target_frac"],
        "strategyqa": DATASET_CONFIGS["strategyqa"]["target_frac"],
        "arc":        DATASET_CONFIGS["arc"]["target_frac"],
        "logiqa":     DATASET_CONFIGS["logiqa"]["target_frac"],
        "openorca":   OPENORCA_CONFIG["target_frac"],
    }
    final_pool = stratified_sample(all_examples, TOTAL_TARGET, target_fracs)

    # ── Stratified train/val split ────────────────────────────────────────────
    print(f"\n[Split] Stratified 90/10 train/val split ...")
    final_train, final_val = stratified_split(final_pool, train_frac=TRAIN_FRAC)

    # ── Write output files ────────────────────────────────────────────────────
    print(f"\n[Output] Writing files ...")
    write_jsonl(output_dir / "finetune_train.jsonl", final_train)
    write_jsonl(output_dir / "finetune_val.jsonl", final_val)

    # generation_stats.json
    stats_out = {}
    for ds_name, s in all_stats.items():
        scores = s.get("scores", [])
        stats_out[ds_name] = {
            "generated": s["generated"],
            "filtered": s.get("filtered", 0),
            "final": s["generated"] - s.get("filtered", 0),
            "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
            "min_score": round(min(scores), 4) if scores else None,
            "max_score": round(max(scores), 4) if scores else None,
            "filter_reasons": s.get("filter_reasons", {}),
        }
    stats_out["split"] = {
        "train": len(final_train),
        "val": len(final_val),
        "total": len(final_train) + len(final_val),
    }

    stats_path = output_dir / "generation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print(f"  Wrote stats -> {stats_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary_table(all_stats, final_train, final_val)

    print("\n[Done] Dataset generation complete.")
    print(f"  Train: {output_dir / 'finetune_train.jsonl'}")
    print(f"  Val  : {output_dir / 'finetune_val.jsonl'}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
