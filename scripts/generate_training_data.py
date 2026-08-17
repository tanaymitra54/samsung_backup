"""
==============================================================================
FILE: scripts/generate_training_data.py
ROLE: QUBO-Curated Training Dataset Generator
BRANCH ADDITION (abhyuday): Newly created dataset generation engine that processes GSM8K,
StrategyQA, ARC-Challenge, and LogiQA using the QUBO pipeline (sampling -> NLI verification ->
QUBO optimization -> annealing selection) to create high-quality training & validation JSONL data
(`finetune_train.jsonl`, `finetune_val.jsonl`) with complete candidate chain pools.
==============================================================================

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
import re
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
from pipeline.device_utils import resolve_device

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path": "gsm8k",
        "hf_name": "main",
        "split": "train",
        "n": 2000,
        "task_type": "math",
        "target_frac": 0.40,
    },
    # MMLU CAUTION: cais/mmlu has no "train" split, and the evaluation suite
    # scores on split="test". Training on "test" would be direct contamination of
    # the benchmark we report, so this uses "auxiliary_train" -- MMLU's own
    # designated training pool, disjoint from test.
    #
    # n defaults to 0 (see FULL_RUN_DEFAULTS): MMLU is off unless explicitly
    # requested. Raise it only with auxiliary_train, never with test.
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "split": "auxiliary_train",
        "n": 0,
        "task_type": "commonsense",
        "target_frac": 0.25,
    },
    # MATH (Hendrycks) — competition-level multi-step math.
    #
    # WHY: evaluation covers MATH-500 and AIME, but training was GSM8K-only, i.e.
    # grade-school arithmetic used to prepare for competition algebra. This is the
    # largest train/eval capability gap in the suite.
    #
    # nlile/hendrycks-MATH-benchmark ships train=12000 / test=500. Its test split
    # is MATH-500; load_math() additionally drops the handful of train problems
    # that appear in HuggingFaceH4/MATH-500, which is what the evaluator scores.
    "math": {
        "hf_path": "nlile/hendrycks-MATH-benchmark",
        "hf_name": None,
        "split": "train",
        "n": 0,
        "task_type": "math",
        "target_frac": 0.20,
    },
    "arc": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split": "train",
        "n": 800,
        "task_type": "commonsense",
        "target_frac": 0.15,
    },
    "strategyqa": {
        # ChilleD/StrategyQA ships disjoint train/test splits; evaluation uses test.
        "hf_path": "ChilleD/StrategyQA",
        "hf_name": None,
        "split": "train",
        "n": 800,
        "task_type": "commonsense",
        "target_frac": 0.12,
    },
    "logiqa": {
        "hf_path": "lucasmccabe/logiqa",
        "hf_name": None,
        "split": "train",
        "n": 600,
        "task_type": "commonsense",
        "target_frac": 0.08,
    },
}

OPENORCA_CONFIG = {
    "hf_path": "Open-Orca/OpenOrca",
    "split": "train",
    "n": 0,          # Disabled: irrelevant NLP annotation tasks hurt reasoning SFT
    "target_frac": 0.0,
}

# A curated example whose best chain scores below this is treated as "hard" and
# recorded for the next round to revisit. Chosen just under the SFT quality gate
# (0.70) so it captures items that barely made the cut as well as those that missed.
HARD_ITEM_SCORE_THRESHOLD = 0.70

# Keys injected into an item during processing; stripped before it is recorded as
# a hard item so the focus file stays a clean loader-shaped question record.
_RUNTIME_ITEM_KEYS = ("greedy_would_have_failed", "full_chains_pool")


def _strip_runtime_keys(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in _RUNTIME_ITEM_KEYS}


# Sizes used when no --n-* flag and no --quick is given. Kept here rather than as
# argparse defaults so --quick can distinguish "unset" from "explicitly set".
FULL_RUN_DEFAULTS = {
    "gsm8k": 1000,
    "math": 0,        # off by default; raise it to close the competition-math gap
    "mmlu": 0,        # off by default; see the MMLU caution in DATASET_CONFIGS
    "arc": 400,
    "logiqa": 300,
    "strategyqa": 400,
    "openorca": 0,
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

    # A params file marked `stale: true` is provenance only -- applying it would
    # silently overwrite config.yaml with values from a superseded search. The
    # current file is stale because it was tuned with the gold answer visible and
    # over a cardinality_penalty range that can only ever select one chain.
    if best.get("stale", False):
        print(f"\n[Config] {qubo_params_path} is marked stale -- NOT applying it.")
        print("         Reason:", str(best.get("notes", "")).strip().splitlines()[0]
              if best.get("notes") else "superseded search")
        print("         Using the QUBO settings from", config_path)
        print("         Regenerate with: python scripts/tune_qubo_params.py --num-questions 300")
        config["model"]["load_in_4bit"] = False
        config["pipeline"]["sampling_max_new_tokens"] = min(
            512, config["pipeline"].get("max_new_tokens", 512)
        )
        return config

    # Override QUBO section with tuned hyperparameters.
    #
    # Every fallback uses .get() with an explicit default: a config.yaml that
    # predates the answer-aware/cardinality QUBO terms simply omits those keys,
    # and direct subscripting made this function raise KeyError on such a file.
    # The defaults mirror QUBOBuilder's own.
    qubo_cfg = config.setdefault("qubo", {})
    qubo_overrides = {
        "penalty_weight":      best.get("penalty_weight",      qubo_cfg.get("penalty_weight", 2.0)),
        "diversity_bonus":     best.get("diversity_bonus",     qubo_cfg.get("diversity_bonus", 0.5)),
        "cardinality_penalty": best.get("cardinality_penalty", qubo_cfg.get("cardinality_penalty", 0.1)),
        "answer_agree_weight": best.get("answer_agree_weight", qubo_cfg.get("answer_agree_weight", 0.4)),
        "answer_sim_weight":   best.get("answer_sim_weight",   qubo_cfg.get("answer_sim_weight", 0.6)),
    }
    qubo_cfg.update(qubo_overrides)

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


def load_model_bfloat16(config: dict, device: str, adapter_path: str | None = None) -> tuple:
    """Load model in bfloat16 on a single CUDA device.

    When `adapter_path` is given, the LoRA adapter is merged into the base weights
    before the model is returned. This is what closes the architecture's feedback
    loop: round N+1 generates its candidate reasoning paths using the adapter
    trained in round N, rather than always re-sampling from the base SLM.
    """
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

    target_device = resolve_device(device)
    use_cuda = target_device.type == "cuda"
    if use_cuda:
        device_index = target_device.index or 0
        torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats()
    else:
        target_device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_cfg.get("cache_dir"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        print(f"[Model] Merging LoRA adapter: {adapter_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        print("[Model] Adapter merged.")

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


def load_math(n: int) -> list:
    """Hendrycks MATH training problems, disjoint from the MATH-500 eval set.

    NUMERIC FILTER: MATH answers are often symbolic (e.g. "\\frac{\\pi}{2}").
    Training-data curation scores chains against gold via the verifier's numeric
    path, which cannot compare symbolic forms, so problems whose answer is not a
    plain number are skipped here. They would otherwise be scored as wrong
    regardless of the reasoning, poisoning the curated set.
    """
    from datasets import load_dataset

    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=DATASET_CONFIGS["math"]["split"])
    rows = list(ds)

    # Drop any problem that also appears in the evaluator's MATH-500 set.
    def _norm(s: str) -> str:
        return " ".join(str(s).split())[:200]

    try:
        eval_problems = {
            _norm(r["problem"])
            for r in load_dataset("HuggingFaceH4/MATH-500", split="test")
        }
        before = len(rows)
        rows = [r for r in rows if _norm(r["problem"]) not in eval_problems]
        if before != len(rows):
            print(f"[MATH] Dropped {before - len(rows)} problem(s) present in MATH-500.")
    except Exception as e:
        print(f"[MATH] WARNING: could not verify disjointness with MATH-500 ({e}).")

    numeric = re.compile(r"^-?\d+(?:\.\d+)?$")
    kept = [r for r in rows if numeric.fullmatch(str(r.get("answer", "")).strip())]
    print(f"[MATH] {len(kept)}/{len(rows)} problems have plain-numeric answers.")

    random.seed(SEED)
    random.shuffle(kept)
    kept = kept[:n]

    out = []
    for row in kept:
        out.append({
            "question": row["problem"],
            "gold": str(row["answer"]).strip(),
            "task_type": "math",
            "source": "math",
            "prompt_question": row["problem"],
        })
    print(f"[MATH] Loaded {len(out)} questions")
    return out


def load_strategyqa(n: int) -> list:
    """StrategyQA training questions from the split the evaluator does not use.

    Source is ChilleD/StrategyQA (train=1603, test=687, disjoint by construction).
    The evaluator reads the TEST split, so nothing here overlaps it. The previous
    sources are broken under datasets>=3: wics/strategy-qa is a loading script and
    voidful/StrategyQA fails schema validation part-way through generation.
    """
    from datasets import load_dataset

    split = DATASET_CONFIGS["strategyqa"]["split"]
    if split != "train":
        raise ValueError(
            f"StrategyQA training data must come from the 'train' split, got {split!r}. "
            "The evaluation suite scores on 'test'."
        )
    ds = load_dataset("ChilleD/StrategyQA", split=split)
    rows = list(ds)

    # The published splits share exactly one question. Drop it so the training
    # pool is strictly disjoint from what the evaluator scores.
    try:
        eval_questions = {
            str(r["question"]).strip()
            for r in load_dataset("ChilleD/StrategyQA", split="test")
        }
        before = len(rows)
        rows = [r for r in rows if str(r["question"]).strip() not in eval_questions]
        if before != len(rows):
            print(f"[StrategyQA] Dropped {before - len(rows)} question(s) also present in the test split.")
    except Exception as e:
        print(f"[StrategyQA] WARNING: could not verify split disjointness ({e}).")

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

    # ARC's published train and test splits genuinely share a couple of items
    # (7 match on question stem; 2 are true duplicates once options are compared).
    # Drop them so the ARC-Challenge eval stays clean.
    def _ident(r) -> str:
        ch = r.get("choices", {})
        texts = " | ".join(map(str, ch.get("text", []))) if isinstance(ch, dict) else ""
        return " ".join(f"{r['question']} || {texts}".split()).lower()[:200]

    try:
        eval_ids = {
            _ident(r)
            for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        }
        before = len(rows)
        rows = [r for r in rows if _ident(r) not in eval_ids]
        if before != len(rows):
            print(f"[ARC-Challenge] Dropped {before - len(rows)} item(s) also in the test split.")
    except Exception as e:
        print(f"[ARC-Challenge] WARNING: could not verify split disjointness ({e}).")

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


def load_mmlu(n: int) -> list:
    """Load MMLU questions (MCQ format, same structure as ARC).

    Reads the split from DATASET_CONFIGS rather than hardcoding it. That split is
    "auxiliary_train"; the evaluation suite scores on "test", so training here
    must never touch "test" or the reported MMLU number is contaminated.
    """
    from datasets import load_dataset

    split = DATASET_CONFIGS["mmlu"]["split"]
    if split == "test":
        raise ValueError(
            "Refusing to build MMLU training data from the 'test' split: "
            "evaluation scores on that same split. Use 'auxiliary_train'."
        )

    try:
        ds = load_dataset("cais/mmlu", "all", split=split, streaming=True)
        rows = []
        for row in ds:
            rows.append(row)
            if len(rows) >= max(n * 5, 500):
                break
    except Exception as e:
        print(f"[MMLU] Streaming failed ({e}), trying standard load...")
        try:
            ds = load_dataset("cais/mmlu", "all", split=split)
            rows = list(ds)
        except Exception as e2:
            print(f"[MMLU] Standard load also failed ({e2}). Skipping MMLU.")
            return []

    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:n]
    letters = ["A", "B", "C", "D"]
    out = []
    for row in rows:
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)   # 0-indexed int
        if not choices or answer_idx >= len(choices):
            continue
        gold = letters[answer_idx]
        options_str = "  ".join(f"{letters[i]}) {c}" for i, c in enumerate(choices) if i < 4)
        subject = row.get("subject", "")
        subject_str = f" ({subject.replace('_', ' ')})" if subject else ""
        prompt_q = f"{row['question']}{subject_str}\n\nOptions: {options_str}"
        out.append({
            "question": row["question"],
            "gold": gold,
            "task_type": "commonsense",
            "source": "mmlu",
            "prompt_question": prompt_q,
            "options_str": options_str,
        })
    print(f"[MMLU] Loaded {len(out)} questions")
    return out


def load_logiqa(n: int) -> list:
    # lucasmccabe/logiqa ships a logiqa.py loading script that is fully banned
    # in newer datasets versions, and trust_remote_code is also rejected.
    # The datasets library also intercepts https://huggingface.co URLs and
    # rewrites them as hf:// paths with broken URL-encoding, so we bypass it
    # entirely: download the HF auto-parquet with urllib and read it with
    # pyarrow (guaranteed present as a datasets/transformers dependency).
    import io
    import urllib.request
    import pyarrow.parquet as pq

    _LOGIQA_PARQUET = (
        "https://huggingface.co/datasets/lucasmccabe/logiqa"
        "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )
    print("[LogiQA] Fetching auto-parquet from HuggingFace via urllib...")
    req = urllib.request.Request(_LOGIQA_PARQUET,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        buf = io.BytesIO(resp.read())
    table = pq.read_table(buf)
    # Convert PyArrow table to list of row dicts matching HF datasets output
    rows = table.to_pylist()

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
            "greedy_would_have_failed": item.get("greedy_would_have_failed", False),
            "full_chains_pool": item.get("full_chains_pool", []),
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

    # 7. Quality filter: score threshold >= 0.5
    if top_score < 0.5:
        stats["filtered"] = True
        stats["filter_reason"] = f"low_score:{top_score:.3f}"
        return [], stats

    best_pre_qubo = max(chains, key=lambda c: c.get("correctness_score", 0.0))
    best_pre_qubo_score = best_pre_qubo.get("correctness_score", 0.0)
    best_pre_qubo_answer = best_pre_qubo.get("answer", "").strip()
    
    final_answer = selected_chains[0].get("answer", "").strip()
    greedy_would_have_failed = (best_pre_qubo_score < 0.5) or (best_pre_qubo_answer != final_answer)
    stats["greedy_would_have_failed"] = greedy_would_have_failed
    stats["full_chains_pool"] = chains

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
        "greedy_failed_count": sum(1 for ex in examples if ex["metadata"].get("greedy_would_have_failed", False)),
        # Items the pipeline struggled on, in loader-item shape so a later round
        # can be pointed straight back at them (see --focus-file). These come
        # from the TRAIN split only, so re-using them is not contamination.
        "hard_items": [],
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
                # Produced no usable chain at all -- the hardest possible outcome.
                agg_stats["hard_items"].append(_strip_runtime_keys(item))
                continue

            # Inject new stats into the item so make_finetuning_example can read it
            item["greedy_would_have_failed"] = stats.get("greedy_would_have_failed", False)
            item["full_chains_pool"] = stats.get("full_chains_pool", [])

            # Use the best selected chain (already ranked by correctness_score)
            ex = make_finetuning_example(item, selected_chains[0])
            examples.append(ex)
            agg_stats["scores"].append(stats["top_score"])
            if stats.get("greedy_would_have_failed", False):
                agg_stats["greedy_failed_count"] += 1
            # Kept, but the best chain was still weak -- worth revisiting.
            if stats["top_score"] < HARD_ITEM_SCORE_THRESHOLD:
                agg_stats["hard_items"].append(_strip_runtime_keys(item))

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
    total_greedy_failed = 0
    scores_all = []

    for name, s in all_stats.items():
        gen = s.get("generated", 0)
        filt = s.get("filtered", 0)
        greedy_failed = s.get("greedy_failed_count", 0)
        final_n = gen - filt
        scores = s.get("scores", [])
        avg_score = f"{sum(scores)/len(scores):.3f}" if scores else "N/A"
        print(f"{name:<15} | {gen:>9} | {filt:>8} | {final_n:>6} | {avg_score:>9}")
        total_gen += gen
        total_filt += filt
        total_greedy_failed += greedy_failed
        scores_all.extend(scores)

    avg_all = f"{sum(scores_all)/len(scores_all):.3f}" if scores_all else "N/A"
    print("-" * 65)
    print(f"{'TOTAL':<15} | {total_gen:>9} | {total_filt:>8} | {'':>6} | {avg_all:>9}")
    print("=" * 65)
    print(f"\nFinal training set : {len(final_train)} examples")
    print(f"Final validation set: {len(final_val)} examples")
    
    total_final = len(final_train) + len(final_val)
    if total_final > 0:
        print(f"Fraction where Greedy would have failed: {total_greedy_failed / total_final:.2%} ({total_greedy_failed}/{total_final})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate QUBO-curated fine-tuning dataset")
    parser.add_argument("--config",      default="config/config.yaml",           help="Path to config.yaml")
    parser.add_argument("--qubo-params", default="config/best_qubo_params.yaml", help="Path to best_qubo_params.yaml")
    parser.add_argument("--output-dir",  default="data",                          help="Output directory")
    parser.add_argument("--resume",      action="store_true",                      help="Resume from cached intermediate files")
    parser.add_argument("--device",      default=None,                             help="Device override (cuda/cpu)")
    parser.add_argument("--adapter-path", default=None,
                        help="LoRA adapter to merge before sampling (closes the retrain loop). "
                             "Falls back to the QUBO_ADAPTER_PATH env var.")
    parser.add_argument("--focus-file", default=None,
                        help="JSON file of hard questions from a previous round "
                             "(hard_questions.json). They are added to this round's pool "
                             "so curation concentrates on what the model struggles with.")
    parser.add_argument("--focus-repeat", type=int, default=2,
                        help="How many times each focus question is added (default: 2)")
    # ── Fast-run size overrides ───────────────────────────────────────────────
    parser.add_argument("--quick",       action="store_true",
                        help="Use reduced dataset sizes for a fast run "
                             "(gsm8k=300, mmlu=200, arc=150, logiqa=100, strategyqa=100, openorca=0). "
                             "Individual --n-* flags override this.")
    # NOTE: these all default to None on purpose so --quick can tell "unset" from
    # "explicitly requested". FULL_RUN_DEFAULTS supplies the real defaults below.
    parser.add_argument("--n-gsm8k",       type=int, default=None, help="# GSM8K questions  (default: 1000)")
    parser.add_argument("--n-math",        type=int, default=None, help="# MATH (Hendrycks) problems (default: 0)")
    parser.add_argument("--n-mmlu",        type=int, default=None, help="# MMLU questions (default: 0, see MMLU note)")
    parser.add_argument("--n-arc",         type=int, default=None, help="# ARC-Challenge questions (default: 400)")
    parser.add_argument("--n-logiqa",      type=int, default=None, help="# LogiQA questions  (default: 300)")
    parser.add_argument("--n-strategyqa",  type=int, default=None, help="# StrategyQA questions (default: 400)")
    parser.add_argument("--n-openorca",    type=int, default=None, help="# OpenOrca examples  (default: 0, disabled)")
    args = parser.parse_args()

    # ── Apply --quick defaults, then let explicit --n-* override ─────────────
    #
    # The "is None" test below only works because every --n-* argument defaults
    # to None (see parse_args). They previously carried numeric defaults, so the
    # test was never true and --quick silently did nothing while printing that it
    # had -- a "quick" run sampled 2100 questions instead of 850.
    QUICK_DEFAULTS = {
        "gsm8k": 300, "math": 200, "mmlu": 200, "arc": 150,
        "logiqa": 100, "strategyqa": 100, "openorca": 0
    }
    if args.quick:
        for ds, val in QUICK_DEFAULTS.items():
            if getattr(args, f"n_{ds}", None) is None:
                setattr(args, f"n_{ds}", val)
        print("[Quick] Using reduced dataset sizes for fast run:")
        for ds in QUICK_DEFAULTS:
            print(f"  {ds}: {getattr(args, f'n_{ds}')}")

    # Anything still unset falls back to the full-run defaults.
    for ds, full_default in FULL_RUN_DEFAULTS.items():
        if getattr(args, f"n_{ds}", None) is None:
            setattr(args, f"n_{ds}", full_default)

    # Apply overrides to DATASET_CONFIGS and OPENORCA_CONFIG in-place
    _ds_map = {"gsm8k": "gsm8k", "math": "math", "mmlu": "mmlu", "arc": "arc",
               "logiqa": "logiqa", "strategyqa": "strategyqa"}
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
    adapter_path = args.adapter_path or os.environ.get("QUBO_ADAPTER_PATH") or None
    shared_model, shared_tokenizer = load_model_bfloat16(config, device, adapter_path)

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

    # ── Load focus questions from a previous round (optional) ────────────────
    focus_items: dict[str, list] = {}
    if args.focus_file:
        focus_path = Path(args.focus_file)
        if focus_path.exists():
            with open(focus_path, encoding="utf-8") as f:
                for item in json.load(f):
                    focus_items.setdefault(item.get("source", ""), []).append(item)
            total_focus = sum(len(v) for v in focus_items.values())
            print(f"[Focus] Loaded {total_focus} hard questions from {focus_path}: "
                  + ", ".join(f"{k}={len(v)}" for k, v in sorted(focus_items.items())))
        else:
            print(f"[Focus] WARNING: {focus_path} not found; continuing without focus questions.")

    # ── Process QUBO datasets (1-4) ──────────────────────────────────────────
    all_examples = {}
    all_stats = {}

    dataset_loaders = {
        "gsm8k":      lambda: load_gsm8k(DATASET_CONFIGS["gsm8k"]["n"]),
        "math":       lambda: load_math(DATASET_CONFIGS["math"]["n"]),
        "mmlu":       lambda: load_mmlu(DATASET_CONFIGS["mmlu"]["n"]),
        "arc":        lambda: load_arc(DATASET_CONFIGS["arc"]["n"]),
        "strategyqa": lambda: load_strategyqa(DATASET_CONFIGS["strategyqa"]["n"]),
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

        # Failure-targeted curriculum: prepend the questions this dataset
        # struggled on last round so they get fresh attempts from the improved
        # model. focus_items is keyed by source, and only ever holds TRAIN-split
        # questions, so this cannot contaminate the evaluation benchmarks.
        extra = focus_items.get(ds_name, [])
        if extra:
            repeated = extra * max(1, args.focus_repeat)
            items = repeated + items
            print(f"[{ds_name}] Focus: {len(extra)} hard questions x{args.focus_repeat} "
                  f"prepended (pool now {len(items)}).")

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

    # ── Process OpenOrca (disabled by default — hurts reasoning SFT) ─────────
    if OPENORCA_CONFIG["n"] > 0:
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
            "generated": len(orca_examples), "filtered": 0, "scores": [], "filter_reasons": {}
        }
    else:
        print("\n[OpenOrca] Skipped (n=0 — disabled for reasoning-focused SFT).")

    # ── Stratified sampling ───────────────────────────────────────────────────
    print(f"\n[Sampling] Applying stratified sampling (target: {TOTAL_TARGET} total) ...")
    # Build target_fracs dynamically — only include datasets that have data
    target_fracs = {ds: cfg["target_frac"] for ds, cfg in DATASET_CONFIGS.items()
                    if all_examples.get(ds)}
    if all_examples.get("openorca"):
        target_fracs["openorca"] = OPENORCA_CONFIG["target_frac"]
    # Re-normalise so fracs sum to 1.0
    total_frac = sum(target_fracs.values())
    if total_frac > 0:
        target_fracs = {k: v / total_frac for k, v in target_fracs.items()}
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

    # hard_questions.json -- TRAIN-split questions this round struggled on, for a
    # later round to revisit via --focus-file.
    hard_items = []
    for ds_name, s in all_stats.items():
        hard_items.extend(s.get("hard_items", []))
    hard_path = output_dir / "hard_questions.json"
    with open(hard_path, "w", encoding="utf-8") as f:
        json.dump(hard_items, f, indent=2, ensure_ascii=False)
    stats_out["hard_questions"] = len(hard_items)
    print(f"  Wrote {len(hard_items)} hard questions -> {hard_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary_table(all_stats, final_train, final_val)

    print("\n[Done] Dataset generation complete.")
    print(f"  Train: {output_dir / 'finetune_train.jsonl'}")
    print(f"  Val  : {output_dir / 'finetune_val.jsonl'}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
