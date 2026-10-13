#!/usr/bin/env python3
"""Curate a fixed 40-question eval panel and a disjoint 40-question train panel.

8 questions each from GSM8K, BBH, StrategyQA, ARC-Challenge, and MMLU.
Writes data/budget40/{eval,train}_manifest.jsonl. Does not touch prepare_training_data.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
PER_BENCH = 8
OUT_DIR = Path("data/budget40")

MMLU_SUBJECTS = [
    "abstract_algebra",
    "college_computer_science",
    "college_physics",
    "electrical_engineering",
    "machine_learning",
]

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "logical_deduction_three_objects",
    "navigate",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "sports_understanding",
    "temporal_sequences",
    "web_of_lies",
]


def _row(
    benchmark: str,
    source_id: str,
    question: str,
    gold: str,
    task_type: str,
    is_mcq: bool,
    split: str,
) -> dict:
    return {
        "benchmark": benchmark,
        "source_id": str(source_id),
        "question": question,
        "gold": gold,
        "task_type": task_type,
        "is_mcq": bool(is_mcq),
        "split": split,
    }


def _format_mcq(question: str, labels: list[str], texts: list[str]) -> str:
    options = [f"{label}. {text}" for label, text in zip(labels, texts)]
    return f"Question: {question}\n" + "\n".join(options) + "\nAnswer:"


def _format_mmlu(question: str, choices: list[str]) -> str:
    return (
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )


def _take_disjoint(pool: list[dict], n: int, rng: random.Random, used: set[str]) -> list[dict]:
    available = [r for r in pool if r["source_id"] not in used]
    rng.shuffle(available)
    chosen = available[:n]
    for r in chosen:
        used.add(r["source_id"])
    if len(chosen) < n:
        raise RuntimeError(
            f"Need {n} rows for {pool[0]['benchmark'] if pool else '?'}, got {len(chosen)}"
        )
    return chosen


def load_gsm8k_pool(split: str) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for i, item in enumerate(ds):
        gold = item["answer"]
        if "####" in gold:
            gold = gold.split("####")[-1].strip()
        rows.append(
            _row("gsm8k", f"gsm8k:{split}:{i}", item["question"], gold, "math", False, split)
        )
    return rows


def load_arc_pool(split: str) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    rows = []
    for i, item in enumerate(ds):
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        q = _format_mcq(item["question"], labels, texts)
        gold = str(item["answerKey"]).strip().upper()
        sid = item.get("id") or f"{split}:{i}"
        rows.append(
            _row("arc_challenge", f"arc:{sid}", q, gold, "commonsense", True, split)
        )
    return rows


def load_mmlu_pool(split: str) -> list[dict]:
    """Build a pool with subjects interleaved so sampling spreads across STEM."""
    by_subject: list[list[dict]] = []
    for subject in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split=split)
        sub_rows = []
        for i, item in enumerate(ds):
            q = _format_mmlu(item["question"], item["choices"])
            gold = ["A", "B", "C", "D"][item["answer"]]
            sub_rows.append(
                _row(
                    "mmlu",
                    f"mmlu:{subject}:{split}:{i}",
                    q,
                    gold,
                    "commonsense",
                    True,
                    split,
                )
            )
        by_subject.append(sub_rows)
    # Round-robin so a shuffle still covers subjects early.
    interleaved: list[dict] = []
    max_len = max(len(s) for s in by_subject)
    for i in range(max_len):
        for sub_rows in by_subject:
            if i < len(sub_rows):
                interleaved.append(sub_rows[i])
    return interleaved


def load_bbh_pool() -> list[dict]:
    """BBH only publishes a test split. One row per task for diversity."""
    rows = []
    for task in BBH_TASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception:
            continue
        for i, item in enumerate(ds):
            rows.append(
                _row(
                    "bbh",
                    f"bbh:{task}:{i}",
                    item["input"],
                    str(item["target"]).strip(),
                    "math",
                    False,
                    "test",
                )
            )
    return rows


def load_strategyqa_pool() -> list[dict]:
    """voidful/StrategyQA only has train. Eval and train panels are disjoint slices."""
    try:
        ds = load_dataset("voidful/StrategyQA", split="train", streaming=True)
        raw = []
        for row in ds:
            raw.append(row)
            if len(raw) >= 2000:
                break
    except Exception:
        ds = load_dataset("voidful/StrategyQA", split="train")
        raw = list(ds)

    rows = []
    for i, row in enumerate(raw):
        gold_bool = row["answer"]
        gold = "yes" if gold_bool else "no"
        rows.append(
            _row(
                "strategyqa",
                f"strategyqa:{i}",
                row["question"],
                gold,
                "commonsense",
                False,
                "train",
            )
        )
    return rows


def sample_bbh(n: int, rng: random.Random, used: set[str]) -> list[dict]:
    """Exactly one question from each of n distinct tasks."""
    by_task: dict[str, list[dict]] = {}
    for row in load_bbh_pool():
        task = row["source_id"].split(":")[1]
        by_task.setdefault(task, []).append(row)
    tasks = [t for t in by_task if any(r["source_id"] not in used for r in by_task[t])]
    rng.shuffle(tasks)
    chosen = []
    for task in tasks:
        if len(chosen) >= n:
            break
        avail = [r for r in by_task[task] if r["source_id"] not in used]
        if not avail:
            continue
        pick = rng.choice(avail)
        used.add(pick["source_id"])
        chosen.append(pick)
    if len(chosen) < n:
        raise RuntimeError(f"BBH needed {n} tasks, got {len(chosen)}")
    return chosen


def sample_mmlu(n: int, rng: random.Random, used: set[str], pool: list[dict]) -> list[dict]:
    """At least one per STEM subject when possible, then fill."""
    by_subject: dict[str, list[dict]] = {}
    for row in pool:
        if row["source_id"] in used:
            continue
        subject = row["source_id"].split(":")[1]
        by_subject.setdefault(subject, []).append(row)
    chosen = []
    subjects = list(MMLU_SUBJECTS)
    rng.shuffle(subjects)
    for subject in subjects:
        if len(chosen) >= n:
            break
        avail = by_subject.get(subject) or []
        if not avail:
            continue
        pick = rng.choice(avail)
        used.add(pick["source_id"])
        chosen.append(pick)
        by_subject[subject] = [r for r in avail if r["source_id"] != pick["source_id"]]
    remaining = [r for r in pool if r["source_id"] not in used]
    rng.shuffle(remaining)
    while len(chosen) < n and remaining:
        pick = remaining.pop()
        used.add(pick["source_id"])
        chosen.append(pick)
    if len(chosen) < n:
        raise RuntimeError(f"MMLU needed {n}, got {len(chosen)}")
    return chosen


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(path: Path) -> None:
    path.write_text(
        """# Budget-40 panel

Fixed panels for the matched-budget outcomes table and the architecture QLoRA loop.

| Manifest | Size | Role |
|---|---:|---|
| `eval_manifest.jsonl` | 40 | Test / held-out graded questions (8 × 5 benchmarks) |
| `train_manifest.jsonl` | 40 | Train-split prompts for QLoRA (disjoint source IDs) |

Seed: 42.

HF sources:
- GSM8K: `openai/gsm8k` (eval=test, train=train)
- ARC-Challenge: `allenai/ai2_arc` / `ARC-Challenge` (eval=test, train=train)
- MMLU: `cais/mmlu` STEM subjects (eval=test, train=validation or test-disjoint)
- BBH: `lukaemon/bbh` (test only; eval and train are disjoint IDs)
- StrategyQA: `voidful/StrategyQA` (train only; eval and train are disjoint IDs)

Regenerate with:

```bash
python scripts/curate_budget40.py
```
""",
        encoding="utf-8",
    )


def build_panels(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    used: set[str] = set()

    gsm8k_test = load_gsm8k_pool("test")
    gsm8k_train = load_gsm8k_pool("train")
    arc_test = load_arc_pool("test")
    arc_train = load_arc_pool("train")
    mmlu_test = load_mmlu_pool("test")
    try:
        mmlu_train = load_mmlu_pool("validation")
    except Exception:
        mmlu_train = load_mmlu_pool("test")
    strategy_pool = load_strategyqa_pool()

    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    # Order matches the budget table top-to-bottom.
    eval_rows.extend(_take_disjoint(gsm8k_test, PER_BENCH, rng, used))
    train_rows.extend(_take_disjoint(gsm8k_train, PER_BENCH, rng, used))

    eval_rows.extend(sample_bbh(PER_BENCH, rng, used))
    train_rows.extend(sample_bbh(PER_BENCH, rng, used))

    # StrategyQA: first take eval slice, then train from remainder.
    eval_rows.extend(_take_disjoint(strategy_pool, PER_BENCH, rng, used))
    train_rows.extend(_take_disjoint(strategy_pool, PER_BENCH, rng, used))

    eval_rows.extend(_take_disjoint(arc_test, PER_BENCH, rng, used))
    train_rows.extend(_take_disjoint(arc_train, PER_BENCH, rng, used))

    eval_rows.extend(sample_mmlu(PER_BENCH, rng, used, mmlu_test))
    train_rows.extend(sample_mmlu(PER_BENCH, rng, used, mmlu_train))

    return eval_rows, train_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate fixed budget-40 manifests")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    eval_rows, train_rows = build_panels(args.seed)
    assert len(eval_rows) == 40, len(eval_rows)
    assert len(train_rows) == 40, len(train_rows)
    eval_ids = {r["source_id"] for r in eval_rows}
    train_ids = {r["source_id"] for r in train_rows}
    assert not (eval_ids & train_ids), "eval/train ID overlap"

    out = args.out_dir
    write_jsonl(out / "eval_manifest.jsonl", eval_rows)
    write_jsonl(out / "train_manifest.jsonl", train_rows)
    write_readme(out / "README.md")

    counts = {}
    for r in eval_rows:
        counts[r["benchmark"]] = counts.get(r["benchmark"], 0) + 1
    print(f"Wrote {out}/eval_manifest.jsonl ({len(eval_rows)}) {counts}")
    print(f"Wrote {out}/train_manifest.jsonl ({len(train_rows)})")


if __name__ == "__main__":
    main()
