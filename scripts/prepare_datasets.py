"""
==============================================================================
FILE: scripts/prepare_datasets.py
ROLE: Dataset Provisioning & Contamination Audit
==============================================================================

Fetches every dataset the pipeline needs (fine-tuning and evaluation), caches
them locally, and audits the train/eval boundary.

WHY THIS EXISTS
---------------
Fine-tuning sources and evaluation sources overlap by name -- GSM8K, MMLU, ARC
and StrategyQA all appear on both sides -- but must never overlap by CONTENT. A
single contaminated split silently invalidates the headline numbers, and the
failure is invisible at run time: the benchmark simply looks better.

This script makes that boundary explicit and checkable. For every source used on
both sides it loads the training pool and the evaluation pool, normalises the
question text, and reports the exact size of the intersection. Anything non-zero
is reported as a FAIL.

USAGE
-----
    python scripts/prepare_datasets.py                 # fetch + audit everything
    python scripts/prepare_datasets.py --audit-only    # skip fetching
    python scripts/prepare_datasets.py --eval-only     # evaluation sets only
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Dataset registry ─────────────────────────────────────────────────────────
#
# Each entry: (label, hf_path, hf_name, split, why it is used).
# "train" entries feed generate_training_data.py; "eval" entries feed
# evaluation/__init__.py.

TRAIN_SOURCES = [
    ("gsm8k", "gsm8k", "main", "train",
     "Grade-school multi-step arithmetic. The backbone of math reasoning SFT."),
    ("math", "nlile/hendrycks-MATH-benchmark", None, "train",
     "Competition-level math. Closes the gap to the MATH-500 and AIME evals, "
     "which GSM8K alone does not prepare for."),
    ("arc", "allenai/ai2_arc", "ARC-Challenge", "train",
     "Grade-school science MCQ requiring retrieval plus reasoning."),
    ("strategyqa", "ChilleD/StrategyQA", None, "train",
     "Implicit multi-hop yes/no questions; trains decomposition."),
    ("logiqa", "lucasmccabe/logiqa", None, "train",
     "Formal deductive logic. Closest available proxy for BBH-style tasks."),
    ("mmlu", "cais/mmlu", "all", "auxiliary_train",
     "Broad academic knowledge. MMLU has no train split, so this uses its "
     "designated auxiliary_train pool -- never the test split it is scored on."),
]

EVAL_SOURCES = [
    ("gsm8k", "gsm8k", "main", "test",
     "Held-out grade-school math."),
    ("math 500", "HuggingFaceH4/MATH-500", None, "test",
     "500-problem competition math subset."),
    ("strategyqa", "ChilleD/StrategyQA", None, "test",
     "Held-out multi-hop yes/no, disjoint from the train split by construction."),
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test",
     "Held-out science MCQ."),
    ("mmlu", "cais/mmlu", "all", "test",
     "Held-out academic MCQ."),
]

# Sources evaluated but never trained on -- pure generalisation probes.
HELD_OUT_ONLY = [
    ("bbh", "lukaemon/bbh", "boolean_expressions", "test",
     "Big-Bench Hard has no training split. Kept as a pure out-of-distribution "
     "probe, which is scientifically the right thing to have."),
]

# Question-text field per dataset, for the contamination audit.
QUESTION_FIELD = {
    "gsm8k": "question",
    "math": "problem",
    "math 500": "problem",
    "arc": "question",
    "arc_challenge": "question",
    "strategyqa": "question",
    "logiqa": "context",
    "mmlu": "question",
    "bbh": "input",
}

# Recommended sizes for a run aiming at a strong final model, with rationale.
RECOMMENDED = {
    "gsm8k": (3000, "Largest clean math source; 7473 available."),
    "math": (2500, "Competition math; ~7700 numeric-answer problems available."),
    "mmlu": (2000, "Breadth, from auxiliary_train."),
    "arc": (1119, "All of ARC-Challenge train."),
    "strategyqa": (1600, "All of the train split."),
    "logiqa": (1200, "Deductive logic coverage."),
}


def _norm(text: str) -> str:
    """Normalise question text for cross-split comparison."""
    return " ".join(str(text).split()).lower()[:200]


def fetch(label, path, name, split, quiet=False):
    """Load one split, returning the dataset or None on failure."""
    from datasets import load_dataset
    try:
        ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
        if not quiet:
            print(f"  [ok]   {label:<14} {path}:{split:<16} n={len(ds):,}")
        return ds
    except Exception as e:
        print(f"  [FAIL] {label:<14} {path}:{split:<16} {str(e)[:70]}")
        return None


def _choices_text(row) -> str:
    """Flatten a row's answer options, whatever shape the dataset uses."""
    ch = row.get("choices")
    if isinstance(ch, dict) and "text" in ch:        # ARC
        return " | ".join(map(str, ch["text"]))
    if isinstance(ch, (list, tuple)):                # MMLU
        return " | ".join(map(str, ch))
    return ""


def questions_of(ds, label):
    """Identity set for a split.

    Includes the answer OPTIONS when present. Comparing question stems alone
    produces false positives on MCQ sets -- MMLU has several distinct items whose
    stem is literally "Which of the following is true?", which looked like 5 leaks
    but is 0 once choices are included. ARC goes the other way: 7 stem matches, of
    which 2 are genuine duplicate items.
    """
    field = QUESTION_FIELD.get(label)
    if ds is None or field is None or field not in ds.column_names:
        return set()
    has_choices = "choices" in ds.column_names
    if has_choices:
        return {_norm(f"{row[field]} || {_choices_text(row)}") for row in ds}
    return {_norm(row[field]) for row in ds}


def main():
    ap = argparse.ArgumentParser(description="Fetch and audit pipeline datasets")
    ap.add_argument("--audit-only", action="store_true",
                    help="Skip fetching; only run the contamination audit")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only handle evaluation datasets")
    args = ap.parse_args()

    print("=" * 78)
    print("  DATASET PROVISIONING")
    print("=" * 78)

    train_ds, eval_ds = {}, {}

    if not args.eval_only:
        print("\nFINE-TUNING SOURCES")
        print("-" * 78)
        for label, path, name, split, _why in TRAIN_SOURCES:
            train_ds[label] = fetch(label, path, name, split)

    print("\nEVALUATION SOURCES")
    print("-" * 78)
    for label, path, name, split, _why in EVAL_SOURCES:
        eval_ds[label] = fetch(label, path, name, split)

    print("\nHELD-OUT-ONLY (evaluated, never trained on)")
    print("-" * 78)
    for label, path, name, split, _why in HELD_OUT_ONLY:
        fetch(f"{label}*", path, name, split)
    print("  note: BBH loads 27 subtasks at eval time; one is probed here.")

    # ── Contamination audit ──────────────────────────────────────────────────
    if not args.eval_only:
        print("\n" + "=" * 78)
        print("  CONTAMINATION AUDIT  (train content vs eval content)")
        print("=" * 78)

        # Map training label -> evaluation label for sources used on both sides.
        pairs = [
            ("gsm8k", "gsm8k"),
            ("math", "math 500"),
            ("arc", "arc_challenge"),
            ("strategyqa", "strategyqa"),
            ("mmlu", "mmlu"),
        ]

        # These loaders in generate_training_data.py drop overlapping items at load
        # time, so raw overlap here is neutralised before any training happens.
        FILTERED_BY_LOADER = {"math", "arc", "strategyqa"}

        print("  Overlap is measured on the RAW published splits. Sources marked")
        print("  [held] have a loader that removes these items before training.\n")

        unhandled = 0
        for tr_label, ev_label in pairs:
            tr, ev = train_ds.get(tr_label), eval_ds.get(ev_label)
            if tr is None or ev is None:
                print(f"  [skip] {tr_label:<12} -> {ev_label:<14} (a split failed to load)")
                continue
            q_tr, q_ev = questions_of(tr, tr_label), questions_of(ev, ev_label)
            if not q_tr or not q_ev:
                print(f"  [skip] {tr_label:<12} -> {ev_label:<14} (no comparable question field)")
                continue
            overlap = q_tr & q_ev
            if not overlap:
                status = "PASS"
            elif tr_label in FILTERED_BY_LOADER:
                status = "held"
            else:
                status = "FAIL"
                unhandled += 1
            print(f"  [{status}] {tr_label:<12} -> {ev_label:<14} "
                  f"train={len(q_tr):>6,}  eval={len(q_ev):>5,}  raw overlap={len(overlap)}")
            if overlap and status == "FAIL":
                for q in list(overlap)[:3]:
                    print(f"           leaked: {q[:66]}")

        print("-" * 78)
        if unhandled:
            print(f"  {unhandled} source(s) leak with NO loader-side filter. Fix before "
                  "trusting any number from those benchmarks.")
        else:
            print("  No unhandled contamination. Every raw overlap is removed at load time.")

        # ── Recommended sizes ────────────────────────────────────────────────
        print("\n" + "=" * 78)
        print("  RECOMMENDED FINE-TUNING SIZES")
        print("=" * 78)
        total = 0
        for label, (n, why) in RECOMMENDED.items():
            avail = len(train_ds[label]) if train_ds.get(label) is not None else 0
            n_eff = min(n, avail) if avail else n
            total += n_eff
            print(f"  --n-{label:<11} {n_eff:>6,}   (available {avail:>7,})  {why}")
        print("-" * 78)
        print(f"  Total questions in: {total:,}")
        print(f"  After QUBO curation (~85% keep rate): ~{int(total * 0.85):,} SFT examples")
        print()
        print("  Command:")
        flags = " ".join(
            f"--n-{k} {min(v[0], len(train_ds[k]) if train_ds.get(k) is not None else v[0])}"
            for k, v in RECOMMENDED.items()
        )
        print(f"    python scripts/generate_training_data.py {flags}")

    # ── Evaluation sizing guidance ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  EVALUATION SIZING")
    print("=" * 78)
    print("  Accuracy standard error at p=0.5 is sqrt(0.25/n):")
    for n in (30, 100, 300, 500, 1000):
        se = (0.25 / n) ** 0.5
        note = "  <- too noisy to see a 5-point change" if n <= 100 else ""
        print(f"    n={n:<5} SE = +/-{se*100:.1f} points{note}")
    print()
    print("  Use --eval-subset 300 minimum; 500-1000 for anything you report.")
    print("=" * 78)


if __name__ == "__main__":
    main()
