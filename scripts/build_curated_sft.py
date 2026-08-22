"""
==============================================================================
FILE: scripts/build_curated_sft.py
ROLE: Curate SFT data three ways, so selection can be tested where it belongs
==============================================================================

THE HYPOTHESIS
--------------
Set selection failed at inference. Measured on MATH-500, a QUBO subset scored
52.3% against 61.0% for weighting the whole pool, and came in BELOW randomly
chosen subsets of the same size at every k. The mechanism was structural rather
than a tuning failure: the objective penalises chains that agree with each
other, so it selects a mutually-disagreeing subset, which is then collapsed by
voting. Selecting for disagreement and then asking for consensus are opposed
operations.

That result says diversity is a liability when producing ONE answer -- there,
concentration of evidence is what aggregation rewards. It says nothing about
curating TRAINING data, where the incentives invert:

  * twenty near-identical correct chains give the same gradient signal twenty
    times over, which is wasted capacity, not reinforcement
  * genuinely different correct derivations teach several valid routes to the
    same answer
  * diversity in instruction-tuning corpora is a well-established lever

So this is the same machinery aimed at the task it actually suits.

WHAT MAKES THE COMPARISON FAIR
------------------------------
Every strategy gets the SAME questions and the SAME number of examples; only
WHICH chains are kept differs. Without that, a win could just mean "more data".
Chains are filtered to CORRECT ones first (legitimate here -- gold at training
time is rejection sampling, not leakage), so no strategy can win merely by
picking more accurate chains. The question is purely: given several correct
derivations, which should the model learn from?

THE CURATION OBJECTIVE IS NOT THE INFERENCE OBJECTIVE
-----------------------------------------------------
The inference QUBO penalised ANSWER AGREEMENT. Here every candidate is already
correct, so they all agree by construction and that term is meaningless. The
off-diagonal instead penalises TEXTUAL redundancy: two chains that reach the
right answer by the same words teach one lesson twice. That is the term that
should have been there all along for this task.

    diagonal      -PRM quality        (learn from well-reasoned chains)
    off-diagonal  +lexical overlap    (avoid teaching the same route twice)

STRATEGIES
----------
    qubo    quality + non-redundancy, jointly optimised
    greedy  top-k by PRM alone -- quality with no diversity pressure
    random  control; isolates whether either signal matters at all

USAGE
-----
    python scripts/build_curated_sft.py \\
        --chains results/cached_chains_math500_300q.json \\
        --prm-cache results/prm_math500_300q.json \\
        --k 3 --out-dir data/curation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Answer handling ──────────────────────────────────────────────────────────

def numeric_match(pred: Optional[float], gold: float) -> bool:
    if pred is None:
        return False
    return abs(pred - gold) < max(0.01, 0.01 * abs(gold))


def last_number(text: str) -> Optional[float]:
    m = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(m[-1]) if m else None


def chain_answer(chain: dict) -> Optional[float]:
    a = last_number(chain.get("answer") or "")
    return a if a is not None else last_number(chain.get("reason") or "")


# ── Lexical redundancy (no embedding model needed) ───────────────────────────
#
# Deliberately model-free so this runs in the minimal PRM venv, and so the
# redundancy notion is transparent rather than hidden in an embedding space.

_TOKEN = re.compile(r"[a-z0-9]+")


def bigrams(text: str) -> set:
    toks = _TOKEN.findall(str(text).lower())
    return set(zip(toks, toks[1:])) if len(toks) > 1 else set()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def set_diversity(chains: list[dict]) -> float:
    """Mean pairwise bigram DISTANCE across a selected set. Higher = more varied."""
    if len(chains) < 2:
        return 0.0
    grams = [bigrams(c.get("reason", "")) for c in chains]
    pairs = [1.0 - jaccard(grams[i], grams[j])
             for i, j in combinations(range(len(grams)), 2)]
    return float(np.mean(pairs))


# ── Curation strategies ──────────────────────────────────────────────────────

def pick_greedy(cands: list[dict], k: int) -> list[dict]:
    return sorted(cands, key=lambda c: -c["_prm"])[:k]


def pick_random(cands: list[dict], k: int, rng) -> list[dict]:
    idx = rng.choice(len(cands), size=min(k, len(cands)), replace=False)
    return [cands[i] for i in idx]


def pick_qubo(cands: list[dict], k: int, redundancy_w: float, card_w: float,
              seed: int) -> list[dict]:
    """Quality on the diagonal, textual redundancy off it, solved by annealing."""
    n = len(cands)
    if n <= k:
        return list(cands)

    grams = [bigrams(c.get("reason", "")) for c in cands]
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i][i] = -cands[i]["_prm"] + card_w * (1 - 2 * k)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i][j] = Q[j][i] = redundancy_w * jaccard(grams[i], grams[j]) + 2 * card_w

    rng = np.random.default_rng(seed)
    n_reads = 40
    states = rng.integers(0, 2, size=(n_reads, n)).astype(np.float64)
    energies = np.einsum("ri,ij,rj->r", states, Q, states)
    for step in range(200):
        T = 10.0 * (0.001 / 10.0) ** (step / 199)
        flip = rng.integers(0, n, size=n_reads)
        cand = states.copy()
        cand[np.arange(n_reads), flip] = 1.0 - cand[np.arange(n_reads), flip]
        new_e = np.einsum("ri,ij,rj->r", cand, Q, cand)
        d = new_e - energies
        acc = (d < 0) | (rng.random(n_reads) < np.exp(-d / max(T, 1e-9)))
        states[acc] = cand[acc]
        energies[acc] = new_e[acc]

    best = states[int(np.argmin(energies))]
    picked = [cands[i] for i in range(n) if best[i] == 1]

    # Force exactly k so every strategy contributes the same example count --
    # otherwise a win could be explained by dataset size rather than curation.
    if len(picked) > k:
        picked = sorted(picked, key=lambda c: -c["_prm"])[:k]
    elif len(picked) < k:
        rest = sorted((c for c in cands if c not in picked), key=lambda c: -c["_prm"])
        picked += rest[: k - len(picked)]
    return picked


# ── SFT record ───────────────────────────────────────────────────────────────

def make_example(question: str, chain: dict, gold, source: str) -> dict:
    """Match the schema fast_finetune_pipeline.py filters on."""
    reason = str(chain.get("reason", "")).strip()
    answer = str(chain.get("answer", "")).strip() or str(gold)
    return {
        "messages": [
            {"role": "user",
             "content": f"Solve the following problem step by step.\n\nQuestion: {question}"},
            {"role": "assistant",
             "content": f"{reason}\n\nAnswer: {answer}" if reason else f"Answer: {answer}"},
        ],
        "metadata": {
            "source": source,
            "gold": str(gold),
            # The trainer keeps examples at >= 0.70; PRM score is the quality
            # measure here, floored so a well-verified chain is never dropped by
            # a threshold meant for the old consensus-based score.
            "correctness_score": round(max(float(chain.get("_prm", 0.0)), 0.70), 4),
            "consensus_score": 0.0,
            "qubo_selected": True,
            "greedy_would_have_failed": False,
            "full_chains_pool": [],
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--prm-cache", required=True)
    ap.add_argument("--k", type=int, default=3,
                    help="Correct chains kept per question, identical for every strategy.")
    ap.add_argument("--out-dir", default="data/curation")
    ap.add_argument("--redundancy-w", type=float, default=1.0)
    ap.add_argument("--card-w", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.load(open(args.chains, encoding="utf-8"))
    prm = json.load(open(args.prm_cache, encoding="utf-8"))["scores"]
    chains_all, golds = data["chains"], data["golds"]
    questions = data["questions"]
    source = data.get("dataset", "math500")
    n = min(len(chains_all), len(golds), len(prm))

    rng = np.random.default_rng(args.seed)
    strategies = ("qubo", "greedy", "random")
    picked_sets = {s: [] for s in strategies}
    div = {s: [] for s in strategies}
    n_usable = 0
    n_correct_total = 0

    for qi in range(n):
        gold = golds[qi]
        cands = []
        for c, p in zip(chains_all[qi], prm[qi]):
            a = chain_answer(c)
            if a is not None and numeric_match(a, gold):
                cc = dict(c)
                cc["_prm"] = float(p)
                cands.append(cc)
        if not cands:
            continue        # nothing correct to learn from
        n_usable += 1
        n_correct_total += len(cands)

        sel = {
            "qubo": pick_qubo(cands, args.k, args.redundancy_w, args.card_w, args.seed + qi),
            "greedy": pick_greedy(cands, args.k),
            "random": pick_random(cands, args.k, rng),
        }
        for s in strategies:
            div[s].append(set_diversity(sel[s]))
            for ch in sel[s]:
                picked_sets[s].append(make_example(questions[qi], ch, gold, source))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"  CURATION  ({source}: {n} questions, k={args.k} chains kept each)")
    print("=" * 66)
    print(f"  questions with >=1 correct chain : {n_usable}/{n}")
    print(f"  correct chains available          : {n_correct_total} "
          f"({n_correct_total / max(n_usable,1):.1f} per usable question)")
    print()
    print(f"  {'strategy':<10}{'examples':>10}{'mean set diversity':>21}")
    print("  " + "-" * 42)
    for s in strategies:
        print(f"  {s:<10}{len(picked_sets[s]):>10}{np.mean(div[s]):>21.3f}")
    print("  " + "-" * 42)
    print("  Diversity = mean pairwise bigram distance within each kept set.")
    print("  QUBO above greedy here confirms the objective is doing what it claims;")
    print("  if they are equal, the redundancy term is not biting and the")
    print("  downstream comparison cannot say anything about diversity.")

    sizes = {s: len(v) for s, v in picked_sets.items()}
    if len(set(sizes.values())) != 1:
        print(f"\n  [WARN] unequal dataset sizes {sizes} -- the comparison is no longer"
              "\n         controlled; a win could be explained by data volume.")

    # Identical split across strategies: same questions train, same questions
    # validate, so only curation differs.
    for s in strategies:
        rows = picked_sets[s]
        idx = np.arange(len(rows))
        rng2 = np.random.default_rng(args.seed)
        rng2.shuffle(idx)
        cut = int(len(rows) * (1 - args.val_frac))
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        for name, part in (("finetune_train.jsonl", idx[:cut]),
                           ("finetune_val.jsonl", idx[cut:])):
            with open(d / name, "w", encoding="utf-8") as f:
                for i in part:
                    f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        print(f"\n  {s:<8} -> {d}  (train {cut}, val {len(rows)-cut})")

    print("\n" + "=" * 66)
    print("  NEXT: fine-tune one adapter per strategy with IDENTICAL hyper-")
    print("  parameters, then evaluate all three plus the base model.")
    print("=" * 66)
    for s in strategies:
        print(f"  python scripts/fast_finetune_pipeline.py --skip-datagen --skip-eval \\")
        print(f"      --data-dir {out / s} --run-name curate-{s} --epochs 3")
    print()
    print("  Then evaluate each adapter (and the base) on held-out benchmarks:")
    print("  QUBO_ADAPTER_PATH=checkpoints/curate-qubo/final_adapter \\")
    print("      python scripts/run_all_benchmarks.py --condition-label qubo \\")
    print("      --benchmarks gsm8k mmlu bbh --subset-size 300 --fresh")


if __name__ == "__main__":
    main()
