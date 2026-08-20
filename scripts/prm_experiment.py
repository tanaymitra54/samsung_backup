"""
==============================================================================
FILE: scripts/prm_experiment.py
ROLE: Decisive test -- can a PRM beat consensus where consensus provably cannot?
==============================================================================

THE QUESTION THIS ANSWERS
-------------------------
Measured on results/cached_chains_300q_goldfree.json (300 GSM8K questions x 20
chains, all scored gold-free):

    single chain (no selection) .................. 78.7%
    plain majority vote over all 20 .............. 90.3%
    quality-weighted vote over all 20 ............ 90.3%
    QUBO-selected subset vote (tuned, 81 combos).. 90.3%
    ORACLE (any of the 20 chains correct) ........ 96.7%

Every selection strategy lands on exactly 271/300 because the dominant quality
term is cross-chain consensus -- which IS majority voting, so anything built on
it reproduces the majority answer by construction. The entire 6.3-point gap to
oracle lives in 19 questions where the majority is wrong but a correct chain
exists, and on those the current signal ranks the correct chains above the
wrong-majority chains only 3/19 times (16%): on exactly those questions,
"correct" and "popular" are anti-correlated.

So this script does not ask "is the PRM good". It asks the only question that
matters: ON THOSE 19 RECOVERABLE QUESTIONS, does a PRM rank the correct chains
above the popular-but-wrong ones? If yes, the selection stage finally has a
signal that can beat voting and the QUBO objective has something real to
optimise. If no, PRM scoring is not the answer and we should stop before
spending a fine-tuning cycle on it.

WHY IT REUSES THE CACHE
-----------------------
The chains are already generated and stored as text. A PRM grades text, so this
experiment needs no re-sampling from the policy model -- only one PRM forward
pass per chain. PRM scores are themselves cached to disk, so the analysis can be
re-run (different aggregation, different blend weights) for free afterwards.

USAGE
-----
    # One-time scoring pass (the only GPU-heavy step), then analysis:
    python scripts/prm_experiment.py --prm-model Qwen/Qwen2.5-Math-PRM-7B

    # Re-analyse an existing score cache without touching the GPU:
    python scripts/prm_experiment.py --analyze-only

    # Try a different step-aggregation rule (requires re-scoring):
    python scripts/prm_experiment.py --aggregation mean --prm-cache results/prm_mean.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Answer extraction / matching (mirrors tune_qubo_params.py) ───────────────

def extract_gsm8k_gold(answer_field: str) -> Optional[float]:
    if "####" not in answer_field:
        return None
    tail = answer_field.split("####")[-1].strip().replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", tail)
    return float(m[-1]) if m else None


def numeric_match(pred: Optional[float], gold: float) -> bool:
    if pred is None:
        return False
    return abs(pred - gold) < max(0.01, 0.01 * abs(gold))


def last_number(text: str) -> Optional[float]:
    m = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(m[-1]) if m else None


def chain_answer(chain: dict) -> Optional[float]:
    """A chain's final answer: its parsed answer field, else its reasoning tail."""
    a = last_number(chain.get("answer") or "")
    return a if a is not None else last_number(chain.get("reason") or "")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_golds(n: int) -> list[float]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    golds = []
    for item in ds:
        g = extract_gsm8k_gold(item["answer"])
        if g is not None:
            golds.append(g)
        if len(golds) == n:
            break
    return golds


def load_questions(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    out = []
    for item in ds:
        if extract_gsm8k_gold(item["answer"]) is not None:
            out.append(item["question"])
        if len(out) == n:
            break
    return out


# ── PRM scoring pass ─────────────────────────────────────────────────────────

def score_with_prm(questions, chains_all, args) -> list[list[float]]:
    """One PRM forward pass per chain. Cached to disk -- this is the only slow part."""
    from pipeline.prm_scorer import PRMScorer
    import time

    print(f"[prm] loading {args.prm_model} (aggregation={args.aggregation}) ...", flush=True)
    scorer = PRMScorer(
        model_name=args.prm_model,
        device=args.device,
        aggregation=args.aggregation,
        cache_dir=args.cache_dir,
    )
    print("[prm] loaded.", flush=True)

    all_scores: list[list[float]] = []
    t0 = time.time()
    total = len(chains_all)
    for qi, (question, chains) in enumerate(zip(questions, chains_all)):
        scores = []
        for chain in chains:
            s, _steps = scorer.score_chain(question, chain.get("reason", ""))
            scores.append(s)
        all_scores.append(scores)

        if (qi + 1) % 10 == 0:
            el = time.time() - t0
            rate = (qi + 1) / el
            eta = (total - qi - 1) / rate if rate else float("nan")
            print(
                f"[prm] {qi+1}/{total} questions  ({el:.0f}s elapsed, ETA {eta:.0f}s)",
                flush=True,
            )
    return all_scores


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyse(questions, chains_all, golds, prm_scores, blend_weights):
    """Compare every selection strategy, then zoom in on the recoverable set."""
    n = len(golds)

    counts = {
        "single chain (no selection)": 0,
        "plain majority vote (all 20)": 0,
        "consensus-weighted vote (all 20)": 0,
        "PRM-weighted vote (all 20)": 0,
        "PRM best-of-N (argmax, all 20)": 0,
        "ORACLE (any chain correct)": 0,
    }
    recoverable = []

    for qi, (chains, gold) in enumerate(zip(chains_all, golds)):
        prm = prm_scores[qi]
        rows = []
        for c, p in zip(chains, prm):
            a = chain_answer(c)
            if a is not None:
                rows.append({
                    "ans": a,
                    "prm": float(p),
                    "cons": float(c.get("correctness_score", 0.0)),
                    "ok": numeric_match(a, gold),
                })
        if not rows:
            continue

        counts["single chain (no selection)"] += rows[0]["ok"]

        tally = Counter(r["ans"] for r in rows)
        mv = tally.most_common(1)[0][0]
        maj_ok = numeric_match(mv, gold)
        counts["plain majority vote (all 20)"] += maj_ok

        def weighted(key):
            w = {}
            for r in rows:
                w[r["ans"]] = w.get(r["ans"], 0.0) + r[key]
            return max(w.items(), key=lambda kv: kv[1])[0]

        counts["consensus-weighted vote (all 20)"] += numeric_match(weighted("cons"), gold)
        counts["PRM-weighted vote (all 20)"] += numeric_match(weighted("prm"), gold)
        counts["PRM best-of-N (argmax, all 20)"] += max(rows, key=lambda r: r["prm"])["ok"]

        any_ok = any(r["ok"] for r in rows)
        counts["ORACLE (any chain correct)"] += any_ok

        if not maj_ok and any_ok:
            corr = [r for r in rows if r["ok"]]
            majr = [r for r in rows if r["ans"] == mv]
            recoverable.append({
                "qi": qi,
                "n_correct": len(corr),
                "n_valid": len(rows),
                "n_majority": len(majr),
                "prm_correct": sum(r["prm"] for r in corr) / len(corr),
                "prm_majority": sum(r["prm"] for r in majr) / len(majr),
                "cons_correct": sum(r["cons"] for r in corr) / len(corr),
                "cons_majority": sum(r["cons"] for r in majr) / len(majr),
                "prm_argmax_ok": max(rows, key=lambda r: r["prm"])["ok"],
            })

    # ── Headline table ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  SELECTION STRATEGIES  (300 GSM8K questions, 20 chains each)")
    print("=" * 64)
    for name, c in counts.items():
        marker = ""
        if name.startswith("plain majority"):
            marker = "   <- baseline to beat"
        if name.startswith("ORACLE"):
            marker = "   <- ceiling"
        print(f"  {name:<38}{c/n:>7.1%}{marker}")

    base = counts["plain majority vote (all 20)"] / n
    prm_vote = counts["PRM-weighted vote (all 20)"] / n
    prm_bon = counts["PRM best-of-N (argmax, all 20)"] / n
    print("-" * 64)
    print(f"  PRM-weighted vote vs plain vote : {(prm_vote-base)*100:+.1f} points")
    print(f"  PRM best-of-N     vs plain vote : {(prm_bon-base)*100:+.1f} points")

    # ── The part that actually decides it ────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  RECOVERABLE SET -- {len(recoverable)} questions where the majority is")
    print("  WRONG but at least one chain is RIGHT. This is the entire")
    print("  opportunity space; voting cannot win any of these by construction.")
    print("=" * 64)
    if not recoverable:
        print("  (none -- nothing to recover)")
        return

    print(f"  {'q':>4} {'#right':>7} {'#major':>7} "
          f"{'PRM right':>10} {'PRM major':>10} {'sep?':>6} {'cons sep?':>10}")
    print("  " + "-" * 60)
    prm_sep = cons_sep = 0
    for r in recoverable:
        p_ok = r["prm_correct"] > r["prm_majority"]
        c_ok = r["cons_correct"] > r["cons_majority"]
        prm_sep += p_ok
        cons_sep += c_ok
        print(f"  {r['qi']:>4} {r['n_correct']:>7} {r['n_majority']:>7} "
              f"{r['prm_correct']:>10.3f} {r['prm_majority']:>10.3f} "
              f"{'YES' if p_ok else 'no':>6} {'YES' if c_ok else 'no':>10}")

    m = len(recoverable)
    print("  " + "-" * 60)
    print(f"\n  Correct chains ranked ABOVE the wrong majority:")
    print(f"    by PRM score       : {prm_sep}/{m}  ({prm_sep/m:.0%})")
    print(f"    by current signal  : {cons_sep}/{m}  ({cons_sep/m:.0%})   <- what we have today")
    n_argmax = sum(r["prm_argmax_ok"] for r in recoverable)
    print(f"\n  PRM argmax picks a CORRECT chain on {n_argmax}/{m} recoverable questions.")
    print(f"  Upper bound if selection used PRM perfectly here: "
          f"{(base + m/n)*100:.1f}%")

    # ── Verdict ──────────────────────────────────────────────────────────────
    #
    # Beating the CURRENT signal is not a sufficient bar. The current signal is
    # 65% consensus, so on this set -- where correct and popular are
    # anti-correlated by definition -- it scores ~16%, i.e. WORSE than chance.
    # A purely random scorer gets ~50% here and would clear that bar while
    # adding nothing (verified: a random scorer produced 42% separation and
    # -10.3 points on best-of-N). So the verdict requires the PRM to beat
    # CHANCE on separation AND to actually move headline accuracy.
    sep_rate = prm_sep / m
    beats_chance = sep_rate > 0.5
    helps_accuracy = (prm_vote > base) or (prm_bon > base)

    print("\n  VERDICT")
    print(f"    separation {sep_rate:.0%} vs 50% chance ....... "
          f"{'PASS' if beats_chance else 'FAIL'}")
    print(f"    improves headline accuracy ............ "
          f"{'PASS' if helps_accuracy else 'FAIL'} "
          f"(best {max(prm_vote, prm_bon)*100:.1f}% vs {base*100:.1f}%)")

    if beats_chance and helps_accuracy:
        print("\n    -> The PRM carries real signal where consensus cannot. Wire it")
        print("       into the QUBO diagonal and re-tune.")
    elif helps_accuracy and not beats_chance:
        print("\n    -> Accuracy improved but per-question separation is at chance;")
        print("       the gain may be noise on 19 questions. Re-check on a larger")
        print("       pool before committing to it.")
    else:
        print("\n    -> Not usable as-is. Try another aggregation")
        print("       (--aggregation mean/prod/last), a stronger PRM, or inspect")
        print("       step segmentation -- a PRM given bad step boundaries grades")
        print("       spans that are not really reasoning steps.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", default="results/cached_chains_300q_goldfree.json",
                    help="Cached scored chains from tune_qubo_params.py")
    ap.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B",
                    help="PRM to score chains with. 7B is recommended on an 80GB card.")
    ap.add_argument("--prm-cache", default="results/prm_scores_300q.json",
                    help="Where PRM scores are cached, so analysis can re-run free.")
    ap.add_argument("--aggregation", default="min", choices=["min", "prod", "mean", "last"],
                    help="How per-step probabilities collapse to one chain score.")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu")
    ap.add_argument("--cache-dir", default="./cache/models")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip PRM scoring; analyse an existing --prm-cache.")
    ap.add_argument("--blend", nargs="*", type=float, default=[0.5],
                    help="Reserved: PRM/consensus blend weights to sweep in analysis.")
    args = ap.parse_args()

    chains_path = Path(args.chains)
    if not chains_path.exists():
        print(f"[error] {chains_path} not found. Run tune_qubo_params.py first "
              "(its pre-cache phase produces this file).", file=sys.stderr)
        sys.exit(1)

    data = json.load(open(chains_path, encoding="utf-8"))
    chains_all = data["chains"]
    n = len(chains_all)
    print(f"[data] {n} questions x {len(chains_all[0])} chains from {chains_path}")

    golds = load_golds(n)
    questions = load_questions(n)
    if len(golds) < n:
        print(f"[warn] only {len(golds)} golds for {n} cached questions; truncating.")
        chains_all = chains_all[:len(golds)]
        n = len(golds)

    prm_cache = Path(args.prm_cache)
    if args.analyze_only:
        if not prm_cache.exists():
            print(f"[error] --analyze-only but {prm_cache} does not exist.", file=sys.stderr)
            sys.exit(1)
        prm_scores = json.load(open(prm_cache, encoding="utf-8"))["scores"]
        print(f"[prm] loaded cached scores from {prm_cache}")
    elif prm_cache.exists():
        prm_scores = json.load(open(prm_cache, encoding="utf-8"))["scores"]
        print(f"[prm] reusing existing {prm_cache} (delete it to re-score)")
    else:
        prm_scores = score_with_prm(questions, chains_all, args)
        prm_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(prm_cache, "w", encoding="utf-8") as f:
            json.dump({"model": args.prm_model,
                       "aggregation": args.aggregation,
                       "scores": prm_scores}, f)
        print(f"[prm] scores cached -> {prm_cache}")

    analyse(questions, chains_all, golds, prm_scores, args.blend)


if __name__ == "__main__":
    main()
