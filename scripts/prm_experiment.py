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

def sweep_aggregation_rules(chains_all, golds, prm_scores) -> None:
    """How should PRM scores be turned into ONE answer per question?

    The default (sum of PRM over chains sharing an answer) is linear in vote
    count, so it cannot overcome a count disadvantage no matter how confident
    the PRM is. Measured on q89 of this pool: one correct chain at PRM 0.984
    against seventeen majority chains at 0.816 gives 0.98 vs 13.9 -- the
    majority wins by arithmetic, not by being right. With AUC 0.815 the ranking
    signal is clearly there; the aggregation rule is what is throwing it away.

    Each rule below scores every candidate answer differently:
      count        plain majority (the baseline)
      sum          count weighted by PRM -- linear, the current behaviour
      max          an answer is as good as its single best-supported chain,
                   which lets one strongly-verified chain outvote a large but
                   weakly-verified crowd
      mean         average support, ignoring how many chains back it
      top2/top3    sum of only the k best chains per answer, capping how much
                   raw popularity can accumulate
      pow{p}       sum of PRM**p -- sharpens toward confident chains as p grows
      gated{t}     plain count, but only chains scoring above t may vote
      blend{a}     a * normalised count + (1-a) * normalised max

    CAUTION: this sweeps many rules against the SAME 300 questions, so the top
    entry is optimistically biased -- with ~12 rules, a couple of points of
    apparent gain can be selection noise. Treat the winner as a hypothesis to
    confirm on a held-out pool, not as a measured result.
    """
    from collections import defaultdict

    def rules_for(rows):
        """rows: list of dicts with 'ans' and 'prm'. Returns {rule_name: answer}."""
        by_ans = defaultdict(list)
        for r in rows:
            by_ans[r["ans"]].append(r["prm"])

        out = {}
        out["count"] = max(by_ans.items(), key=lambda kv: len(kv[1]))[0]
        out["sum"] = max(by_ans.items(), key=lambda kv: sum(kv[1]))[0]
        out["max"] = max(by_ans.items(), key=lambda kv: max(kv[1]))[0]
        out["mean"] = max(by_ans.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
        for k in (2, 3):
            out[f"top{k}"] = max(
                by_ans.items(), key=lambda kv: sum(sorted(kv[1], reverse=True)[:k])
            )[0]
        for p in (2, 4, 8):
            out[f"pow{p}"] = max(
                by_ans.items(), key=lambda kv: sum(v ** p for v in kv[1])
            )[0]
        for t in (0.5, 0.8, 0.9):
            gated = {a: [v for v in vs if v >= t] for a, vs in by_ans.items()}
            gated = {a: vs for a, vs in gated.items() if vs}
            # Every chain filtered out -> fall back to plain count rather than
            # abstaining, so the rule is never worse than the baseline by default.
            src = gated if gated else by_ans
            out[f"gated{t}"] = max(src.items(), key=lambda kv: len(kv[1]))[0]

        n_tot = sum(len(v) for v in by_ans.values())
        max_all = max(max(v) for v in by_ans.values()) or 1.0
        for a_w in (0.3, 0.5, 0.7):
            out[f"blend{a_w}"] = max(
                by_ans.items(),
                key=lambda kv: a_w * (len(kv[1]) / n_tot)
                + (1 - a_w) * (max(kv[1]) / max_all),
            )[0]
        return out

    # Per-question outcomes, not just totals: every rule is evaluated on the
    # SAME questions, so the comparison against plain voting is PAIRED and must
    # be tested as such (see the McNemar note below).
    outcomes: dict[str, list[bool]] = defaultdict(list)
    n = 0
    for qi, (chains, gold) in enumerate(zip(chains_all, golds)):
        rows = []
        for c, p in zip(chains, prm_scores[qi]):
            a = chain_answer(c)
            if a is not None:
                rows.append({"ans": a, "prm": float(p)})
        if not rows:
            continue
        n += 1
        for rule, ans in rules_for(rows).items():
            outcomes[rule].append(bool(numeric_match(ans, gold)))

    tallies = {r: sum(v) for r, v in outcomes.items()}
    base = tallies["count"] / n
    baseline_outcomes = outcomes["count"]

    def mcnemar_p(rule_outcomes: list[bool]) -> tuple[int, int, float]:
        """Exact two-sided McNemar test against the plain-voting baseline.

        b = rule right where baseline wrong; c = rule wrong where baseline right.
        Only discordant questions carry information -- the ones both methods get
        right or both get wrong say nothing about which is better. Under the null
        each discordant question is a fair coin, so the exact binomial tail is the
        p-value. This is far more powerful than comparing two independent
        proportions, which is what an unpaired standard error assumes and which
        needlessly discards the pairing this design already has.
        """
        from math import comb
        b = sum(1 for r, base_ok in zip(rule_outcomes, baseline_outcomes) if r and not base_ok)
        c = sum(1 for r, base_ok in zip(rule_outcomes, baseline_outcomes) if not r and base_ok)
        tot = b + c
        if tot == 0:
            return b, c, 1.0
        k = min(b, c)
        p = 2 * sum(comb(tot, i) for i in range(k + 1)) / (2 ** tot)
        return b, c, min(p, 1.0)

    print("\n" + "=" * 72)
    print("  VOTE-AGGREGATION SWEEP  (same PRM scores, different rules)")
    print("=" * 72)
    print(f"  {'rule':<11}{'acc':>7}{'vs count':>10}{'b':>5}{'c':>5}{'McNemar p':>11}  {'verdict':<14}")
    print("  " + "-" * 66)
    for rule, correct in sorted(tallies.items(), key=lambda kv: -kv[1]):
        acc = correct / n
        d = (acc - base) * 100
        if rule == "count":
            print(f"  {rule:<11}{acc:>6.1%}{d:>+9.1f}{'-':>5}{'-':>5}{'-':>11}  {'<- baseline':<14}")
            continue
        b, c, p = mcnemar_p(outcomes[rule])
        if p < 0.01:
            verdict = "SIGNIFICANT**" if d > 0 else "WORSE**"
        elif p < 0.05:
            verdict = "SIGNIFICANT*" if d > 0 else "WORSE*"
        else:
            verdict = "not significant"
        print(f"  {rule:<11}{acc:>6.1%}{d:>+9.1f}{b:>5}{c:>5}{p:>11.4f}  {verdict:<14}")
    print("  " + "-" * 66)
    print(f"  n={n}.  b = rule right where voting wrong;  c = rule wrong where voting right.")
    print("  Exact two-sided McNemar on the discordant questions only -- the correct")
    print("  test for two methods scored on the SAME questions.")
    print("\n  Sweeping ~14 rules on one pool biases the top entry upward, and running")
    print("  14 tests inflates the false-positive rate: at p<0.05 you expect ~0.7")
    print("  spurious hits by chance. Treat a winner as a hypothesis and confirm it")
    print("  on a fresh pool before adopting it.")


def _explain_vendored_code_failure(exc: Exception, args, stage: str) -> None:
    """Turn an opaque transformers-compat crash into an actionable next step.

    Qwen2.5-Math-PRM ships its own modeling code via trust_remote_code, written
    against an older transformers. Each incompatibility only surfaces when
    execution reaches that line, so fixing one reveals the next
    (config.pad_token_id -> DynamicCache.from_legacy_cache -> ...). Two have
    been patched in pipeline/prm_scorer.py; this explains the escape routes
    rather than leaving a raw traceback to interpret.
    """
    import transformers
    print("\n" + "!" * 70, file=sys.stderr)
    print(f"  PRM failed during {stage}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("!" * 70, file=sys.stderr)
    print(f"""
  This is almost certainly a transformers-version incompatibility, not a bug
  in your data or setup. {args.prm_model} vendors its own modeling code
  (trust_remote_code=True) written against an older transformers than the
  {transformers.__version__} installed here, and each removed API it calls only
  fails once execution reaches it.

  Two such breakages are already patched in pipeline/prm_scorer.py
  (config.pad_token_id, DynamicCache.from_legacy_cache). If you have hit a
  third, the highest-value fix is to stop patching one at a time:

  OPTION 1 -- pin transformers for PRM scoring only (most reliable).
    The PRM run is a standalone offline pass, so it can use its own env:
        python -m venv .venv_prm
        .venv_prm/bin/pip install "transformers==4.46.3" torch accelerate datasets
        .venv_prm/bin/python scripts/prm_experiment.py --prm-model {args.prm_model}
    Nothing else in the pipeline needs to change; only the cached score file
    is consumed downstream.

  OPTION 2 -- use a PRM that does not need trust_remote_code.
        --prm-model Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
    Different families use different step-separator conventions, so
    pipeline/prm_scorer.py may need an adapter for the new format -- it
    currently targets the Qwen '<extra_0>' scheme and will say so clearly
    rather than scoring nonsense.

  Send me the traceback either way and I will tell you which is the shorter path.
""", file=sys.stderr)


def score_with_prm(questions, chains_all, args) -> list[list[float]]:
    """One PRM forward pass per chain. Cached to disk -- this is the only slow part."""
    import time

    # Load prm_scorer.py BY FILE PATH rather than as pipeline.prm_scorer.
    #
    # `from pipeline.prm_scorer import ...` executes pipeline/__init__.py, which
    # imports qubo_builder -> sentence_transformers -> the full training stack.
    # This script is designed to run in a throwaway pinned-transformers venv
    # holding only what the PRM needs, so dragging in those dependencies would
    # force that venv to mirror the entire project environment -- defeating the
    # point of isolating it. prm_scorer.py imports nothing from this package
    # (only torch + transformers), so loading the file directly is safe.
    import importlib.util
    _prm_path = _PROJECT_ROOT / "pipeline" / "prm_scorer.py"
    _spec = importlib.util.spec_from_file_location("_prm_scorer_standalone", _prm_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PRMScorer = _mod.PRMScorer

    print(f"[prm] loading {args.prm_model} (aggregation={args.aggregation}) ...", flush=True)
    try:
        scorer = PRMScorer(
            model_name=args.prm_model,
            device=args.device,
            aggregation=args.aggregation,
            cache_dir=args.cache_dir,
        )
    except (AttributeError, TypeError) as e:
        _explain_vendored_code_failure(e, args, stage="loading")
        raise
    print("[prm] loaded.", flush=True)

    all_scores: list[list[float]] = []
    t0 = time.time()
    total = len(chains_all)
    shown = False
    for qi, (question, chains) in enumerate(zip(questions, chains_all)):
        scores = []
        for ci, chain in enumerate(chains):
            try:
                s, steps = scorer.score_chain(question, chain.get("reason", ""))
            except (AttributeError, TypeError) as e:
                _explain_vendored_code_failure(e, args, stage="the first forward pass")
                raise
            scores.append(s)

            # On a --limit smoke run, show one chain's per-step vector. A PRM
            # given bad step boundaries still returns numbers, so the only way
            # to catch mis-segmentation is to look at the steps it graded.
            if args.limit and not shown and ci == 0:
                shown = True
                print(f"\n[prm] sample grading, question {qi}, chain 0:")
                print(f"      aggregate ({scorer.aggregation}) = {s:.3f}")
                print(f"      {len(steps)} steps graded: "
                      f"{[round(p, 3) for p in steps]}")
                for j, seg in enumerate(scorer.split_steps(chain.get('reason', ''))[:6]):
                    p = f"{steps[j]:.3f}" if j < len(steps) else "  -  "
                    print(f"        [{p}] {seg[:70]}")
                print()
        all_scores.append(scores)

        if (qi + 1) % 10 == 0:
            el = time.time() - t0
            rate = (qi + 1) / el
            eta = (total - qi - 1) / rate if rate else float("nan")
            print(
                f"[prm] {qi+1}/{total} questions  ({el:.0f}s elapsed, ETA {eta:.0f}s)",
                flush=True,
            )

    # Always report throughput, including for short --limit runs that never hit
    # the every-10 checkpoint. This number decides whether PRM scoring is
    # affordable INSIDE the training loop (scoring every chain, every round) or
    # only as an offline analysis pass -- a real architectural fork.
    el = time.time() - t0
    n_chains = sum(len(c) for c in chains_all)
    print(
        f"\n[prm] scored {n_chains} chains from {total} questions in {el:.0f}s "
        f"({el / max(n_chains, 1):.2f}s per chain)",
        flush=True,
    )
    print(
        f"[prm] extrapolated to a full 300-question pool (6000 chains): "
        f"{el / max(n_chains, 1) * 6000 / 60:.0f} min",
        flush=True,
    )
    return all_scores


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyse(questions, chains_all, golds, prm_scores, blend_weights, dataset="unknown"):
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

    # ── Discrimination: can the PRM tell correct chains from wrong ones? ─────
    #
    # Reported before the strategy table because it is the precondition for all
    # of them. A PRM that scores nearly everything ~1.0 carries no ranking
    # information no matter how it is aggregated or blended, and the smoke run
    # showed scores clustered in 0.977-1.0 on easy questions. AUC here is the
    # probability that a randomly chosen CORRECT chain outranks a randomly
    # chosen INCORRECT one: 0.5 is coin-flip, 1.0 is perfect separation.
    pos, neg = [], []
    for qi, (chains, gold) in enumerate(zip(chains_all, golds)):
        for c, p in zip(chains, prm_scores[qi]):
            a = chain_answer(c)
            if a is None:
                continue
            (pos if numeric_match(a, gold) else neg).append(float(p))

    print("\n" + "=" * 64)
    print("  PRM DISCRIMINATION  (all chains pooled)")
    print("=" * 64)
    if pos and neg:
        import statistics
        auc = sum(
            1.0 if p > q else 0.5 if p == q else 0.0
            for p in pos for q in neg
        ) / (len(pos) * len(neg))
        print(f"  correct chains   n={len(pos):<6} mean={statistics.mean(pos):.3f} "
              f"sd={statistics.pstdev(pos):.3f}")
        print(f"  incorrect chains n={len(neg):<6} mean={statistics.mean(neg):.3f} "
              f"sd={statistics.pstdev(neg):.3f}")
        print(f"  AUC (P[correct scores above incorrect]) = {auc:.3f}")
        if auc < 0.6:
            print("  -> WEAK. The PRM barely ranks correct above incorrect, so no")
            print("     aggregation or blend will rescue it. Suspect segmentation or")
            print("     try --aggregation prod/mean before anything else.")
        elif auc < 0.75:
            print("  -> MODERATE. Real signal, but expect modest gains.")
        else:
            print("  -> STRONG. This is a usable selection signal.")
    else:
        print("  (need both correct and incorrect chains to measure separation)")

    # ── Headline table ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    n_chains = len(chains_all[0]) if chains_all else 0
    print(f"  SELECTION STRATEGIES  ({dataset}: {n} questions x {n_chains} chains)")
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
    ap.add_argument("--limit", type=int, default=None,
                    help="Only score the first N questions. Use a small value (e.g. 5) "
                         "to confirm the model loads and produces sane per-step scores "
                         "before committing to the full pass. Scores from a limited run "
                         "are written to a --prm-cache suffixed with the limit, so they "
                         "can never be mistaken for a complete scoring pass.")
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

    # Prefer questions/golds stored INSIDE the cache (build_chain_cache.py writes
    # them). Re-deriving them by reloading GSM8K -- the only option for the
    # original cache, which did not store them -- assumes the consumer filters
    # and orders the dataset identically to the producer. That happens to hold
    # for GSM8K but silently misaligns golds with chains for any other dataset,
    # which would corrupt every number without raising anything.
    if "golds" in data and "questions" in data:
        golds = data["golds"]
        questions = data["questions"]
        print(f"[data] using questions/golds stored in the cache "
              f"(dataset={data.get('dataset', 'unknown')})")
    else:
        print("[data] cache has no inline golds; falling back to re-deriving them "
              "from GSM8K. Valid only if this pool IS GSM8K in the original order.")
        golds = load_golds(n)
        questions = load_questions(n)

    if len(golds) < n:
        print(f"[warn] only {len(golds)} golds for {n} cached questions; truncating.")
        chains_all = chains_all[:len(golds)]
        n = len(golds)

    prm_cache = Path(args.prm_cache)
    if args.limit:
        n = min(args.limit, n)
        chains_all, golds, questions = chains_all[:n], golds[:n], questions[:n]
        # Separate cache file: a partial pass must never be reused as if it were
        # the full 300-question scoring run.
        prm_cache = prm_cache.with_name(f"{prm_cache.stem}_limit{n}{prm_cache.suffix}")
        print(f"[data] --limit {args.limit}: scoring only the first {n} questions")
        print(f"[data] partial results -> {prm_cache}")
        print("[data] NOTE: the recoverable-set analysis needs the full 300 to be "
              "meaningful; this run is a plumbing check only.")
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

    analyse(questions, chains_all, golds, prm_scores, args.blend,
            dataset=data.get("dataset", "gsm8k (assumed)"))
    sweep_aggregation_rules(chains_all, golds, prm_scores)


if __name__ == "__main__":
    main()
