"""
==============================================================================
FILE: scripts/qubo_prm_experiment.py
ROLE: Does QUBO SET-SELECTION beat simply weighting the votes?
==============================================================================

THE QUESTION
------------
On MATH-500, PRM-weighted voting is now a genuine, statistically significant
improvement over plain majority voting (exact McNemar, surviving
Benjamini-Hochberg correction across 14 rules):

    plain majority vote ....... 57.3%
    PRM-weighted (pow8) ....... 61.0%   +3.7   p=0.0127
    ORACLE .................... 78.0%

But that winning method uses NO QUBO. It weights every chain's vote and never
selects a subset. So the project's central claim is still unproven: does
optimising over SETS of chains beat scoring them independently?

This is the honest test of that claim, and it is designed so QUBO can lose.

WHY SET SELECTION COULD BEAT WEIGHTING
--------------------------------------
Every method measured so far -- majority voting, PRM-weighted voting,
best-of-N -- scores each chain INDEPENDENTLY and then aggregates. None can
express a preference over combinations. A QUBO can:

    diagonal      per-chain quality           (now the PRM, not consensus)
    off-diagonal  pairwise redundancy         (what makes a PAIR wasteful)

If the off-diagonal encodes something real, the selected subset is more than
its parts. If it does not, QUBO degenerates into top-k-by-diagonal and there is
no reason to expect it to beat weighting -- which is exactly what happened when
the diagonal was 65% consensus.

FAILURE-LOCUS COMPLEMENTARITY (the novel term)
----------------------------------------------
With per-step PRM vectors, redundancy can mean something sharper than "these
two chains read similarly". Two chains that break down at the SAME point in
their derivation share a failure mode: keeping both buys one error twice. Two
chains whose weak steps sit at DIFFERENT loci cover for each other -- where one
is unreliable the other may not be.

Chains are sampled independently, so step j of chain A has no correspondence to
step j of chain B and the vectors cannot be compared elementwise. What IS
comparable is the normalised position of each chain's weakest step:

    locus(i) = argmin(step_scores_i) / (len(step_scores_i) - 1)   in [0, 1]

Two chains with nearby loci fail in the same region of the derivation and are
penalised as redundant; distant loci are left unpenalised. This is a genuine
set-level property -- invisible to any per-chain score, and therefore something
voting structurally cannot use.

USAGE
-----
    python scripts/qubo_prm_experiment.py \\
        --chains results/cached_chains_math500_300q.json \\
        --prm-cache results/prm_math500_300q.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Shared helpers (kept local so this runs in the minimal PRM venv) ──────────

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


def mcnemar_p(a_ok: list[bool], b_ok: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar: is method A better than method B on paired data?"""
    from math import comb
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if not x and y)
    tot = b + c
    if tot == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(2 * sum(comb(tot, i) for i in range(k + 1)) / (2 ** tot), 1.0)


# ── Vote rules ───────────────────────────────────────────────────────────────

def vote(rows: list[dict], rule: str, key: str = "prm") -> Optional[float]:
    """Collapse a set of chains into one answer under the named rule."""
    if not rows:
        return None
    by_ans = defaultdict(list)
    for r in rows:
        by_ans[r["ans"]].append(r[key])
    if rule == "count":
        return max(by_ans.items(), key=lambda kv: len(kv[1]))[0]
    if rule == "sum":
        return max(by_ans.items(), key=lambda kv: sum(kv[1]))[0]
    if rule.startswith("pow"):
        p = float(rule[3:])
        return max(by_ans.items(), key=lambda kv: sum(v ** p for v in kv[1]))[0]
    raise ValueError(f"unknown vote rule {rule!r}")


# ── Failure locus ────────────────────────────────────────────────────────────

def failure_locus(step_scores: list[float]) -> float:
    """Normalised position of a chain's weakest step, in [0, 1].

    0.0 means it breaks down immediately, 1.0 at the very end. Chains with a
    single step (or none) return 0.5 -- neutral, since there is no positional
    information to extract and forcing them to one end would fabricate signal.
    """
    if not step_scores or len(step_scores) < 2:
        return 0.5
    return float(int(np.argmin(step_scores)) / (len(step_scores) - 1))


# ── QUBO construction and solve ──────────────────────────────────────────────

def build_qubo(rows, k, diag_w, penalty_w, agree_w, locus_w, card_w):
    """Q for one question. Lower energy = better subset.

    diagonal      -quality + cardinality offset
    off-diagonal  redundancy: answer agreement, failure-locus proximity,
                  plus the cardinality cross term
    """
    n = len(rows)
    Q = np.zeros((n, n))

    for i in range(n):
        Q[i][i] = -diag_w * rows[i]["prm"] + card_w * (1 - 2 * k)

    for i in range(n):
        for j in range(i + 1, n):
            agree = 1.0 if rows[i]["ans"] == rows[j]["ans"] else 0.0
            # Loci close together => same failure region => redundant.
            locus_sim = 1.0 - abs(rows[i]["locus"] - rows[j]["locus"])
            pen = penalty_w * (agree_w * agree + locus_w * locus_sim) + 2 * card_w
            Q[i][j] = Q[j][i] = pen
    return Q


def solve_qubo(Q, n_reads=60, iters=250, seed=0):
    """Simulated annealing, vectorised over parallel reads.

    Self-contained rather than importing pipeline.solver, so this script runs in
    the minimal PRM venv (which has no sentence-transformers).
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    states = rng.integers(0, 2, size=(n_reads, n)).astype(np.float64)
    energies = np.einsum("ri,ij,rj->r", states, Q, states)

    temp0, temp1 = 10.0, 0.01
    for step in range(iters):
        T = temp0 * (temp1 / temp0) ** (step / max(iters - 1, 1))
        idx = rng.integers(0, n, size=n_reads)
        flipped = states.copy()
        flipped[np.arange(n_reads), idx] = 1.0 - flipped[np.arange(n_reads), idx]
        new_e = np.einsum("ri,ij,rj->r", flipped, Q, flipped)
        delta = new_e - energies
        accept = (delta < 0) | (rng.random(n_reads) < np.exp(-delta / max(T, 1e-9)))
        states[accept] = flipped[accept]
        energies[accept] = new_e[accept]

    best = int(np.argmin(energies))
    return states[best]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--prm-cache", required=True)
    ap.add_argument("--k", type=int, default=6, help="Target subset size.")
    ap.add_argument("--vote-rule", default="pow8",
                    help="How the selected subset is collapsed to an answer.")
    ap.add_argument("--diag-w", type=float, default=1.0)
    ap.add_argument("--penalty-w", type=float, default=0.5)
    ap.add_argument("--agree-w", type=float, default=0.5)
    ap.add_argument("--locus-w", type=float, default=0.5)
    ap.add_argument("--card-w", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true",
                    help="Run the controls that separate the two possible causes of a "
                         "QUBO loss: INFORMATION LOSS from voting over a subset at all, "
                         "versus BAD SELECTION of which chains to keep. Compares QUBO-k "
                         "against random-k and greedy-top-k at several k, all collapsed "
                         "with the same vote rule.")
    args = ap.parse_args()

    data = json.load(open(args.chains, encoding="utf-8"))
    chains_all = data["chains"]
    golds = data["golds"]
    dataset = data.get("dataset", "unknown")

    prm_blob = json.load(open(args.prm_cache, encoding="utf-8"))
    prm_scores = prm_blob["scores"]
    step_scores = prm_blob.get("step_scores")
    if step_scores is None:
        print("[error] this PRM cache has no per-step vectors, so the "
              "failure-locus term cannot be computed.\n"
              "        Re-run prm_experiment.py to regenerate it (a few minutes):\n"
              f"        rm {args.prm_cache} && python scripts/prm_experiment.py "
              f"--chains {args.chains} --prm-cache {args.prm_cache}",
              file=sys.stderr)
        sys.exit(1)

    n = min(len(chains_all), len(golds))
    print(f"[data] {dataset}: {n} questions x {len(chains_all[0])} chains")
    print(f"[qubo] k={args.k}  diag={args.diag_w}  penalty={args.penalty_w} "
          f"(agree={args.agree_w}, locus={args.locus_w})  card={args.card_w}")
    print(f"[qubo] subset collapsed with vote rule: {args.vote_rule}\n")

    outcomes = {name: [] for name in
                ("plain vote (all)", "PRM-weighted (all)", "QUBO+PRM subset",
                 "QUBO no-locus subset", "ORACLE")}
    sizes = []

    for qi in range(n):
        gold = golds[qi]
        rows = []
        for c, p, steps in zip(chains_all[qi], prm_scores[qi], step_scores[qi]):
            a = chain_answer(c)
            if a is None:
                continue
            rows.append({"ans": a, "prm": float(p), "locus": failure_locus(steps)})
        if not rows:
            for v in outcomes.values():
                v.append(False)
            continue

        outcomes["plain vote (all)"].append(
            numeric_match(vote(rows, "count"), gold))
        outcomes["PRM-weighted (all)"].append(
            numeric_match(vote(rows, args.vote_rule), gold))
        outcomes["ORACLE"].append(any(numeric_match(r["ans"], gold) for r in rows))

        # With the failure-locus complementarity term
        Q = build_qubo(rows, args.k, args.diag_w, args.penalty_w,
                       args.agree_w, args.locus_w, args.card_w)
        state = solve_qubo(Q, seed=args.seed + qi)
        picked = [rows[i] for i in range(len(rows)) if state[i] == 1] or rows
        sizes.append(len(picked))
        outcomes["QUBO+PRM subset"].append(
            numeric_match(vote(picked, args.vote_rule), gold))

        # Ablation: identical objective with the locus term switched off, so any
        # difference is attributable to that term alone.
        Q0 = build_qubo(rows, args.k, args.diag_w, args.penalty_w,
                        args.agree_w, 0.0, args.card_w)
        state0 = solve_qubo(Q0, seed=args.seed + qi)
        picked0 = [rows[i] for i in range(len(rows)) if state0[i] == 1] or rows
        outcomes["QUBO no-locus subset"].append(
            numeric_match(vote(picked0, args.vote_rule), gold))

        if (qi + 1) % 50 == 0:
            print(f"  ... {qi+1}/{n}", flush=True)

    print(f"\n[qubo] mean selected subset size: {np.mean(sizes):.2f} (target {args.k})")

    print("\n" + "=" * 70)
    print(f"  SET SELECTION vs INDEPENDENT WEIGHTING  ({dataset}, n={n})")
    print("=" * 70)
    base = outcomes["PRM-weighted (all)"]
    print(f"  {'method':<26}{'acc':>8}{'vs PRM-wtd':>12}{'b':>4}{'c':>4}{'p':>9}")
    print("  " + "-" * 62)
    for name, oks in outcomes.items():
        acc = sum(oks) / len(oks)
        if name == "PRM-weighted (all)":
            print(f"  {name:<26}{acc:>7.1%}{'  <- baseline':>12}")
            continue
        b, c, p = mcnemar_p(oks, base)
        d = (acc - sum(base) / len(base)) * 100
        print(f"  {name:<26}{acc:>7.1%}{d:>+11.1f}{b:>4}{c:>4}{p:>9.4f}")
    print("  " + "-" * 62)
    print("\n  The claim under test is whether QUBO+PRM beats PRM-weighted voting.")
    print("  Equal accuracy means set-selection adds nothing over independent")
    print("  weighting here, however good the underlying signal is.")
    print("  QUBO+PRM vs QUBO no-locus isolates the failure-locus term itself.")

    if args.diagnose:
        diagnose_subset_size(chains_all, golds, prm_scores, step_scores, args, n)


def diagnose_subset_size(chains_all, golds, prm_scores, step_scores, args, n):
    """Separate INFORMATION LOSS from BAD SELECTION.

    A QUBO subset losing to weighting over the whole pool has two very different
    possible causes, and they call for opposite responses:

      INFORMATION LOSS -- voting over k chains is simply worse than voting over
        all 20, regardless of which k are chosen, because a vote is an estimate
        and estimates sharpen with sample size. If so, RANDOM-k performs about
        as well as QUBO-k, accuracy climbs monotonically with k, and the whole
        idea of selecting a subset before aggregating is misconceived for this
        task -- no better objective can rescue it.

      BAD SELECTION -- subsetting is fine but the QUBO picks the wrong chains.
        If so, QUBO-k sits clearly above RANDOM-k, and greedy-top-k by PRM is a
        useful reference for whether the combinatorial machinery earns its cost
        over simple ranking.

    The two are confounded in a single k=6 number, which is why they are pulled
    apart here rather than guessed at.
    """
    rng = np.random.default_rng(args.seed)
    ks = [2, 3, 6, 10, 15, 20]

    rows_per_q = []
    for qi in range(n):
        rows = []
        for c, p, steps in zip(chains_all[qi], prm_scores[qi], step_scores[qi]):
            a = chain_answer(c)
            if a is None:
                continue
            rows.append({"ans": a, "prm": float(p), "locus": failure_locus(steps)})
        rows_per_q.append(rows)

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC: is the loss information loss, or bad selection?")
    print("=" * 70)
    print(f"  {'k':>3}  {'QUBO-k':>9}{'greedy-k':>11}{'random-k':>11}   "
          f"{'QUBO vs random':>15}")
    print("  " + "-" * 62)

    for k in ks:
        acc = {"qubo": 0, "greedy": 0, "random": 0}
        for qi, rows in enumerate(rows_per_q):
            if not rows:
                continue
            gold = golds[qi]

            # QUBO-selected k
            Q = build_qubo(rows, k, args.diag_w, args.penalty_w,
                           args.agree_w, args.locus_w, args.card_w)
            st = solve_qubo(Q, seed=args.seed + qi)
            picked = [rows[i] for i in range(len(rows)) if st[i] == 1] or rows
            acc["qubo"] += numeric_match(vote(picked, args.vote_rule), gold)

            # Greedy top-k by PRM -- ranking, no combinatorial structure
            greedy = sorted(rows, key=lambda r: -r["prm"])[:k]
            acc["greedy"] += numeric_match(vote(greedy, args.vote_rule), gold)

            # Random k -- isolates the cost of subsetting itself
            idx = rng.choice(len(rows), size=min(k, len(rows)), replace=False)
            rand = [rows[i] for i in idx]
            acc["random"] += numeric_match(vote(rand, args.vote_rule), gold)

        q, g, r = (acc["qubo"] / n, acc["greedy"] / n, acc["random"] / n)
        print(f"  {k:>3}  {q:>8.1%}{g:>10.1%}{r:>10.1%}   {(q - r) * 100:>+14.1f}")

    print("  " + "-" * 62)
    print("  READ:")
    print("    accuracy rising steadily with k, and QUBO ~ random at every k")
    print("      -> INFORMATION LOSS. Subsetting before aggregating is the wrong")
    print("         operation for this task; a better objective cannot fix it.")
    print("    QUBO clearly above random at small k")
    print("      -> selection works; compare against greedy to see whether the")
    print("         combinatorial objective earns its cost over plain ranking.")


if __name__ == "__main__":
    main()
