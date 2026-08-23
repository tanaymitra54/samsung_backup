"""
==============================================================================
FILE: scripts/compare_curation_arms.py
ROLE: Held-out, paired comparison of SFT adapters trained on differently
      curated data (qubo / greedy / random)
==============================================================================

WHY THIS EXISTS
---------------
The three-arm curation experiment asks one question: does QUBO subset
selection produce better SFT data than greedy or random selection? Answering
it needs two things the per-epoch training probe cannot give:

1. ENOUGH QUESTIONS. The in-training probe ran on 40 GSM8K items. At n=40 a
   single question is 2.5 points and the 95% interval around 25% is roughly
   [14%, 40%] -- wide enough to swallow the entire spread between the three
   arms. Detecting a true 25%->35% difference at 80% power needs ~330
   questions per arm.

2. A PAIRED TEST. All arms answer the SAME questions, so the arms are not
   independent samples and comparing two independent proportions throws away
   the pairing. McNemar's exact test conditions on the discordant questions
   -- the ones where the two arms disagree -- which is where all the
   information about which arm is better actually lives.

This script runs each adapter over one shared held-out set via
run_all_benchmarks.py, then compares every pair of arms with exact McNemar on
the per-question outcomes.

USAGE
-----
    python scripts/compare_curation_arms.py \
        --arms qubo greedy random \
        --benchmark gsm8k --n 300

    # adapters already evaluated -- just redo the statistics
    python scripts/compare_curation_arms.py --skip-eval

Adapters default to checkpoints/curate-<arm>/final_adapter, matching the
--run-name convention used when training the arms.
"""

import argparse
import json
import os
import subprocess
import sys
from itertools import combinations
from math import comb, sqrt
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Statistics ───────────────────────────────────────────────────────────────

def wilson95(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Unlike the normal approximation it stays inside
    [0, 1] at small n and near the extremes, which is exactly where a
    few-hundred-question accuracy lives."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test between two paired outcome vectors.

    b = A right where B wrong; c = A wrong where B right. Questions both arms
    get right, or both get wrong, carry no information about which is better
    and drop out. Under the null each discordant question is a fair coin, so
    the exact binomial tail is the p-value -- no normal approximation, which
    matters because discordant counts here are often small.
    """
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    tot = b + c
    if tot == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(tot, i) for i in range(k + 1)) / (2 ** tot)
    return b, c, min(p, 1.0)


def min_detectable_effect(n: int, p_base: float = 0.30) -> float:
    """Roughly the smallest accuracy gap this n could resolve at 80% power.

    Printed so a null result is read as 'could not resolve' rather than 'no
    difference' when n was too small to tell those apart. This is the
    two-independent-proportions approximation, which is CONSERVATIVE here:
    McNemar exploits the pairing and so has somewhat more power than this
    number implies. Treat it as an upper bound on the resolvable gap.
    """
    if n <= 0:
        return 1.0
    return (1.96 + 0.84) * sqrt(2 * p_base * (1 - p_base)) / sqrt(n)


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_arm(arm: str, adapter: Path, out_dir: Path, benchmark: str, n: int,
            seed: int, device: str | None, extra: list[str]) -> None:
    """Evaluate one adapter with run_all_benchmarks.py, labelled by arm."""
    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "run_all_benchmarks.py"),
        "--output-dir", str(out_dir),
        "--condition-label", arm,
        "--benchmarks", benchmark,
        "--subset-size", str(n),
        "--seed", str(seed),          # same seed for every arm -> same questions
    ]
    if device:
        cmd += ["--device", device]
    cmd += extra

    env = dict(os.environ)
    env["QUBO_ADAPTER_PATH"] = str(adapter)

    print(f"\n{'=' * 70}\n[{arm}] adapter: {adapter}\n[{arm}] {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
    if result.returncode != 0:
        sys.exit(f"[ERROR] Arm '{arm}' failed with exit code {result.returncode}")


def load_outcomes(out_dir: Path, arm: str, benchmark: str, mode: str) -> dict[int, bool]:
    """Per-question correctness for one arm, keyed by question id so the arms
    can be aligned. Missing file -> empty, reported by the caller."""
    path = out_dir / f"{arm}_{benchmark}_results.jsonl"
    if not path.exists():
        return {}
    outcomes = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = row.get(f"correct_{mode}")
            if val is not None and "id" in row:
                outcomes[row["id"]] = bool(val)
    return outcomes


# ── Reporting ────────────────────────────────────────────────────────────────

def report(out_dir: Path, arms: list[str], benchmark: str, mode: str) -> None:
    per_arm = {a: load_outcomes(out_dir, a, benchmark, mode) for a in arms}

    missing = [a for a, o in per_arm.items() if not o]
    if missing:
        print(f"\n[WARN] No results found for: {', '.join(missing)}")
        print(f"       Expected {out_dir}/<arm>_{benchmark}_results.jsonl")
    present = [a for a in arms if per_arm[a]]
    if len(present) < 2:
        print("[ERROR] Need at least two arms with results to compare.")
        return

    # Compare only questions every arm actually answered, so all pairs are
    # scored on identical items and the accuracies below are directly
    # comparable to each other.
    shared = set.intersection(*(set(per_arm[a]) for a in present))
    if not shared:
        print("[ERROR] Arms share no question ids -- were they run with the same --seed and --n?")
        return
    ids = sorted(shared)
    n = len(ids)

    dropped = {a: len(per_arm[a]) - n for a in present}
    if any(dropped.values()):
        print(f"\n[note] Restricted to the {n} questions common to all arms "
              f"(dropped per arm: {dropped}).")

    print("\n" + "=" * 70)
    print(f"  HELD-OUT ACCURACY -- {benchmark}, decode mode '{mode}', n={n}")
    print("=" * 70)
    print(f"  {'arm':<10}{'acc':>8}{'correct':>10}   95% CI")
    print("  " + "-" * 52)
    for a in sorted(present, key=lambda x: -sum(per_arm[x][i] for i in ids)):
        k = sum(per_arm[a][i] for i in ids)
        lo, hi = wilson95(k, n)
        print(f"  {a:<10}{k / n:>7.1%}{k:>7}/{n:<4}  [{lo:.1%}, {hi:.1%}]")

    print("\n" + "=" * 70)
    print("  PAIRWISE EXACT McNEMAR  (paired -- same questions per arm)")
    print("=" * 70)
    print(f"  {'comparison':<22}{'delta':>8}{'b':>5}{'c':>5}{'p':>9}  verdict")
    print("  " + "-" * 60)
    for a, b_arm in combinations(present, 2):
        va = [per_arm[a][i] for i in ids]
        vb = [per_arm[b_arm][i] for i in ids]
        delta = (sum(va) - sum(vb)) / n
        bb, cc, p = mcnemar_exact(va, vb)
        if p < 0.01:
            verdict = "SIGNIFICANT**"
        elif p < 0.05:
            verdict = "significant*"
        else:
            verdict = "not resolved"
        print(f"  {a + ' vs ' + b_arm:<22}{delta:>+7.1%}{bb:>5}{cc:>5}{p:>9.3f}  {verdict}")

    mde = min_detectable_effect(n)
    print("\n  b = first arm right where second wrong; c = the reverse.")
    print(f"  At n={n} this design resolves gaps of roughly {mde:.1%} or larger "
          f"(80% power).")
    print("  'not resolved' means the data cannot separate the arms -- it is NOT")
    print("  evidence that they perform the same.")
    print(f"\n  Raw per-question results: {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Held-out paired comparison of curation-arm SFT adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--arms", nargs="+", default=["qubo", "greedy", "random"],
                    help="Arm names (default: qubo greedy random)")
    ap.add_argument("--adapter-root", default="checkpoints",
                    help="Adapters are <root>/curate-<arm>/final_adapter (default: checkpoints)")
    ap.add_argument("--adapter-pattern", default="curate-{arm}/final_adapter",
                    help="Adapter path pattern relative to --adapter-root")
    ap.add_argument("--benchmark", default="gsm8k", help="Benchmark to evaluate on")
    ap.add_argument("--n", type=int, default=300,
                    help="Questions per arm (default: 300). Below ~300 the design "
                         "cannot resolve the ~5-10 point differences at stake here.")
    ap.add_argument("--mode", default="greedy", choices=["greedy", "cot", "qubo"],
                    help="Which decode mode's correctness to compare (default: greedy). "
                         "Use 'greedy' to isolate what fine-tuning changed, without the "
                         "inference-time QUBO pipeline on top of it.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shared across arms so every arm sees the same questions")
    ap.add_argument("--output-dir", default="results/curation_compare")
    ap.add_argument("--device", default=None,
                    help="cuda:N / cpu. Default: auto-pick the GPU with the most free VRAM.")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip evaluation; just recompute statistics from existing results")
    ap.add_argument("--eval-arg", action="append", default=[],
                    help="Extra flag passed through to run_all_benchmarks.py (repeatable)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_eval:
        if args.n < 300:
            print(f"[warn] --n {args.n} is below the ~300 needed to resolve the "
                  f"differences this experiment is testing for.")
        for arm in args.arms:
            adapter = Path(args.adapter_root) / args.adapter_pattern.format(arm=arm)
            if not adapter.exists():
                sys.exit(f"[ERROR] Adapter not found for arm '{arm}': {adapter}")
        for arm in args.arms:
            adapter = Path(args.adapter_root) / args.adapter_pattern.format(arm=arm)
            run_arm(arm, adapter, out_dir, args.benchmark, args.n,
                    args.seed, args.device, args.eval_arg)

    report(out_dir, args.arms, args.benchmark, args.mode)


if __name__ == "__main__":
    main()
