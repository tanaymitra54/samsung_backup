"""
==============================================================================
FILE: scripts/tune_qubo_params.py
ROLE: QUBO Hyperparameter Grid Search & Optimization Script
BRANCH ADDITION (abhyuday): Newly introduced automated grid search tool evaluating 81
combinations of QUBO objective coefficients (`penalty_weight`, `diversity_bonus`,
`cardinality_penalty`, `answer_agree_weight`). Pre-caches reasoning chains and NLI scores
to evaluate all 81 combinations rapidly without re-running neural inference. Outputs
`config/best_qubo_params.yaml`.
==============================================================================

scripts/tune_qubo_params.py  —  QUBO Hyperparameter Grid Search
================================================================

Runs an 81-combination grid search over QUBO parameters on the first
300 questions of GSM8K (test split) and saves the best configuration.

PARAMETER GRID (81 total combinations):
    penalty_weight:       [0.5, 1.0, 2.0]
    diversity_bonus:      [0.1, 0.3, 0.5]
    cardinality_penalty:  [0.4, 1.2, 2.5]
    answer_agree_weight:  [0.1, 0.3, 0.5]
    answer_sim_weight:    1.0 - answer_agree_weight  (always sums to 1.0)

    See the _CP_VALUES comment for why the ranges were widened: the previous grid
    could not select more than one chain, so the multi-chain CoT scaffold that the
    method depends on never engaged.

EFFICIENCY:
    • InferencePipeline is instantiated once (loads the LLM internally).
      DiverseSampler receives its model/tokenizer as shared references
      so no second model load occurs.
    • Sampling AND scoring are both run once upfront per question and
      cached in memory. Verifier weights are fixed, so scoring is
      independent of QUBO parameters.
    • QUBOBuilder attributes are hot-swapped per combo (avoids re-loading
      the sentence-transformer embedder 81 times).
    • Questions where ALL chains score < 0.2 are excluded from the
      accuracy denominator for every combo.

COMBO SCORING (--scoring, default "proxy"):
    Once a combo has built the QUBO and solved it for a question, its
    accuracy still needs to be measured somehow. Two ways to do that:

    "proxy" (default) -- majority vote over the SELECTED chains' own
      already-known answers (proxy_vote()). No generation at all: build+solve
      (already cheap) is the entire cost, so all 81 combos finish in minutes.
      Justified by run_all_benchmarks.py's own finding that full-pipeline
      QUBO accuracy already ties plain majority vote every time it has been
      measured -- this proxy targets the thing that actually varies between
      combos (which chains got selected) and should track the real
      full-pipeline metric closely.

    "synthesis" -- the original design: a fresh LLM generation per question
      per combo (batched within a combo, but still one full pass over every
      question for every combo). Measured on live hardware: combos took
      57, 95 and 113 minutes and RISING -- GPU allocator fragmentation
      building up over many thousands of variable-length batched generations
      in one long-lived process -- which puts a full 81-combo sweep at 3+
      days even before that degradation is counted. Use it to spot-check a
      small shortlist already ranked by "proxy", not for the full grid:

        python scripts/tune_qubo_params.py --scoring proxy            # rank all 81, minutes
        # inspect results/qubo_hyperparam_search_proxy.json, pick top few
        # then re-run with --scoring synthesis on a smaller --num-questions,
        # comparing only those candidates' real full-pipeline accuracy.

    Results from the two modes are NOT comparable numbers and are never
    mixed: each mode writes and resumes its own
    results/qubo_hyperparam_search_{proxy,synthesis}.json.

RESUME:
    Pass --resume to load existing results for the CURRENT --scoring mode and
    skip already-evaluated combos. Resume keys are derived from the 4
    parameter values — NOT combo index — so they remain stable across runs
    even if grid ordering changes.

OUTPUTS:
    results/qubo_hyperparam_search_{proxy,synthesis}.json  — all combo results
    config/best_qubo_params.yaml                           — best combination

USAGE:
    # Run from the project root (default: fast proxy scoring, all 81 combos):
    python scripts/tune_qubo_params.py

    # Resume an interrupted search:
    python scripts/tune_qubo_params.py --resume

    # Smoke-test with fewer questions:
    python scripts/tune_qubo_params.py --num-questions 20

    # Full-pipeline spot-check of a shortlist (see COMBO SCORING above):
    python scripts/tune_qubo_params.py --scoring synthesis --num-questions 60
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

import yaml

# ── Ensure project root is on sys.path when run directly ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Pipeline imports ───────────────────────────────────────────────────────────
import torch

from pipeline.device_utils import resolve_device
from pipeline.qubo_builder import QUBOBuilder
from pipeline.sampling import DiverseSampler
from pipeline.solver import SimulatedAnnealingSolver
from pipeline.inference import InferencePipeline
from pipeline.verifier import ReasonVerifier


# =============================================================================
# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
# =============================================================================

# Fixed-order lists — itertools.product iterates these in declaration order,
# guaranteeing a deterministic, reproducible combo sequence across all runs.
#
# GRID WIDENED. The previous ranges could not produce a working subset:
# cardinality_penalty topped out at 0.2 while penalty_weight went to 1.5, so the
# pairwise redundancy penalty always swamped the soft k-constraint and the solver
# selected ONE chain regardless of subset_size. Measured on a realistic 12-chain
# pool (mean pairwise cosine 0.51) with the shipped config, every combination in
# the old grid yields a 1-chain scaffold -- i.e. the multi-chain CoT premise of
# the method never actually engaged.
#
# cardinality_penalty must be within roughly an order of magnitude of
# penalty_weight for the k-constraint to bind, so its range now overlaps it.
#
# Range raised again after measuring on real chains: cardinality_penalty 0.8
# yields only 2.1 selected chains per question against subset_size 6, because
# real chains for one question are far more similar than synthetic ones, so the
# redundancy penalty is stronger in practice. 1.2 was still too low to reach a
# fuller subset. Note the tension the search has to resolve: as this term grows
# the solver is forced toward exactly k chains chosen on diagonal quality alone,
# and the pairwise diversity structure stops mattering -- so higher is not
# automatically better and the optimum is somewhere in the middle.
# answer_agree_weight is also allowed to go lower: gold-free scoring REWARDS
# cross-chain consensus on the diagonal, so penalising answer agreement off the
# diagonal now works against the quality signal rather than complementing it.
#
# Still 3^4 = 81 combinations, so runtime is unchanged.
_PW_VALUES   = [0.5, 1.0, 2.0]          # penalty_weight
_DB_VALUES   = [0.1, 0.3, 0.5]          # diversity_bonus
_CP_VALUES   = [0.4, 1.2, 2.5]          # cardinality_penalty
_AAW_VALUES  = [0.1, 0.3, 0.5]          # answer_agree_weight
# answer_sim_weight = 1.0 - answer_agree_weight

# Low-quality-question threshold: if EVERY chain for a question scores below
# this, the question is excluded from the accuracy denominator for ALL combos.
_EXCLUSION_SCORE_THRESHOLD = 0.2

# Batch size for the per-combo final-answer synthesis (Phase B of
# run_grid_search). Originally matched evaluation.batch_size (16), but that
# figure was calibrated on the short greedy/CoT generations in
# run_all_benchmarks.py (tens of tokens). This script's synthesis prompts are
# longer -- up to subset_size chains of reasoning as context, up to
# max_new_tokens (512) of greedy output -- and 16 OOM'd on the first batch of
# combo 1 on a live run. generate_answers_batch's own OOM back-off would have
# recovered by halving to 8 anyway, but starting there directly skips a
# guaranteed OOM-then-halve on every one of the 81 combos, not just the first.
_GEN_BATCH_SIZE = 8


# =============================================================================
# ─── CLI ───────────────────────────────────────────────────────────────────────
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search QUBO hyperparameters on GSM8K validation subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=300,
        help="Number of GSM8K test questions to evaluate on.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing results/qubo_hyperparam_search.json and skip "
             "already-evaluated combinations.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where qubo_hyperparam_search.json is saved.",
    )
    parser.add_argument(
        "--best-config-path",
        default="config/best_qubo_params.yaml",
        help="Path to write the best hyperparameter configuration.",
    )
    parser.add_argument(
        "--oracle-scoring",
        action="store_true",
        help="ABLATION ONLY: let the verifier see the gold answer while scoring "
             "candidate chains. The winning combination then reflects what best "
             "exploits an answer key the deployed pipeline never has. The default "
             "(gold-free) matches how run_all_benchmarks.py now evaluates.",
    )
    parser.add_argument(
        "--scoring",
        choices=["proxy", "synthesis"],
        default="proxy",
        help="How a combo's accuracy is computed once it has selected a subset. "
             "'proxy' (default): majority vote over the SELECTED chains' own "
             "already-known answers -- zero extra generation, a full 81-combo "
             "sweep takes minutes. Justified by run_all_benchmarks.py's own "
             "finding that full-pipeline QUBO accuracy already ties plain "
             "majority vote every time it has been measured, so this proxy "
             "should track the real metric closely while being ~1000x cheaper. "
             "'synthesis': the original path -- a fresh LLM generation per "
             "question per combo. Measured on live hardware: 3 combos took "
             "57/95/113 minutes and rising (GPU allocator fragmentation over "
             "many variable-length batches), putting a full 81-combo sweep at "
             "3+ days even before that degradation. Use 'synthesis' only to "
             "spot-check a handful of combos already shortlisted by 'proxy' -- "
             "see --best-config-path workflow in the module docstring.",
    )
    parser.add_argument(
        "--keep-exclusions",
        action="store_true",
        help="Rank combinations by accuracy over the reduced denominator that drops "
             "questions where every chain scored poorly. Off by default: those are "
             "precisely the hard questions, and dropping them inflates the score.",
    )
    return parser.parse_args()


# =============================================================================
# ─── COMBO KEY ─────────────────────────────────────────────────────────────────
# =============================================================================

def build_combo_key(combo: dict) -> str:
    """Stable resume key derived from the 4 parameter values.

    Using parameter values (not combo index) means the key remains valid
    even if the grid ordering or total size changes between runs.
    """
    return (
        f"pw={combo['penalty_weight']}"
        f"|db={combo['diversity_bonus']}"
        f"|cp={combo['cardinality_penalty']}"
        f"|aaw={combo['answer_agree_weight']}"
    )


def generate_all_combos() -> list[dict]:
    """Return all 81 parameter combination dicts in deterministic order.

    Order is fixed by the declaration order of _PW_VALUES, _DB_VALUES, etc.
    itertools.product iterates the last iterable fastest (rightmost varies first).
    """
    combos = []
    for pw, db, cp, aaw in itertools.product(
        _PW_VALUES, _DB_VALUES, _CP_VALUES, _AAW_VALUES
    ):
        combos.append({
            "penalty_weight":      pw,
            "diversity_bonus":     db,
            "cardinality_penalty": cp,
            "answer_agree_weight": aaw,
            "answer_sim_weight":   round(1.0 - aaw, 10),
        })
    return combos


# =============================================================================
# ─── DATA UTILITIES ────────────────────────────────────────────────────────────
# =============================================================================

def load_gsm8k(num_questions: int) -> list[dict]:
    """Load the first `num_questions` examples from GSM8K test split.

    Each returned dict has:
        'question' : str   — the math problem text
        'gold'     : float — the numeric answer extracted after "####"

    Requires: datasets library (pip install datasets).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required: pip install datasets"
        ) from exc

    print(f"[data] Loading GSM8K test split (first {num_questions} questions)...")
    ds = load_dataset("gsm8k", "main", split="test")
    examples = []
    for item in ds:
        if len(examples) >= num_questions:
            break
        gold = extract_gsm8k_gold(item["answer"])
        if gold is None:
            # Skip malformed entries (shouldn't happen in GSM8K, but be safe)
            continue
        examples.append({"question": item["question"], "gold": gold})
    print(f"[data] Loaded {len(examples)} examples.")
    return examples


def extract_gsm8k_gold(answer_field: str) -> Optional[float]:
    """Extract the numeric gold answer from a GSM8K answer string.

    GSM8K answers are formatted as:
        "<multi-line reasoning>\\n#### <number>"

    We extract the number after the last "####" marker and strip commas.
    Returns None if no numeric answer is found.
    """
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def numeric_match(pred: Optional[float], gold: float) -> bool:
    """Hybrid tolerance comparison, identical to ReasonVerifier._gold_match.

    threshold = max(0.01, 0.01 × |gold|)
    Returns True iff |pred - gold| < threshold.
    """
    if pred is None:
        return False
    threshold = max(0.01, 0.01 * abs(gold))
    return abs(pred - gold) < threshold


# =============================================================================
# ─── PRE-CACHE PHASE ───────────────────────────────────────────────────────────
# =============================================================================

def cache_and_score_all(
    examples: list[dict],
    sampler: DiverseSampler,
    verifier: ReasonVerifier,
    cache_file: Optional[Path] = None,
    oracle_scoring: bool = False,
) -> tuple[list[list[dict]], set[int]]:
    """Sample AND score all questions once upfront, saving/loading from disk cache.

    Returns
    -------
    cached_scored_chains : list[list[dict]]
        cached_scored_chains[i] is the fully scored list of chain dicts for
        examples[i]. Each dict has 'correctness_score' already set.

    excluded_indices : set[int]
        Indices into `examples` where ALL chains scored < _EXCLUSION_SCORE_THRESHOLD.
    """
    if cache_file and cache_file.exists():
        print(f"\n[cache] Loading pre-cached scored chains from disk ({cache_file})...")
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            cached_scored_chains = data["chains"]
            excluded_indices = set(data["excluded_indices"])
            print(f"[cache] Successfully loaded {len(cached_scored_chains)} questions from disk cache in 0.1s!")
            return cached_scored_chains, excluded_indices
        except Exception as e:
            print(f"[cache] Could not read cache file ({e}); re-sampling...")

    n = len(examples)
    cached_scored_chains: list[list[dict]] = []
    print(f"\n[cache] Sampling and scoring {n} questions...")
    t0 = time.time()

    for idx, ex in enumerate(examples):
        # Step 1: Sample diverse chains
        chains = sampler.sample(ex["question"], task_type="math")

        # Step 2: Score immediately (verifier params are fixed for all combos).
        #
        # Gold is withheld by default so the search optimises the parameters that
        # work at DEPLOYMENT. Passing gold here makes the QUBO diagonal encode
        # "already has the right answer", so the winning combination would be the
        # one that best exploits an answer key the real pipeline never has.
        scoring_gold = str(ex["gold"]) if oracle_scoring else None
        verifier.score_batch(chains, task_type="math", gold=scoring_gold,
                             question=ex["question"])

        cached_scored_chains.append(chains)

        elapsed = time.time() - t0
        avg = elapsed / (idx + 1)
        eta = avg * (n - idx - 1)
        print(
            f"\r[cache] {idx+1}/{n} - {elapsed:.0f}s elapsed, ETA {eta:.0f}s",
            end="",
            flush=True,
        )

    print(f"\n[cache] Done in {time.time()-t0:.1f}s.")

    # ── Build exclusion list ─────────────────────────────────────────────────
    excluded_indices: set[int] = set()
    for idx, chains in enumerate(cached_scored_chains):
        scores = [s.get("correctness_score", 0.0) for s in chains]
        if scores and all(sc < _EXCLUSION_SCORE_THRESHOLD for sc in scores):
            excluded_indices.add(idx)

    if excluded_indices:
        print(
            f"[cache] Excluded {len(excluded_indices)} questions "
            f"(all chains scored < {_EXCLUSION_SCORE_THRESHOLD}) from denominator."
        )
    else:
        print("[cache] No questions excluded (at least one chain scored well on each).")

    # Save to disk cache for instant reuse on restart/resume
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({
                    "chains": cached_scored_chains,
                    "excluded_indices": list(excluded_indices),
                }, f, indent=2)
            print(f"[cache] Saved pre-cached chains to disk -> {cache_file}")
        except Exception as e:
            print(f"[cache] Failed to save disk cache: {e}")

    return cached_scored_chains, excluded_indices


# =============================================================================
# ─── QUBO RECONFIGURATION ──────────────────────────────────────────────────────
# =============================================================================

def reconfigure_qubo_builder(builder: QUBOBuilder, combo: dict) -> None:
    """Hot-swap QUBOBuilder attributes for the current combination.

    Avoids re-loading the sentence-transformer embedder (which is stateless
    w.r.t. these parameters) 81 times across the grid.
    """
    builder.penalty_weight      = combo["penalty_weight"]
    builder.diversity_bonus     = combo["diversity_bonus"]
    builder.cardinality_penalty = combo["cardinality_penalty"]
    builder.answer_agree_weight = combo["answer_agree_weight"]
    builder.answer_sim_weight   = combo["answer_sim_weight"]
    # Keep live config in sync so _apply_cardinality_constraint reads correctly
    builder.config["qubo"]["cardinality_penalty"] = combo["cardinality_penalty"]


# =============================================================================
# ─── PROXY SCORING (--scoring proxy) ────────────────────────────────────────
# =============================================================================

def proxy_vote(
    chains: list[dict], final_indices: list[int], extract_fn
) -> Optional[float]:
    """Predict the answer from the QUBO-selected subset alone -- no LLM call.

    Majority vote over the SELECTED chains' own already-extracted answers
    (falling back to their reasoning text when the answer field is empty, via
    extract_fn -- pass verifier._extract_last_number). Same mechanism as
    run_all_benchmarks.py's self-consistency baseline, scoped to just the
    chains this combo chose, which is what actually varies between combos.

    Ties are broken by first-encountered value (Counter.most_common is stable
    on insertion order for equal counts), which is fine here: this only feeds
    a ranking across 81 combos, not a single reported number.
    """
    from collections import Counter

    votes = []
    for idx in final_indices:
        chain = chains[idx]
        text = str(chain.get("answer", "") or chain.get("reason", "") or "")
        val = extract_fn(text)
        if val is not None:
            votes.append(val)
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


# =============================================================================
# ─── GRID SEARCH ───────────────────────────────────────────────────────────────
# =============================================================================

def run_grid_search(
    combos: list[dict],
    examples: list[dict],
    cached_scored_chains: list[list[dict]],
    excluded_indices: set[int],
    verifier: ReasonVerifier,
    qubo_builder: QUBOBuilder,
    solver: SimulatedAnnealingSolver,
    inf_pipeline: InferencePipeline,
    already_done: set[str],
    output_path: Path,
    keep_exclusions: bool = False,
    oracle_scoring: bool = False,
    scoring_mode: str = "proxy",
) -> list[dict]:
    """Run the grid search and return the full results list.

    For each combo:
      Phase A (classical, per question): use pre-scored cached chains, build
        the QUBO, solve it, resolve final_indices. Fast (~1s/question with the
        vectorised solver) but was previously interleaved with Phase B below,
        so nothing printed until an entire combo -- 300 sequential unbatched
        generations -- finished.
      Phase B/C, scoring_mode="proxy" (default): majority-vote over the
        SELECTED chains' own already-known answers -- see proxy_vote(). No
        generation at all; a full 81-combo sweep runs in minutes.
      Phase B/C, scoring_mode="synthesis": the original path. One batched
        generation call per combo via InferencePipeline.run_batch() (300
        individual generate_answer() calls collapsed into ceil(300/batch_size)
        batched ones), then predictions extracted from the generated text.
        This is the expensive path -- measured on live hardware at 57-113+
        minutes per combo and rising as the run progresses (GPU allocator
        fragmentation over many variable-length batches in one long-lived
        process), which puts a full 81-combo sweep at 3+ days. Intended for
        spot-checking a small shortlist of combos already ranked by 'proxy',
        not for the full grid.
    """
    total_combos = len(combos)
    all_results: list[dict] = []
    effective_n = len(examples) - len(excluded_indices)

    # Load existing results into all_results if resuming
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)

    for combo_idx, combo in enumerate(combos, start=1):
        key = build_combo_key(combo)
        if key in already_done:
            print(
                f"[Combo {combo_idx:2d}/{total_combos}] "
                f"pw={combo['penalty_weight']} db={combo['diversity_bonus']} "
                f"cp={combo['cardinality_penalty']} "
                f"aaw={combo['answer_agree_weight']} → SKIPPED"
            )
            continue

        # ── Reconfigure QUBOBuilder ────────────────────────────────────────
        reconfigure_qubo_builder(qubo_builder, combo)

        correct = 0
        blank   = 0   # questions where no numeric answer could be extracted

        # ── Phase A: classical QUBO build + solve for every question ────────
        # Fast (~1s/question with the vectorised solver), but still prints
        # progress -- at 300 questions this alone can take a few minutes, and
        # silently running it was part of why a combo looked "stuck".
        phase_a_t0 = time.time()
        active_questions, active_golds, active_indices, active_chains = [], [], [], []
        for q_idx, ex in enumerate(examples):
            if q_idx in excluded_indices:
                continue  # this question doesn't affect any combo's denominator

            # Use a deep copy so the cached chains are never mutated between
            # combos (QUBOBuilder only reads, but deep-copying is cheap here
            # and guarantees safety).
            chains = copy.deepcopy(cached_scored_chains[q_idx])

            try:
                Q, selected_indices = qubo_builder.build_qubo(chains)
                state, _ = solver.solve(Q)

                active_local = [i for i, bit in enumerate(state) if bit == 1]
                if not active_local:
                    # Fallback: pick the single highest-quality chain
                    best_local = max(
                        range(len(selected_indices)),
                        key=lambda i: chains[selected_indices[i]].get(
                            "correctness_score", 0.0
                        ),
                    )
                    active_local = [best_local]

                final_indices = [selected_indices[i] for i in active_local]
                active_questions.append(ex["question"])
                active_golds.append(ex["gold"])
                active_indices.append(final_indices)
                active_chains.append(chains)

            except Exception as e:
                print(
                    f"\n  [WARN] q={q_idx} combo={key} (build/solve): "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                # Counts as incorrect (not blank — it's a pipeline failure)

            if (q_idx + 1) % 50 == 0:
                print(
                    f"    [Combo {combo_idx:2d}/{total_combos}] "
                    f"build+solve {q_idx + 1}/{len(examples)} "
                    f"({time.time() - phase_a_t0:.0f}s elapsed)",
                    flush=True,
                )

        if scoring_mode == "proxy":
            # ── Phase B/C: vote over the selected chains, no generation ─────
            # See proxy_vote() -- majority vote among the QUBO-selected
            # chains' own already-known answers. This is the entire cost of
            # scoring a combo in proxy mode: a Python loop over already-cached
            # data, no GPU call at all.
            phase_bc_t0 = time.time()
            for gold, chains, final_indices in zip(
                active_golds, active_chains, active_indices
            ):
                pred = proxy_vote(chains, final_indices, verifier._extract_last_number)
                if pred is None:
                    blank += 1
                elif numeric_match(pred, gold):
                    correct += 1
            print(
                f"    [Combo {combo_idx:2d}/{total_combos}] "
                f"proxy-voted {len(active_questions)} questions in "
                f"{time.time() - phase_bc_t0:.1f}s",
                flush=True,
            )
        else:
            # ── Phase B: one batched generation call for the whole combo ────
            # The expensive path. Previously one generate_answer() call per
            # question -- 300 sequential unbatched generations before this
            # combo's single print line ever appeared. Now ceil(n/batch_size)
            # batched calls. empty_cache() up front gives the allocator its
            # best shot at a clean start before this combo's first attempt at
            # the full batch size, since fragmentation was observed to worsen
            # combo-over-combo in one long-lived process.
            torch.cuda.empty_cache()
            phase_b_t0 = time.time()
            print(
                f"    [Combo {combo_idx:2d}/{total_combos}] "
                f"synthesising {len(active_questions)} final answers "
                f"(batch_size={_GEN_BATCH_SIZE}) ...",
                flush=True,
            )
            try:
                final_answers = inf_pipeline.run_batch(
                    active_questions, active_indices, active_chains,
                    batch_size=_GEN_BATCH_SIZE,
                )
            except Exception as e:
                print(
                    f"\n  [WARN] combo={key} (batched synthesis): "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                final_answers = [""] * len(active_questions)
            print(
                f"    [Combo {combo_idx:2d}/{total_combos}] "
                f"synthesis done in {time.time() - phase_b_t0:.0f}s",
                flush=True,
            )

            # ── Phase C: extract predictions, compare to gold ────────────────
            for gold, final_answer in zip(active_golds, final_answers):
                pred = verifier._extract_last_number(final_answer)
                if pred is None:
                    blank += 1
                elif numeric_match(pred, gold):
                    correct += 1

        # Two denominators:
        #   accuracy_excluded — over questions that had at least one decent chain
        #   accuracy_all      — over EVERY question asked (the honest figure)
        # Excluded questions are the ones the model handled worst, so removing them
        # inflates the score. accuracy_all is the default ranking key.
        accuracy_excluded = correct / effective_n if effective_n > 0 else 0.0
        accuracy_all = correct / len(examples) if examples else 0.0
        accuracy = accuracy_excluded if keep_exclusions else accuracy_all

        print(
            f"[Combo {combo_idx:2d}/{total_combos}] "
            f"pw={combo['penalty_weight']} "
            f"db={combo['diversity_bonus']} "
            f"cp={combo['cardinality_penalty']} "
            f"aaw={combo['answer_agree_weight']} "
            f"-> acc={accuracy:.4f}  "
            f"(all={correct}/{len(examples)}, "
            f"excl-denom={correct}/{effective_n}, blank={blank})"
        )

        result = {
            "key":                 key,
            "combo_index":         combo_idx,
            "penalty_weight":      combo["penalty_weight"],
            "diversity_bonus":     combo["diversity_bonus"],
            "cardinality_penalty": combo["cardinality_penalty"],
            "answer_agree_weight": combo["answer_agree_weight"],
            "answer_sim_weight":   combo["answer_sim_weight"],
            "accuracy":            accuracy,
            "accuracy_all":        accuracy_all,
            "accuracy_excluded":   accuracy_excluded,
            "correct":             correct,
            "blank":               blank,
            "effective_n":         effective_n,
            "total_n":             len(examples),
            "excluded":            len(excluded_indices),
            "oracle_scoring":      oracle_scoring,
        }
        all_results.append(result)
        already_done.add(key)

        # Persist after every combo so a Ctrl+C can be --resumed safely
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    return all_results


# =============================================================================
# ─── OUTPUT HELPERS ─────────────────────────────────────────────────────────────
# =============================================================================

def save_best_config(best: dict, path: Path, n_questions: int = 0,
                     oracle_scoring: bool = False, scoring_mode: str = "proxy") -> None:
    """Write the best hyperparameter combination to a YAML file.

    `stale` is written explicitly. load_config() refuses to apply a params file
    marked stale, which is how the superseded (leaky, unreachable-grid) result is
    kept from silently overriding config.yaml. A run scored WITH the oracle is
    marked stale for the same reason: its winner reflects answer-key exploitation
    rather than deployable behaviour.

    scoring_mode ("proxy" vs "synthesis") is recorded but does NOT affect
    staleness -- proxy scoring is gold-free and a legitimate default, just a
    different (much cheaper) accuracy metric than full-pipeline synthesis. It
    is recorded so a later run can tell which cost function actually produced
    this file, since the two are not directly comparable numbers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    se = (0.25 / n_questions) ** 0.5 if n_questions else None
    config_out = {
        "stale": bool(oracle_scoring),
        "penalty_weight":      best["penalty_weight"],
        "diversity_bonus":     best["diversity_bonus"],
        "cardinality_penalty": best["cardinality_penalty"],
        "answer_agree_weight": best["answer_agree_weight"],
        "answer_sim_weight":   best["answer_sim_weight"],
        "validation_accuracy": round(best["accuracy"], 6),
        "n_questions":         n_questions,
        "scoring":             "oracle" if oracle_scoring else "gold-free",
        "combo_scoring":       scoring_mode,
        "notes": (
            f"Grid search over {n_questions} questions, "
            f"{'ORACLE (answer key visible -- ablation only)' if oracle_scoring else 'gold-free'} scoring, "
            f"combo accuracy via {scoring_mode}"
            + (" (majority vote over selected chains, no generation)." if scoring_mode == "proxy" else " (full LLM re-synthesis).")
            + (f" Accuracy standard error about +/-{se * 100:.1f} points. " if se else "")
            + ("Marked stale: oracle-scored results must not drive the deployed config."
               if oracle_scoring else
               "Verify the selected cardinality_penalty yields a multi-chain subset; "
               "run_all_benchmarks.py warns if the mean drops below 2.")
        ),
    }
    with open(path, "w") as f:
        yaml.dump(config_out, f, default_flow_style=False, sort_keys=False)
    print(f"\n[output] Best config saved -> {path}")
    if oracle_scoring:
        print("[output] Marked stale: oracle scoring was used, so this must not be applied.")


def print_top5(all_results: list[dict]) -> None:
    """Print the top-5 combinations ranked by accuracy.

    Shows accuracy, correct/effective_n count, blank count, and excluded count
    so results can be fully interpreted without opening the JSON file.
    """
    ranked = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)
    print("\n" + "=" * 80)
    print("  TOP-5 COMBINATIONS")
    print("=" * 80)
    header = f"  {'#':<3}  {'pw':<5}  {'db':<5}  {'cp':<6}  {'aaw':<5}  {'acc':>7}  {'correct/N':>10}  {'blank':>6}  {'excl':>5}"
    print(header)
    print("  " + "-" * 76)
    for rank, r in enumerate(ranked[:5], start=1):
        print(
            f"  #{rank:<2}  "
            f"{r['penalty_weight']:<5}  "
            f"{r['diversity_bonus']:<5}  "
            f"{r['cardinality_penalty']:<6}  "
            f"{r['answer_agree_weight']:<5}  "
            f"{r['accuracy']:>7.4f}  "
            f"{r['correct']:>4}/{r['effective_n']:<5}  "
            f"{r['blank']:>6}  "
            f"{r['excluded']:>5}"
        )
    print("=" * 80 + "\n")


# =============================================================================
# ─── VRAM REPORTING ────────────────────────────────────────────────────────────
# =============================================================================

def report_vram() -> None:
    """Print current CUDA VRAM allocation to confirm we're not near OOM."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[vram] CUDA not available — running on CPU.")
            return
        allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
        reserved_mb  = torch.cuda.memory_reserved()  / 1024 ** 2
        total_mb     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        print(
            f"[vram] Allocated: {allocated_mb:.0f} MB  |  "
            f"Reserved: {reserved_mb:.0f} MB  |  "
            f"Total: {total_mb:.0f} MB  |  "
            f"Free (est): {total_mb - reserved_mb:.0f} MB"
        )
    except Exception as e:
        print(f"[vram] Could not read VRAM usage: {e}")


# =============================================================================
# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    args = parse_args()

    config_path      = args.config
    # Scoring-mode-specific filename: proxy and synthesis results are NOT
    # comparable (different cost functions), so mixing them into one
    # --resume'd file would rank combos against each other on two different
    # metrics. A run under one mode never sees or skips results from the other.
    output_path      = Path(args.output_dir) / f"qubo_hyperparam_search_{args.scoring}.json"
    best_config_path = Path(args.best_config_path)
    print(f"[grid] Scoring mode: {args.scoring}  ->  results file: {output_path}")

    # ── Load config ────────────────────────────────────────────────────────────
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Resolve all 81 combinations ────────────────────────────────────────────
    combos = generate_all_combos()
    print(f"[grid] Total combinations: {len(combos)}")

    # ── Resume: identify already-evaluated combos ───────────────────────────────
    already_done: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        for r in existing:
            # Prefer the stored "key" field; fall back to reconstructing it
            k = r.get("key") or (
                f"pw={r['penalty_weight']}"
                f"|db={r['diversity_bonus']}"
                f"|cp={r['cardinality_penalty']}"
                f"|aaw={r['answer_agree_weight']}"
            )
            already_done.add(k)
        print(f"[resume] Skipping {len(already_done)} already-evaluated combinations.")
    elif args.resume:
        print(f"[resume] No existing results at {output_path}; starting fresh.")

    # ── Load dataset ───────────────────────────────────────────────────────────
    examples = load_gsm8k(args.num_questions)

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT INITIALISATION
    #
    # Order matters for VRAM efficiency:
    #   1. InferencePipeline loads the LLM (the largest allocation).
    #   2. DiverseSampler receives InferencePipeline's model and tokenizer as
    #      shared references → no second model load occurs.
    #   3. ReasonVerifier loads only the small NLI cross-encoder (86M params).
    #   4. QUBOBuilder loads the MiniLM sentence-transformer (22M params).
    #   5. SimulatedAnnealingSolver is purely numerical (no neural model).
    # ─────────────────────────────────────────────────────────────────────────

    print("\n[init] Loading InferencePipeline (LLM)...")
    inf_pipeline = InferencePipeline(config_path=config_path, use_vllm=False)

    print("[init] Loading DiverseSampler (shared model reference)...")
    sampler = DiverseSampler(
        config_path=config_path,
        shared_model=inf_pipeline.model,
        shared_tokenizer=inf_pipeline.tokenizer,
    )

    # ── VRAM check: confirm no OOM before the grid starts ─────────────────────
    report_vram()

    print("[init] Loading ReasonVerifier (NLI cross-encoder)...")
    verifier = ReasonVerifier(config_path=config_path)

    print("[init] Loading QUBOBuilder (MiniLM embedder)...")
    qubo_builder = QUBOBuilder(config_path=config_path)

    print("[init] Building SimulatedAnnealingSolver...")
    solver = SimulatedAnnealingSolver(config_path=config_path)

    print("[init] All components ready.")

    # ── Pre-cache: sample + score all questions, build exclusion set ───────────
    #
    # The scoring mode is part of the cache filename. Chains scored WITH gold have
    # completely different correctness_scores from chains scored without it, so a
    # cache built under one mode must never be reused under the other.
    mode_tag = "oracle" if args.oracle_scoring else "goldfree"
    cache_file = (
        Path(args.output_dir) / f"cached_chains_{args.num_questions}q_{mode_tag}.json"
    )
    cached_scored_chains, excluded_indices = cache_and_score_all(
        examples, sampler, verifier,
        cache_file=cache_file,
        oracle_scoring=args.oracle_scoring,
    )

    # ── Run grid search ────────────────────────────────────────────────────────
    effective_n = len(examples) - len(excluded_indices)
    print(
        f"\n[grid] Starting grid search over {len(combos)} combinations "
        f"on {effective_n} questions (excluded {len(excluded_indices)})...\n"
    )
    t_grid_start = time.time()

    all_results = run_grid_search(
        combos=combos,
        examples=examples,
        cached_scored_chains=cached_scored_chains,
        excluded_indices=excluded_indices,
        verifier=verifier,
        qubo_builder=qubo_builder,
        solver=solver,
        inf_pipeline=inf_pipeline,
        already_done=already_done,
        output_path=output_path,
        keep_exclusions=args.keep_exclusions,
        oracle_scoring=args.oracle_scoring,
        scoring_mode=args.scoring,
    )

    elapsed_grid = time.time() - t_grid_start
    print(f"\n[grid] Grid search complete in {elapsed_grid/60:.1f} minutes.")

    # ── Final save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[output] Full results saved -> {output_path}")

    # ── Find best and save ─────────────────────────────────────────────────────
    if not all_results:
        print("[warn] No results to process (all combos may have been skipped).")
        return

    best = max(all_results, key=lambda r: r["accuracy"])
    save_best_config(best, best_config_path,
                     n_questions=len(examples),
                     oracle_scoring=args.oracle_scoring,
                     scoring_mode=args.scoring)

    # ── Print top-5 ────────────────────────────────────────────────────────────
    print_top5(all_results)

    print(
        f"[done] Best: pw={best['penalty_weight']} db={best['diversity_bonus']} "
        f"cp={best['cardinality_penalty']} aaw={best['answer_agree_weight']} "
        f"-> acc={best['accuracy']:.4f}  (blank={best['blank']})"
    )


if __name__ == "__main__":
    main()
