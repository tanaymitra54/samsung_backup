"""
scripts/tune_qubo_params.py  —  QUBO Hyperparameter Grid Search
================================================================

Runs an 81-combination grid search over QUBO parameters on the first
300 questions of GSM8K (test split) and saves the best configuration.

PARAMETER GRID (81 total combinations):
    penalty_weight:       [0.5, 1.0, 1.5]
    diversity_bonus:      [0.1, 0.2, 0.3]
    cardinality_penalty:  [0.05, 0.1, 0.2]
    answer_agree_weight:  [0.3, 0.4, 0.5]
    answer_sim_weight:    1.0 - answer_agree_weight  (always sums to 1.0)

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

RESUME:
    Pass --resume to load existing results from
    results/qubo_hyperparam_search.json and skip already-evaluated combos.
    Resume keys are derived from the 4 parameter values — NOT combo index —
    so they remain stable across runs even if grid ordering changes.

OUTPUTS:
    results/qubo_hyperparam_search.json   — all combo results
    config/best_qubo_params.yaml          — best combination details

USAGE:
    # Run from the project root:
    python scripts/tune_qubo_params.py

    # Resume an interrupted search:
    python scripts/tune_qubo_params.py --resume

    # Smoke-test with fewer questions:
    python scripts/tune_qubo_params.py --num-questions 20
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
_PW_VALUES   = [0.5, 1.0, 1.5]          # penalty_weight
_DB_VALUES   = [0.1, 0.2, 0.3]          # diversity_bonus
_CP_VALUES   = [0.05, 0.1, 0.2]         # cardinality_penalty
_AAW_VALUES  = [0.3, 0.4, 0.5]          # answer_agree_weight
# answer_sim_weight = 1.0 - answer_agree_weight

# Low-quality-question threshold: if EVERY chain for a question scores below
# this, the question is excluded from the accuracy denominator for ALL combos.
_EXCLUSION_SCORE_THRESHOLD = 0.2


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

        # Step 2: Score immediately (verifier params are fixed for all combos)
        gold_str = str(ex["gold"])
        verifier.score_batch(chains, task_type="math", gold=gold_str,
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
) -> list[dict]:
    """Run the grid search and return the full results list.

    For each combo:
      For each non-excluded question:
        1. Use pre-scored cached chains (correctness_score already set)
        2. Build QUBO Q matrix (hot-swapped params)
        3. Solve with SA solver → state bits → selected_indices
        4. Run InferencePipeline.run() → final answer text
        5. Extract last number from answer
        6. Compare to gold with hybrid tolerance → correct/blank/incorrect
      Accuracy = correct / (n_questions - len(excluded_indices))
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

        for q_idx, ex in enumerate(examples):
            if q_idx in excluded_indices:
                continue  # this question doesn't affect any combo's denominator

            question = ex["question"]
            gold     = ex["gold"]

            # Use a deep copy so the cached chains are never mutated between
            # combos (QUBOBuilder and InferencePipeline only read, but
            # deep-copying is cheap here and guarantees safety).
            chains = copy.deepcopy(cached_scored_chains[q_idx])

            try:
                # Step 2: Build QUBO with current combo's parameters
                Q, selected_indices = qubo_builder.build_qubo(chains)

                # Step 3: Solve → binary state vector
                state, _ = solver.solve(Q)

                # Map state bits → sample indices in the original chain list
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

                # Step 4: Generate final answer from QUBO-selected reasons
                final_answer = inf_pipeline.run(question, final_indices, chains)

                # Step 5: Extract numeric prediction from generated text
                pred = verifier._extract_last_number(final_answer)

                if pred is None:
                    blank += 1
                elif numeric_match(pred, gold):
                    correct += 1

            except Exception as e:
                print(
                    f"\n  [WARN] q={q_idx} combo={key}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                # Counts as incorrect (not blank — it's a pipeline failure)

        accuracy = correct / effective_n if effective_n > 0 else 0.0

        print(
            f"[Combo {combo_idx:2d}/{total_combos}] "
            f"pw={combo['penalty_weight']} "
            f"db={combo['diversity_bonus']} "
            f"cp={combo['cardinality_penalty']} "
            f"aaw={combo['answer_agree_weight']} "
            f"-> acc={accuracy:.4f}  "
            f"({correct}/{effective_n}, blank={blank})"
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
            "correct":             correct,
            "blank":               blank,
            "effective_n":         effective_n,
            "excluded":            len(excluded_indices),
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

def save_best_config(best: dict, path: Path) -> None:
    """Write the best hyperparameter combination to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config_out = {
        "penalty_weight":      best["penalty_weight"],
        "diversity_bonus":     best["diversity_bonus"],
        "cardinality_penalty": best["cardinality_penalty"],
        "answer_agree_weight": best["answer_agree_weight"],
        "answer_sim_weight":   best["answer_sim_weight"],
        "validation_accuracy": round(best["accuracy"], 6),
    }
    with open(path, "w") as f:
        yaml.dump(config_out, f, default_flow_style=False, sort_keys=False)
    print(f"\n[output] Best config saved -> {path}")


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
    output_path      = Path(args.output_dir) / "qubo_hyperparam_search.json"
    best_config_path = Path(args.best_config_path)

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
    cache_file = Path(args.output_dir) / f"cached_chains_{args.num_questions}q.json"
    cached_scored_chains, excluded_indices = cache_and_score_all(
        examples, sampler, verifier, cache_file=cache_file
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
    save_best_config(best, best_config_path)

    # ── Print top-5 ────────────────────────────────────────────────────────────
    print_top5(all_results)

    print(
        f"[done] Best: pw={best['penalty_weight']} db={best['diversity_bonus']} "
        f"cp={best['cardinality_penalty']} aaw={best['answer_agree_weight']} "
        f"-> acc={best['accuracy']:.4f}  (blank={best['blank']})"
    )


if __name__ == "__main__":
    main()
