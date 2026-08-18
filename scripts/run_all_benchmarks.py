"""
==============================================================================
FILE: scripts/run_all_benchmarks.py
ROLE: Benchmark Evaluation Suite & Execution Harness
BRANCH ADDITION (abhyuday): Added question-level partial caching and resume support,
allowing evaluation runs to skip previously evaluated questions, recover safely from
interruptions, and write continuous evaluation metrics.
==============================================================================
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


def _mask_single_gpu_from_argv():
    if "--multi-gpu" in sys.argv:
        return

    for idx, arg in enumerate(sys.argv):
        if arg == "--device" and idx + 1 < len(sys.argv):
            device_arg = sys.argv[idx + 1]
            if device_arg.startswith("cuda:"):
                os.environ["CUDA_VISIBLE_DEVICES"] = device_arg.split(":", 1)[1]
            return


_mask_single_gpu_from_argv()

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import BenchmarkRunner
from evaluation.answer_utils import (
    extract_gsm8k_gold,
    extract_predicted_answer,
    is_correct_prediction,
)
from evaluation.answer_utils_v2 import (
    check_repetition,
    extract_predicted_answer_v2,
)
from pipeline.device_utils import resolve_device
from pipeline.inference import InferencePipeline
from pipeline.qubo_builder import QUBOBuilder
from pipeline.sampling import DiverseSampler
from pipeline.solver import SimulatedAnnealingSolver
from pipeline.verifier import ReasonVerifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QUBO pipeline across all configured benchmarks"
    )
    parser.add_argument(
        "--subset-size", type=int, default=None, help="Override config subset_size"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run on full datasets (ignore subset_size)"
    )
    parser.add_argument(
        "--output-dir", default="results/eval", help="Directory for output files"
    )
    parser.add_argument(
        "--condition-label", default="base", help="Condition label for JSONL output (e.g. A, B, base)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Specific benchmarks to run (default: all in config)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible runs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for batched inference"
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batched inference (force per-question)",
    )
    parser.add_argument(
        "--use-vllm", action="store_true", help="Use vLLM backend for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Single device for benchmark execution (default: evaluation.device or cuda:0)",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Distribute benchmarks across available GPUs",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name for tracking",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save first 5 raw model outputs for debugging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-question QUBO detail (pool size, variables, chains selected, "
             "score spread). Useful for diagnosing a run; noisy for a long one.",
    )
    parser.add_argument(
        "--oracle-selection",
        action="store_true",
        help="ABLATION ONLY: let the verifier see the gold answer when scoring "
             "candidate chains. This leaks the answer key into QUBO selection and "
             "produces accuracies that are NOT reproducible at deployment. Use it "
             "to report the oracle ceiling alongside the real number, never alone.",
    )
    return parser.parse_args()


TASK_TYPE = {
    "gsm8k": "math",
    # BBH was typed "math", but its golds are ~63% parenthesised MCQ letters plus
    # booleans and word-sorting strings; only a couple of its 27 subtasks are
    # arithmetic. Scoring it as math meant chain quality came from arithmetic
    # consistency on traces that contain no arithmetic, and consensus compared
    # extracted numbers where there are none. "commonsense" routes it to the
    # NLI/coverage/structure scorer and to string-based consensus.
    "bbh": "commonsense",
    "strategyqa": "commonsense",
    "mmlu": "commonsense",
    "arc_challenge": "commonsense",
    "math 500": "math",
    "gpqa diamond": "commonsense",
    "aime": "math",
    "mmlu pro": "commonsense",
}

IS_MCQ = {"mmlu", "arc_challenge", "gpqa diamond", "mmlu pro"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Single letters that are also ordinary English words. Appearing bare in prose,
# these say nothing about which option was chosen -- "the shape is a triangle"
# is not a vote for choice A, and "I believe" is not a vote for choice I.
# They are only accepted when delimited, explicitly tagged, or the whole answer.
_AMBIGUOUS_BARE_LETTERS = {"A", "I"}

_MCQ_LETTERS = "A-J"


def extract_mcq_choice(text: str, question: str = "") -> str:
    """Extract the selected multiple-choice letter, abstaining when ambiguous.

    Returns "" if no choice can be identified with confidence.

    Runs answer_utils_v2's extractor first when a `question` is available. That
    stage adds two things this module cannot do alone:
      * VALUE MATCHING -- a model that answers with the option's text ("The answer
        is 4") instead of its letter is mapped back to the letter by comparing
        against the options parsed out of the question.
      * \\boxed{}, [A] and trailing "(A)" forms.
    It also rejects degenerate repetition loops outright.

    If that stage yields no single letter, the staged patterns below run as a
    fallback, ending in an abstention rather than a guess.

    ABSTAINING IS DELIBERATE. The previous implementation ended with "return the
    last A-J letter found anywhere in the text", which fired on the article "a"
    and the pronoun "I" -- so any prose answer without an explicit tag was scored
    as choice A. That both added noise and biased results toward A. Returning ""
    scores the item wrong, which is the honest outcome when the model did not
    state a choice we can read.

    Patterns are tried strongest-first; each is anchored on a delimiter or an
    explicit answer word so ordinary prose cannot trigger it.
    """
    if not text:
        return ""

    # Degenerate repetition means the model never settled on an answer.
    if check_repetition(text):
        return ""

    # Stage 0: answer_utils_v2 (explicit tags, boxed/bracket forms, value matching).
    try:
        v2 = extract_predicted_answer_v2(text, is_mcq=True, question=question)
    except Exception:
        v2 = None
    if v2 and re.fullmatch(rf"[{_MCQ_LETTERS}]", str(v2).strip().upper()):
        return str(v2).strip().upper()

    upper = text.strip().upper()
    cls = f"[{_MCQ_LETTERS}]"

    # 0. The whole response is just a letter, optionally decorated: "C", "(C)", "**C**", "C."
    #    This is the common shape of a greedy MCQ answer and is unambiguous, so
    #    ambiguous-letter filtering does not apply.
    solo = re.fullmatch(rf"[\(\[\*\s]*({cls})[\)\]\*\.\s]*", upper)
    if solo:
        return solo.group(1)

    # 1. Explicit answer tag: "ANSWER: C", "THE ANSWER IS (C)", "FINAL ANSWER - C"
    tagged = re.search(rf"\bANSWER\b\s*(?:IS|=|:|-)?\s*[\(\[\*]*({cls})\b", upper)
    if tagged:
        return tagged.group(1)

    # 2. "THE CORRECT ANSWER IS X" / "RIGHT OPTION: X"
    explicit = re.search(
        rf"\b(?:CORRECT|RIGHT)\s+(?:ANSWER|CHOICE|OPTION)?\s*(?:IS\s+|:\s*)?[\(\[\*]*({cls})\b",
        upper,
    )
    if explicit:
        return explicit.group(1)

    # Everything below only inspects the conclusion, where the decision is stated.
    tail = upper[-300:] if len(upper) > 300 else upper

    # 3. "OPTION C" / "CHOICE C"
    labelled = re.findall(rf"\b(?:OPTION|CHOICE)\s*[\(\[\*]*({cls})\b", tail)
    if labelled:
        return labelled[-1]

    # 4. Delimited letter: "(C)", "[C]", "**C**", or "C)" at a token boundary.
    #    A delimiter means the letter was written as a label, not as a word.
    delimited = re.findall(
        rf"\(\s*({cls})\s*\)|\[\s*({cls})\s*\]|\*\*\s*({cls})\s*\*\*|(?:^|\s)({cls})\)",
        tail,
    )
    flat = [g for groups in delimited for g in groups if g]
    if flat:
        return flat[-1]

    # 5. Bare standalone letter, excluding the ones that are English words.
    bare = [c for c in re.findall(rf"\b({cls})\b", tail)
            if c not in _AMBIGUOUS_BARE_LETTERS]
    if bare:
        return bare[-1]

    # 6. No identifiable choice -- abstain rather than guess.
    return ""


def _mcq_prompt(question: str) -> str:
    return f"{question}\n\nOutput only the correct answer choice letter (e.g., A, B, C, etc.):"


def _mcq_cot_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step. At the end, output the correct answer choice letter (e.g., A, B, C, etc.)."


def baseline_greedy(
    inference: InferencePipeline, question: str, benchmark: str = ""
) -> str:
    """Greedy baseline - direct answer without reasoning."""
    if benchmark in IS_MCQ:
        prompt = _mcq_prompt(question)
    else:
        prompt = f"{question}\n\nProvide only the final numerical answer, nothing else."
    return inference.generate_answer(prompt)


def baseline_cot(
    inference: InferencePipeline, question: str, benchmark: str = ""
) -> str:
    """Chain-of-Thought baseline with step-by-step reasoning."""
    if benchmark in IS_MCQ:
        prompt = _mcq_cot_prompt(question)
    else:
        prompt = f"{question}\n\nLet's think step by step, then provide the final answer in the format: #### [answer]"
    return inference.generate_answer(prompt)


def make_batch_greedy(inference: InferencePipeline):
    def fn(questions: list[str], benchmark: str = "") -> list[str]:
        if benchmark in IS_MCQ:
            prompts = [_mcq_prompt(q) for q in questions]
        else:
            prompts = [
                f"{q}\n\nProvide only the final numerical answer, nothing else."
                for q in questions
            ]
        return inference.generate_answers_batch(prompts)

    return fn


def make_batch_cot(inference: InferencePipeline):
    def fn(questions: list[str], benchmark: str = "") -> list[str]:
        if benchmark in IS_MCQ:
            prompts = [_mcq_cot_prompt(q) for q in questions]
        else:
            prompts = [
                f"{q}\n\nLet's think step by step, then provide the final answer in the format: #### [answer]"
                for q in questions
            ]
        return inference.generate_answers_batch(prompts)

    return fn


def run_qubo_pipeline(
    sampler: DiverseSampler,
    verifier: ReasonVerifier,
    qubo_builder: QUBOBuilder,
    solver: SimulatedAnnealingSolver,
    inference: InferencePipeline,
    question: str,
    task_type: str = "math",
    gold: str = "",
    oracle_selection: bool = False,
) -> str:
    t_sample = time.time()
    samples = sampler.sample(question, task_type=task_type)
    _STAGE_TIMES["sample"] += time.time() - t_sample
    if not samples:
        if _VERBOSE:
            print("      [qubo] sampler returned no chains", flush=True)
        return ""

    # The gold answer is deliberately WITHHELD from chain scoring here.
    #
    # verify_math() gives answer_match a 0.63 weight, so feeding it the gold
    # answer lets the QUBO diagonal encode "this chain already has the right
    # answer" — the solver then selects chains using the answer key and the
    # reported accuracy is not reproducible at deployment. The greedy and CoT
    # baselines get no such help, so it also makes the comparison unfair.
    #
    # With gold=None the verifier falls back to gold-free scoring: arithmetic
    # consistency plus cross-chain consensus (see ReasonVerifier.score_batch).
    #
    # oracle_selection=True restores the old behaviour for ABLATION ONLY, so the
    # oracle-selected ceiling can be reported alongside the deployable number.
    # Never enable it for a headline result.
    scoring_gold = gold if oracle_selection else None
    t_verify = time.time()
    samples = verifier.score_batch(
        samples, task_type=task_type, gold=scoring_gold, question=question
    )
    _STAGE_TIMES["verify"] += time.time() - t_verify

    t_qubo = time.time()
    Q, qubo_var_indices = qubo_builder.build_qubo(samples)
    _STAGE_TIMES["qubo_build"] += time.time() - t_qubo

    t_solve = time.time()
    state, _ = solver.solve(Q)
    _STAGE_TIMES["solve"] += time.time() - t_solve

    selected_indices = [qubo_var_indices[i] for i in range(len(state)) if state[i] == 1]
    if not selected_indices:
        selected_indices = list(range(min(inference.subset_size, len(samples))))

    _CHAIN_STATS["pools"] += 1
    _CHAIN_STATS["chains"] += len(samples)
    _CHAIN_STATS["empty_answer"] += sum(
        1 for s_ in samples if not str(s_.get("answer", "")).strip()
    )
    _cons = [float(s_.get("consensus_score", 0.0)) for s_ in samples]
    _CHAIN_STATS["consensus_sum"] += sum(_cons)
    _CHAIN_STATS["score_sum"] += sum(
        float(s_.get("correctness_score", 0.0)) for s_ in samples
    )
    if max(_cons, default=0.0) == 0.0:
        _CHAIN_STATS["zero_consensus_pools"] += 1

    if _VERBOSE:
        scores = [s.get("correctness_score", 0.0) for s in samples]
        print(
            f"      [qubo] pool={len(samples)} vars={len(qubo_var_indices)} "
            f"selected={len(selected_indices)} "
            f"score min/mean/max={min(scores):.2f}/{sum(scores)/len(scores):.2f}/{max(scores):.2f}",
            flush=True,
        )

    # Record how many chains actually reached the CoT scaffold. If this sits at 1
    # while subset_size is 6, the QUBO is collapsing to a single chain and the
    # multi-chain scaffold the method depends on is not happening -- usually
    # cardinality_penalty being far too small relative to penalty_weight.
    _SELECTION_SIZES.append(len(selected_indices))

    t_final = time.time()
    answer = inference.run(question, selected_indices, samples)
    _STAGE_TIMES["final_answer"] += time.time() - t_final
    return answer


# Running tally of QUBO subset sizes, summarised at the end of a benchmark run.
_SELECTION_SIZES: list[int] = []

# Cumulative wall time per pipeline stage. Printed at the end of a run so a slow
# run can be attributed to a stage instead of guessed at.
_STAGE_TIMES: dict[str, float] = {
    "sample": 0.0,
    "verify": 0.0,
    "qubo_build": 0.0,
    "solve": 0.0,
    "final_answer": 0.0,
}

# Chain-quality diagnostics. The gold-free score is 65% cross-chain consensus,
# so if chains rarely produce a parseable answer the consensus term collapses to
# zero and selection runs on process quality alone. These counters make that
# visible instead of leaving it to be inferred from low scores.
_CHAIN_STATS = {
    "chains": 0,
    "empty_answer": 0,
    "consensus_sum": 0.0,
    "score_sum": 0.0,
    "zero_consensus_pools": 0,
    "pools": 0,
}

# Per-question detail. Set by --verbose.
_VERBOSE = False


def report_selection_sizes(subset_size: int):
    """Warn if the QUBO is not producing multi-chain scaffolds."""
    if not _SELECTION_SIZES:
        return
    mean_sel = sum(_SELECTION_SIZES) / len(_SELECTION_SIZES)
    print(
        f"\n[QUBO] Selected {mean_sel:.2f} chains per question on average "
        f"(target subset_size={subset_size}, n={len(_SELECTION_SIZES)})."
    )
    if mean_sel < 2.0 and subset_size >= 3:
        print(
            "[QUBO] WARNING: the solver is collapsing to a near-single-chain scaffold, "
            "so the final prompt carries almost no reasoning diversity. Raise "
            "qubo.cardinality_penalty (try 0.5-1.5) or lower qubo.penalty_weight, "
            "then re-run scripts/tune_qubo_params.py."
        )


def report_chain_quality():
    """Health of the candidate pools that gold-free selection depends on.

    Gold-free scoring is 65% cross-chain consensus. Consensus only exists when
    chains produce comparable answers, so a high empty-answer rate or a mean
    consensus near zero means selection is effectively running on process
    quality alone -- a weak signal that will underperform plain CoT.
    """
    n = _CHAIN_STATS["chains"]
    if not n:
        return
    empty_pct = _CHAIN_STATS["empty_answer"] / n
    mean_cons = _CHAIN_STATS["consensus_sum"] / n
    mean_score = _CHAIN_STATS["score_sum"] / n
    pools = _CHAIN_STATS["pools"]
    dead = _CHAIN_STATS["zero_consensus_pools"]

    print("")
    print(f"[Chains] {n} chains across {pools} questions:")
    print(f"  empty answer field       : {empty_pct:.1%}")
    print(f"  mean consensus           : {mean_cons:.3f}")
    print(f"  mean quality score       : {mean_score:.3f}")
    print(f"  pools with zero consensus: {dead}/{pools}")

    if empty_pct > 0.25:
        print("  WARNING: many chains never emitted a parseable answer. They are")
        print("  likely truncating before the 'Answer:' line -- raise")
        print("  pipeline.sampling_max_new_tokens (try 384-512).")
    if mean_cons < 0.15:
        print("  WARNING: cross-chain consensus is near zero, so 65% of the")
        print("  gold-free quality score carries no information and the QUBO is")
        print("  selecting on process quality alone. Expect it to trail plain CoT")
        print("  until this is fixed.")


def report_stage_times():
    """Where the wall clock actually went, per pipeline stage."""
    total = sum(_STAGE_TIMES.values())
    if total <= 0:
        return
    print(f"\n[Timing] QUBO pipeline stage breakdown (total {total / 60:.1f} min):")
    for stage, secs in sorted(_STAGE_TIMES.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(40 * secs / total)
        print(f"  {stage:<13} {secs / 60:7.1f} min  {secs / total:5.1%}  {bar}")
    n = len(_SELECTION_SIZES)
    if n:
        print(f"  per question : {total / n:.1f}s across {n} questions")


def extract_answer(pred: str, benchmark: str) -> str:
    if not pred:
        return ""
    if benchmark == "gsm8k":
        return extract_predicted_answer(pred)
    if benchmark in {"math 500", "aime"}:
        if "####" in pred:
            return pred.split("####")[-1].strip()
        return pred.strip()
    return pred.strip()


# Markers after which a model states its conclusion. Grading looks only at the
# text following the LAST marker, so a gold token that merely appears mid-working
# ("False and True = False, therefore True") cannot be mistaken for the answer.
_FINAL_MARKERS = re.compile(
    r"####|FINAL\s+ANSWER|THE\s+ANSWER\s+IS|\bANSWER\s*[:=]|\bTHEREFORE\b|\bTHUS\b|\bHENCE\b|\bSO\s+THE\s+ANSWER\b",
    re.IGNORECASE,
)

_BOOLEAN_GOLDS = {"yes", "no", "true", "false", "valid", "invalid"}

# Polarity words that mean the same verdict. A model asked a yes/no question may
# answer "True", and StrategyQA's gold is stored as a bool, so the two phrasings
# must compare equal. "valid"/"invalid" are BBH's formal_fallacies phrasing.
_POLARITY_CANON = {
    "yes": "+", "true": "+", "valid": "+",
    "no": "-", "false": "-", "invalid": "-",
}


def _conclusion_span(pred: str) -> str:
    """The part of `pred` that states the final answer.

    Text after the last final-answer marker if one exists, otherwise the last
    non-empty line. Grading against this span instead of the whole prediction is
    what stops mid-reasoning mentions from counting as the answer.
    """
    if not pred:
        return ""
    matches = list(_FINAL_MARKERS.finditer(pred))
    if matches:
        tail = pred[matches[-1].end():].strip()
        if tail:
            return tail
    lines = [ln.strip() for ln in pred.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else pred.strip()


def _normalise(text: str) -> str:
    """Lowercase and collapse to alphanumeric tokens for tolerant comparison."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_correct(pred: str, gold: str, benchmark: str, question: str = "") -> bool:
    """Grade a prediction against gold, dispatching on the FORMAT of the gold.

    `question` is optional but improves MCQ grading: it lets the extractor map an
    answer given as option TEXT back to its letter.

    Benchmark-name dispatch alone is not enough: BBH mixes parenthesised MCQ
    letters ("(A)"), booleans ("False"), free numbers ("-50") and word-sorting
    strings across its subtasks, so the gold itself decides how to compare.

    The previous fallback accepted `gold in pred`, which scored
    "False and True = False, therefore the answer is True" as correct for
    gold="False", and matched gold="no" inside "I do not know". Both are fixed
    by grading the conclusion span with format-aware comparison.
    """
    if not pred:
        return False

    gold_s = gold.strip()
    if not gold_s:
        return False

    # GSM8K keeps its dedicated numeric path (handles the '#### N' gold format).
    if benchmark == "gsm8k":
        return is_correct_prediction(pred, extract_gsm8k_gold(gold))

    # 1. Parenthesised MCQ letter, e.g. BBH "(A)" -- compare extracted choices.
    paren = re.fullmatch(r"\(([A-Ja-j])\)", gold_s)
    if paren:
        return extract_mcq_choice(pred, question) == paren.group(1).upper()

    # 2. Benchmarks declared multiple-choice: gold is a bare letter.
    if benchmark in IS_MCQ:
        extracted = extract_mcq_choice(pred, question)
        return bool(extracted and extracted == gold_s.upper())

    span = _conclusion_span(pred)
    gold_l = gold_s.lower()

    # 3. Boolean / polarity golds: take the LAST polarity word in the conclusion
    #    so the stated verdict wins over anything mentioned while working, and
    #    compare canonical polarity so "True" matches gold "yes".
    if gold_l in _BOOLEAN_GOLDS:
        stated = [t for t in re.findall(r"[a-z]+", span.lower()) if t in _BOOLEAN_GOLDS]
        return bool(stated) and _POLARITY_CANON[stated[-1]] == _POLARITY_CANON[gold_l]

    # 4. Numeric golds: compare the last number in the conclusion.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", gold_s):
        nums = re.findall(r"-?\d+(?:\.\d+)?", span.replace(",", ""))
        if not nums:
            return False
        try:
            return abs(float(nums[-1]) - float(gold_s)) < 1e-6
        except ValueError:
            return False

    # 5. Free-form gold: require the conclusion to be, or end with, the gold
    #    phrase -- not merely to contain it somewhere.
    span_n, gold_n = _normalise(span), _normalise(gold_s)
    if not gold_n:
        return False
    return span_n == gold_n or span_n.endswith(gold_n)


def write_summary_json(path: str, results: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_summary_markdown(path: str, results: dict, config_benchmarks: list[str]):
    lines = [
        "# Multi-Benchmark Evaluation Report",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Benchmarks: {', '.join(results.keys())}",
        "",
        "## Accuracy Summary",
        "",
        "| Benchmark | Samples | Greedy | CoT | QUBO | Δ vs Greedy | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for b in config_benchmarks:
        if b not in results:
            continue
        r = results[b]
        # Check if benchmark has accuracy data or if it failed
        if "error" in r and "accuracy" not in r:
            lines.append(f"| {b} | 0 | N/A | N/A | N/A | N/A | ❌ Failed |")
        else:
            a = r.get("accuracy", {"greedy": 0.0, "cot": 0.0, "qubo": 0.0})
            gain = r.get("abs_gain_vs_greedy", 0.0)
            lines.append(
                f"| {b} | {r.get('num_samples', 0)} "
                f"| {a.get('greedy', 0.0):.2%} "
                f"| {a.get('cot', 0.0):.2%} "
                f"| {a.get('qubo', 0.0):.2%} "
                f"| {gain:+.2%} | ✓ Complete |"
            )

    lines.extend(["", "## Benchmark Details", ""])
    for b in config_benchmarks:
        if b not in results:
            continue
        r = results[b]
        # Check if benchmark has accuracy data or if it failed
        if "error" in r and "accuracy" not in r:
            lines.extend(
                [
                    f"### {b}",
                    "",
                    f"**Status: Failed**",
                    f"- Error: {r['error']}",
                    "",
                ]
            )
        else:
            a = r.get("accuracy", {"greedy": 0.0, "cot": 0.0, "qubo": 0.0})
            lines.extend(
                [
                    f"### {b}",
                    "",
                    f"- Samples: {r.get('num_samples', 0)}",
                    f"- Failed samples: {r.get('failed_samples', 0)}",
                    f"- Greedy accuracy: {a.get('greedy', 0.0):.2%}",
                    f"- CoT accuracy: {a.get('cot', 0.0):.2%}",
                    f"- QUBO pipeline accuracy: {a.get('qubo', 0.0):.2%}",
                    f"- Absolute gain vs Greedy: {r.get('abs_gain_vs_greedy', 0.0):+.2%}",
                    f"- CoT gain over Greedy: {r.get('cot_gain_over_greedy', 0.0):+.2%}",
                    "",
                ]
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_benchmark_on_gpu(
    gpu_id: int,
    benchmark_name: str,
    config_path: str,
    seed: int,
    subset_size: int,
    full_eval: bool,
    use_batch: bool,
    batch_size: int,
    use_vllm: bool,
    device: str | None,
    oracle_selection: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_seed(seed)

    worker_device = "cuda:0"

    runner = BenchmarkRunner(config_path)
    if subset_size is not None:
        runner.subset_size = subset_size
    if full_eval:
        runner.full_eval = True

    inference = InferencePipeline(config_path, device=worker_device, use_vllm=use_vllm)
    runtime_device = str(inference.device)

    sampler = DiverseSampler(
        config_path,
        device=runtime_device,
        shared_model=inference.model,
        shared_tokenizer=inference.tokenizer,
    )
    verifier = ReasonVerifier(config_path, device=runtime_device)
    qubo_builder = QUBOBuilder(config_path, device=runtime_device)
    solver = SimulatedAnnealingSolver(config_path, device=runtime_device)

    task_type = TASK_TYPE.get(benchmark_name, "math")
    questions, gold_answers = runner.load_benchmark(benchmark_name)

    results_rows = []
    correct_greedy = 0
    correct_cot = 0
    correct_qubo = 0
    total = 0
    failed = 0

    if use_batch and batch_size > 1 and torch.cuda.is_available():
        batch_greedy_fn = make_batch_greedy(inference)
        batch_cot_fn = make_batch_cot(inference)
        for i in range(0, len(questions), batch_size):
            batch_q = questions[i : i + batch_size]
            batch_gold = gold_answers[i : i + batch_size]
            try:
                t0 = time.time()
                preds_g = batch_greedy_fn(batch_q)
                t1 = time.time()
                preds_c = batch_cot_fn(batch_q)
                t2 = time.time()
                for j, q in enumerate(batch_q):
                    gold = batch_gold[j]
                    pred_q = run_qubo_pipeline(
                        sampler,
                        verifier,
                        qubo_builder,
                        solver,
                        inference,
                        q,
                        task_type,
                        gold=gold,
                        oracle_selection=oracle_selection,
                    )
                    pred_qubo_n = extract_answer(pred_q, benchmark_name)
                    pred_g_n = extract_answer(preds_g[j], benchmark_name)
                    pred_c_n = extract_answer(preds_c[j], benchmark_name)
                    c_g = int(is_correct(pred_g_n, gold, benchmark_name, q))
                    c_c = int(is_correct(pred_c_n, gold, benchmark_name, q))
                    c_q = int(is_correct(pred_qubo_n, gold, benchmark_name, q))
                    correct_greedy += c_g
                    correct_cot += c_c
                    correct_qubo += c_q
                    total += 1
                    results_rows.append(
                        {
                            "benchmark": benchmark_name,
                            "id": i + j,
                            "question": q,
                            "gold": gold,
                            "pred_greedy": pred_g_n,
                            "pred_cot": pred_c_n,
                            "pred_qubo": pred_qubo_n,
                            "correct_greedy": c_g,
                            "correct_cot": c_c,
                            "correct_qubo": c_q,
                            "runtime_greedy_s": round((t1 - t0) / len(batch_q), 4),
                            "runtime_cot_s": round((t2 - t1) / len(batch_q), 4),
                            "runtime_qubo_s": 0.0,
                            "error": "",
                        }
                    )
            except Exception as e:
                failed += len(batch_q)
                for j in range(len(batch_q)):
                    results_rows.append(
                        {
                            "benchmark": benchmark_name,
                            "id": i + j,
                            "question": batch_q[j],
                            "gold": batch_gold[j],
                            "pred_greedy": "",
                            "pred_cot": "",
                            "pred_qubo": "",
                            "correct_greedy": 0,
                            "correct_cot": 0,
                            "correct_qubo": 0,
                            "runtime_greedy_s": 0.0,
                            "runtime_cot_s": 0.0,
                            "runtime_qubo_s": 0.0,
                            "error": str(e),
                        }
                    )
    else:
        for idx, (q, gold) in enumerate(zip(questions, gold_answers)):
            try:
                t0 = time.time()
                pred_greedy = baseline_greedy(inference, q)
                t1 = time.time()
                pred_cot = baseline_cot(inference, q)
                t2 = time.time()
                pred_qubo = run_qubo_pipeline(
                    sampler,
                    verifier,
                    qubo_builder,
                    solver,
                    inference,
                    q,
                    task_type,
                    gold=gold,
                    oracle_selection=oracle_selection,
                )
                t3 = time.time()
                pred_g_n = extract_answer(pred_greedy, benchmark_name)
                pred_c_n = extract_answer(pred_cot, benchmark_name)
                pred_q_n = extract_answer(pred_qubo, benchmark_name)
                c_g = int(is_correct(pred_g_n, gold, benchmark_name, q))
                c_c = int(is_correct(pred_c_n, gold, benchmark_name, q))
                c_q = int(is_correct(pred_q_n, gold, benchmark_name, q))
                correct_greedy += c_g
                correct_cot += c_c
                correct_qubo += c_q
                total += 1
                results_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "id": idx,
                        "question": q,
                        "gold": gold,
                        "pred_greedy": pred_g_n,
                        "pred_cot": pred_c_n,
                        "pred_qubo": pred_q_n,
                        "correct_greedy": c_g,
                        "correct_cot": c_c,
                        "correct_qubo": c_q,
                        "runtime_greedy_s": round(t1 - t0, 4),
                        "runtime_cot_s": round(t2 - t1, 4),
                        "runtime_qubo_s": round(t3 - t2, 4),
                        "error": "",
                    }
                )
            except Exception as e:
                failed += 1
                results_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "id": idx,
                        "question": q,
                        "gold": gold,
                        "pred_greedy": "",
                        "pred_cot": "",
                        "pred_qubo": "",
                        "correct_greedy": 0,
                        "correct_cot": 0,
                        "correct_qubo": 0,
                        "runtime_greedy_s": 0.0,
                        "runtime_cot_s": 0.0,
                        "runtime_qubo_s": 0.0,
                        "error": str(e),
                    }
                )

    acc_g = (correct_greedy / total) if total else 0.0
    acc_c = (correct_cot / total) if total else 0.0
    acc_q = (correct_qubo / total) if total else 0.0

    return {
        "benchmark": benchmark_name,
        "accuracy": {"greedy": acc_g, "cot": acc_c, "qubo": acc_q},
        "num_samples": total,
        "failed_samples": failed,
        "abs_gain_vs_greedy": acc_q - acc_g,
        "cot_gain_over_greedy": acc_c - acc_g,
        "rows": results_rows,
    }


def main():
    args = parse_args()
    global _VERBOSE
    _VERBOSE = args.verbose
    requested_device = args.device
    if requested_device and requested_device.startswith("cuda:") and not args.multi_gpu:
        selected_device = "cuda:0"
    else:
        selected_device = requested_device
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runner = BenchmarkRunner()
    selected_device = str(
        resolve_device(
            selected_device or runner.config.get("evaluation", {}).get("device")
        )
    )
    if args.subset_size is not None:
        runner.subset_size = args.subset_size
    if args.full:
        runner.full_eval = True

    benchmark_list = args.benchmarks if args.benchmarks else runner.benchmarks
    unknown = [b for b in benchmark_list if b not in runner.benchmarks]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s): {unknown}. Allowed: {runner.benchmarks}"
        )

    use_batch = (
        not args.no_batch
        and selected_device.startswith("cuda")
        and torch.cuda.is_available()
    )
    # Use smaller default batch size (4) to prevent OOM errors; can be overridden with --batch-size
    batch_size = args.batch_size or runner.config.get("evaluation", {}).get(
        "batch_size", 4
    )

    if requested_device and requested_device.startswith("cuda:") and not args.multi_gpu:
        print(f"Benchmark device: {requested_device} -> visible as {selected_device}")
    else:
        print(f"Benchmark device: {selected_device}")
    print(
        f"CUDA available: {torch.cuda.is_available()} | visible GPUs: {torch.cuda.device_count()}"
    )

    if args.wandb_project:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                config={
                    "model": runner.config["model"]["name"],
                    "benchmarks": benchmark_list,
                    "subset_size": runner.subset_size,
                    "full_eval": runner.full_eval,
                    "batch_size": batch_size,
                    "use_batch": use_batch,
                    "use_vllm": args.use_vllm,
                    "seed": args.seed,
                },
            )
        except ImportError:
            print("  WARNING: wandb not installed. Install with `pip install wandb`.")
            args.wandb_project = None

    num_gpus = torch.cuda.device_count() if args.multi_gpu else 0
    summary = {}
    all_rows = []

    if num_gpus > 1 and len(benchmark_list) > 1:
        print(
            f"  Distributing {len(benchmark_list)} benchmarks across {num_gpus} GPUs..."
        )
        chunk_size = max(1, len(benchmark_list) // num_gpus)
        gpu_assignments = {}
        for i, b in enumerate(benchmark_list):
            gpu_id = i % num_gpus
            gpu_assignments.setdefault(gpu_id, []).append(b)

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, benches in gpu_assignments.items():
                for b in benches:
                    futures.append(
                        executor.submit(
                            run_benchmark_on_gpu,
                            gpu_id,
                            b,
                            "config/config.yaml",
                            args.seed,
                            runner.subset_size,
                            runner.full_eval,
                            use_batch,
                            batch_size,
                            args.use_vllm,
                            args.device,
                            args.oracle_selection,
                        )
                    )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Multi-GPU"
            ):
                result = future.result()
                summary[result["benchmark"]] = {
                    k: v for k, v in result.items() if k != "rows"
                }
                all_rows.extend(result["rows"])
                a = result["accuracy"]
                print(
                    f"  [{result['benchmark']}] GPU | Greedy: {a['greedy']:.2%} | CoT: {a['cot']:.2%} | QUBO: {a['qubo']:.2%}"
                )
    else:
        print(f"\n{'=' * 70}")
        print("INITIALIZATION")
        print(f"{'=' * 70}")
        print("[1/6] Loading inference model...")
        sys.stdout.flush()
        _adapter_path = os.environ.get("QUBO_ADAPTER_PATH") or None
        if _adapter_path:
            print(f"[Eval] LoRA adapter: {_adapter_path}")
        inference = InferencePipeline(device=selected_device, use_vllm=args.use_vllm, adapter_path=_adapter_path)
        runtime_device = str(inference.device)

        print("[2/6] Loading sampler (sharing model)...")
        sys.stdout.flush()
        sampler = DiverseSampler(
            device=runtime_device,
            shared_model=inference.model,
            shared_tokenizer=inference.tokenizer,
        )
        print("[3/6] Loading verifier...")
        sys.stdout.flush()
        verifier = ReasonVerifier(device=runtime_device)
        print("[4/6] Loading QUBO builder...")
        sys.stdout.flush()
        qubo_builder = QUBOBuilder(device=runtime_device)
        print("[5/6] Loading solver...")
        sys.stdout.flush()
        solver = SimulatedAnnealingSolver(device=runtime_device)
        print("[6/6] Initialization complete!")
        print("")

        print(f"Runtime device: {runtime_device}")
        print(
            f"Inference model device: {inference.model_input_device if not inference.use_vllm else inference.device}"
        )
        print(
            f"Generation device: {inference.generation_input_device if not inference.use_vllm else inference.device}"
        )
        print(f"Sampler device: {sampler.device}")
        print(f"Verifier device: {verifier.device}")
        print(f"Solver device: {solver.device}")

        # Effective configuration. Printed so a run can be attributed to exact
        # settings from the log alone -- which model, which QUBO weights, and
        # crucially whether chain selection is gold-free or oracle.
        _mc = runner.config.get("model", {})
        _pc = runner.config.get("pipeline", {})
        _qc = runner.config.get("qubo", {})
        print(f"\n{'=' * 60}")
        print("  EFFECTIVE CONFIG")
        print(f"{'=' * 60}")
        print(f"  model            : {_mc.get('name')}")
        print(f"  adapter          : {_adapter_path or 'none (base model)'}")
        print(f"  chain pool       : {_pc.get('num_answers', 3)} samples x 4 perturbations "
              f"= {_pc.get('num_answers', 3) * 4}")
        print(f"  subset_size      : {_pc.get('subset_size')}   max_new_tokens: {_pc.get('max_new_tokens')}")
        print(f"  penalty_weight   : {_qc.get('penalty_weight')}   "
              f"cardinality_penalty: {_qc.get('cardinality_penalty')}")
        print(f"  answer_agree_w   : {_qc.get('answer_agree_weight')}   "
              f"diversity_bonus: {_qc.get('diversity_bonus')}")
        print(f"  chain scoring    : "
              f"{'ORACLE (gold visible -- ABLATION, not deployable)' if args.oracle_selection else 'gold-free (consensus + consistency)'}")
        print(f"  benchmarks       : {', '.join(benchmark_list)}")
        print(f"  questions each   : {'full' if runner.full_eval else runner.subset_size}")
        print(f"{'=' * 60}\n", flush=True)

        csv_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.csv")
        fieldnames = [
            "benchmark",
            "id",
            "question",
            "gold",
            "pred_greedy",
            "pred_cot",
            "pred_qubo",
            "correct_greedy",
            "correct_cot",
            "correct_qubo",
            "runtime_greedy_s",
            "runtime_cot_s",
            "runtime_qubo_s",
            "error",
        ]
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for b in benchmark_list:
            jsonl_path = os.path.join(args.output_dir, f"{args.condition_label}_{b}_results.jsonl")
            print(f"\n{'=' * 60}")
            print(f"Benchmark: {b}")
            print(f"{'=' * 60}")
            print(f"  [STAGE 1/5] Loading benchmark data...")
            sys.stdout.flush()

            try:
                questions, gold_answers = runner.load_benchmark(b)
            except Exception as e:
                print(f"  ❌ Failed to load benchmark {b}: {e}")
                summary[b] = {
                    "error": str(e),
                    "num_samples": 0,
                    "accuracy": {"greedy": 0.0, "cot": 0.0, "qubo": 0.0},
                    "abs_gain_vs_greedy": 0.0,
                    "cot_gain_over_greedy": 0.0,
                }
                continue

            task_type = TASK_TYPE.get(b, "math")
            print(f"  ✓ Loaded {len(questions)} questions")

            # Check if completed or partial cached results exist for this benchmark & condition
            cached_rows = []
            if os.path.exists(jsonl_path):
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as f_check:
                        cached_rows = [json.loads(line) for line in f_check if line.strip()]
                except Exception as cache_err:
                    print(f"  ⚠️ Could not read cache {jsonl_path}: {cache_err}. Starting fresh...")
                    cached_rows = []

            if len(cached_rows) >= len(questions) and len(questions) > 0:
                cached_rows = cached_rows[:len(questions)]
                c_g = sum(r.get("correct_greedy", 0) for r in cached_rows)
                c_c = sum(r.get("correct_cot", 0) for r in cached_rows)
                c_q = sum(r.get("correct_qubo", 0) for r in cached_rows)
                tot = len(cached_rows)
                acc_greedy = c_g / tot if tot else 0.0
                acc_cot = c_c / tot if tot else 0.0
                acc_qubo = c_q / tot if tot else 0.0
                summary[b] = {
                    "accuracy": {"greedy": acc_greedy, "cot": acc_cot, "qubo": acc_qubo},
                    "num_samples": tot,
                    "failed_samples": 0,
                    "abs_gain_vs_greedy": acc_qubo - acc_greedy,
                    "cot_gain_over_greedy": acc_cot - acc_greedy,
                }
                print(f"  [Cache] Found {tot}/{len(questions)} completed results in {jsonl_path}. Skipping generation.")
                print(f"  [{b}] Greedy: {acc_greedy:.2%} | CoT: {acc_cot:.2%} | QUBO: {acc_qubo:.2%}")
                for idx, r in enumerate(cached_rows):
                    csv_row = {
                        "benchmark": b,
                        "id": idx,
                        "question": r.get("question", ""),
                        "gold": r.get("gold", ""),
                        "pred_greedy": r.get("pred_greedy", ""),
                        "pred_cot": r.get("pred_cot", ""),
                        "pred_qubo": r.get("pred_qubo", ""),
                        "correct_greedy": r.get("correct_greedy", 0),
                        "correct_cot": r.get("correct_cot", 0),
                        "correct_qubo": r.get("correct_qubo", 0),
                        "runtime_greedy_s": 0.0,
                        "runtime_cot_s": 0.0,
                        "runtime_qubo_s": 0.0,
                        "error": "",
                    }
                    writer.writerow(csv_row)
                continue

            num_cached = len(cached_rows)
            if num_cached > 0:
                print(f"  [Cache] Resuming {b} from question {num_cached + 1}/{len(questions)} ({num_cached} cached).")
                correct_greedy = sum(r.get("correct_greedy", 0) for r in cached_rows)
                correct_cot = sum(r.get("correct_cot", 0) for r in cached_rows)
                correct_qubo = sum(r.get("correct_qubo", 0) for r in cached_rows)
                total = num_cached
                failed = 0
                for idx, r in enumerate(cached_rows):
                    csv_row = {
                        "benchmark": b,
                        "id": idx,
                        "question": r.get("question", ""),
                        "gold": r.get("gold", ""),
                        "pred_greedy": r.get("pred_greedy", ""),
                        "pred_cot": r.get("pred_cot", ""),
                        "pred_qubo": r.get("pred_qubo", ""),
                        "correct_greedy": r.get("correct_greedy", 0),
                        "correct_cot": r.get("correct_cot", 0),
                        "correct_qubo": r.get("correct_qubo", 0),
                        "runtime_greedy_s": 0.0,
                        "runtime_cot_s": 0.0,
                        "runtime_qubo_s": 0.0,
                        "error": "",
                    }
                    writer.writerow(csv_row)
                questions = questions[num_cached:]
                gold_answers = gold_answers[num_cached:]
                jsonl_file = open(jsonl_path, "a", encoding="utf-8")
            else:
                correct_greedy = 0
                correct_cot = 0
                correct_qubo = 0
                total = 0
                failed = 0
                jsonl_file = open(jsonl_path, "w", encoding="utf-8")

            print(f"  [STAGE 2/5] Starting evaluation...")
            sys.stdout.flush()
            correct_greedy = 0
            correct_cot = 0
            correct_qubo = 0
            total = 0
            failed = 0

            if use_batch and batch_size > 1:
                print(f"  Using batched inference (batch_size={batch_size})")
                sys.stdout.flush()
                batch_greedy_fn = make_batch_greedy(inference)
                batch_cot_fn = make_batch_cot(inference)
                batch_total = (len(questions) + batch_size - 1) // batch_size
                for i in range(0, len(questions), batch_size):
                    batch_num = i // batch_size + 1
                    batch_q = questions[i : i + batch_size]
                    batch_gold = gold_answers[i : i + batch_size]

                    print(
                        f"  [Batch {batch_num}/{batch_total}] Processing questions {i + 1}-{i + len(batch_q)}..."
                    )
                    sys.stdout.flush()

                    try:
                        t0 = time.time()
                        print(f"    → Greedy inference...", end="", flush=True)
                        preds_g = batch_greedy_fn(batch_q)
                        t1 = time.time()
                        print(f" done ({t1 - t0:.1f}s)")

                        print(f"    → CoT inference...", end="", flush=True)
                        preds_c = batch_cot_fn(batch_q)
                        t2 = time.time()
                        print(f" done ({t2 - t1:.1f}s)")

                        for j, q in enumerate(batch_q):
                            print(
                                f"    → QUBO pipeline (Q{i + j + 1})...",
                                end="",
                                flush=True,
                            )
                            tq = time.time()
                            gold = batch_gold[j]
                            pred_qubo = run_qubo_pipeline(
                                sampler,
                                verifier,
                                qubo_builder,
                                solver,
                                inference,
                                q,
                                task_type,
                                gold=gold,
                                oracle_selection=args.oracle_selection,
                            )
                            tq_end = time.time()
                            print(f" done ({tq_end - tq:.1f}s)", flush=True)
                            pred_g_n = extract_answer(preds_g[j], b)
                            pred_c_n = extract_answer(preds_c[j], b)
                            pred_q_n = extract_answer(pred_qubo, b)
                            c_g = int(is_correct(pred_g_n, gold, b, q))
                            c_c = int(is_correct(pred_c_n, gold, b, q))
                            c_q = int(is_correct(pred_q_n, gold, b, q))
                            correct_greedy += c_g
                            correct_cot += c_c
                            correct_qubo += c_q
                            total += 1
                            row = {
                                "benchmark": b,
                                "id": i + j,
                                "question": q,
                                "gold": gold,
                                "pred_greedy": pred_g_n,
                                "pred_cot": pred_c_n,
                                "pred_qubo": pred_q_n,
                                "correct_greedy": c_g,
                                "correct_cot": c_c,
                                "correct_qubo": c_q,
                                "runtime_greedy_s": round((t1 - t0) / len(batch_q), 4),
                                "runtime_cot_s": round((t2 - t1) / len(batch_q), 4),
                                "runtime_qubo_s": round(tq_end - tq, 4),
                                "error": "",
                            }
                            writer.writerow(row)
                            
                            jsonl_row = {
                                "id": i + j,
                                "question": q,
                                "gold": gold,
                                "pred_greedy_raw": preds_g[j],
                                "pred_cot_raw": preds_c[j],
                                "pred_qubo_raw": pred_qubo,
                                "pred_greedy": pred_g_n,
                                "pred_cot": pred_c_n,
                                "pred_qubo": pred_q_n,
                                "correct_greedy": c_g,
                                "correct_cot": c_c,
                                "correct_qubo": c_q,
                            }
                            jsonl_file.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")
                            jsonl_file.flush()

                            all_rows.append(row)
                            if len(all_rows) <= 3:
                                print(
                                    f"  [DEBUG #{len(all_rows)}] {b} gold='{gold}' raw_g='{repr(preds_g[j][:200])}' raw_c='{repr(preds_c[j][:200])}' ext_g='{pred_g_n}' ext_c='{pred_c_n}'"
                                )
                        acc_g = (correct_greedy / total) * 100 if total else 0.0
                        acc_c = (correct_cot / total) * 100 if total else 0.0
                        acc_q = (correct_qubo / total) * 100 if total else 0.0
                    except Exception as e:
                        import traceback

                        print(f"\n  ⚠️  BATCH ERROR (batch {batch_num}): {e}")
                        traceback.print_exc()
                        failed += len(batch_q)
                        for j in range(len(batch_q)):
                            row = {
                                "benchmark": b,
                                "id": i + j,
                                "question": batch_q[j],
                                "gold": batch_gold[j],
                                "pred_greedy": "",
                                "pred_cot": "",
                                "pred_qubo": "",
                                "correct_greedy": 0,
                                "correct_cot": 0,
                                "correct_qubo": 0,
                                "runtime_greedy_s": 0.0,
                                "runtime_cot_s": 0.0,
                                "runtime_qubo_s": 0.0,
                                "error": str(e),
                            }
                            writer.writerow(row)

                    acc_g = (correct_greedy / total) * 100 if total else 0.0
                    acc_c = (correct_cot / total) * 100 if total else 0.0
                    acc_q = (correct_qubo / total) * 100 if total else 0.0
                    bar_len = 10
                    filled = int(bar_len * batch_num / batch_total)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(
                        f"  {b:>8} {bar} {batch_num:>2}/{batch_total}"
                        f"  |  g:{acc_g:>5.1f}%  c:{acc_c:>5.1f}%  q:{acc_q:>5.1f}%",
                        flush=True,
                    )
            else:
                print(f"  Using per-question inference (no batching)")
                sys.stdout.flush()
                for idx, (q, gold) in enumerate(list(zip(questions, gold_answers))):
                    print(
                        f"  [Question {idx + 1}/{len(questions)}] Processing...",
                        flush=True,
                    )
                    try:
                        t0 = time.time()
                        print(f"    → Greedy...", end="", flush=True)
                        pred_greedy = baseline_greedy(inference, q)
                        t1 = time.time()
                        print(f" {t1 - t0:.1f}s", flush=True)

                        print(f"    → CoT...", end="", flush=True)
                        pred_cot = baseline_cot(inference, q)
                        t2 = time.time()
                        print(f" {t2 - t1:.1f}s", flush=True)

                        print(f"    → QUBO pipeline...", end="", flush=True)
                        pred_qubo = run_qubo_pipeline(
                            sampler,
                            verifier,
                            qubo_builder,
                            solver,
                            inference,
                            q,
                            task_type,
                            gold=gold,
                            oracle_selection=args.oracle_selection,
                        )
                        t3 = time.time()
                        print(f" {t3 - t2:.1f}s", flush=True)
                        pred_g_n = extract_answer(pred_greedy, b)
                        pred_c_n = extract_answer(pred_cot, b)
                        pred_q_n = extract_answer(pred_qubo, b)
                        c_g = int(is_correct(pred_g_n, gold, b, q))
                        c_c = int(is_correct(pred_c_n, gold, b, q))
                        c_q = int(is_correct(pred_q_n, gold, b, q))
                        correct_greedy += c_g
                        correct_cot += c_c
                        correct_qubo += c_q
                        total += 1
                        row = {
                            "benchmark": b,
                            "id": idx,
                            "question": q,
                            "gold": gold,
                            "pred_greedy": pred_g_n,
                            "pred_cot": pred_c_n,
                            "pred_qubo": pred_q_n,
                            "correct_greedy": c_g,
                            "correct_cot": c_c,
                            "correct_qubo": c_q,
                            "runtime_greedy_s": round(t1 - t0, 4),
                            "runtime_cot_s": round(t2 - t1, 4),
                            "runtime_qubo_s": round(t3 - t2, 4),
                            "error": "",
                        }
                        writer.writerow(row)
                        
                        jsonl_row = {
                            "id": idx,
                            "question": q,
                            "gold": gold,
                            "pred_greedy_raw": pred_greedy,
                            "pred_cot_raw": pred_cot,
                            "pred_qubo_raw": pred_qubo,
                            "pred_greedy": pred_g_n,
                            "pred_cot": pred_c_n,
                            "pred_qubo": pred_q_n,
                            "correct_greedy": c_g,
                            "correct_cot": c_c,
                            "correct_qubo": c_q,
                        }
                        jsonl_file.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")
                        jsonl_file.flush()

                        all_rows.append(row)
                        if len(all_rows) <= 3:
                            print(
                                f"  [DEBUG #{len(all_rows)}] {b} gold='{gold}' raw_g='{repr(pred_greedy[:200])}' raw_c='{repr(pred_cot[:200])}' ext_g='{pred_g_n}' ext_c='{pred_c_n}'"
                            )
                    except Exception as e:
                        import traceback

                        print(f"  ⚠️  ERROR (question {idx}): {e}")
                        traceback.print_exc()
                        failed += 1
                        row = {
                            "benchmark": b,
                            "id": idx,
                            "question": q,
                            "gold": gold,
                            "pred_greedy": "",
                            "pred_cot": "",
                            "pred_qubo": "",
                            "correct_greedy": 0,
                            "correct_cot": 0,
                            "correct_qubo": 0,
                            "runtime_greedy_s": 0.0,
                            "runtime_cot_s": 0.0,
                            "runtime_qubo_s": 0.0,
                            "error": str(e),
                        }
                        writer.writerow(row)

            acc_greedy = (correct_greedy / total) if total else 0.0
            acc_cot = (correct_cot / total) if total else 0.0
            acc_qubo = (correct_qubo / total) if total else 0.0
            summary[b] = {
                "accuracy": {"greedy": acc_greedy, "cot": acc_cot, "qubo": acc_qubo},
                "num_samples": total,
                "failed_samples": failed,
                "abs_gain_vs_greedy": acc_qubo - acc_greedy,
                "cot_gain_over_greedy": acc_cot - acc_greedy,
            }
            print(
                f"  [{b}] Greedy: {acc_greedy:.2%} | CoT: {acc_cot:.2%} | QUBO: {acc_qubo:.2%} | samples={total} failed={failed}"
            )
            jsonl_file.close()

            if args.wandb_project:
                try:
                    import wandb

                    wandb.log(
                        {
                            f"{b}/accuracy_greedy": acc_greedy,
                            f"{b}/accuracy_cot": acc_cot,
                            f"{b}/accuracy_qubo": acc_qubo,
                            f"{b}/abs_gain_vs_greedy": acc_qubo - acc_greedy,
                            f"{b}/samples": total,
                        }
                    )
                except ImportError:
                    pass

        csv_file.close()

    json_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.json")
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    summary_meta = {
        "timestamp": timestamp,
        "seed": args.seed,
        "benchmarks": benchmark_list,
        "model": cfg.get("model", {}).get("name", "unknown"),
        "device": selected_device,
        "use_batch": use_batch,
        "batch_size": batch_size,
        "use_vllm": args.use_vllm,
        "multi_gpu": num_gpus > 1,
        "results": summary,
    }
    write_summary_json(json_path, summary_meta)

    md_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.md")
    write_summary_markdown(md_path, summary, benchmark_list)

    report_selection_sizes(runner.config.get("pipeline", {}).get("subset_size", 6))
    report_chain_quality()
    report_stage_times()

    print(f"\n{'=' * 60}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"{'=' * 60}")

    if args.wandb_project:
        try:
            import wandb

            wandb.log({"summary": summary_meta})
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
