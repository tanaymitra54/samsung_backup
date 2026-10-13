"""Regrade saved evidence; paired inference clusters repeated seeds by question."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from evaluation.selection_protocol import PROTOCOL_VERSION, select_methods
from evaluation.strict_answers import grade

PRIMARY = ("cot16", "cot16_synthesis", "greedy6", "random6", "weighted_vote16", "qubo6_vote", "greedy6_vote", "random6_vote")


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    # Recurrence in log space remains usable for large discordant counts.
    k = min(wins, losses)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    largest = max(terms)
    return min(1.0, 2 * math.exp(largest) * sum(math.exp(v - largest) for v in terms))


def paired_inference(differences: list[float], seed: int = 2718, draws: int = 5000, permutation_draws: int = 20000) -> dict:
    n = len(differences)
    mean = sum(differences) / n if n else 0.0
    if not n:
        return dict(delta=0.0, ci95=[0.0, 0.0], p=1.0)
    if all(d == 0 for d in differences):
        return dict(delta=0.0, ci95=[0.0, 0.0], p=1.0)
    import numpy as np
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=float)
    boot = []
    for start in range(0, draws, 256):
        indices = rng.integers(0, n, size=(min(256, draws - start), n))
        boot.extend(values[indices].mean(axis=1).tolist())
    boot.sort()
    statistic = abs(sum(differences))
    if n <= 16:
        hits = sum(abs(sum(d if (mask >> i) & 1 else -d for i, d in enumerate(differences))) >= statistic - 1e-12 for mask in range(1 << n))
        p = hits / (1 << n)
    else:
        hits = 0
        for start in range(0, permutation_draws, 256):
            signs = rng.integers(0, 2, size=(min(256, permutation_draws - start), n), dtype=np.int8) * 2 - 1
            hits += int(np.count_nonzero(np.abs(signs @ values) >= statistic - 1e-12))
        p = (hits + 1) / (permutation_draws + 1)
    return dict(delta=mean, ci95=[boot[int(0.025 * draws)], boot[min(draws - 1, int(0.975 * draws))]], p=p,
                permutation_draws=(1 << n) if n <= 16 else permutation_draws)


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def verified_record(record: dict, weights: dict) -> dict:
    if record["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol")
    if record.get("error"):
        return record
    example = record["example"]
    from evaluation.diagnostic_data import question_hash
    if example["question_hash"] != question_hash(example["question"]):
        raise ValueError("Question fingerprint disagrees with question text")
    if record.get("model_identity") != weights["model_identity"]:
        raise ValueError("Record model differs from calibration")
    if example.get("role") != "eval":
        raise ValueError("Only reserved evaluation questions can be reported as evaluation")
    if example["question_hash"] in set(weights.get("fit_question_hashes", []) + weights.get("tune_question_hashes", [])):
        raise ValueError("Evaluation question appeared in calibration")
    samples = [dict(s, canonical_answer=grade(s["raw_text"], example)["answer"]) for s in record["samples"]]
    expected = select_methods(samples, record["similarity"], weights, record["seed"])
    if expected["indices"] != record["selection"]["indices"] or expected["votes"] != record["selection"]["votes"]:
        raise ValueError("Saved selections or votes do not match reproducible evidence")
    if abs(expected["qubo_energy"] - record["selection"]["qubo_energy"]) > 1e-8:
        raise ValueError("Saved QUBO energy mismatch")
    methods = dict(record["methods"])
    for name, value in expected["votes"].items():
        methods[name] = {"raw_text": value}
    for name, method in methods.items():
        recomputed = grade(method["raw_text"], example)
        if name in record["methods"] and any(record["methods"][name].get(k) != v for k, v in recomputed.items()):
            raise ValueError("Saved correctness disagrees with raw prediction")
        method.update(recomputed)
    return dict(record, samples=samples, methods=methods, selection=expected)


def failure_category(record: dict, method="qubo6") -> str:
    if record.get("error"):
        return "execution_error"
    example = record["example"]
    correct_indices = {i for i, s in enumerate(record["samples"]) if grade(s["raw_text"], example)["correct"]}
    selected = set(record["selection"]["indices"][method.replace("6", "")])
    if record["methods"][method]["correct"]:
        return "correct_final_without_correct_selected_answer" if not (selected & correct_indices) else "correct_final"
    if not correct_indices:
        return "no_correct_answer_in_pool"
    if not (selected & correct_indices):
        return "selection_missed_correct_answer"
    return "wrong_final_despite_correct_selected_answer"


def summarize(records: list[dict], weights: dict, expected_seeds: list[int], min_questions=30) -> dict:
    records = [verified_record(r, weights) for r in records]
    by_question = defaultdict(list)
    for row in records:
        by_question[row["example"]["question_hash"]].append(row)
    if not by_question:
        raise ValueError("No evaluation records")
    for rows in by_question.values():
        seeds = [r["seed"] for r in rows]
        if sorted(seeds) != sorted(expected_seeds):
            raise ValueError("Missing or duplicated question/seed records; do not summarize a partial run")
        if any(r["example"] != rows[0]["example"] for r in rows):
            raise ValueError("Question metadata or gold changes across seeds")
    groups = defaultdict(list)
    for digest, rows in by_question.items():
        groups[("overall", "all")].append(rows)
        groups[("topic", rows[0]["example"].get("topic", "unlabelled"))].append(rows)
        words = len(rows[0]["example"]["question"].split())
        groups[("observed_question_length", "<=50 words" if words <= 50 else "51-100 words" if words <= 100 else ">100 words")].append(rows)
    comparisons = []
    for (group_type, group), questions in groups.items():
        errors = sum(bool(r.get("error")) for rows in questions for r in rows)
        for control in PRIMARY:
            qrate, crate, differences = [], [], []
            for rows in questions:
                q = sum(int(not r.get("error") and r["methods"]["qubo6"]["correct"]) for r in rows) / len(rows)
                c = sum(int(not r.get("error") and r["methods"][control]["correct"]) for r in rows) / len(rows)
                qrate.append(q)
                crate.append(c)
                differences.append(q - c)
            per_seed = []
            for seed in expected_seeds:
                subset = [r for rows in questions for r in rows if r["seed"] == seed]
                wins = sum(not r.get("error") and r["methods"]["qubo6"]["correct"] and not r["methods"][control]["correct"] for r in subset)
                losses = sum(not r.get("error") and not r["methods"]["qubo6"]["correct"] and r["methods"][control]["correct"] for r in subset)
                per_seed.append(dict(seed=seed, wins=wins, losses=losses, mcnemar_p=exact_mcnemar(wins, losses), delta=(wins - losses) / len(subset)))
            comparisons.append(dict(group_type=group_type, group=group, comparator=control,
                                    n_questions=len(questions), n_records=len(questions) * len(expected_seeds), errors=errors,
                                    qubo_accuracy=sum(qrate) / len(qrate), comparator_accuracy=sum(crate) / len(crate),
                                    per_seed=per_seed, **paired_inference(differences, permutation_draws=max(20000, 100 * len(groups) * len(PRIMARY)))))
    adjusted = holm_adjust([c["p"] for c in comparisons])
    for comparison, p in zip(comparisons, adjusted):
        comparison["holm_p"] = p
        stable = all(s["delta"] > 0 for s in comparison["per_seed"]) or all(s["delta"] < 0 for s in comparison["per_seed"])
        adequate = comparison["n_questions"] >= min_questions and len(expected_seeds) >= 3
        comparison["evidence"] = "inconclusive"
        if adequate and stable and comparison["errors"] == 0 and p <= 0.05:
            if comparison["ci95"][0] > 0:
                comparison["evidence"] = "repeatable_advantage_on_this_panel"
            elif comparison["ci95"][1] < 0:
                comparison["evidence"] = "repeatable_disadvantage_on_this_panel"
    failures = defaultdict(int)
    trace_health = dict(traces=0, invalid_answers=0, incorrect_answers=0, output_token_limits=0,
                        high_score_wrong_answers=0, high_score_threshold=0.8, brier_score=None)
    squared_errors = []
    final_limits = defaultdict(int)
    question_details = []
    for digest, rows in by_question.items():
        counts = {name: sum(int(not r.get("error") and r["methods"][name]["correct"]) for r in rows) for name in ("qubo6", "cot16", "greedy6")}
        categories = [failure_category(r) for r in rows]
        for category in categories:
            failures[category] += 1
        for r in rows:
            if r.get("error"):
                continue
            for sample, score in zip(r["samples"], r["selection"]["scores"]):
                label = grade(sample["raw_text"], r["example"])
                trace_health["traces"] += 1
                trace_health["invalid_answers"] += int(not label["valid"])
                trace_health["incorrect_answers"] += int(not label["correct"])
                trace_health["output_token_limits"] += int(sample.get("token_limit_reached", False))
                trace_health["high_score_wrong_answers"] += int(score >= 0.8 and not label["correct"])
                squared_errors.append((score - int(label["correct"])) ** 2)
            for method, result in r["methods"].items():
                final_limits[method] += int(result.get("token_limit_reached", False))
        question_details.append(dict(question_hash=digest, source_id=rows[0]["example"]["source_id"],
                                     topic=rows[0]["example"].get("topic", "unlabelled"), correct_by_method=counts,
                                     n_seeds=len(rows), categories=categories))
    trace_health["brier_score"] = sum(squared_errors) / len(squared_errors) if squared_errors else None
    return dict(protocol=PROTOCOL_VERSION, n_questions=len(by_question), seeds=expected_seeds,
                errors=sum(bool(r.get("error")) for r in records), comparisons=comparisons,
                failure_counts=dict(failures), questions=question_details,
                trace_health=trace_health, final_output_token_limits=dict(final_limits),
                limitations=["Answer correctness does not verify reasoning truth or detect hallucinations.",
                             "Topic results are exploratory; confirm on a fresh reserved panel after diagnosis.",
                             "Unlabelled datasets need independently reviewed topic annotations.",
                             "CoT majority voting uses 16 generations; QUBO/Greedy/random each add one synthesis generation.",
                             "CoT-16 synthesis is a separate 16-trace plus one generation control; actual input token costs still differ.",
                             "Repeated seeds are clustered by question, not treated as independent questions."])


def write_report(directory: Path, records: list[dict], weights: dict, seeds: list[int]) -> dict:
    summary = summarize(records, weights, seeds)
    (directory / "diagnosis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Verified QUBO diagnostic", "", f"Questions: {summary['n_questions']}; seeds: {seeds}; execution errors: {summary['errors']}.", "",
             "QUBO selects exactly six from the same 16 CoT traces used by all controls. Scoring is calibrated and frozen.", "",
             "| Group | Control | N questions | QUBO | Control | Delta | Paired 95% CI | Holm p | Evidence |",
             "|---|---|---:|---:|---:|---:|---|---:|---|"]
    for c in summary["comparisons"]:
        lo, hi = c["ci95"]
        lines.append(f"| {c['group_type']}: {c['group']} | {c['comparator']} | {c['n_questions']} | {c['qubo_accuracy']:.1%} | {c['comparator_accuracy']:.1%} | {c['delta']:+.1%} | [{lo:+.1%}, {hi:+.1%}] | {c['holm_p']:.4f} | {c['evidence']} |")
    lines += ["", "## Observed failure counts", ""] + [f"- {k}: {v}" for k, v in summary["failure_counts"].items()]
    lines += ["", "## Trace and generation health", ""] + [f"- {k}: {v}" for k, v in summary["trace_health"].items()]
    lines += ["- High-score wrong answers are confidence errors; these counts do not establish hallucinations."]
    lines += ["", "## Limits", ""] + [f"- {limit}" for limit in summary["limitations"]]
    lines += ["", "Evidence: results.jsonl contains raw chains, raw final answers, selections, QUBO matrices, seeds, token counts and scores. human_review.csv is an unfilled annotation sheet, not a hallucination verdict."]
    (directory / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ["source_id", "seed", "topic", "question", "gold", "raw_qubo", "raw_cot_vote", "raw_greedy", "failure_category", "selected_trace_text", "reasoning_verified", "unsupported_claim", "reviewer", "evidence_notes"]
    with (directory / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            e = r["example"]
            methods = r.get("methods", {})
            indices = r.get("selection", {}).get("indices", {}).get("qubo", [])
            writer.writerow(dict(source_id=e["source_id"], seed=r["seed"], topic=e.get("topic", "unlabelled"),
                                 question=e["question"], gold=e["gold"], raw_qubo=methods.get("qubo6", {}).get("raw_text", ""),
                                 raw_cot_vote=methods.get("cot16", {}).get("raw_text", ""), raw_greedy=methods.get("greedy6", {}).get("raw_text", ""),
                                 failure_category=failure_category(r), selected_trace_text="\n\n".join(r["samples"][i]["raw_text"] for i in indices)))
    return summary
