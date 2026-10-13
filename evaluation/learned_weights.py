"""Fit trace scores on fit questions and choose redundancy penalty on tune questions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from evaluation.selection_protocol import FEATURES, PROTOCOL_VERSION, select_methods, validate_pool
from evaluation.strict_answers import grade


def artifact_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_records(records: list[dict], role: str) -> None:
    if not records:
        raise ValueError(f"No {role} records")
    for record in records:
        if record["example"].get("role") != role or record.get("error"):
            raise ValueError(f"Need complete {role} records; test results and failed questions cannot be used")
        validate_pool(record["samples"], record["similarity"])
        if record.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("Incompatible protocol")


def fit_weights(fit: list[dict], tune: list[dict], seed=42, candidates=16) -> dict:
    import numpy as np
    validate_records(fit, "fit")
    validate_records(tune, "tune")
    fit_ids = {r["example"]["question_hash"] for r in fit}
    tune_ids = {r["example"]["question_hash"] for r in tune}
    if fit_ids & tune_ids:
        raise ValueError("Fit/tune question overlap")
    identities = {artifact_hash(r["model_identity"]) for r in fit + tune}
    if len(identities) != 1:
        raise ValueError("Calibration mixes model/adapter/generation settings")
    revisions = {artifact_hash(r.get("resolved_revisions", {})) for r in fit + tune}
    if len(revisions) != 1:
        raise ValueError("Calibration mixes resolved model revisions")
    implementations = {r.get("implementation_sha256") for r in fit + tune}
    if len(implementations) != 1:
        raise ValueError("Calibration mixes generation/scoring code versions")
    if len(fit_ids) < 20 or len(tune_ids) < 20:
        raise ValueError("Need at least 20 distinct questions in each calibration partition")
    x, y, sample_weights = [], [], []
    counts = {}
    for record in fit:
        key = record["example"]["question_hash"]
        counts[key] = counts.get(key, 0) + 1
    for record in fit:
        for sample in record["samples"]:
            x.append([1.0] + [float(sample[f]) for f in FEATURES])
            y.append(float(grade(sample["raw_text"], record["example"])["correct"]))
            sample_weights.append(1.0 / counts[record["example"]["question_hash"]])
    x, y, sw = np.array(x), np.array(y), np.array(sample_weights)
    if len(set(y)) != 2:
        raise ValueError("Score learning requires both correct and incorrect fit traces")
    coeff = np.zeros(5)
    # Convex logistic loss; projected gradient enforces nonnegative signal coefficients.
    # No manually picked SC/NLI/confidence/arithmetic weights are loaded.
    step = 1.0 / (0.25 * float(np.linalg.norm(x * np.sqrt(sw[:, None]), 2) ** 2 / sw.sum()) + 0.001)
    for _ in range(4000):
        prob = 1.0 / (1.0 + np.exp(-np.clip(x @ coeff, -60, 60)))
        gradient = x.T @ ((prob - y) * sw) / sw.sum()
        gradient[1:] += 0.001 * coeff[1:]
        next_coeff = coeff - step * gradient
        next_coeff[1:] = np.maximum(0.0, next_coeff[1:])
        if np.max(np.abs(next_coeff - coeff)) < 1e-7:
            coeff = next_coeff
            break
        coeff = next_coeff
    artifact = dict(protocol=PROTOCOL_VERSION, intercept=float(coeff[0]),
                    coefficients={f: float(v) for f, v in zip(FEATURES, coeff[1:])},
                    model_identity=fit[0]["model_identity"], fit_question_hashes=sorted(fit_ids),
                    resolved_revisions=fit[0].get("resolved_revisions", {}),
                    implementation_sha256=fit[0].get("implementation_sha256"),
                    tune_question_hashes=sorted(tune_ids), calibration_seed=seed,
                    benchmark_coverage=sorted({r["example"]["benchmark"] for r in fit + tune}),
                    score_objective="regularized_nonnegative_logistic_answer_correctness",
                    selection_objective="tune_selected_six_majority_answer_accuracy",
                    limitation="Penalty is calibrated on selected-answer votes, not end-to-end synthesis accuracy.")
    rng = np.random.default_rng(seed)
    penalties = [0.0] + sorted(float(v) for v in np.exp(rng.uniform(math.log(0.001), math.log(1.0), candidates - 1)))
    trials = []
    for penalty in penalties:
        artifact["penalty_weight"] = penalty
        per_question = {}
        for record in tune:
            # Reparse cached predictions to prevent stale or fabricated canonical labels.
            samples = [dict(s, canonical_answer=grade(s["raw_text"], record["example"])["answer"]) for s in record["samples"]]
            result = select_methods(samples, record["similarity"], artifact, record["seed"])
            correct = grade(result["votes"]["qubo6_vote"], record["example"])["correct"]
            per_question.setdefault(record["example"]["question_hash"], []).append(int(correct))
        metric = sum(sum(v) / len(v) for v in per_question.values()) / len(per_question)
        trials.append(dict(penalty_weight=penalty, tune_accuracy=metric))
    best = max(trials, key=lambda trial: (trial["tune_accuracy"], -trial["penalty_weight"]))
    artifact.update(best)
    artifact["penalty_trials"] = trials
    artifact["input_hashes"] = {"fit": artifact_hash({"records": fit}), "tune": artifact_hash({"records": tune})}
    return artifact


def load_weights(path: str | Path, identity: dict, examples: list[dict]) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value["protocol"] != PROTOCOL_VERSION or value["model_identity"] != identity:
        raise ValueError("Weights belong to a different model, adapter, or sampling configuration; recalibrate")
    numbers = [float(value["intercept"]), float(value["penalty_weight"])] + [float(value["coefficients"][f]) for f in FEATURES]
    if not all(math.isfinite(v) for v in numbers) or any(v < 0 for v in numbers[1:]):
        raise ValueError("Non-finite or negative learned coefficients")
    calibration = set(value["fit_question_hashes"] + value["tune_question_hashes"])
    if calibration & {e["question_hash"] for e in examples}:
        raise ValueError("Evaluation overlaps calibration questions")
    if {e["benchmark"] for e in examples} - set(value["benchmark_coverage"]):
        raise ValueError("Benchmark was absent from weight calibration")
    return value
