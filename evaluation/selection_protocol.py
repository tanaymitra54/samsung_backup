"""Full 16-trace exact-six selection, independent of benchmark gold labels."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from pipeline.qubo_math import exhaustive_solve

FEATURES = ("sc_score", "nli_score", "conf_score", "arith_score")
N_TRACES, K = 16, 6
PROTOCOL_VERSION = "shared16-fullpool-exact6-v1"


def validate_pool(samples: list[dict], similarity: list[list[float]]) -> None:
    if len(samples) != N_TRACES or len(similarity) != N_TRACES or any(len(r) != N_TRACES for r in similarity):
        raise ValueError("Protocol requires exactly 16 traces and a 16x16 similarity matrix")
    for i, sample in enumerate(samples):
        for name in FEATURES:
            value = float(sample[name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name} at trace {i}")
        for j in range(N_TRACES):
            value = float(similarity[i][j])
            if not math.isfinite(value) or abs(value) > 1.000001 or abs(value - float(similarity[j][i])) > 1e-6:
                raise ValueError("Non-finite, asymmetric or out-of-range similarity")


def learned_score(sample: dict, weights: dict) -> float:
    value = float(weights["intercept"]) + sum(float(weights["coefficients"][f]) * float(sample[f]) for f in FEATURES)
    return 1.0 / (1.0 + math.exp(-min(60.0, max(-60.0, value))))


def vote(samples: list[dict], indices: list[int], scores: list[float] | None = None) -> str:
    totals = defaultdict(float)
    for i in indices:
        answer = samples[i].get("canonical_answer", "")
        if answer:
            totals[answer] += scores[i] if scores is not None else 1.0
    # Insertion order breaks ties by first trace. No gold or verifier tie-break for CoT.
    return max(totals, key=totals.get) if totals else ""


def build_objective(samples: list[dict], similarity: list[list[float]], scores: list[float], penalty: float) -> list[list[float]]:
    n = len(samples)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = -scores[i]
        for j in range(i + 1, n):
            a, b = samples[i].get("canonical_answer", ""), samples[j].get("canonical_answer", "")
            # Preserve the repository objective's within-answer redundancy penalty.
            cost = penalty * max(0.0, float(similarity[i][j])) if a and a == b else 0.0
            matrix[i][j] = matrix[j][i] = cost
    return matrix


def select_methods(samples, similarity, weights, seed):
    validate_pool(samples, similarity)
    scores = [learned_score(s, weights) for s in samples]
    matrix = build_objective(samples, similarity, scores, float(weights["penalty_weight"]))
    state, energy = exhaustive_solve(matrix, K)
    qubo = [i for i, bit in enumerate(state) if bit]
    greedy = sorted(range(N_TRACES), key=lambda i: (-scores[i], i))[:K]
    random_indices = sorted(random.Random(seed).sample(range(N_TRACES), K))
    if len(qubo) != K:
        raise AssertionError("Exact-six invariant failed")
    return {"scores": scores, "qubo_matrix": matrix, "qubo_energy": energy,
            "indices": {"qubo": qubo, "greedy": greedy, "random": random_indices},
            "votes": {"cot16": vote(samples, list(range(N_TRACES))),
                      "weighted_vote16": vote(samples, list(range(N_TRACES)), scores),
                      "qubo6_vote": vote(samples, qubo), "greedy6_vote": vote(samples, greedy),
                      "random6_vote": vote(samples, random_indices)}}
