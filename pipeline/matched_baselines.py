"""Budget-matched controls from the same sampled traces as QUBO."""

from __future__ import annotations

from collections import Counter


def majority_vote_answers(answers: list[str]) -> str:
    cleaned = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def best_of_n_answer(samples: list[dict]) -> str:
    if not samples:
        return ""
    best = max(samples, key=lambda s: float(s.get("correctness_score") or 0.0))
    return str(best.get("answer") or "").strip()
