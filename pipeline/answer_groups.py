"""Group reasoning traces by extracted final answer."""

from __future__ import annotations


def normalise_answer(answer: str | None) -> str:
    return (answer or "").strip().lower()


def group_indices_by_answer(samples: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, sample in enumerate(samples):
        key = normalise_answer(sample.get("answer"))
        if not key:
            continue
        groups.setdefault(key, []).append(i)
    return groups


def pick_winning_group(samples: list[dict], weighted: bool = True) -> list[int]:
    """Winning answer group.

    weighted=True (default): highest sum of correctness_score — a
    confidence-weighted vote, so a small high-confidence group can beat a
    larger low-confidence one. Falls back to a plain majority vote when every
    score is missing/zero.

    weighted=False: plain majority (count), tie-broken by mean correctness.
    """
    groups = group_indices_by_answer(samples)
    if not groups:
        return list(range(len(samples)))

    def score_stats(idxs: list[int]) -> tuple[float, int, float]:
        scores = [float(samples[i].get("correctness_score") or 0.0) for i in idxs]
        return sum(scores), len(idxs), sum(scores) / len(scores)

    def key(item: tuple[str, list[int]]) -> tuple[float, int, float]:
        total, count, mean = score_stats(item[1])
        if weighted:
            return (total, count, mean)
        return (0.0, count, mean)

    _winner, idxs = max(groups.items(), key=key)
    return idxs


def keep_one_answer_group(samples: list[dict], indices: list[int]) -> list[int]:
    """Drop selected traces that conflict with the majority selected answer."""
    if not indices:
        return []
    selected = [samples[i] for i in indices]
    local_winner = pick_winning_group(selected)
    if not selected or not any(normalise_answer(s.get("answer")) for s in selected):
        return list(indices)
    winner_keys = {
        normalise_answer(selected[j].get("answer")) for j in local_winner
    }
    return [
        i
        for i in indices
        if normalise_answer(samples[i].get("answer")) in winner_keys
    ]
