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


def pick_winning_group(samples: list[dict]) -> list[int]:
    """Majority answer group. Tie-break by mean correctness_score."""
    groups = group_indices_by_answer(samples)
    if not groups:
        return list(range(len(samples)))

    def key(item: tuple[str, list[int]]) -> tuple[int, float]:
        idxs = item[1]
        mean = sum(float(samples[i].get("correctness_score") or 0.0) for i in idxs) / len(
            idxs
        )
        return (len(idxs), mean)

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
