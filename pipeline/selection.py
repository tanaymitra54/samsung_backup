"""Map a QUBO bitstring back to sample indices."""

from __future__ import annotations


def select_best_reasoning(
    state,
    qubo_var_indices: list[int],
    n_samples: int,
    subset_size: int,
    scores: list[float] | None = None,
) -> list[int]:
    selected = [
        qubo_var_indices[i]
        for i in range(len(state))
        if int(state[i]) == 1 and i < len(qubo_var_indices)
    ]
    if selected:
        return selected
    print(
        "WARNING: QUBO returned an empty selection; "
        "falling back to top-score traces, not first-N by index."
    )
    if scores is None:
        scores = [0.0] * n_samples
    ranked = sorted(range(n_samples), key=lambda i: scores[i], reverse=True)
    return ranked[: min(subset_size, n_samples)]
