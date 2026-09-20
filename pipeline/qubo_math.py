"""QUBO energy and exact solvers. No torch / sentence-transformers."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence


def qubo_energy(Q: Sequence[Sequence[float]], x: Sequence[int]) -> float:
    """E = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j.

    Off-diagonal Q_ij is the intended pairwise cost once. Do not use x^T Q x
    on a symmetric Q — that doubles every pair.
    """
    n = len(x)
    energy = 0.0
    for i in range(n):
        if not x[i]:
            continue
        energy += float(Q[i][i])
        for j in range(i + 1, n):
            if x[j]:
                energy += float(Q[i][j])
    return energy


def pairwise_penalty(
    cos: float, answers_agree: bool, penalty_weight: float
) -> float:
    """Penalise duplicate rationales inside one answer group only."""
    if not answers_agree:
        return 0.0
    return float(cos) * float(penalty_weight)


def exhaustive_solve(
    Q: Sequence[Sequence[float]], k: int | None = None
) -> tuple[list[int], float]:
    """Ground-truth minimiser. If k is set, only size-k subsets are allowed."""
    n = len(Q)
    if n == 0:
        return [], 0.0
    if k is not None:
        k = min(max(int(k), 0), n)
        if k == 0:
            return [0] * n, 0.0
        index_sets = combinations(range(n), k)
    else:
        index_sets = (
            combo
            for r in range(0, n + 1)
            for combo in combinations(range(n), r)
        )

    best_state = [0] * n
    best_energy = 0.0
    found = False
    for combo in index_sets:
        state = [0] * n
        for i in combo:
            state[i] = 1
        energy = qubo_energy(Q, state)
        if not found or energy < best_energy:
            best_energy = energy
            best_state = state
            found = True
    return best_state, best_energy


def exact_k_swap_anneal(
    Q: Sequence[Sequence[float]],
    k: int,
    *,
    initial_temp: float = 100.0,
    final_temp: float = 0.01,
    cooling_rate: float = 0.99,
    iterations: int = 500,
    num_reads: int = 8,
    rng_seed: int | None = 0,
) -> tuple[list[int], float]:
    """SA that never leaves the exact-k Hamming slice. For n > 20 only."""
    import random

    n = len(Q)
    if n == 0:
        return [], 0.0
    k = min(max(int(k), 0), n)
    if k == 0:
        return [0] * n, 0.0

    rng = random.Random(rng_seed)
    best_state = None
    best_energy = float("inf")

    for _ in range(num_reads):
        chosen = set(rng.sample(range(n), k))
        state = [1 if i in chosen else 0 for i in range(n)]
        energy = qubo_energy(Q, state)
        temp = initial_temp
        for _step in range(iterations):
            ones = [i for i in range(n) if state[i] == 1]
            zeros = [i for i in range(n) if state[i] == 0]
            if not ones or not zeros:
                break
            off_i = rng.choice(ones)
            on_j = rng.choice(zeros)
            state[off_i] = 0
            state[on_j] = 1
            new_energy = qubo_energy(Q, state)
            delta = new_energy - energy
            if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temp, 1e-8)):
                energy = new_energy
            else:
                state[off_i] = 1
                state[on_j] = 0
            temp = max(final_temp, temp * cooling_rate)
        if energy < best_energy:
            best_energy = energy
            best_state = state[:]

    return best_state if best_state is not None else [0] * n, best_energy


def solve_qubo(
    Q: Sequence[Sequence[float]],
    k: int | None = None,
    *,
    exhaustive_max_n: int = 20,
) -> tuple[list[int], float]:
    n = len(Q)
    if k is not None and n > exhaustive_max_n:
        return exact_k_swap_anneal(Q, k)
    return exhaustive_solve(Q, k=k)
