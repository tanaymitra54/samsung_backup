"""Protocol checks for Debanjan review: exact-K, energy, groups, labels, CIs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.answer_groups import (
    keep_one_answer_group,
    normalise_answer,
    pick_winning_group,
)
from pipeline.preference_labels import chain_matches_gold, select_verified_pair
from pipeline.matched_baselines import best_of_n_answer, majority_vote_answers
from pipeline.qubo_math import exhaustive_solve, pairwise_penalty, qubo_energy
from pipeline.selection import select_best_reasoning
from pipeline.stats_utils import binomial_ci, expected_calibration_error


def test_qubo_energy_counts_pairwise_cost_once():
    Q = [[-1.0, 0.5], [0.5, -1.0]]
    x = [1, 1]
    # Diagonal -1 + -1 and one pairwise 0.5, not 1.0 from x^T Q x.
    assert abs(qubo_energy(Q, x) - (-1.5)) < 1e-9


def test_exhaustive_exact_k_always_selects_k():
    Q = [
        [-1.0, 0.2, 0.2, 0.2],
        [0.2, -0.8, 0.2, 0.2],
        [0.2, 0.2, -0.5, 0.2],
        [0.2, 0.2, 0.2, -0.1],
    ]
    state, _ = exhaustive_solve(Q, k=2)
    assert sum(state) == 2
    assert set(state) <= {0, 1}


def test_exhaustive_finds_known_minimum():
    Q = [
        [-2.0, 5.0, 5.0],
        [5.0, -0.1, 5.0],
        [5.0, 5.0, -0.1],
    ]
    state, energy = exhaustive_solve(Q, k=None)
    assert state == [1, 0, 0]
    assert abs(energy - (-2.0)) < 1e-9


def test_pairwise_penalty_only_within_same_answer():
    assert pairwise_penalty(cos=0.9, answers_agree=True, penalty_weight=2.0) == 1.8
    assert pairwise_penalty(cos=0.9, answers_agree=False, penalty_weight=2.0) == 0.0


def test_empty_answers_are_not_one_group():
    samples = [
        {"answer": "", "correctness_score": 0.9},
        {"answer": "", "correctness_score": 0.8},
        {"answer": "12", "correctness_score": 0.4},
        {"answer": "12", "correctness_score": 0.3},
    ]
    assert pick_winning_group(samples) == [2, 3]


def test_winning_group_is_majority_then_score():
    samples = [
        {"answer": "A", "correctness_score": 0.2},
        {"answer": "A", "correctness_score": 0.2},
        {"answer": "B", "correctness_score": 0.99},
    ]
    assert pick_winning_group(samples) == [0, 1]


def test_final_prompt_drops_conflicting_answers():
    samples = [
        {"answer": "4", "reason": "two plus two"},
        {"answer": "5", "reason": "wrong"},
        {"answer": "4", "reason": "2+2=4"},
    ]
    kept = keep_one_answer_group(samples, [0, 1, 2])
    assert kept == [0, 2]


def test_empty_qubo_fallback_is_top_score_not_first_indices():
    state = [0, 0, 0]
    scores = [0.1, 0.9, 0.2]
    selected = select_best_reasoning(
        state, [0, 1, 2], n_samples=3, subset_size=1, scores=scores
    )
    assert selected == [1]


def test_binomial_ci_contains_rate():
    p, lo, hi = binomial_ci(70, 100)
    assert abs(p - 0.7) < 1e-9
    assert 0.5 < lo < 0.7 < hi < 0.9


def test_ece_zero_when_perfectly_calibrated():
    conf = [0.0, 0.0, 1.0, 1.0]
    correct = [0, 0, 1, 1]
    assert expected_calibration_error(conf, correct, n_bins=2) == 0.0


def test_preference_pair_requires_verified_correct_and_incorrect():
    gold = "42"
    pool = [
        {"reason": "good", "answer": "42", "correctness_score": 0.4},
        {"reason": "bad", "answer": "7", "correctness_score": 0.9},
    ]
    pair = select_verified_pair(pool, gold, task_type="math")
    assert pair is not None
    chosen, rejected = pair
    assert chain_matches_gold(chosen, gold, "math")
    assert not chain_matches_gold(rejected, gold, "math")
    # Highest self-score is the wrong chain; label must ignore that.
    assert chosen["answer"] == "42"
    assert rejected["answer"] == "7"


def test_preference_pair_none_without_wrong_chain():
    gold = "42"
    pool = [
        {"reason": "a", "answer": "42", "correctness_score": 0.9},
        {"reason": "b", "answer": "42", "correctness_score": 0.1},
    ]
    assert select_verified_pair(pool, gold, task_type="math") is None


def test_normalise_answer_is_stable():
    assert normalise_answer("  Yes ") == normalise_answer("yes")


def test_exhaustive_n16_k6_selects_exactly_six():
    n, k = 16, 6
    Q = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Q[i][i] = -float(n - i)
    state, _ = exhaustive_solve(Q, k=k)
    assert sum(state) == 6
    assert state[:6] == [1, 1, 1, 1, 1, 1]


def test_majority_and_best_of_share_pool():
    assert majority_vote_answers(["4", "4", "5"]) == "4"
    samples = [
        {"answer": "5", "correctness_score": 0.9},
        {"answer": "4", "correctness_score": 0.1},
    ]
    assert best_of_n_answer(samples) == "5"


if __name__ == "__main__":
    names = [n for n, fn in globals().items() if n.startswith("test_") and callable(fn)]
    failed = 0
    for name in names:
        try:
            globals()[name]()
            print("ok", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, type(exc).__name__, exc)
    if failed:
        raise SystemExit(1)
    print(f"{len(names)} passed")
