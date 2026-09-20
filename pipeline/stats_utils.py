"""Eval statistics: binomial CI and expected calibration error."""

from __future__ import annotations

import math


def binomial_ci(
    successes: int, n: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Wilson score interval. Returns (rate, low, high)."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)


def expected_calibration_error(
    confidences: list[float], correct: list[int], n_bins: int = 10
) -> float:
    if not confidences or len(confidences) != len(correct) or n_bins <= 0:
        return 0.0
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    conf_bins: list[list[float]] = [[] for _ in range(n_bins)]
    for conf, y in zip(confidences, correct):
        c = min(max(float(conf), 0.0), 1.0)
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append(int(y))
        conf_bins[idx].append(c)
    n = len(confidences)
    ece = 0.0
    for ys, cs in zip(bins, conf_bins):
        if not ys:
            continue
        acc = sum(ys) / len(ys)
        mean_p = sum(cs) / len(cs)
        ece += (len(ys) / n) * abs(acc - mean_p)
    return ece
