"""DPO labels: independently checked correct vs incorrect, not self-score."""

from __future__ import annotations

import re


def _last_number(text: str | None) -> float | None:
    if not text:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def chain_matches_gold(chain: dict, gold, task_type: str = "math") -> bool:
    pred = chain.get("answer") or ""
    gold_s = "" if gold is None else str(gold)
    if task_type == "math":
        pred_n = _last_number(pred) or _last_number(chain.get("reason"))
        gold_n = _last_number(gold_s)
        if pred_n is None or gold_n is None:
            return False
        tol = max(0.01, 0.01 * abs(gold_n))
        return abs(pred_n - gold_n) <= tol
    pred_n = _norm(pred)
    gold_n = _norm(gold_s)
    if not pred_n or not gold_n:
        return False
    if pred_n == gold_n:
        return True
    yes_no = {"yes": "true", "true": "yes", "no": "false", "false": "no"}
    return yes_no.get(pred_n) == gold_n or gold_n in pred_n or pred_n in gold_n


def select_verified_pair(
    pool: list[dict], gold, task_type: str = "math"
) -> tuple[dict, dict] | None:
    correct = [c for c in pool if chain_matches_gold(c, gold, task_type)]
    incorrect = [c for c in pool if not chain_matches_gold(c, gold, task_type)]
    if not correct or not incorrect:
        return None
    chosen = max(correct, key=lambda c: float(c.get("correctness_score") or 0.0))
    rejected = min(incorrect, key=lambda c: float(c.get("correctness_score") or 0.0))
    if (chosen.get("reason") or "").strip() == (rejected.get("reason") or "").strip() and (
        chosen.get("answer") or ""
    ).strip() == (rejected.get("answer") or "").strip():
        return None
    return chosen, rejected
