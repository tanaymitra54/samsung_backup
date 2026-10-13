"""DPO labels: independently checked correct vs incorrect, not self-score."""

from __future__ import annotations

import re
from evaluation.strict_answers import canonical_answer


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
        kind = "numeric"
    elif gold_s.strip().lower() in {"yes", "no", "true", "false"}:
        kind = "boolean"
    elif re.fullmatch(r"\(?[A-Z]\)?", gold_s.strip()):
        kind = "choice"
    else:
        kind = "text"
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    predicted = canonical_answer(pred or chain.get("reason", ""), kind, labels)
    target = canonical_answer(gold_s, kind, labels)
    return bool(predicted and target and predicted == target)


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
