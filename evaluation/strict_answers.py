"""Conservative benchmark grading. Never infer a label from arbitrary prose."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def final_segment(text: str) -> str:
    text = str(text or "").strip()
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    tags = list(re.finditer(
        r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*([^\n]+)", text, re.I
    ))
    if tags:
        return tags[-1].group(1).strip()
    tags = list(re.finditer(r"\b(?:the\s+)?(?:final\s+)?answer\s+is\s+([^\n]+)", text, re.I))
    return tags[-1].group(1).strip() if tags else text


def canonical_answer(text: str, answer_type: str, labels: list[str] | None = None) -> str:
    value = final_segment(text).strip().strip("`").strip()
    if answer_type == "choice":
        match = re.fullmatch(r"\(?([A-Z0-9])\)?[.]?", value.upper())
        return match.group(1) if match and match.group(1) in (labels or list("ABCD")) else ""
    if answer_type == "boolean":
        return {"yes": "yes", "true": "yes", "no": "no", "false": "no"}.get(value.lower().rstrip("."), "")
    if answer_type == "numeric":
        value = value.replace(",", "").strip()
        match = re.fullmatch(r"[$]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*[.]?", value)
        if not match:
            return ""
        try:
            number = Decimal(match.group(1))
            return format(number.normalize(), "f") if number.is_finite() else ""
        except InvalidOperation:
            return ""
    if answer_type == "text":
        return re.sub(r"\s+", " ", value).strip().lower()
    raise ValueError(f"Unsupported answer_type: {answer_type}")


def grade(text: str, example: dict) -> dict:
    kind = example["answer_type"]
    labels = example.get("labels")
    gold = canonical_answer(str(example["gold"]), kind, labels)
    if not gold:
        raise ValueError(f"Unparseable gold answer for {example.get('source_id')}")
    pred = canonical_answer(text, kind, labels)
    return {"answer": pred, "valid": bool(pred), "correct": bool(pred and pred == gold)}
