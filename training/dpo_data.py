"""Pure validation of answer-verified, training-only DPO pairs."""
from __future__ import annotations

from evaluation.diagnostic_data import question_hash, read_jsonl
from evaluation.strict_answers import grade


def validate_pairs(path):
    pairs = read_jsonl(path)
    if not pairs:
        raise ValueError("No preference pairs")
    seen = set()
    for pair in pairs:
        if pair.get("role") != "fit":
            raise ValueError("DPO pairs must come from reserved fit questions, never evaluation or tune questions")
        for key in ("question", "prompt", "chosen", "rejected", "chosen_answer", "rejected_answer", "gold", "answer_type"):
            if key not in pair or not str(pair[key]).strip():
                raise ValueError(f"Missing DPO field: {key}")
        if not grade(pair["chosen_answer"], pair)["correct"]:
            raise ValueError("Chosen answer does not match gold")
        rejected = grade(pair["rejected_answer"], pair)
        if not rejected["valid"] or rejected["correct"]:
            raise ValueError("Rejected answer must be parseable and incorrect")
        if not grade(pair["chosen"], pair)["correct"] or not grade(pair["rejected"], pair)["valid"] or grade(pair["rejected"], pair)["correct"]:
            raise ValueError("Completion contradicts preference label")
        if pair["chosen"].strip() == pair["rejected"].strip():
            raise ValueError("Identical chosen/rejected completions")
        digest = question_hash(pair["question"])
        if digest in seen:
            raise ValueError("Repeated preference question; keep one verified pair per question")
        seen.add(digest)
        pair["question_hash"] = digest
    return pairs
