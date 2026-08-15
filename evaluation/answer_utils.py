"""
==============================================================================
FILE: evaluation/answer_utils.py
ROLE: Answer Extraction & Normalization Utilities
BRANCH ADDITION (abhyuday): Enhanced Multiple-Choice Question (MCQ) extraction logic
in `extract_predicted_answer()`. Replaced naive regex matching with a robust 5-stage
heuristic sequence (explicit tags, choice phrase matching, conclusion lookback in the
last 300 chars, standalone letter detection, and fallback scan).
==============================================================================
"""

import re


def extract_gsm8k_gold(answer_text: str) -> str:
    if not answer_text:
        return ""
    if "####" in answer_text:
        answer_text = answer_text.split("####")[-1]
    return normalize_numeric_answer(answer_text)


def extract_predicted_answer(prediction: str, is_mcq: bool = False) -> str:
    """Extract answer from prediction.

    Args:
        prediction: Model's raw output
        is_mcq: If True, extract MCQ choice (A/B/C/D), else extract numeric answer
    """
    if not prediction:
        return ""

    if is_mcq:
        # MCQ: Look for A, B, C, D, E, F, G, H, I, J
        upper = prediction.strip().upper()
        # 1. Look for explicit "ANSWER: X" or "ANSWER IS X" pattern
        tagged = re.search(r"ANSWER\s*[:\-]?\s*([A-J])\b", upper)
        if tagged:
            return tagged.group(1)
        # 2. Look for "CORRECT ANSWER IS X" or "OPTION X IS CORRECT"
        explicit = re.search(
            r"(?:CORRECT|RIGHT)\s+(?:ANSWER|CHOICE|OPTION)?\s*(?:IS\s+|:\s*)?([A-J])\b", upper
        )
        if explicit:
            return explicit.group(1)
        # 3. Look for "OPTION X" or "CHOICE X" in conclusion (last 300 chars)
        last_chunk = upper[-300:] if len(upper) > 300 else upper
        last_option = re.search(r"(?:OPTION|CHOICE)\s*([A-J])\b", last_chunk)
        if last_option:
            return last_option.group(1)
        # 4. Search for standalone letter in the last 200 characters (where final decision is stated)
        standalone_end = re.findall(r"\b([A-J])\b", last_chunk)
        if standalone_end:
            return standalone_end[-1]
        # 5. Fallback: last single letter found anywhere in the text
        all_letters = re.findall(r"\b([A-J])\b", upper)
        return all_letters[-1] if all_letters else ""
    else:
        # Numerical answer
        if "####" in prediction:
            prediction = prediction.split("####")[-1]
        return normalize_numeric_answer(prediction)


def normalize_numeric_answer(text: str) -> str:
    cleaned = text.strip().replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return cleaned.lower()
    value = matches[-1]
    if "." in value:
        try:
            f = float(value)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except ValueError:
            return value
    return value


def is_correct_prediction(pred: str, gold: str, is_mcq: bool = False) -> bool:
    """Check if prediction is correct.

    Args:
        pred: Predicted answer
        gold: Gold answer
        is_mcq: If True, compare as MCQ (case-insensitive letter match)
    """
    if is_mcq:
        return pred.strip().upper() == gold.strip().upper()
    return pred == gold
