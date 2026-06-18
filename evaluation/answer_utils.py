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
        # MCQ: Look for A, B, C, D
        upper = prediction.strip().upper()
        # Look for explicit "ANSWER: X" or "CORRECT ANSWER IS X"
        tagged = re.search(r"ANSWER\s*[:\-]?\s*([A-D])\b", upper)
        if tagged:
            return tagged.group(1)
        # Look for explicit "CORRECT ANSWER IS X" pattern
        explicit = re.search(r"(?:CORRECT|RIGHT)\s+ANSWER\s+(?:IS\s+|:\s*)?([A-D])\b", upper)
        if explicit:
            return explicit.group(1)
        # Look for single letter answer (first occurrence)
        direct = re.search(r"\b([A-D])\b", upper)
        if direct:
            return direct.group(1)
        # Last resort: return first letter found anywhere
        all_letters = re.findall(r"[A-D]", upper)
        return all_letters[0] if all_letters else ""
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
