import re


def extract_gsm8k_gold(answer_text: str) -> str:
    if not answer_text:
        return ""
    if "####" in answer_text:
        answer_text = answer_text.split("####")[-1]
    return normalize_numeric_answer(answer_text)


def extract_predicted_answer(prediction: str) -> str:
    if not prediction:
        return ""
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


def is_correct_prediction(pred: str, gold: str) -> bool:
    return pred == gold
