import re
import yaml
import numpy as np
import torch
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.device_utils import resolve_device


class ReasonVerifier:
    def __init__(self, config_path: str = "config/config.yaml", device: str | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        verifier_cfg = self.config["verifier"]
        self.math_mode = verifier_cfg["math_mode"]
        self.nli_threshold = verifier_cfg["nli_threshold"]

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        self.device = resolve_device(preferred_device)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            verifier_cfg["nli_model"],
        )
        self.nli_model = self.nli_model.to("cpu")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            verifier_cfg["nli_model"]
        )
        self.nli_model.eval()

    def _extract_arithmetic(self, text: str) -> list[float]:
        pattern = r"(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)"
        matches = re.findall(pattern, text)
        results = []
        for a, op, b in matches:
            a, b = float(a), float(b)
            if op == "+":
                results.append(a + b)
            elif op == "-":
                results.append(a - b)
            elif op == "*":
                results.append(a * b)
            elif op == "/" and b != 0:
                results.append(a / b)
        return results

    def _extract_last_number(self, text: str) -> Optional[float]:
        cleaned = text.strip().replace(",", "")
        matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if not matches:
            return None
        try:
            v = float(matches[-1])
            return int(v) if v.is_integer() else v
        except ValueError:
            return None

    def verify_math(self, reason: str, expected_answer: Optional[str] = None, predicted_answer: Optional[str] = None) -> float:
        computations = self._extract_arithmetic(reason)
        if not computations:
            base = 0.3
        else:
            consistency = 1.0
            if len(computations) > 1:
                diffs = [abs(computations[i] - computations[i + 1]) for i in range(len(computations) - 1)]
                consistency = 1.0 / (1.0 + np.mean(diffs))
            base = min(1.0, consistency)

        # TANAY'S FIX: Check both reason and answer fields for gold match
        if expected_answer is not None:
            gold_num = self._extract_last_number(expected_answer)
            if gold_num is not None:
                # First check if reason contains the gold answer
                pred_num = self._extract_last_number(reason)
                if pred_num is not None and abs(pred_num - gold_num) < 0.01:
                    return 0.6 * 1.0 + 0.4 * base
                # Then check predicted_answer field if provided
                if predicted_answer:
                    pred_num = self._extract_last_number(predicted_answer)
                    if pred_num is not None and abs(pred_num - gold_num) < 0.01:
                        return 0.6 * 1.0 + 0.4 * base
                # No match - return penalized score
                return 0.6 * 0.0 + 0.4 * base

        return base

    def verify_commonsense(self, reason: str, answer: str) -> float:
        premise = reason
        hypothesis = answer
        inputs = self.nli_tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        entail_prob = probs[0][0].item()
        return entail_prob

    def verify(self, reason: str, task_type: str = "math", answer: Optional[str] = None, gold: Optional[str] = None) -> float:
        if task_type == "math":
            return self.verify_math(reason, expected_answer=gold, predicted_answer=answer)
        else:
            return self.verify_commonsense(reason, answer or "")

    def score_batch(self, samples: list[dict], task_type: str = "math", gold: Optional[str] = None) -> list[dict]:
        for sample in samples:
            score = self.verify(
                sample["reason"],
                task_type=task_type,
                answer=sample.get("answer"),
                gold=gold,
            )
            sample["correctness_score"] = score
        return samples
