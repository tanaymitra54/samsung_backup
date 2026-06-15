import re
import yaml
import numpy as np
import torch
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ReasonVerifier:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        verifier_cfg = self.config["verifier"]
        self.math_mode = verifier_cfg["math_mode"]
        self.nli_threshold = verifier_cfg["nli_threshold"]

        device_map = verifier_cfg.get("device_map_verifier", "auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            verifier_cfg["nli_model"],
            device_map=device_map if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.nli_model = self.nli_model.to(self.device)
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

    def verify_math(self, reason: str, expected_answer: Optional[str] = None) -> float:
        computations = self._extract_arithmetic(reason)
        if not computations:
            return 0.3
        consistency = 1.0
        if len(computations) > 1:
            diffs = [abs(computations[i] - computations[i + 1]) for i in range(len(computations) - 1)]
            consistency = 1.0 / (1.0 + np.mean(diffs))
        return min(1.0, consistency)

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

    def verify(self, reason: str, task_type: str = "math", answer: Optional[str] = None) -> float:
        if task_type == "math":
            return self.verify_math(reason)
        else:
            return self.verify_commonsense(reason, answer or "")

    def score_batch(self, samples: list[dict], task_type: str = "math") -> list[dict]:
        for sample in samples:
            score = self.verify(
                sample["reason"],
                task_type=task_type,
                answer=sample.get("answer"),
            )
            sample["correctness_score"] = score
        return samples
