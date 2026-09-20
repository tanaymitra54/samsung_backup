import math
import re
from typing import Optional

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.device_utils import resolve_device


DEFAULT_WEIGHTS = {
    "math": {"sc": 0.4, "nli": 0.3, "conf": 0.2, "arith": 0.1},
    "commonsense": {"sc": 0.5, "nli": 0.3, "conf": 0.2, "arith": 0.0},
}

_EQUATION = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)


def arithmetic_consistency(text: str) -> float:
    """Passed explicit equations divided by equations found in the trace."""
    if not text:
        return 0.0
    matches = _EQUATION.findall(text)
    if not matches:
        return 0.0
    passed = 0
    for raw_a, op, raw_b, raw_c in matches:
        a, b, c = float(raw_a), float(raw_b), float(raw_c)
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op == "*":
            expected = a * b
        elif op == "/" and b != 0:
            expected = a / b
        else:
            continue
        if abs(expected - c) < 0.01:
            passed += 1
    return passed / len(matches)


def answer_key(sample: dict, task_type: str) -> str:
    """Canonical predicted answer used for self-consistency counts."""
    answer = (sample.get("answer") or "").strip()
    reason = sample.get("reason") or ""
    text = answer or reason
    if not text.strip():
        return f"__empty_{id(sample)}"
    if task_type == "math":
        number = _last_number(text)
        if number is None and answer:
            number = _last_number(reason)
        if number is not None:
            return str(number)
    return re.sub(r"\s+", " ", text.strip().lower())


def self_consistency_scores(keys: list[str]) -> list[float]:
    """Frequency of each trace's predicted answer over the full sample set."""
    total = len(keys)
    if total == 0:
        return []
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return [counts[key] / total for key in keys]


def fuse_signals(
    sc: float,
    nli: float,
    conf: float,
    arith: float,
    weights: dict,
) -> float:
    """S_final = w_SC S_SC + w_NLI S_NLI + w_Conf S_Conf + w_Arith S_Arith."""
    score = (
        float(weights["sc"]) * sc
        + float(weights["nli"]) * nli
        + float(weights["conf"]) * conf
        + float(weights["arith"]) * arith
    )
    return float(min(1.0, max(0.0, score)))


def geometric_mean_probability(token_probs: list[float]) -> float:
    """exp(mean log P(t_i)) over generated-token probabilities."""
    if not token_probs:
        return 0.0
    log_sum = 0.0
    for prob in token_probs:
        log_sum += math.log(min(1.0, max(float(prob), 1e-12)))
    return float(min(1.0, max(0.0, math.exp(log_sum / len(token_probs)))))


def _last_number(text: str) -> Optional[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    value = float(matches[-1])
    return int(value) if value.is_integer() else value


class ReasonVerifier:
    def __init__(self, config_path: str = "config/config.yaml", device: str | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        verifier_cfg = self.config["verifier"]
        self.math_mode = verifier_cfg["math_mode"]
        self.nli_threshold = verifier_cfg["nli_threshold"]
        self.weights = self._load_weights(verifier_cfg.get("weights"))

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        self.device = resolve_device(preferred_device)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            verifier_cfg["nli_model"],
        )
        self.nli_model = self.nli_model.to(self.device)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(verifier_cfg["nli_model"])
        self.nli_model.eval()
        self.entailment_index = self._entailment_index()

    def _load_weights(self, configured: Optional[dict]) -> dict:
        weights = {task: dict(values) for task, values in DEFAULT_WEIGHTS.items()}
        if not configured:
            return weights
        for task, values in configured.items():
            base = dict(weights.get(task, weights["commonsense"]))
            base.update(values or {})
            weights[task] = base
        return weights

    def _entailment_index(self) -> int:
        id2label = getattr(self.nli_model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
        return 1

    def _weights_for(self, task_type: str) -> dict:
        if task_type in self.weights:
            return self.weights[task_type]
        return self.weights["commonsense"]

    def _nli_entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        premises = [premise.strip() or "." for premise, _ in pairs]
        hypotheses = [hypothesis.strip() or "." for _, hypothesis in pairs]
        inputs = self.nli_tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return [float(value) for value in probs[:, self.entailment_index].tolist()]

    def score_batch(self, samples: list[dict], task_type: str = "math", gold: str | None = None) -> list[dict]:
        """Score traces without the gold answer.

        S_SC is how often this trace's predicted answer appears.
        S_NLI is the entailment probability of reason -> answer.
        S_Conf is the geometric mean of generated-token probabilities.
        S_Arith is passed equations / equations found, and is weighted only for math.
        """
        if not samples:
            return samples

        weights = self._weights_for(task_type)
        keys = [answer_key(sample, task_type) for sample in samples]
        sc_scores = self_consistency_scores(keys)
        nli_pairs = [
            (sample.get("reason") or "", sample.get("answer") or "")
            for sample in samples
        ]
        nli_scores = self._nli_entailment(nli_pairs)

        for sample, sc, nli in zip(samples, sc_scores, nli_scores):
            conf = float(sample.get("confidence_score") or 0.0)
            conf = min(1.0, max(0.0, conf))
            trace_text = "\n".join(
                part for part in (sample.get("reason") or "", sample.get("answer") or "") if part
            )
            arith = arithmetic_consistency(trace_text) if task_type == "math" else 0.0
            sample["sc_score"] = sc
            sample["nli_score"] = nli
            sample["conf_score"] = conf
            sample["arith_score"] = arith
            sample["correctness_score"] = fuse_signals(sc, nli, conf, arith, weights)
        return samples
