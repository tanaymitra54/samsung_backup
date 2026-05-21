import yaml
import json
import torch
import re
from pathlib import Path
from datasets import load_dataset


class BenchmarkRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        eval_cfg = self.config["evaluation"]
        self.benchmarks = eval_cfg["benchmarks"]
        self.subset_size = eval_cfg["subset_size"]
        self.full_eval = eval_cfg["full_eval"]

    def _extract_answer_gsm8k(self, text: str) -> str:
        if "####" in text:
            return text.split("####")[-1].strip()
        return text.strip()

    def _extract_answer_bbh(self, text: str) -> str:
        if "answer is" in text.lower():
            return text.lower().split("answer is")[-1].strip()
        return text.strip()

    def load_gsm8k(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("gsm8k", "main", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        questions = [item["question"] for item in dataset]
        answers = [item["answer"] for item in dataset]
        return questions, answers

    def load_bbh(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("lukaemon/bbh", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        questions = [item["input"] for item in dataset]
        answers = [item["target"] for item in dataset]
        return questions, answers

    def load_strategyqa(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("taesiri/strategy_qa", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        questions = [item["question"] for item in dataset]
        answers = [str(item["answer"]) for item in dataset]
        return questions, answers

    def load_mmlu(self) -> tuple[list[str], list[str]]:
        subjects = [
            "abstract_algebra",
            "college_computer_science",
            "college_physics",
            "electrical_engineering",
            "machine_learning",
        ]
        questions = []
        answers = []
        per_subject = self.subset_size // len(subjects) if not self.full_eval else None
        for subject in subjects:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            if per_subject:
                dataset = dataset.select(range(min(per_subject, len(dataset))))
            for item in dataset:
                choices = item["choices"]
                formatted_q = (
                    f"Question: {item['question']}\n"
                    f"A. {choices[0]}\n"
                    f"B. {choices[1]}\n"
                    f"C. {choices[2]}\n"
                    f"D. {choices[3]}\n"
                    f"Answer:"
                )
                questions.append(formatted_q)
                answers.append(["A", "B", "C", "D"][item["answer"]])
        return questions, answers

    def load_arc_challenge(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))

        questions = []
        answers = []
        for item in dataset:
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            options = [f"{label}. {text}" for label, text in zip(labels, texts)]
            formatted_q = f"Question: {item['question']}\n" + "\n".join(options) + "\nAnswer:"
            questions.append(formatted_q)
            answers.append(str(item["answerKey"]).strip().upper())
        return questions, answers

    def load_benchmark(self, name: str) -> tuple[list[str], list[str]]:
        loaders = {
            "gsm8k": self.load_gsm8k,
            "bbh": self.load_bbh,
            "strategyqa": self.load_strategyqa,
            "mmlu": self.load_mmlu,
            "arc_challenge": self.load_arc_challenge,
        }
        if name not in loaders:
            raise ValueError(f"Unknown benchmark: {name}")
        return loaders[name]()

    def compute_accuracy(self, predictions: list[str], ground_truth: list[str]) -> float:
        correct = 0
        total = len(predictions)
        for pred, truth in zip(predictions, ground_truth):
            if pred.strip().lower() == truth.strip().lower():
                correct += 1
            elif truth.strip().lower() in pred.strip().lower():
                correct += 1
        return correct / total if total > 0 else 0.0

    def _extract_mcq_choice(self, text: str) -> str:
        if not text:
            return ""

        upper = text.strip().upper()
        direct = re.search(r"\b([A-D])\b", upper)
        if direct:
            return direct.group(1)

        tagged = re.search(r"ANSWER\s*[:\-]?\s*([A-D])\b", upper)
        if tagged:
            return tagged.group(1)

        return ""

    def compute_accuracy_mcq(self, predictions: list[str], ground_truth: list[str]) -> float:
        correct = 0
        total = len(predictions)
        for pred, truth in zip(predictions, ground_truth):
            extracted = self._extract_mcq_choice(pred)
            if extracted and extracted == truth.strip().upper():
                correct += 1
        return correct / total if total > 0 else 0.0

    def run_all(self, pipeline_fn) -> dict:
        results = {}
        for benchmark_name in self.benchmarks:
            print(f"Running {benchmark_name}...")
            questions, answers = self.load_benchmark(benchmark_name)

            predictions = []
            for q in questions:
                pred = pipeline_fn(q)
                predictions.append(pred)

            if benchmark_name in {"mmlu", "arc_challenge"}:
                accuracy = self.compute_accuracy_mcq(predictions, answers)
            else:
                accuracy = self.compute_accuracy(predictions, answers)
            results[benchmark_name] = {
                "accuracy": accuracy,
                "num_samples": len(questions),
            }
            print(f"  {benchmark_name}: {accuracy:.2%}")

        return results
