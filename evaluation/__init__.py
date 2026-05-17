import yaml
import json
import torch
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

    def load_benchmark(self, name: str) -> tuple[list[str], list[str]]:
        loaders = {
            "gsm8k": self.load_gsm8k,
            "bbh": self.load_bbh,
            "strategyqa": self.load_strategyqa,
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

    def run_all(self, pipeline_fn) -> dict:
        results = {}
        for benchmark_name in self.benchmarks:
            print(f"Running {benchmark_name}...")
            questions, answers = self.load_benchmark(benchmark_name)

            predictions = []
            for q in questions:
                pred = pipeline_fn(q)
                predictions.append(pred)

            accuracy = self.compute_accuracy(predictions, answers)
            results[benchmark_name] = {
                "accuracy": accuracy,
                "num_samples": len(questions),
            }
            print(f"  {benchmark_name}: {accuracy:.2%}")

        return results
