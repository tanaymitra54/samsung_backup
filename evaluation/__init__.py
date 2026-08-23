"""
==============================================================================
FILE: evaluation/__init__.py
ROLE: Benchmark Loader & Dataset Evaluation Runner Module
BRANCH ADDITION (abhyuday): Fixed AI2 ARC Challenge dataset HF loader namespace
(`allenai/ai2_arc`) to ensure robust benchmark dataset fetching across remote servers.
==============================================================================
"""

import json
import re
from pathlib import Path

import torch
import yaml
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
        bbh_configs = [
            "boolean_expressions",
            "causal_judgement",
            "date_understanding",
            "disambiguation_qa",
            "dyck_languages",
            "formal_fallacies",
            "geometric_shapes",
            "hyperbaton",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
            "logical_deduction_three_objects",
            "movie_recommendation",
            "multistep_arithmetic_two",
            "navigate",
            "object_counting",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "ruin_names",
            "salient_translation_error_detection",
            "snarks",
            "sports_understanding",
            "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
            "web_of_lies",
            "word_sorting",
        ]
        questions = []
        answers = []
        per_config = max(
            1, (self.subset_size if not self.full_eval else 9999) // len(bbh_configs)
        )
        for config in bbh_configs:
            try:
                dataset = load_dataset("lukaemon/bbh", config, split="test")
                dataset = dataset.select(range(min(per_config, len(dataset))))
                questions.extend([item["input"] for item in dataset])
                answers.extend([item["target"] for item in dataset])
            except Exception:
                pass
        return questions, answers

    def load_strategyqa(self) -> tuple[list[str], list[str]]:
        """StrategyQA multi-hop yes/no questions.

        Uses ChilleD/StrategyQA, which ships pre-defined disjoint train/test
        splits. The two sources previously tried here are both broken under
        datasets>=3: wics/strategy-qa is a loading script (no longer supported)
        and voidful/StrategyQA fails schema validation mid-generation -- which is
        why this loader used to raise NotImplementedError.

        Evaluation reads the TEST split; scripts/generate_training_data.py reads
        the TRAIN split, so the two never overlap.

        Gold is normalised to "yes"/"no". is_correct() treats true/false as the
        same polarity pair, so a model answering "True" still grades correctly.
        """
        dataset = load_dataset("ChilleD/StrategyQA", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))

        questions = []
        answers = []
        for item in dataset:
            q = str(item["question"]).strip()
            questions.append(
                f"Question: {q}\n"
                "Answer this yes/no question. Think step by step, then state your "
                "final answer as either 'yes' or 'no'."
            )
            answers.append("yes" if bool(item["answer"]) else "no")
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
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))

        questions = []
        answers = []
        for item in dataset:
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            options = [f"{label}. {text}" for label, text in zip(labels, texts)]
            formatted_q = (
                f"Question: {item['question']}\n" + "\n".join(options) + "\nAnswer:"
            )
            questions.append(formatted_q)
            answers.append(str(item["answerKey"]).strip().upper())
        return questions, answers

    def load_math_500(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        questions = [item["problem"] for item in dataset]
        answers = [str(item["answer"]).strip() for item in dataset]
        return questions, answers

    # Answers that survive this are plain numbers, matching the filter
    # build_chain_cache.py applied when the training pool was built -- so the
    # holdout is graded the same way the training questions were.
    _MATH_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")

    @staticmethod
    def _clean_math_answer(ans: str) -> str:
        """Strip LaTeX wrappers that hide an otherwise plain number.
        Kept byte-identical to build_chain_cache._clean so the two agree on
        which questions are numeric."""
        s = str(ans).strip()
        s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
        s = s.replace("\\!", "").replace("\\,", "").replace("$", "").replace(",", "")
        s = s.replace("\\%", "").replace("%", "")
        return s.strip()

    def load_math500_holdout(self) -> tuple[list[str], list[str]]:
        """MATH-500 questions that are NOT in the cached training-chain pool.

        WHY THIS EXISTS
        ---------------
        The SFT curation arms are trained on chains for the first 300
        numeric-answer questions of MATH-500. load_math_500() returns the first
        `subset_size` questions in raw order, so evaluating "math 500" at any
        normal subset size scores the model on the very questions its training
        chains were derived from. Every arm would look strong and the
        comparison would be meaningless.

        This loader takes the complement instead: MATH-500 minus every question
        text present in the chain cache. Disjointness is established by exact
        question-string match against the same file that produced the training
        data, so it cannot drift out of sync with what was actually trained on.

        Raises rather than silently falling back if the cache is missing -- a
        quiet fallback here produces contaminated numbers that look fine.
        """
        cache_path = Path(
            self.config.get("evaluation", {}).get(
                "math500_chain_cache", "results/cached_chains_math500_300q.json"
            )
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"math500 holdout needs the training-chain cache to exclude, but "
                f"{cache_path} does not exist. Without it the holdout cannot be "
                f"proven disjoint from training, and a contaminated eval is worse "
                f"than none. Set evaluation.math500_chain_cache in the config if "
                f"the pool lives elsewhere."
            )
        with open(cache_path, encoding="utf-8") as f:
            trained_on = {q.strip() for q in json.load(f).get("questions", [])}
        if not trained_on:
            raise ValueError(f"{cache_path} contains no 'questions' to exclude.")

        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        questions, answers, seen, non_numeric = [], [], 0, 0
        for item in dataset:
            problem = item["problem"]
            if problem.strip() in trained_on:
                seen += 1
                continue
            raw = self._clean_math_answer(item.get("answer", ""))
            if not self._MATH_NUMERIC.fullmatch(raw):
                non_numeric += 1
                continue
            questions.append(problem)
            answers.append(raw)

        print(
            f"[math500 holdout] {len(dataset)} total - {seen} trained-on "
            f"- {non_numeric} non-numeric = {len(questions)} available"
        )
        if seen == 0:
            raise ValueError(
                f"No MATH-500 question matched the chain cache at {cache_path}. "
                f"The cache is probably for a different dataset, so nothing was "
                f"actually excluded and this eval would be contaminated."
            )
        if not self.full_eval:
            questions = questions[: self.subset_size]
            answers = answers[: self.subset_size]
        if len(questions) < 100:
            print(
                f"[math500 holdout] WARNING: only {len(questions)} questions. "
                f"At this size the eval resolves differences of roughly "
                f"{(2.8 * (2 * 0.3 * 0.7) ** 0.5 / max(len(questions), 1) ** 0.5):.0%} "
                f"or larger -- treat a null result as 'could not resolve'."
            )
        return questions, answers

    def load_gpqa_diamond(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        questions = []
        answers = []
        import random

        for item in dataset:
            correct = item["Correct Answer"]
            incorrects = [
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]
            choices = [correct] + incorrects
            random.Random(42).shuffle(choices)  # consistent shuffle
            correct_idx = choices.index(correct)
            correct_letter = ["A", "B", "C", "D"][correct_idx]

            formatted_q = (
                f"Question: {item['Question']}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
                f"Answer:"
            )
            questions.append(formatted_q)
            answers.append(correct_letter)
        return questions, answers

    def load_aime(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        questions = [item["problem"] for item in dataset]
        answers = [str(item["answer"]).strip() for item in dataset]
        return questions, answers

    def load_mmlu_pro(self) -> tuple[list[str], list[str]]:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        if not self.full_eval:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        questions = []
        answers = []
        for item in dataset:
            choices = item["options"]
            formatted_q = f"Question: {item['question']}\n"
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for idx, choice in enumerate(choices):
                formatted_q += f"{letters[idx]}. {choice}\n"
            formatted_q += "Answer:"
            questions.append(formatted_q)
            answers.append(item["answer"])
        return questions, answers

    def load_benchmark(self, name: str) -> tuple[list[str], list[str]]:
        loaders = {
            "gsm8k": self.load_gsm8k,
            "bbh": self.load_bbh,
            "strategyqa": self.load_strategyqa,
            "mmlu": self.load_mmlu,
            "arc_challenge": self.load_arc_challenge,
            "math 500": self.load_math_500,
            "math500 holdout": self.load_math500_holdout,
            "gpqa diamond": self.load_gpqa_diamond,
            "aime": self.load_aime,
            "mmlu pro": self.load_mmlu_pro,
        }
        if name not in loaders:
            raise ValueError(f"Unknown benchmark: {name}")
        return loaders[name]()

    def compute_accuracy(
        self, predictions: list[str], ground_truth: list[str]
    ) -> float:
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
        direct = re.search(r"\b([A-J])\b", upper)
        if direct:
            return direct.group(1)

        tagged = re.search(r"ANSWER\s*[:\-]?\s*([A-J])\b", upper)
        if tagged:
            return tagged.group(1)

        return ""

    def compute_accuracy_mcq(
        self, predictions: list[str], ground_truth: list[str]
    ) -> float:
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

            if benchmark_name in {"mmlu", "arc_challenge", "gpqa diamond", "mmlu pro"}:
                accuracy = self.compute_accuracy_mcq(predictions, answers)
            else:
                accuracy = self.compute_accuracy(predictions, answers)
            results[benchmark_name] = {
                "accuracy": accuracy,
                "num_samples": len(questions),
            }
            print(f"  {benchmark_name}: {accuracy:.2%}")

        return results

    def run_all_batched(
        self,
        greedy_fn,
        cot_fn,
        qubo_fn,
        batch_fn_greedy=None,
        batch_fn_cot=None,
        batch_size: int = 8,
    ) -> dict:
        results = {}
        for benchmark_name in self.benchmarks:
            print(f"Running {benchmark_name} (batched, size={batch_size})...")
            questions, answers = self.load_benchmark(benchmark_name)

            preds_greedy = []
            preds_cot = []
            preds_qubo = []

            for i in range(0, len(questions), batch_size):
                batch_q = questions[i : i + batch_size]
                if batch_fn_greedy:
                    preds_greedy.extend(batch_fn_greedy(batch_q))
                else:
                    for q in batch_q:
                        preds_greedy.append(greedy_fn(q))

                if batch_fn_cot:
                    preds_cot.extend(batch_fn_cot(batch_q))
                else:
                    for q in batch_q:
                        preds_cot.append(cot_fn(q))

                for q in batch_q:
                    preds_qubo.append(qubo_fn(q))

            if benchmark_name in {"mmlu", "arc_challenge", "gpqa diamond", "mmlu pro"}:
                acc_g = self.compute_accuracy_mcq(preds_greedy, answers)
                acc_c = self.compute_accuracy_mcq(preds_cot, answers)
                acc_q = self.compute_accuracy_mcq(preds_qubo, answers)
            else:
                acc_g = self.compute_accuracy(preds_greedy, answers)
                acc_c = self.compute_accuracy(preds_cot, answers)
                acc_q = self.compute_accuracy(preds_qubo, answers)

            results[benchmark_name] = {
                "accuracy": {"greedy": acc_g, "cot": acc_c, "qubo": acc_q},
                "num_samples": len(questions),
                "abs_gain_vs_greedy": acc_q - acc_g,
                "cot_gain_over_greedy": acc_c - acc_g,
            }
            print(
                f"  [{benchmark_name}] Greedy: {acc_g:.2%} | CoT: {acc_c:.2%} | QUBO: {acc_q:.2%}"
            )
            print(
                f"  [{benchmark_name}] Greedy: {acc_g:.2%} | CoT: {acc_c:.2%} | QUBO: {acc_q:.2%}"
            )

        return results
