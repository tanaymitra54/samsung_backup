import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import BenchmarkRunner
from evaluation.answer_utils import extract_predicted_answer, is_correct_prediction
from pipeline.inference import InferencePipeline
from pipeline.qubo_builder import QUBOBuilder
from pipeline.sampling import DiverseSampler
from pipeline.solver import SimulatedAnnealingSolver
from pipeline.verifier import ReasonVerifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QUBO pipeline across all configured benchmarks"
    )
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Override config subset_size")
    parser.add_argument("--full", action="store_true",
                        help="Run on full datasets (ignore subset_size)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for output files")
    parser.add_argument("--benchmarks", nargs="*", default=None,
                        help="Specific benchmarks to run (default: all in config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible runs")
    return parser.parse_args()


TASK_TYPE = {
    "gsm8k": "math",
    "bbh": "math",
    "strategyqa": "commonsense",
    "mmlu": "commonsense",
    "arc_challenge": "commonsense",
}

IS_MCQ = {"mmlu", "arc_challenge"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_mcq_choice(text: str) -> str:
    if not text:
        return ""
    upper = text.strip().upper()
    import re
    direct = re.search(r"\b([A-D])\b", upper)
    if direct:
        return direct.group(1)
    tagged = re.search(r"ANSWER\s*[:\-]?\s*([A-D])\b", upper)
    if tagged:
        return tagged.group(1)
    return ""


def baseline_greedy(inference: InferencePipeline, question: str) -> str:
    prompt = f"Question: {question}\nAnswer:"
    return inference.generate_answer(prompt)


def baseline_cot(inference: InferencePipeline, question: str) -> str:
    prompt = (
        "Let's think step by step and provide the final answer.\n"
        f"Question: {question}\nAnswer:"
    )
    return inference.generate_answer(prompt)


def run_qubo_pipeline(
    sampler: DiverseSampler,
    verifier: ReasonVerifier,
    qubo_builder: QUBOBuilder,
    solver: SimulatedAnnealingSolver,
    inference: InferencePipeline,
    question: str,
    task_type: str = "math",
) -> str:
    samples = sampler.sample(question)
    if not samples:
        return ""
    samples = verifier.score_batch(samples, task_type=task_type)
    Q, qubo_var_indices = qubo_builder.build_qubo(samples)
    state, _ = solver.solve(Q)
    selected_indices = [qubo_var_indices[i] for i in range(len(state)) if state[i] == 1]
    if not selected_indices:
        selected_indices = list(range(min(inference.subset_size, len(samples))))
    return inference.run(question, selected_indices, samples)


def extract_answer(pred: str, benchmark: str) -> str:
    if not pred:
        return ""
    if benchmark == "gsm8k":
        return extract_predicted_answer(pred)
    return pred.strip()


def is_correct(pred: str, gold: str, benchmark: str) -> bool:
    if not pred:
        return False
    if benchmark in IS_MCQ:
        extracted = extract_mcq_choice(pred)
        return bool(extracted and extracted == gold.strip().upper())
    if benchmark == "gsm8k":
        return is_correct_prediction(pred, gold)
    return pred.strip().lower() == gold.strip().lower() or gold.strip().lower() in pred.strip().lower()


def write_summary_json(path: str, results: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_summary_markdown(path: str, results: dict, config_benchmarks: list[str]):
    lines = [
        "# Multi-Benchmark Evaluation Report",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Benchmarks: {', '.join(results.keys())}",
        "",
        "## Accuracy Summary",
        "",
        "| Benchmark | Samples | Greedy | CoT | QUBO | Δ vs Greedy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b in config_benchmarks:
        if b not in results:
            continue
        r = results[b]
        a = r["accuracy"]
        lines.append(
            f"| {b} | {r['num_samples']} "
            f"| {a['greedy']:.2%} "
            f"| {a['cot']:.2%} "
            f"| {a['qubo']:.2%} "
            f"| {r['abs_gain_vs_greedy']:+.2%} |"
        )

    lines.extend(["", "## Benchmark Details", ""])
    for b in config_benchmarks:
        if b not in results:
            continue
        r = results[b]
        a = r["accuracy"]
        lines.extend([
            f"### {b}",
            "",
            f"- Samples: {r['num_samples']}",
            f"- Greedy accuracy: {a['greedy']:.2%}",
            f"- CoT accuracy: {a['cot']:.2%}",
            f"- QUBO pipeline accuracy: {a['qubo']:.2%}",
            f"- Absolute gain vs Greedy: {r['abs_gain_vs_greedy']:+.2%}",
            f"- CoT gain over Greedy: {r['cot_gain_over_greedy']:+.2%}",
            "",
        ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runner = BenchmarkRunner()
    if args.subset_size is not None:
        runner.subset_size = args.subset_size
    if args.full:
        runner.full_eval = True

    benchmark_list = args.benchmarks if args.benchmarks else runner.benchmarks
    unknown = [b for b in benchmark_list if b not in runner.benchmarks]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s): {unknown}. Allowed: {runner.benchmarks}"
        )

    sampler = DiverseSampler()
    verifier = ReasonVerifier()
    qubo_builder = QUBOBuilder()
    solver = SimulatedAnnealingSolver()
    inference = InferencePipeline()

    summary = {}

    csv_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.csv")
    fieldnames = [
        "benchmark", "id", "question", "gold", "pred_greedy", "pred_cot",
        "pred_qubo", "correct_greedy", "correct_cot", "correct_qubo",
        "runtime_greedy_s", "runtime_cot_s", "runtime_qubo_s", "error",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for b in benchmark_list:
        print(f"\n{'='*60}")
        print(f"Benchmark: {b}")
        print(f"{'='*60}")

        try:
            questions, gold_answers = runner.load_benchmark(b)
        except Exception as e:
            print(f"  Failed to load benchmark {b}: {e}")
            summary[b] = {"error": str(e), "num_samples": 0}
            continue

        task_type = TASK_TYPE.get(b, "math")
        print(f"  Loaded {len(questions)} questions")
        correct_greedy = 0
        correct_cot = 0
        correct_qubo = 0
        total = 0
        failed = 0

        for idx, (q, gold) in enumerate(tqdm(list(zip(questions, gold_answers)), desc=f"{b}", leave=False)):
            try:
                t0 = time.time()
                pred_greedy = baseline_greedy(inference, q)
                t1 = time.time()
                pred_cot = baseline_cot(inference, q)
                t2 = time.time()
                pred_qubo = run_qubo_pipeline(
                    sampler, verifier, qubo_builder, solver, inference, q, task_type
                )
                t3 = time.time()

                pred_greedy_n = extract_answer(pred_greedy, b)
                pred_cot_n = extract_answer(pred_cot, b)
                pred_qubo_n = extract_answer(pred_qubo, b)

                c_g = int(is_correct(pred_greedy_n, gold, b))
                c_c = int(is_correct(pred_cot_n, gold, b))
                c_q = int(is_correct(pred_qubo_n, gold, b))
                correct_greedy += c_g
                correct_cot += c_c
                correct_qubo += c_q
                total += 1

                row = {
                    "benchmark": b,
                    "id": idx,
                    "question": q,
                    "gold": gold,
                    "pred_greedy": pred_greedy_n,
                    "pred_cot": pred_cot_n,
                    "pred_qubo": pred_qubo_n,
                    "correct_greedy": c_g,
                    "correct_cot": c_c,
                    "correct_qubo": c_q,
                    "runtime_greedy_s": round(t1 - t0, 4),
                    "runtime_cot_s": round(t2 - t1, 4),
                    "runtime_qubo_s": round(t3 - t2, 4),
                    "error": "",
                }
            except Exception as e:
                failed += 1
                row = {
                    "benchmark": b,
                    "id": idx,
                    "question": q,
                    "gold": gold,
                    "pred_greedy": "",
                    "pred_cot": "",
                    "pred_qubo": "",
                    "correct_greedy": 0,
                    "correct_cot": 0,
                    "correct_qubo": 0,
                    "runtime_greedy_s": 0.0,
                    "runtime_cot_s": 0.0,
                    "runtime_qubo_s": 0.0,
                    "error": str(e),
                }
            writer.writerow(row)

        acc_greedy = (correct_greedy / total) if total else 0.0
        acc_cot = (correct_cot / total) if total else 0.0
        acc_qubo = (correct_qubo / total) if total else 0.0

        summary[b] = {
            "accuracy": {
                "greedy": acc_greedy,
                "cot": acc_cot,
                "qubo": acc_qubo,
            },
            "num_samples": total,
            "failed_samples": failed,
            "abs_gain_vs_greedy": acc_qubo - acc_greedy,
            "cot_gain_over_greedy": acc_cot - acc_greedy,
        }

        print(f"  [{b}] Greedy: {acc_greedy:.2%} | CoT: {acc_cot:.2%} | QUBO: {acc_qubo:.2%}")

    csv_file.close()

    json_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.json")
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    summary_meta = {
        "timestamp": timestamp,
        "seed": args.seed,
        "benchmarks": benchmark_list,
        "model": cfg.get("model", {}).get("name", "unknown"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "results": summary,
    }
    write_summary_json(json_path, summary_meta)

    md_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.md")
    write_summary_markdown(md_path, summary, benchmark_list)

    print(f"\n{'='*60}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
