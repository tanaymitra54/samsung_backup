import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import BenchmarkRunner
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
    return parser.parse_args()


TASK_TYPE = {
    "gsm8k": "math",
    "bbh": "math",
    "strategyqa": "commonsense",
    "mmlu": "commonsense",
    "arc_challenge": "commonsense",
}

IS_MCQ = {"mmlu", "arc_challenge"}


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
        from evaluation.answer_utils import extract_predicted_answer
        return extract_predicted_answer(pred)
    return pred.strip()


def is_correct(pred: str, gold: str, benchmark: str) -> bool:
    if not pred:
        return False
    if benchmark in IS_MCQ:
        from evaluation import BenchmarkRunner
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        extracted = runner._extract_mcq_choice(pred)
        return bool(extracted and extracted == gold.strip().upper())
    if benchmark == "gsm8k":
        from evaluation.answer_utils import is_correct_prediction
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
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runner = BenchmarkRunner()
    if args.subset_size is not None:
        runner.subset_size = args.subset_size
    if args.full:
        runner.full_eval = True

    benchmark_list = args.benchmarks if args.benchmarks else runner.benchmarks

    sampler = DiverseSampler()
    verifier = ReasonVerifier()
    qubo_builder = QUBOBuilder()
    solver = SimulatedAnnealingSolver()
    inference = InferencePipeline()

    all_rows = {b: [] for b in benchmark_list}
    summary = {}

    for b in benchmark_list:
        print(f"\n{'='*60}")
        print(f"Benchmark: {b}")
        print(f"{'='*60}")

        questions, gold_answers = runner.load_benchmark(b)
        task_type = TASK_TYPE.get(b, "math")
        print(f"  Loaded {len(questions)} questions")

        for idx, (q, gold) in enumerate(zip(questions, gold_answers)):
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

            row = {
                "id": idx,
                "question": q,
                "gold": gold,
                "pred_greedy": pred_greedy_n,
                "pred_cot": pred_cot_n,
                "pred_qubo": pred_qubo_n,
                "correct_greedy": int(is_correct(pred_greedy_n, gold, b)),
                "correct_cot": int(is_correct(pred_cot_n, gold, b)),
                "correct_qubo": int(is_correct(pred_qubo_n, gold, b)),
                "runtime_greedy_s": round(t1 - t0, 4),
                "runtime_cot_s": round(t2 - t1, 4),
                "runtime_qubo_s": round(t3 - t2, 4),
            }
            all_rows[b].append(row)

            if (idx + 1) % 10 == 0:
                print(f"  [{b}] Processed {idx + 1}/{len(questions)}")

        total = len(all_rows[b])
        acc_greedy = sum(r["correct_greedy"] for r in all_rows[b]) / total if total else 0.0
        acc_cot = sum(r["correct_cot"] for r in all_rows[b]) / total if total else 0.0
        acc_qubo = sum(r["correct_qubo"] for r in all_rows[b]) / total if total else 0.0

        summary[b] = {
            "accuracy": {
                "greedy": acc_greedy,
                "cot": acc_cot,
                "qubo": acc_qubo,
            },
            "num_samples": total,
            "abs_gain_vs_greedy": acc_qubo - acc_greedy,
            "cot_gain_over_greedy": acc_cot - acc_greedy,
        }

        print(f"  [{b}] Greedy: {acc_greedy:.2%} | CoT: {acc_cot:.2%} | QUBO: {acc_qubo:.2%}")

    csv_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "benchmark", "id", "question", "gold", "pred_greedy", "pred_cot",
            "pred_qubo", "correct_greedy", "correct_cot", "correct_qubo",
            "runtime_greedy_s", "runtime_cot_s", "runtime_qubo_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in benchmark_list:
            for row in all_rows[b]:
                row["benchmark"] = b
                writer.writerow(row)

    json_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.json")
    write_summary_json(json_path, summary)

    md_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.md")
    write_summary_markdown(md_path, summary, benchmark_list)

    print(f"\n{'='*60}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
