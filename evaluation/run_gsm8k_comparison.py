import argparse
import csv
import json
import os
import time
from datetime import datetime

from datasets import load_dataset

from evaluation.answer_utils import (
    extract_gsm8k_gold,
    extract_predicted_answer,
    is_correct_prediction,
)
from pipeline.inference import InferencePipeline
from pipeline.qubo_builder import QUBOBuilder
from pipeline.sampling import DiverseSampler
from pipeline.solver import SimulatedAnnealingSolver
from pipeline.verifier import ReasonVerifier


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GSM8K baseline vs QUBO pipeline")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def baseline_greedy(inference: InferencePipeline, question: str) -> str:
    prompt = f"Question: {question}\nAnswer:"
    return inference.generate_answer(prompt)


def baseline_cot(inference: InferencePipeline, question: str) -> str:
    prompt = (
        "Let's think step by step and provide the final numeric answer.\n"
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
) -> str:
    samples = sampler.sample(question)
    if not samples:
        return ""
    samples = verifier.score_batch(samples, task_type="math")
    Q, qubo_var_indices = qubo_builder.build_qubo(samples)
    state, _ = solver.solve(Q)
    selected_indices = [qubo_var_indices[i] for i in range(len(state)) if state[i] == 1]
    if not selected_indices:
        selected_indices = list(range(min(inference.subset_size, len(samples))))
    return inference.run(question, selected_indices, samples)


def compute_summary(rows):
    total = len(rows)
    acc = {}
    for method in ["greedy", "cot", "qubo"]:
        correct = sum(1 for r in rows if r[f"correct_{method}"] == 1)
        acc[method] = (correct / total) if total else 0.0

    abs_gain_vs_greedy = acc["qubo"] - acc["greedy"]
    rel_gain_vs_greedy = (abs_gain_vs_greedy / acc["greedy"] * 100.0) if acc["greedy"] else 0.0
    cot_gain = acc["cot"] - acc["greedy"]
    two_x_threshold = 2.0 * cot_gain
    meets_2x = abs_gain_vs_greedy >= two_x_threshold

    return {
        "num_samples": total,
        "accuracy": acc,
        "abs_gain_vs_greedy": abs_gain_vs_greedy,
        "rel_gain_vs_greedy_percent": rel_gain_vs_greedy,
        "cot_gain_over_greedy": cot_gain,
        "two_x_threshold_gain": two_x_threshold,
        "meets_two_x_target": meets_2x,
    }


def write_markdown_report(path: str, summary: dict):
    a = summary["accuracy"]
    lines = [
        "# GSM8K Accuracy Comparison",
        "",
        f"Samples: {summary['num_samples']}",
        "",
        "| Method | Accuracy |",
        "|---|---:|",
        f"| Greedy | {a['greedy']:.2%} |",
        f"| CoT | {a['cot']:.2%} |",
        f"| QUBO Pipeline | {a['qubo']:.2%} |",
        "",
        f"- Absolute gain vs Greedy: {summary['abs_gain_vs_greedy']:.2%}",
        f"- Relative gain vs Greedy: {summary['rel_gain_vs_greedy_percent']:.2f}%",
        f"- CoT gain over Greedy: {summary['cot_gain_over_greedy']:.2%}",
        f"- 2x threshold gain: {summary['two_x_threshold_gain']:.2%}",
        f"- Meets 2x target: {summary['meets_two_x_target']}",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = load_dataset("gsm8k", "main", split="test")
    if not args.full:
        dataset = dataset.select(range(min(args.subset_size, len(dataset))))

    sampler = DiverseSampler()
    verifier = ReasonVerifier()
    qubo_builder = QUBOBuilder()
    solver = SimulatedAnnealingSolver()
    inference = InferencePipeline()

    rows = []
    for idx, item in enumerate(dataset):
        q = item["question"]
        gold = extract_gsm8k_gold(item["answer"])

        t0 = time.time()
        pred_greedy = baseline_greedy(inference, q)
        t1 = time.time()
        pred_cot = baseline_cot(inference, q)
        t2 = time.time()
        pred_qubo = run_qubo_pipeline(sampler, verifier, qubo_builder, solver, inference, q)
        t3 = time.time()

        pred_greedy_n = extract_predicted_answer(pred_greedy)
        pred_cot_n = extract_predicted_answer(pred_cot)
        pred_qubo_n = extract_predicted_answer(pred_qubo)

        row = {
            "id": idx,
            "question": q,
            "gold": gold,
            "pred_greedy": pred_greedy_n,
            "pred_cot": pred_cot_n,
            "pred_qubo": pred_qubo_n,
            "correct_greedy": int(is_correct_prediction(pred_greedy_n, gold)),
            "correct_cot": int(is_correct_prediction(pred_cot_n, gold)),
            "correct_qubo": int(is_correct_prediction(pred_qubo_n, gold)),
            "runtime_greedy_s": round(t1 - t0, 4),
            "runtime_cot_s": round(t2 - t1, 4),
            "runtime_qubo_s": round(t3 - t2, 4),
        }
        rows.append(row)

    csv_path = os.path.join(args.output_dir, f"gsm8k_predictions_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"gsm8k_summary_{timestamp}.json")
    md_path = os.path.join(args.output_dir, f"gsm8k_report_{timestamp}.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "id", "question", "gold", "pred_greedy", "pred_cot", "pred_qubo",
            "correct_greedy", "correct_cot", "correct_qubo", "runtime_greedy_s",
            "runtime_cot_s", "runtime_qubo_s"
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = compute_summary(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_markdown_report(md_path, summary)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print("Note: This script has now been defined; run manually when ready.")


if __name__ == "__main__":
    main()
