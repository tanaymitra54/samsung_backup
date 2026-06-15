import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import BenchmarkRunner
from evaluation.answer_utils import extract_predicted_answer, extract_gsm8k_gold, is_correct_prediction
from pipeline.inference import InferencePipeline
from pipeline.qubo_builder import QUBOBuilder
from pipeline.sampling import DiverseSampler
from pipeline.device_utils import resolve_device
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
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for batched inference")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched inference (force per-question)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM backend for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Single device for benchmark execution (default: evaluation.device or cuda:0)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Distribute benchmarks across available GPUs")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Weights & Biases project name for tracking")
    parser.add_argument("--debug", action="store_true",
                        help="Save first 5 raw model outputs for debugging")
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


def make_batch_greedy(inference: InferencePipeline):
    def fn(questions: list[str]) -> list[str]:
        prompts = [f"Question: {q}\nAnswer:" for q in questions]
        return inference.generate_answers_batch(prompts)
    return fn


def make_batch_cot(inference: InferencePipeline):
    def fn(questions: list[str]) -> list[str]:
        prompts = [
            "Let's think step by step and provide the final answer.\n"
            f"Question: {q}\nAnswer:"
            for q in questions
        ]
        return inference.generate_answers_batch(prompts)
    return fn


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
        return is_correct_prediction(pred, extract_gsm8k_gold(gold))
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
        "| Benchmark | Samples | Greedy | CoT | QUBO | Δ vs Greedy | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for b in config_benchmarks:
        if b not in results:
            continue
        r = results[b]
        # Check if benchmark has accuracy data or if it failed
        if "error" in r and "accuracy" not in r:
            lines.append(
                f"| {b} | 0 | N/A | N/A | N/A | N/A | ❌ Failed |"
            )
        else:
            a = r.get("accuracy", {"greedy": 0.0, "cot": 0.0, "qubo": 0.0})
            gain = r.get("abs_gain_vs_greedy", 0.0)
            lines.append(
                f"| {b} | {r.get('num_samples', 0)} "
                f"| {a.get('greedy', 0.0):.2%} "
                f"| {a.get('cot', 0.0):.2%} "
                f"| {a.get('qubo', 0.0):.2%} "
                f"| {gain:+.2%} | ✓ Complete |"
            )

    lines.extend(["", "## Benchmark Details", ""])
    for b in config_benchmarks:
        if b not in results:
            continue
        r = results[b]
        # Check if benchmark has accuracy data or if it failed
        if "error" in r and "accuracy" not in r:
            lines.extend([
                f"### {b}",
                "",
                f"**Status: Failed**",
                f"- Error: {r['error']}",
                "",
            ])
        else:
            a = r.get("accuracy", {"greedy": 0.0, "cot": 0.0, "qubo": 0.0})
            lines.extend([
                f"### {b}",
                "",
                f"- Samples: {r.get('num_samples', 0)}",
                f"- Failed samples: {r.get('failed_samples', 0)}",
                f"- Greedy accuracy: {a.get('greedy', 0.0):.2%}",
                f"- CoT accuracy: {a.get('cot', 0.0):.2%}",
                f"- QUBO pipeline accuracy: {a.get('qubo', 0.0):.2%}",
                f"- Absolute gain vs Greedy: {r.get('abs_gain_vs_greedy', 0.0):+.2%}",
                f"- CoT gain over Greedy: {r.get('cot_gain_over_greedy', 0.0):+.2%}",
                "",
            ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_benchmark_on_gpu(
    gpu_id: int,
    benchmark_name: str,
    config_path: str,
    seed: int,
    subset_size: int,
    full_eval: bool,
    use_batch: bool,
    batch_size: int,
    use_vllm: bool,
    device: str | None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_seed(seed)

    worker_device = "cuda:0"

    runner = BenchmarkRunner(config_path)
    if subset_size is not None:
        runner.subset_size = subset_size
    if full_eval:
        runner.full_eval = True

    inference = InferencePipeline(config_path, device=worker_device, use_vllm=use_vllm)
    runtime_device = str(inference.device)

    sampler = DiverseSampler(
        config_path,
        device=runtime_device,
        shared_model=inference.model,
        shared_tokenizer=inference.tokenizer,
    )
    verifier = ReasonVerifier(config_path, device=runtime_device)
    qubo_builder = QUBOBuilder(config_path, device=runtime_device)
    solver = SimulatedAnnealingSolver(config_path, device=runtime_device)

    task_type = TASK_TYPE.get(benchmark_name, "math")
    questions, gold_answers = runner.load_benchmark(benchmark_name)

    results_rows = []
    correct_greedy = 0
    correct_cot = 0
    correct_qubo = 0
    total = 0
    failed = 0

    if use_batch and batch_size > 1 and torch.cuda.is_available():
        batch_greedy_fn = make_batch_greedy(inference)
        batch_cot_fn = make_batch_cot(inference)
        for i in range(0, len(questions), batch_size):
            batch_q = questions[i:i + batch_size]
            batch_gold = gold_answers[i:i + batch_size]
            try:
                t0 = time.time()
                preds_g = batch_greedy_fn(batch_q)
                t1 = time.time()
                preds_c = batch_cot_fn(batch_q)
                t2 = time.time()
                for j, q in enumerate(batch_q):
                    pred_q = run_qubo_pipeline(
                        sampler, verifier, qubo_builder, solver, inference, q, task_type
                    )
                    pred_qubo_n = extract_answer(pred_q, benchmark_name)
                    pred_g_n = extract_answer(preds_g[j], benchmark_name)
                    pred_c_n = extract_answer(preds_c[j], benchmark_name)
                    gold = batch_gold[j]
                    c_g = int(is_correct(pred_g_n, gold, benchmark_name))
                    c_c = int(is_correct(pred_c_n, gold, benchmark_name))
                    c_q = int(is_correct(pred_qubo_n, gold, benchmark_name))
                    correct_greedy += c_g
                    correct_cot += c_c
                    correct_qubo += c_q
                    total += 1
                    results_rows.append({
                        "benchmark": benchmark_name,
                        "id": i + j,
                        "question": q,
                        "gold": gold,
                        "pred_greedy": pred_g_n,
                        "pred_cot": pred_c_n,
                        "pred_qubo": pred_qubo_n,
                        "correct_greedy": c_g,
                        "correct_cot": c_c,
                        "correct_qubo": c_q,
                        "runtime_greedy_s": round((t1 - t0) / len(batch_q), 4),
                        "runtime_cot_s": round((t2 - t1) / len(batch_q), 4),
                        "runtime_qubo_s": 0.0,
                        "error": "",
                    })
            except Exception as e:
                failed += len(batch_q)
                for j in range(len(batch_q)):
                    results_rows.append({
                        "benchmark": benchmark_name,
                        "id": i + j,
                        "question": batch_q[j],
                        "gold": batch_gold[j],
                        "pred_greedy": "", "pred_cot": "", "pred_qubo": "",
                        "correct_greedy": 0, "correct_cot": 0, "correct_qubo": 0,
                        "runtime_greedy_s": 0.0, "runtime_cot_s": 0.0, "runtime_qubo_s": 0.0,
                        "error": str(e),
                    })
    else:
        for idx, (q, gold) in enumerate(zip(questions, gold_answers)):
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
                pred_g_n = extract_answer(pred_greedy, benchmark_name)
                pred_c_n = extract_answer(pred_cot, benchmark_name)
                pred_q_n = extract_answer(pred_qubo, benchmark_name)
                c_g = int(is_correct(pred_g_n, gold, benchmark_name))
                c_c = int(is_correct(pred_c_n, gold, benchmark_name))
                c_q = int(is_correct(pred_q_n, gold, benchmark_name))
                correct_greedy += c_g
                correct_cot += c_c
                correct_qubo += c_q
                total += 1
                results_rows.append({
                    "benchmark": benchmark_name,
                    "id": idx, "question": q, "gold": gold,
                    "pred_greedy": pred_g_n, "pred_cot": pred_c_n, "pred_qubo": pred_q_n,
                    "correct_greedy": c_g, "correct_cot": c_c, "correct_qubo": c_q,
                    "runtime_greedy_s": round(t1 - t0, 4),
                    "runtime_cot_s": round(t2 - t1, 4),
                    "runtime_qubo_s": round(t3 - t2, 4),
                    "error": "",
                })
            except Exception as e:
                failed += 1
                results_rows.append({
                    "benchmark": benchmark_name, "id": idx, "question": q, "gold": gold,
                    "pred_greedy": "", "pred_cot": "", "pred_qubo": "",
                    "correct_greedy": 0, "correct_cot": 0, "correct_qubo": 0,
                    "runtime_greedy_s": 0.0, "runtime_cot_s": 0.0, "runtime_qubo_s": 0.0,
                    "error": str(e),
                })

    acc_g = (correct_greedy / total) if total else 0.0
    acc_c = (correct_cot / total) if total else 0.0
    acc_q = (correct_qubo / total) if total else 0.0

    return {
        "benchmark": benchmark_name,
        "accuracy": {"greedy": acc_g, "cot": acc_c, "qubo": acc_q},
        "num_samples": total,
        "failed_samples": failed,
        "abs_gain_vs_greedy": acc_q - acc_g,
        "cot_gain_over_greedy": acc_c - acc_g,
        "rows": results_rows,
    }


def main():
    args = parse_args()
    requested_device = args.device
    if requested_device and requested_device.startswith("cuda:") and not args.multi_gpu:
        physical_gpu = requested_device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
        selected_device = "cuda:0"
    else:
        selected_device = requested_device
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runner = BenchmarkRunner()
    selected_device = str(resolve_device(selected_device or runner.config.get("evaluation", {}).get("device")))
    if args.subset_size is not None:
        runner.subset_size = args.subset_size
    if args.full:
        runner.full_eval = True

    benchmark_list = args.benchmarks if args.benchmarks else runner.benchmarks
    unknown = [b for b in benchmark_list if b not in runner.benchmarks]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {unknown}. Allowed: {runner.benchmarks}")

    use_batch = not args.no_batch and selected_device.startswith("cuda") and torch.cuda.is_available()
    # Use smaller default batch size (4) to prevent OOM errors; can be overridden with --batch-size
    batch_size = args.batch_size or runner.config.get("evaluation", {}).get("batch_size", 4)

    if requested_device and requested_device.startswith("cuda:") and not args.multi_gpu:
        print(f"Benchmark device: {requested_device} -> visible as {selected_device}")
    else:
        print(f"Benchmark device: {selected_device}")
    print(f"CUDA available: {torch.cuda.is_available()} | visible GPUs: {torch.cuda.device_count()}")

    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config={
                    "model": runner.config["model"]["name"],
                    "benchmarks": benchmark_list,
                    "subset_size": runner.subset_size,
                    "full_eval": runner.full_eval,
                    "batch_size": batch_size,
                    "use_batch": use_batch,
                    "use_vllm": args.use_vllm,
                    "seed": args.seed,
                },
            )
        except ImportError:
            print("  WARNING: wandb not installed. Install with `pip install wandb`.")
            args.wandb_project = None

    num_gpus = torch.cuda.device_count() if args.multi_gpu else 0
    summary = {}
    all_rows = []

    if num_gpus > 1 and len(benchmark_list) > 1:
        print(f"  Distributing {len(benchmark_list)} benchmarks across {num_gpus} GPUs...")
        chunk_size = max(1, len(benchmark_list) // num_gpus)
        gpu_assignments = {}
        for i, b in enumerate(benchmark_list):
            gpu_id = i % num_gpus
            gpu_assignments.setdefault(gpu_id, []).append(b)

        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, benches in gpu_assignments.items():
                for b in benches:
                    futures.append(executor.submit(
                        run_benchmark_on_gpu, gpu_id, b,
                        "config/config.yaml", args.seed,
                        runner.subset_size, runner.full_eval,
                        use_batch, batch_size, args.use_vllm, args.device,
                    ))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Multi-GPU"):
                result = future.result()
                summary[result["benchmark"]] = {k: v for k, v in result.items() if k != "rows"}
                all_rows.extend(result["rows"])
                a = result["accuracy"]
                print(f"  [{result['benchmark']}] GPU | Greedy: {a['greedy']:.2%} | CoT: {a['cot']:.2%} | QUBO: {a['qubo']:.2%}")
    else:
        inference = InferencePipeline(device=selected_device, use_vllm=args.use_vllm)
        runtime_device = str(inference.device)
        sampler = DiverseSampler(
            device=runtime_device,
            shared_model=inference.model,
            shared_tokenizer=inference.tokenizer,
        )
        verifier = ReasonVerifier(device=runtime_device)
        qubo_builder = QUBOBuilder(device=runtime_device)
        solver = SimulatedAnnealingSolver(device=runtime_device)

        print(f"Runtime device: {runtime_device}")
        print(f"Inference model device: {inference.model_input_device if not inference.use_vllm else inference.device}")
        print(f"Generation device: {inference.generation_input_device if not inference.use_vllm else inference.device}")
        print(f"Sampler device: {sampler.device}")
        print(f"Verifier device: {verifier.device}")
        print(f"Solver device: {solver.device}")

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
                summary[b] = {"error": str(e), "num_samples": 0, "accuracy": {"greedy": 0.0, "cot": 0.0, "qubo": 0.0}, "abs_gain_vs_greedy": 0.0, "cot_gain_over_greedy": 0.0}
                continue

            task_type = TASK_TYPE.get(b, "math")
            print(f"  Loaded {len(questions)} questions")
            correct_greedy = 0
            correct_cot = 0
            correct_qubo = 0
            total = 0
            failed = 0

            if use_batch and batch_size > 1:
                batch_greedy_fn = make_batch_greedy(inference)
                batch_cot_fn = make_batch_cot(inference)
                for i in range(0, len(questions), batch_size):
                    batch_q = questions[i:i + batch_size]
                    batch_gold = gold_answers[i:i + batch_size]
                    try:
                        t0 = time.time()
                        preds_g = batch_greedy_fn(batch_q)
                        t1 = time.time()
                        preds_c = batch_cot_fn(batch_q)
                        t2 = time.time()
                        for j, q in enumerate(batch_q):
                            tq = time.time()
                            pred_qubo = run_qubo_pipeline(
                                sampler, verifier, qubo_builder, solver, inference, q, task_type
                            )
                            tq_end = time.time()
                            pred_g_n = extract_answer(preds_g[j], b)
                            pred_c_n = extract_answer(preds_c[j], b)
                            pred_q_n = extract_answer(pred_qubo, b)
                            gold = batch_gold[j]
                            c_g = int(is_correct(pred_g_n, gold, b))
                            c_c = int(is_correct(pred_c_n, gold, b))
                            c_q = int(is_correct(pred_q_n, gold, b))
                            correct_greedy += c_g
                            correct_cot += c_c
                            correct_qubo += c_q
                            total += 1
                            row = {
                                "benchmark": b, "id": i + j, "question": q, "gold": gold,
                                "pred_greedy": pred_g_n, "pred_cot": pred_c_n, "pred_qubo": pred_q_n,
                                "correct_greedy": c_g, "correct_cot": c_c, "correct_qubo": c_q,
                                "runtime_greedy_s": round((t1 - t0) / len(batch_q), 4),
                                "runtime_cot_s": round((t2 - t1) / len(batch_q), 4),
                                "runtime_qubo_s": round(tq_end - tq, 4),
                                "error": "",
                            }
                            writer.writerow(row)
                            all_rows.append(row)
                            if len(all_rows) <= 3:
                                print(f"  [DEBUG #{len(all_rows)}] {b} gold='{gold}' raw_g='{repr(preds_g[j][:200])}' raw_c='{repr(preds_c[j][:200])}' ext_g='{pred_g_n}' ext_c='{pred_c_n}'")
                    except Exception as e:
                        import traceback
                        print(f"  ⚠️  BATCH ERROR (batch {i//batch_size}): {e}")
                        traceback.print_exc()
                        failed += len(batch_q)
                        for j in range(len(batch_q)):
                            row = {
                                "benchmark": b, "id": i + j, "question": batch_q[j], "gold": batch_gold[j],
                                "pred_greedy": "", "pred_cot": "", "pred_qubo": "",
                                "correct_greedy": 0, "correct_cot": 0, "correct_qubo": 0,
                                "runtime_greedy_s": 0.0, "runtime_cot_s": 0.0, "runtime_qubo_s": 0.0,
                                "error": str(e),
                            }
                            writer.writerow(row)
            else:
                for idx, (q, gold) in enumerate(tqdm(
                    list(zip(questions, gold_answers)), desc=f"{b}", leave=False
                )):
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
                        pred_g_n = extract_answer(pred_greedy, b)
                        pred_c_n = extract_answer(pred_cot, b)
                        pred_q_n = extract_answer(pred_qubo, b)
                        c_g = int(is_correct(pred_g_n, gold, b))
                        c_c = int(is_correct(pred_c_n, gold, b))
                        c_q = int(is_correct(pred_q_n, gold, b))
                        correct_greedy += c_g
                        correct_cot += c_c
                        correct_qubo += c_q
                        total += 1
                        row = {
                            "benchmark": b, "id": idx, "question": q, "gold": gold,
                            "pred_greedy": pred_g_n, "pred_cot": pred_c_n, "pred_qubo": pred_q_n,
                            "correct_greedy": c_g, "correct_cot": c_c, "correct_qubo": c_q,
                            "runtime_greedy_s": round(t1 - t0, 4),
                            "runtime_cot_s": round(t2 - t1, 4),
                            "runtime_qubo_s": round(t3 - t2, 4),
                            "error": "",
                        }
                        writer.writerow(row)
                        all_rows.append(row)
                        if len(all_rows) <= 3:
                            print(f"  [DEBUG #{len(all_rows)}] {b} gold='{gold}' raw_g='{repr(pred_greedy[:200])}' raw_c='{repr(pred_cot[:200])}' ext_g='{pred_g_n}' ext_c='{pred_c_n}'")
                    except Exception as e:
                        import traceback
                        print(f"  ⚠️  ERROR (question {idx}): {e}")
                        traceback.print_exc()
                        failed += 1
                        row = {
                            "benchmark": b, "id": idx, "question": q, "gold": gold,
                            "pred_greedy": "", "pred_cot": "", "pred_qubo": "",
                            "correct_greedy": 0, "correct_cot": 0, "correct_qubo": 0,
                            "runtime_greedy_s": 0.0, "runtime_cot_s": 0.0, "runtime_qubo_s": 0.0,
                            "error": str(e),
                        }
                        writer.writerow(row)

            acc_greedy = (correct_greedy / total) if total else 0.0
            acc_cot = (correct_cot / total) if total else 0.0
            acc_qubo = (correct_qubo / total) if total else 0.0
            summary[b] = {
                "accuracy": {"greedy": acc_greedy, "cot": acc_cot, "qubo": acc_qubo},
                "num_samples": total,
                "failed_samples": failed,
                "abs_gain_vs_greedy": acc_qubo - acc_greedy,
                "cot_gain_over_greedy": acc_cot - acc_greedy,
            }
            print(f"  [{b}] Greedy: {acc_greedy:.2%} | CoT: {acc_cot:.2%} | QUBO: {acc_qubo:.2%} | samples={total} failed={failed}")

            if args.wandb_project:
                try:
                    import wandb
                    wandb.log({
                        f"{b}/accuracy_greedy": acc_greedy,
                        f"{b}/accuracy_cot": acc_cot,
                        f"{b}/accuracy_qubo": acc_qubo,
                        f"{b}/abs_gain_vs_greedy": acc_qubo - acc_greedy,
                        f"{b}/samples": total,
                    })
                except ImportError:
                    pass

        csv_file.close()

    json_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.json")
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    summary_meta = {
        "timestamp": timestamp,
        "seed": args.seed,
        "benchmarks": benchmark_list,
        "model": cfg.get("model", {}).get("name", "unknown"),
        "device": selected_device,
        "use_batch": use_batch,
        "batch_size": batch_size,
        "use_vllm": args.use_vllm,
        "multi_gpu": num_gpus > 1,
        "results": summary,
    }
    write_summary_json(json_path, summary_meta)

    md_path = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.md")
    write_summary_markdown(md_path, summary, benchmark_list)

    print(f"\n{'='*60}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"{'='*60}")

    if args.wandb_project:
        try:
            import wandb
            wandb.log({"summary": summary_meta})
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
