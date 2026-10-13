#!/usr/bin/env python3
"""End-to-end architecture runner with crash-safe matched-budget tables.

Stage A: base matched-budget eval on data/budget40/eval_manifest.jsonl
Stage B: QLoRA on train_manifest traces (append-per-question)
Stage C: re-run matched-budget with the saved adapter

Does not start unless you run this file from the terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from evaluation import BenchmarkRunner
from evaluation.answer_utils import extract_gsm8k_gold
from pipeline.budget_table import BudgetTableWriter, empty_error_row
from pipeline.inference import InferencePipeline
from pipeline.qubo_builder import QUBOBuilder
from pipeline.reasoning import run_reasoning_pipeline
from pipeline.sampling import DiverseSampler
from pipeline.solver import SimulatedAnnealingSolver
from pipeline.verifier import ReasonVerifier
from training.sft import QUBOSFTTrainer

# Import helpers from the sibling benchmark script without requiring a package.
import importlib.util

_bench_path = Path(__file__).resolve().parent / "run_all_benchmarks.py"
_spec = importlib.util.spec_from_file_location("run_all_benchmarks_helpers", _bench_path)
_bench = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bench)
baseline_cot = _bench.baseline_cot
baseline_greedy = _bench.baseline_greedy
extract_answer = _bench.extract_answer
is_correct = _bench.is_correct
run_matched_budget = _bench.run_matched_budget
set_seed = _bench.set_seed

BENCH_ORDER = ["gsm8k", "bbh", "strategyqa", "arc_challenge", "mmlu"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Architecture pipeline: matched budget → QLoRA → re-eval"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-qlora",
        action="store_true",
        help="Only run Stage A (base budget table).",
    )
    return parser.parse_args()


def _write_stage(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_existing_trace_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("source_id") or row.get("id")
            if sid is not None:
                done.add(str(sid))
    return done


def _sort_manifest(rows: list[dict]) -> list[dict]:
    rank = {b: i for i, b in enumerate(BENCH_ORDER)}
    return sorted(rows, key=lambda r: (rank.get(r.get("benchmark"), 99), str(r.get("source_id"))))


def _build_stack(config_path: str, device: str, adapter_path: str | None, load_in_4bit: bool):
    inference = InferencePipeline(
        config_path,
        device=device,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
    )
    runtime = str(inference.device)
    sampler = DiverseSampler(
        config_path,
        device=runtime,
        shared_model=inference.model,
        shared_tokenizer=inference.tokenizer,
    )
    verifier = ReasonVerifier(config_path, device=runtime)
    if load_in_4bit and verifier.device.type == "cuda":
        verifier.nli_model = verifier.nli_model.half()
    qubo_builder = QUBOBuilder(config_path, device=runtime)
    solver = SimulatedAnnealingSolver(config_path, device=runtime)
    return inference, sampler, verifier, qubo_builder, solver


def run_matched_eval(
    *,
    config_path: str,
    eval_rows: list[dict],
    outputs_dir: Path,
    stage: str,
    adapter_path: str | None,
    resume: bool,
    device: str,
    load_in_4bit: bool,
) -> BudgetTableWriter:
    writer = BudgetTableWriter(outputs_dir, stage=stage)
    writer.rebuild()
    inference, sampler, verifier, qubo_builder, solver = _build_stack(
        config_path, device, adapter_path, load_in_4bit
    )
    try:
        for idx, item in enumerate(eval_rows, start=1):
            bench = item["benchmark"]
            source_id = str(item["source_id"])
            question = item["question"]
            gold = item["gold"]
            task_type = item.get("task_type") or "math"
            is_mcq = bool(item.get("is_mcq"))

            if resume and writer.is_done(bench, source_id, stage):
                print(f"  [{stage}] skip {bench} {source_id}")
                continue

            print(f"  [{stage}] {idx}/{len(eval_rows)} {bench} {source_id}")
            try:
                t0 = time.time()
                pred_g = baseline_greedy(inference, question, bench)
                t1 = time.time()
                pred_c = baseline_cot(inference, question, bench)
                t2 = time.time()
                pred_sc, pred_bo, pred_q, _ = run_matched_budget(
                    sampler,
                    verifier,
                    qubo_builder,
                    solver,
                    inference,
                    question,
                    bench,
                    task_type,
                    is_mcq,
                )
                t3 = time.time()

                pred_g_n = extract_answer(pred_g, bench)
                pred_c_n = extract_answer(pred_c, bench)
                # SC/Best/QUBO already extracted inside run_matched_budget
                c_g = int(is_correct(pred_g_n, gold, bench))
                c_c = int(is_correct(pred_c_n, gold, bench))
                c_sc = int(is_correct(pred_sc, gold, bench))
                c_bo = int(is_correct(pred_bo, gold, bench))
                c_q = int(is_correct(pred_q, gold, bench))

                writer.append_row(
                    {
                        "benchmark": bench,
                        "id": source_id,
                        "question": question,
                        "gold": gold,
                        "pred_greedy": pred_g_n,
                        "pred_cot": pred_c_n,
                        "pred_sc16": pred_sc,
                        "pred_best16": pred_bo,
                        "pred_qubo": pred_q,
                        "correct_greedy": c_g,
                        "correct_cot": c_c,
                        "correct_sc16": c_sc,
                        "correct_best16": c_bo,
                        "correct_qubo": c_q,
                        "runtime_greedy_s": round(t1 - t0, 4),
                        "runtime_cot_s": round(t2 - t1, 4),
                        "runtime_matched_s": round(t3 - t2, 4),
                        "error": "",
                        "stage": stage,
                        "adapter": adapter_path or "",
                    }
                )
            except Exception as exc:
                writer.append_row(
                    empty_error_row(
                        benchmark=bench,
                        source_id=source_id,
                        question=question,
                        gold=gold,
                        stage=stage,
                        adapter=adapter_path or "",
                        error=str(exc),
                    )
                )
                print(f"  ERROR {bench} {source_id}: {exc}")
    finally:
        del inference, sampler, verifier, qubo_builder, solver
    return writer


def collect_train_traces(
    *,
    config_path: str,
    train_rows: list[dict],
    round_dir: Path,
    resume: bool,
    device: str,
    load_in_4bit: bool,
) -> list[dict]:
    traces_path = round_dir / "train_traces.jsonl"
    done = _load_existing_trace_ids(traces_path) if resume else set()
    inference, sampler, verifier, qubo_builder, solver = _build_stack(
        config_path, device, None, load_in_4bit
    )
    records: list[dict] = []
    # Reload already-written traces for QLoRA.
    if traces_path.exists():
        with traces_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    try:
        for idx, item in enumerate(train_rows, start=1):
            source_id = str(item["source_id"])
            if source_id in done:
                print(f"  [train] skip {item['benchmark']} {source_id}")
                continue
            bench = item["benchmark"]
            question = item["question"]
            gold = item["gold"]
            task_type = item.get("task_type") or "math"
            print(f"  [train] {idx}/{len(train_rows)} {bench} {source_id}")
            result = run_reasoning_pipeline(
                sampler,
                verifier,
                qubo_builder,
                solver,
                inference,
                question,
                task_type=task_type,
                generate_answer=False,
            )
            if not result or not result.get("selected_traces"):
                continue
            target = gold.strip()
            if bench == "gsm8k" and "####" in target:
                target = extract_gsm8k_gold(target)
            row = {
                "id": len(records) + 1,
                "source_id": source_id,
                "benchmark": bench,
                "question": question,
                "selected_traces": result["selected_traces"],
                "final_prompt": result["final_prompt"],
                "predicted_answer": "",
                "correct_answer": target,
                "correct": 0,
                "energy": result.get("energy", 0.0),
            }
            _append_jsonl(traces_path, row)
            records.append(row)
            _write_stage(
                round_dir / "stage.json",
                {
                    "stage": "collecting_traces",
                    "n_traces": len(records),
                    "adapter": None,
                    "last_source_id": source_id,
                },
            )
    finally:
        del inference, sampler, verifier, qubo_builder, solver
    return records


def main():
    args = parse_args()
    set_seed(args.seed)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    arch = config.get("architecture_run") or {}
    eval_manifest = Path(arch.get("eval_manifest", "data/budget40/eval_manifest.jsonl"))
    train_manifest = Path(arch.get("train_manifest", "data/budget40/train_manifest.jsonl"))
    outputs_dir = Path(arch.get("outputs_dir", "outputs/architecture_40q"))
    resume = (not args.no_resume) and bool(arch.get("resume", True))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not eval_manifest.exists() or not train_manifest.exists():
        raise SystemExit(
            f"Missing manifests. Run: python scripts/curate_budget40.py\n"
            f"  expected {eval_manifest} and {train_manifest}"
        )

    eval_rows = _sort_manifest(BenchmarkRunner.load_manifest(eval_manifest))
    train_rows = _sort_manifest(BenchmarkRunner.load_manifest(train_manifest))
    print(f"Eval panel: {len(eval_rows)}  Train panel: {len(train_rows)}")
    print(f"Outputs: {outputs_dir}  resume={resume}")

    trainer = QUBOSFTTrainer(args.config)
    device = trainer.device
    load_in_4bit = trainer.inference_4bit

    # Stage A
    print("\n=== Stage A: base matched-budget eval ===")
    run_matched_eval(
        config_path=args.config,
        eval_rows=eval_rows,
        outputs_dir=outputs_dir,
        stage="base",
        adapter_path=None,
        resume=resume,
        device=device,
        load_in_4bit=load_in_4bit,
    )
    _write_stage(
        outputs_dir / "round_1" / "stage.json",
        {"stage": "base_eval_done", "adapter": None, "n_traces": 0},
    )

    if args.skip_qlora:
        print("Skipping QLoRA (--skip-qlora).")
        return

    adapter_path = None
    for round_idx in range(1, args.rounds + 1):
        round_dir = outputs_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Stage B: QLoRA round {round_idx} ===")
        records = collect_train_traces(
            config_path=args.config,
            train_rows=train_rows,
            round_dir=round_dir,
            resume=resume,
            device=device,
            load_in_4bit=load_in_4bit,
        )
        if not records:
            print("No training traces; stopping.")
            break

        _write_stage(
            round_dir / "stage.json",
            {"stage": "qlora_starting", "adapter": adapter_path, "n_traces": len(records)},
        )
        dataset = trainer.prepare_dataset_from_pipeline(records)
        adapter_path = trainer.train(
            dataset,
            run_name=f"architecture-sft-round-{round_idx}",
            resume_adapter=adapter_path,
        )
        _write_stage(
            round_dir / "stage.json",
            {
                "stage": "qlora_done",
                "adapter": adapter_path,
                "n_traces": len(records),
            },
        )

        print(f"\n=== Stage C: post-QLoRA matched-budget (round {round_idx}) ===")
        run_matched_eval(
            config_path=args.config,
            eval_rows=eval_rows,
            outputs_dir=outputs_dir,
            stage="post_qlora",
            adapter_path=adapter_path,
            resume=resume,
            device=device,
            load_in_4bit=load_in_4bit,
        )
        _write_stage(
            round_dir / "stage.json",
            {
                "stage": "post_qlora_eval_done",
                "adapter": adapter_path,
                "n_traces": len(records),
            },
        )

    print(f"\nDone. Tables under {outputs_dir}")
    print(f"  {outputs_dir / 'budget_table.md'}")
    print(f"  {outputs_dir / 'budget_table_post_qlora.md'}")
    if adapter_path:
        print(f"  adapter: {adapter_path}")


if __name__ == "__main__":
    main()
