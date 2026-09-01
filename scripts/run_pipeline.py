#!/usr/bin/env python3
"""Run the diagram pipeline: paths -> QUBO -> select -> prompt -> Qwen 3B+LoRA -> outputs.

Examples:
  python scripts/run_pipeline.py --query "What is 15% of 80?"
  python scripts/run_pipeline.py --loop --benchmarks gsm8k --subset-size 8
  python scripts/run_pipeline.py --retrain-from-outputs outputs/round_1
  python scripts/run_pipeline.py --check

Single GPU (recommended on remote):
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_pipeline.py --loop ... --device cuda:0
"""

import argparse
import json
import os
import sys
from datetime import datetime


def _mask_single_gpu_from_argv():
    for idx, arg in enumerate(sys.argv):
        if arg == "--device" and idx + 1 < len(sys.argv):
            device_arg = sys.argv[idx + 1]
            if device_arg.startswith("cuda:"):
                os.environ["CUDA_VISIBLE_DEVICES"] = device_arg.split(":", 1)[1]
            return


_mask_single_gpu_from_argv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import PRISMPipeline, check_flow, load_traces_for_sft


def parse_args():
    parser = argparse.ArgumentParser(description="PRISM diagram pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--query", default=None, help="Single question")
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--gold", default="")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device (cuda:N masks to that GPU; use cuda:0 after masking)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="QLoRA SFT loop: generate traces, train adapters, re-run",
    )
    parser.add_argument("--retrain-from-outputs", default=None, metavar="DIR")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument(
        "--dry-run-sft",
        action="store_true",
        help="Collect traces and skip CUDA QLoRA (no H100 needed)",
    )
    parser.add_argument("--check", action="store_true", help="Run solver/prompt self-check")
    return parser.parse_args()


def _resolve_device(device: str | None) -> str | None:
    if device and device.startswith("cuda:"):
        return "cuda:0"
    return device


def _load_eval_split(config_path: str, benchmarks: list[str] | None, subset_size: int | None):
    from evaluation import BenchmarkRunner

    runner = BenchmarkRunner(config_path)
    if subset_size is not None:
        runner.subset_size = subset_size
    names = benchmarks or ["gsm8k"]
    questions, golds = [], []
    for name in names:
        qs, ans = runner.load_benchmark(name)
        questions.extend(qs)
        golds.extend(ans)
    return questions, golds


def main():
    args = parse_args()
    if args.check:
        check_flow()
        return

    device = _resolve_device(args.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.retrain_from_outputs:
        pipe = PRISMPipeline(args.config, device=device, load_models=False)
        adapter = pipe.retrain_from_outputs(
            args.retrain_from_outputs,
            dry_run=args.dry_run_sft,
            run_name=f"qlora-retrain-{stamp}",
        )
        print(f"LoRA adapter: {adapter}")
        print(f"Traces used: {len(load_traces_for_sft(args.retrain_from_outputs))}")
        return

    if args.loop:
        questions, golds = _load_eval_split(
            args.config, args.benchmarks, args.subset_size
        )
        loop_dir = os.path.join(out_dir, f"loop_{stamp}")
        pipe = PRISMPipeline(args.config, device=device)
        adapter = pipe.run_iterative_loop(
            questions,
            golds,
            output_dir=loop_dir,
            num_rounds=args.rounds,
            dry_run=args.dry_run_sft,
        )
        print(f"Loop done. Adapter: {adapter}")
        print(f"Outputs: {loop_dir}")
        return

    if not args.query:
        raise SystemExit("Pass --query, --loop, --retrain-from-outputs, or --check")

    run_dir = os.path.join(out_dir, f"query_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    pipe = PRISMPipeline(args.config, device=device)
    result = pipe.run_query(
        args.query, initial_prompt=args.initial_prompt, gold=args.gold
    )
    traces_path = pipe.save_output(result, output_dir=run_dir)
    answer_path = os.path.join(run_dir, "final_answer.json")
    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": result["question"],
                "answer": result["answer"],
                "selected_traces": result["selected_traces"],
                "final_prompt": result["final_prompt"],
            },
            f,
            indent=2,
        )
    print(result["answer"])
    print(f"Saved: {answer_path}")
    print(f"Traces: {traces_path}")


if __name__ == "__main__":
    main()
