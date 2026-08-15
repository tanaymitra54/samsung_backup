"""
==============================================================================
FILE: scripts/evaluate_all.py
ROLE: Automated Multi-Stage Model Evaluation Harness (Base vs SFT vs DPO)
BRANCH ADDITION (abhyuday): Newly introduced orchestrator script that runs
comparative evaluations across Base LLM, SFT LoRA checkpoint, and DPO LoRA checkpoint
over standard benchmark sets (GSM8K, MMLU, BBH, ARC).
==============================================================================
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate Base, SFT, and DPO models")
    parser.add_argument("--sft-adapter", help="Path to SFT final_adapter")
    parser.add_argument("--dpo-adapter", help="Path to DPO final_adapter")
    parser.add_argument("--eval-subset", type=int, default=100, help="Subset size for eval")
    parser.add_argument("--benchmarks", nargs="*", default=["gsm8k", "mmlu", "bbh"])
    parser.add_argument("--output-dir", default="results/eval", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bmark_script = "scripts/run_all_benchmarks.py"

    conditions = [
        ("base", None),
        ("sft", args.sft_adapter),
        ("dpo", args.dpo_adapter),
    ]

    for label, adapter_path in conditions:
        if label != "base" and not adapter_path:
            print(f"[Eval] Skipping {label} evaluation (no adapter path provided).")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating Condition: {label.upper()}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, bmark_script,
            "--output-dir", str(out_dir),
            "--condition-label", label,
            "--subset-size", str(args.eval_subset),
            "--benchmarks", *args.benchmarks,
            "--seed", "42"
        ]

        env = os.environ.copy()
        if adapter_path:
            env["QUBO_ADAPTER_PATH"] = adapter_path
        elif "QUBO_ADAPTER_PATH" in env:
            del env["QUBO_ADAPTER_PATH"]

        subprocess.run(cmd, env=env, check=True)

    print("\n[Eval] All evaluations complete. Generating diagnostic report...")
    
    diag_cmd = [sys.executable, "scripts/diagnose_eval.py", "--eval-dir", str(out_dir)]
    subprocess.run(diag_cmd, check=True)

if __name__ == "__main__":
    main()
