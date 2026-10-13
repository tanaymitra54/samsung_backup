"""Run fit -> tune -> learned weights -> evaluation for each selected benchmark."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.diagnostic_data import BENCHMARKS

ENTRYPOINTS = dict(gsm8k="evaluate_gsm8k.py", mmlu="evaluate_mmlu.py", bbh="evaluate_bbh.py",
                   arc_challenge="evaluate_arc.py", strategyqa="evaluate_strategyqa.py")


def combine_reports(output, benchmarks):
    from evaluation.diagnostic_report import holm_adjust
    comparisons = []
    for benchmark in benchmarks:
        report = json.loads((output / benchmark / "eval/diagnosis.json").read_text(encoding="utf-8"))
        for comparison in report["comparisons"]:
            comparisons.append(dict(comparison, benchmark=benchmark, within_report_holm_p=comparison["holm_p"]))
    for comparison, adjusted in zip(comparisons, holm_adjust([c["p"] for c in comparisons])):
        comparison["holm_p"] = adjusted
        if adjusted > 0.05:
            comparison["evidence"] = "inconclusive"
    payload = dict(benchmarks=benchmarks, correction_family="all comparisons across the selected dataset reports", comparisons=comparisons)
    (output / "combined_diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Combined dataset diagnosis", "", "Holm correction covers all comparisons across these dataset reports. Use this combined report when searching for advantages across datasets.", "",
             "| Dataset | Group | Comparator | N questions | Delta | Global Holm p | Evidence |", "|---|---|---|---:|---:|---:|---|"]
    for comparison in comparisons:
        lines.append(f"| {comparison['benchmark']} | {comparison['group_type']}: {comparison['group']} | {comparison['comparator']} | {comparison['n_questions']} | {comparison['delta']:+.1%} | {comparison['holm_p']:.4f} | {comparison['evidence']} |")
    (output / "combined_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--calibration-questions", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print commands without executing")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Evaluation dispatches its own GPU workers; run one dispatcher, not torchrun/srun with multiple tasks")
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir or config.get("diagnostics", {}).get("outputs_dir", "outputs/diagnostics")).resolve()
    for benchmark in args.benchmarks:
        base = output / benchmark
        common = ["--config", str(Path(args.config).resolve())]
        for flag in ("multi_gpu", "no_multi_gpu", "resume"):
            if getattr(args, flag):
                common.append("--" + flag.replace("_", "-"))
        for key in ("device", "adapter"):
            if getattr(args, key):
                common += ["--" + key, str(getattr(args, key))]
        if args.gpu_ids:
            common += ["--gpu-ids"] + [str(i) for i in args.gpu_ids]
        for manifest in args.exclude_manifest:
            common += ["--exclude-manifest", str(Path(manifest).resolve())]
        commands = []
        entry = str(ROOT / "scripts" / ENTRYPOINTS[benchmark])
        for role in ("fit", "tune"):
            command = [sys.executable, entry, "--role", role, "--collect-only", "--output-dir", str(base / role)] + common
            if args.calibration_questions:
                command += ["--questions", str(args.calibration_questions)]
            commands.append(command)
        weights = base / "learned_weights.json"
        commands.append([sys.executable, str(ROOT / "scripts/learn_selection_weights.py"),
                         "--fit", str(base / "fit/results.jsonl"), "--tune", str(base / "tune/results.jsonl"), "--output", str(weights)])
        command = [sys.executable, entry, "--weights", str(weights), "--output-dir", str(base / "eval")] + common
        if args.questions:
            command += ["--questions", str(args.questions)]
        if args.seeds:
            command += ["--seeds"] + [str(i) for i in args.seeds]
        commands.append(command)
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.plan:
                subprocess.run(command, cwd=ROOT, check=True)
    if not args.plan:
        combine_reports(output, args.benchmarks)
        print(f"Across-dataset report: {output / 'combined_diagnosis.md'}")


if __name__ == "__main__":
    main()
