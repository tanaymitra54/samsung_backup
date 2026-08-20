#!/usr/bin/env python3
"""
==============================================================================
FILE: scripts/run_architecture_loop.py
ROLE: Closed-Loop Orchestrator for the Finalised Samsung PRISM Architecture
==============================================================================

Implements the approved architecture diagram end to end, including the two
feedback arrows that the previous single-pass pipeline did not close.

    query + initial prompt
              |
              v
    SLM generates multiple paths          <--------------------+
              |                                                |
              v                                                | "re run
    QUBO mapping + Quantum solver                              |  benchmark"
              |                                                |
              v                                                |
    Select the best reasoning                                  |
              |                                                |
              v                                                |
    generate final prompt                                      |
              |                                                |
              v                                                |
    QLoRA SFT                     <-------------+              |
              |                                 | "Re-train    |
              v                                 |  QLoRA"      |
    SLM Qwen 3B + LoRA Adapters                 |              |
              |                                 |              |
              v                                 |              |
    final answer                                |              |
              |                                 |              |
              v                                 |              |
    Outputs directory --------------------------+--------------+

Each round maps onto the diagram as:

  Stage A  "SLM generates multiple paths" -> "QUBO mapping + Quantum solver"
           -> "Select the best reasoning" -> "generate final prompt"
           = scripts/generate_training_data.py, run with the PREVIOUS round's
             adapter merged in. This is the "re run benchmark" arrow: round N+1
             re-generates its candidate reasoning paths using the model that
             round N produced, instead of always re-sampling from the base SLM.

  Stage B  "QLoRA SFT" -> "SLM Qwen 3B + LoRA Adapters"
           = scripts/fast_finetune_pipeline.py (fine-tune stage only).
             This is the "Re-train QLoRA" arrow.

  Stage C  "final answer" -> "Outputs directory"
           = scripts/run_all_benchmarks.py, evaluated with the new adapter.

A fresh LoRA is trained from the base SLM each round rather than stacking
adapters. The improvement carries across rounds through the DATA (each round's
QUBO-curated traces are drawn from a stronger model), which avoids compounding
adapter drift across rounds.

Usage
-----
    # Three rounds, quick sizes, with a base-model baseline first
    python scripts/run_architecture_loop.py --rounds 3 --quick --baseline

    # Resume an interrupted loop (finished rounds are detected and skipped)
    python scripts/run_architecture_loop.py --rounds 3 --quick --resume
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _run(cmd: list, stage: str, env: dict | None = None) -> float:
    """Run a subprocess, streaming output. Exits the loop on failure."""
    print(f"\n{'=' * 70}")
    print(f"[Loop] {stage}")
    print(f"[Cmd]  {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{stage}' failed with exit code {result.returncode}.")
        print("[ERROR] Stopping the loop so the failure is not silently carried forward.")
        sys.exit(result.returncode)

    print(f"\n[Done] {stage}  ({_fmt_duration(elapsed)})")
    return elapsed


def _read_accuracy(results_dir: Path, label: str, benchmarks: list[str]) -> dict:
    """Per-benchmark accuracy for each decode mode, read from the eval JSONL rows."""
    summary = {}
    for bm in benchmarks:
        path = results_dir / f"{label}_{bm}_results.jsonl"
        if not path.exists():
            continue

        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            continue

        entry = {"n": len(rows)}
        for mode in ("greedy", "cot", "qubo"):
            scored = [r for r in rows if r.get(f"correct_{mode}") is not None]
            entry[mode] = (
                sum(1 for r in scored if r[f"correct_{mode}"]) / len(scored)
                if scored else None
            )
        summary[bm] = entry
    return summary


def _save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ── Stage A: sample -> QUBO -> select -> final prompt ────────────────────────

def stage_generate(args, round_dir: Path, adapter_path: str | None,
                   focus_file: Path | None = None) -> Path:
    """Diagram stages: multiple paths -> QUBO+solver -> best reasoning -> final prompt."""
    data_dir = round_dir / "data"
    train_file = data_dir / "finetune_train.jsonl"

    if args.resume and train_file.exists():
        n = sum(1 for _ in open(train_file, encoding="utf-8"))
        print(f"[Loop] Reusing existing curated data: {train_file} ({n} examples).")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "generate_training_data.py"),
        "--config", str(_REPO_ROOT / args.config),
        "--qubo-params", str(_REPO_ROOT / args.qubo_params),
        "--output-dir", str(data_dir),
    ]
    if adapter_path:
        # The feedback arrow: this round samples from the previous round's model.
        cmd += ["--adapter-path", adapter_path]
    if focus_file and Path(focus_file).exists():
        # Curriculum: concentrate on the train-split questions the previous round
        # struggled with, rather than re-covering the pool uniformly.
        cmd += ["--focus-file", str(focus_file), "--focus-repeat", str(args.focus_repeat)]
    if args.device:
        cmd += ["--device", args.device]
    if args.quick:
        cmd += ["--quick"]
    if args.datagen_args:
        # Passthrough for per-dataset sizes (--n-gsm8k, --n-math, ...). Without
        # this the loop can only use generate_training_data's own defaults, in
        # which MATH is 0 -- so the competition-math source would never be used.
        cmd += shlex.split(args.datagen_args)
    if args.resume:
        cmd += ["--resume"]

    for attr, flag in (
        ("n_gsm8k", "--n-gsm8k"),
        ("n_mmlu", "--n-mmlu"),
        ("n_arc", "--n-arc"),
        ("n_logiqa", "--n-logiqa"),
        ("n_strategyqa", "--n-strategyqa"),
        ("n_openorca", "--n-openorca"),
    ):
        val = getattr(args, attr, None)
        if val is not None:
            cmd += [flag, str(val)]

    src = adapter_path or "base SLM"
    _run(cmd, f"Stage A -- sample + QUBO select + final prompt (from {src})")
    return data_dir


# ── Stage B: QLoRA SFT ───────────────────────────────────────────────────────

def stage_train(args, round_idx: int, data_dir: Path) -> str:
    """Diagram stage: QLoRA SFT -> SLM + LoRA adapters."""
    run_name = f"{args.run_prefix}-round{round_idx}"
    adapter_out = _REPO_ROOT / "checkpoints" / run_name / "final_adapter"

    if args.resume and (adapter_out / "adapter_config.json").exists():
        print(f"[Loop] Reusing existing adapter: {adapter_out}")
        return str(adapter_out)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "fast_finetune_pipeline.py"),
        "--skip-datagen",
        "--skip-eval",
        "--data-dir", str(data_dir),
        "--config", args.config,
        "--run-name", run_name,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.resume:
        # Within-stage resume: pick up the newest per-epoch checkpoint rather than
        # restarting a partially finished fine-tune from scratch.
        cmd += ["--resume-finetune"]
    if args.unsloth:
        cmd += ["--unsloth"]

    _run(cmd, f"Stage B -- QLoRA SFT (round {round_idx})")
    return str(adapter_out)


# ── Stage B2: optional DPO on top of the round's SFT adapter ─────────────────

def stage_dpo(args, round_idx: int, data_dir: Path, sft_adapter: str) -> str | None:
    """Preference-pair extraction followed by DPO, initialised from the SFT policy.

    Returns the DPO adapter path, or None when no usable preference pairs exist
    (which happens when few chains clear the score margin).
    """
    pairs_file = data_dir / "preference_pairs.jsonl"
    run_name = f"{args.run_prefix}-round{round_idx}-dpo"
    dpo_dir = _REPO_ROOT / "checkpoints" / run_name
    dpo_adapter = dpo_dir / "final_adapter"

    if args.resume and (dpo_adapter / "adapter_config.json").exists():
        print(f"[Loop] Reusing existing DPO adapter: {dpo_adapter}")
        return str(dpo_adapter)

    if not (args.resume and pairs_file.exists()):
        _run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "generate_preference_pairs.py"),
                "--input", str(data_dir / "finetune_train.jsonl"),
                "--output", str(pairs_file),
                "--min-margin", str(args.dpo_min_margin),
            ],
            f"Stage B2a -- preference pairs (round {round_idx})",
        )

    n_pairs = 0
    if pairs_file.exists():
        with open(pairs_file, encoding="utf-8") as f:
            n_pairs = sum(1 for line in f if line.strip())

    if n_pairs < args.dpo_min_pairs:
        print(
            f"[Loop] Only {n_pairs} preference pairs (need {args.dpo_min_pairs}); "
            "skipping DPO for this round."
        )
        return None

    dpo_cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_dpo.py"),
        "--config", str(_REPO_ROOT / args.config),
        "--data-file", str(pairs_file),
        "--output-dir", str(dpo_dir),
        "--sft-adapter", sft_adapter,
        "--epochs", str(args.dpo_epochs),
        "--beta", str(args.dpo_beta),
    ]
    if args.resume:
        dpo_cmd += ["--resume"]
    if args.unsloth:
        dpo_cmd += ["--unsloth"]

    _run(dpo_cmd, f"Stage B2b -- DPO from SFT policy (round {round_idx})")
    return str(dpo_adapter)


# ── Regression gate (catastrophic-forgetting guard) ──────────────────────────

def _qubo_acc(acc: dict, benchmark: str) -> float | None:
    return (acc.get(benchmark) or {}).get("qubo")


def mean_qubo(acc: dict, benchmarks: list[str]) -> float:
    vals = [_qubo_acc(acc, b) for b in benchmarks]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else -1.0


def benchmark_deltas(baseline: dict | None, candidate: dict,
                     benchmarks: list[str]) -> dict[str, float]:
    """Per-benchmark change vs the base model, in accuracy points."""
    if not baseline:
        return {}
    out = {}
    for b in benchmarks:
        base_v, cand_v = _qubo_acc(baseline, b), _qubo_acc(candidate, b)
        if base_v is not None and cand_v is not None:
            out[b] = cand_v - base_v
    return out


def worst_regression(deltas: dict[str, float]) -> tuple[str | None, float]:
    """The benchmark that lost the most ground, and by how much (<=0 means loss)."""
    if not deltas:
        return None, 0.0
    b = min(deltas, key=lambda k: deltas[k])
    return b, deltas[b]


def choose_adapter(candidates: list[tuple[str, str, dict]], baseline: dict | None,
                   benchmarks: list[str], max_regression: float) -> tuple[str, str, str]:
    """Pick which arm to carry forward, refusing to reward catastrophic forgetting.

    candidates: list of (arm_name, adapter_path, accuracy_dict).

    Selecting purely on MEAN accuracy -- what this did before -- rewards exactly
    the failure mode we care about: an adapter that gains 15 points on GSM8K and
    loses 10 on MMLU has a better mean than one that gains 3 everywhere, so the
    loop would carry the forgetful model forward and compound the damage over
    rounds. A candidate is "safe" only if no individual benchmark has fallen more
    than max_regression below the BASE model; ties among safe candidates are then
    broken on mean. If nothing is safe, the least-damaging arm is carried forward
    with a loud warning rather than silently.

    Returns (arm_name, adapter_path, reason).
    """
    scored = []
    for name, path, acc in candidates:
        deltas = benchmark_deltas(baseline, acc, benchmarks)
        wb, wd = worst_regression(deltas)
        scored.append({
            "name": name, "path": path, "acc": acc,
            "mean": mean_qubo(acc, benchmarks),
            "worst_bench": wb, "worst_delta": wd,
            "safe": (not deltas) or wd >= -max_regression,
        })

    safe = [s for s in scored if s["safe"]]
    pool = safe if safe else scored
    winner = max(pool, key=lambda s: s["mean"])

    if safe:
        reason = f"best mean among arms with no benchmark down >{max_regression:.1%}"
    else:
        reason = (f"NO arm cleared the regression gate; carrying the least-damaging "
                  f"({winner['worst_bench']} {winner['worst_delta']:+.1%})")
    return winner["name"], winner["path"], reason


def report_regression(label: str, deltas: dict[str, float], max_regression: float) -> None:
    if not deltas:
        return
    parts = []
    for b, d in deltas.items():
        flag = "  <-- REGRESSION" if d < -max_regression else ""
        parts.append(f"      {b:<16}{d:+7.1%}{flag}")
    print(f"    {label} vs base model:")
    print("\n".join(parts))


# ── Failure-targeted curriculum ──────────────────────────────────────────────

def collect_failed_questions(results_dir: Path, label: str, benchmarks: list[str]) -> list[dict]:
    """Questions the QUBO pipeline got wrong, for the next round to concentrate on.

    This is what makes the loop a curriculum rather than a re-run: instead of
    regenerating uniformly over the same pool, the next round can oversample the
    items that actually failed.
    """
    failed = []
    for bm in benchmarks:
        path = results_dir / f"{label}_{bm}_results.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("correct_qubo") is False:
                    failed.append({
                        "benchmark": bm,
                        "id": row.get("id"),
                        "question": row.get("question", ""),
                        "gold": row.get("gold", ""),
                    })
    return failed


# ── Stage C: final answer -> outputs directory ───────────────────────────────

def stage_evaluate(args, results_dir: Path, label: str, adapter_path: str | None) -> dict:
    """Diagram stages: final answer -> Outputs directory."""
    results_dir.mkdir(parents=True, exist_ok=True)

    expected = [results_dir / f"{label}_{bm}_results.jsonl" for bm in args.benchmarks]
    if args.resume and all(p.exists() for p in expected):
        print(f"[Loop] Reusing existing eval results for '{label}' in {results_dir}.")
        return _read_accuracy(results_dir, label, args.benchmarks)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_all_benchmarks.py"),
        "--output-dir", str(results_dir),
        "--condition-label", label,
        "--subset-size", str(args.eval_subset),
        "--benchmarks", *args.benchmarks,
        "--seed", str(args.seed),
    ]
    if args.device:
        cmd += ["--device", args.device]

    # run_all_benchmarks (and its multi-GPU workers) pick the adapter up from the
    # environment, so set it here rather than mutating this process's os.environ.
    env = os.environ.copy()
    if adapter_path:
        env["QUBO_ADAPTER_PATH"] = adapter_path
    else:
        env.pop("QUBO_ADAPTER_PATH", None)

    _run(cmd, f"Stage C -- benchmark eval ({label})", env=env)
    return _read_accuracy(results_dir, label, args.benchmarks)


# ── Reporting ────────────────────────────────────────────────────────────────

def print_loop_report(manifest: dict, benchmarks: list[str]):
    rounds = manifest.get("rounds", [])
    if not rounds:
        print("\n[Loop] No completed rounds to report.")
        return

    print("\n" + "=" * 78)
    print("  CLOSED-LOOP RESULTS  (accuracy of the full QUBO pipeline per round)")
    print("=" * 78)

    header = f"  {'Round':<10} | {'Arm':<6} | " + " | ".join(f"{b:>10}" for b in benchmarks)
    print(header)
    print("-" * len(header))

    def _row(label: str, arm: str, acc: dict):
        cells = []
        for bm in benchmarks:
            val = (acc.get(bm) or {}).get("qubo")
            cells.append(f"{val:>10.2%}" if val is not None else f"{'N/A':>10}")
        print(f"  {label:<10} | {arm:<6} | " + " | ".join(cells))

    for r in rounds:
        name = r.get("label", "?")
        arm = "base" if r.get("adapter") is None else "sft"
        _row(name, arm, r.get("accuracy", {}))
        if r.get("dpo_accuracy"):
            _row("", "dpo", r["dpo_accuracy"])

    print("=" * 78)

    # Round-over-round movement on the QUBO path, which is the pipeline's own
    # output. Each round is represented by whichever arm was carried forward.
    def _best_acc(r: dict) -> dict:
        if r.get("dpo_accuracy") and r.get("carried_forward") == r.get("dpo_adapter"):
            return r["dpo_accuracy"]
        return r.get("accuracy", {})

    if len(rounds) >= 2:
        first, last = rounds[0], rounds[-1]
        print("\n  Net change, first round to last (QUBO decode, best arm per round):")
        for bm in benchmarks:
            a = (_best_acc(first).get(bm) or {}).get("qubo")
            b = (_best_acc(last).get(bm) or {}).get("qubo")
            if a is None or b is None:
                print(f"    {bm:<12} N/A")
            else:
                print(f"    {bm:<12} {a:.2%} -> {b:.2%}  ({b - a:+.2%})")
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Closed-loop orchestrator for the finalised architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rounds", type=int, default=3,
                   help="Number of closed-loop rounds (default: 3)")
    p.add_argument("--baseline", action="store_true",
                   help="Evaluate the un-finetuned base SLM first, as round 0")
    p.add_argument("--resume", action="store_true",
                   help="Skip stages whose outputs already exist")

    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--qubo-params", default="config/best_qubo_params.yaml")
    p.add_argument("--device", default=None, help="cuda:0 / cpu")
    p.add_argument("--loop-dir", default="outputs/architecture_loop",
                   help="Root directory for loop artefacts (default: outputs/architecture_loop)")
    p.add_argument("--run-prefix", default="qubo-loop",
                   help="Checkpoint name prefix (default: qubo-loop)")

    # Data generation sizes
    p.add_argument("--quick", action="store_true", help="Reduced dataset sizes")
    p.add_argument("--n-gsm8k", type=int, default=None)
    p.add_argument("--n-mmlu", type=int, default=None)
    p.add_argument("--n-arc", type=int, default=None)
    p.add_argument("--n-logiqa", type=int, default=None)
    p.add_argument("--n-strategyqa", type=int, default=None)
    p.add_argument("--n-openorca", type=int, default=None)

    # Training hyper-parameters
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--unsloth", action="store_true",
                   help="Pass --unsloth through to both the SFT and DPO stages. Free for "
                        "single-GPU LoRA, which is what both stages do. Not applicable to "
                        "the datagen or evaluation stages -- those don't train a model.")

    p.add_argument("--datagen-args", default=None,
                   help="Extra arguments forwarded verbatim to generate_training_data.py, "
                        "e.g. \"--n-gsm8k 3000 --n-math 2500\". Needed to set per-dataset "
                        "sizes; without it the loop uses that script's own defaults, in "
                        "which MATH is 0.")

    # Failure-targeted curriculum
    p.add_argument("--focus", action="store_true",
                   help="Each round concentrates on the TRAIN-split questions the previous "
                        "round struggled with (hard_questions.json), instead of re-covering "
                        "the pool uniformly.")
    p.add_argument("--focus-repeat", type=int, default=2,
                   help="How many times each hard question is re-added (default: 2)")

    # Catastrophic-forgetting guard
    p.add_argument("--max-regression", type=float, default=0.03,
                   help="Largest per-benchmark drop vs the BASE model that an adapter "
                        "may have and still be carried forward (default 0.03 = 3 points). "
                        "Selection was previously on mean accuracy alone, which rewards "
                        "an adapter that gains 15 points on one benchmark while losing 10 "
                        "on another. Requires --baseline to have a reference to compare "
                        "against. Set to 1.0 to disable the gate entirely.")

    # DPO (optional stage after SFT in each round)
    p.add_argument("--dpo", action="store_true",
                   help="After SFT, extract preference pairs and run DPO from the SFT policy. "
                        "Each round then reports both an SFT and a DPO adapter.")
    p.add_argument("--dpo-epochs", type=int, default=3)
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--dpo-min-margin", type=float, default=0.3,
                   help="Minimum chosen-vs-rejected score gap for a usable pair")
    p.add_argument("--dpo-min-pairs", type=int, default=50,
                   help="Skip DPO for a round if fewer usable pairs than this")

    # Evaluation
    p.add_argument("--benchmarks", nargs="*", default=["gsm8k", "mmlu", "bbh"])
    p.add_argument("--eval-subset", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    if args.rounds < 1:
        print("[ERROR] --rounds must be at least 1.")
        sys.exit(1)

    # Subprocesses run with cwd=_REPO_ROOT, so anchor relative paths there too;
    # otherwise artefacts split across two directories when invoked from elsewhere.
    loop_root = Path(args.loop_dir)
    if not loop_root.is_absolute():
        loop_root = _REPO_ROOT / loop_root
    loop_root.mkdir(parents=True, exist_ok=True)
    manifest_path = loop_root / "loop_manifest.json"

    manifest = {
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "rounds_requested": args.rounds,
        "benchmarks": args.benchmarks,
        "eval_subset": args.eval_subset,
        "rounds": [],
    }

    wall_start = time.time()
    print("\n" + "=" * 70)
    print("  CLOSED-LOOP ARCHITECTURE RUN")
    print(f"  Started    : {manifest['started']}")
    print(f"  Rounds     : {args.rounds}   Baseline: {args.baseline}")
    print(f"  Benchmarks : {', '.join(args.benchmarks)} ({args.eval_subset} questions each)")
    print(f"  Artefacts  : {loop_root}")
    print("=" * 70)

    # ── Round 0: base SLM, no adapter ────────────────────────────────────────
    baseline_accuracy: dict | None = None
    if args.baseline:
        base_results = loop_root / "round0_baseline" / "results"
        accuracy = stage_evaluate(args, base_results, "base", None)
        baseline_accuracy = accuracy
        manifest["rounds"].append({
            "label": "round 0",
            "adapter": None,
            "results_dir": str(base_results),
            "accuracy": accuracy,
        })
        _save_manifest(manifest_path, manifest)
    else:
        print("[Loop] WARNING: no --baseline. Without a base-model reference the "
              "regression gate cannot detect catastrophic forgetting, so adapters "
              "are selected on mean accuracy alone.")

    # ── Closed loop ──────────────────────────────────────────────────────────
    adapter_path: str | None = None
    focus_file: Path | None = None

    for round_idx in range(1, args.rounds + 1):
        round_dir = loop_root / f"round{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 70)
        print(f"#  ROUND {round_idx} / {args.rounds}")
        print(f"#  Sampling model: {adapter_path or 'base SLM (no adapter)'}")
        print("#" * 70)

        # A: generate multiple paths -> QUBO -> select best -> final prompt
        data_dir = stage_generate(args, round_dir, adapter_path, focus_file)

        # B: QLoRA SFT -> SLM + LoRA adapter
        new_adapter = stage_train(args, round_idx, data_dir)

        # C: final answer -> outputs directory
        results_dir = round_dir / "results"
        accuracy = stage_evaluate(args, results_dir, "sft", new_adapter)

        record = {
            "label": f"round {round_idx}",
            "adapter": new_adapter,
            "sampled_from": adapter_path,
            "data_dir": str(data_dir),
            "results_dir": str(results_dir),
            "accuracy": accuracy,
        }

        # B2 + C again: optional DPO arm, evaluated the same way.
        best_adapter = new_adapter
        best_arm = "sft"
        arms: list[tuple[str, str, dict]] = [("sft", new_adapter, accuracy)]

        if args.dpo:
            dpo_adapter = stage_dpo(args, round_idx, data_dir, new_adapter)
            if dpo_adapter:
                dpo_accuracy = stage_evaluate(args, results_dir, "dpo", dpo_adapter)
                record["dpo_adapter"] = dpo_adapter
                record["dpo_accuracy"] = dpo_accuracy
                arms.append(("dpo", dpo_adapter, dpo_accuracy))

        # Carry forward under the regression gate rather than on mean alone.
        print(f"\n[Loop] Round {round_idx} arm comparison:")
        for name, _path, acc in arms:
            deltas = benchmark_deltas(baseline_accuracy, acc, args.benchmarks)
            print(f"    {name.upper():<4} mean QUBO {mean_qubo(acc, args.benchmarks):.2%}")
            report_regression(name.upper(), deltas, args.max_regression)
            record[f"{name}_mean_qubo"] = mean_qubo(acc, args.benchmarks)
            record[f"{name}_deltas_vs_base"] = deltas

        best_arm, best_adapter, why = choose_adapter(
            arms, baseline_accuracy, args.benchmarks, args.max_regression
        )
        record["carry_forward_reason"] = why
        print(f"[Loop] Carrying forward {best_arm.upper()} -- {why}.")

        # A regression that survives the gate still matters: it compounds every
        # round, since the next round samples its training data FROM this model.
        winning_acc = next(acc for name, _p, acc in arms if name == best_arm)
        winning_deltas = benchmark_deltas(baseline_accuracy, winning_acc, args.benchmarks)
        wb, wd = worst_regression(winning_deltas)
        if wb is not None and wd < -args.max_regression:
            print(
                f"[Loop] *** CATASTROPHIC FORGETTING WARNING ***\n"
                f"       {wb} is {wd:+.1%} vs the base model, past the "
                f"{args.max_regression:.1%} tolerance.\n"
                f"       The next round samples its training data from this model, so\n"
                f"       this loss will compound. Consider stopping, widening the\n"
                f"       training mix beyond math, or lowering --lr / --epochs."
            )

        # Record which questions the pipeline still gets wrong, so the next round
        # can be pointed at them (failure-targeted curriculum).
        # Harvest failures from whichever arm is actually being carried forward.
        eval_label = best_arm
        failed = collect_failed_questions(results_dir, eval_label, args.benchmarks)
        failures_path = round_dir / "failed_questions.json"
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        record["n_failed"] = len(failed)
        record["failed_questions"] = str(failures_path)
        print(f"[Loop] {len(failed)} questions still failing; written to {failures_path}")

        record["carried_forward"] = best_adapter
        manifest["rounds"].append(record)
        _save_manifest(manifest_path, manifest)

        # Close the loop: the next round samples from this round's best model,
        # and concentrates on the train-split questions this round found hard.
        adapter_path = best_adapter
        round_hard = data_dir / "hard_questions.json"
        if args.focus and round_hard.exists():
            focus_file = round_hard
            try:
                n_hard = len(json.loads(round_hard.read_text(encoding="utf-8")))
                print(f"[Loop] Next round will focus on {n_hard} hard training questions.")
            except (json.JSONDecodeError, OSError):
                pass

    manifest["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest["wall_seconds"] = round(time.time() - wall_start, 1)
    _save_manifest(manifest_path, manifest)

    print_loop_report(manifest, args.benchmarks)

    print("=" * 70)
    print(f"  Loop complete. Total wall time: {_fmt_duration(time.time() - wall_start)}")
    print(f"  Final adapter : {adapter_path}")
    print(f"  Manifest      : {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
