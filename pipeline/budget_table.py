"""Crash-safe matched-budget table writer.

Appends one row per question with flush+fsync, then rebuilds the readable
budget table (CSV + Markdown) from whatever rows are already on disk.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from pipeline.stats_utils import binomial_ci

BENCH_ORDER = ["gsm8k", "bbh", "strategyqa", "arc_challenge", "mmlu"]
BENCH_LABEL = {
    "gsm8k": "GSM8K",
    "bbh": "BBH",
    "strategyqa": "StrategyQA",
    "arc_challenge": "ARC",
    "mmlu": "MMLU",
}
METHODS = ("greedy", "cot", "sc16", "best16", "qubo")


class BudgetTableWriter:
    def __init__(self, outputs_dir: str | Path, stage: str = "base"):
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.per_question_path = self.outputs_dir / "per_question.jsonl"
        if stage == "base":
            self.table_csv = self.outputs_dir / "budget_table.csv"
            self.table_md = self.outputs_dir / "budget_table.md"
        else:
            self.table_csv = self.outputs_dir / f"budget_table_{stage}.csv"
            self.table_md = self.outputs_dir / f"budget_table_{stage}.md"

    def completed_keys(self) -> set[tuple[str, str, str]]:
        """Return (stage, benchmark, source_id) already flushed."""
        done: set[tuple[str, str, str]] = set()
        if not self.per_question_path.exists():
            return done
        with self.per_question_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(
                    (
                        str(row.get("stage") or "base"),
                        str(row.get("benchmark") or ""),
                        str(row.get("id") or ""),
                    )
                )
        return done

    def is_done(self, benchmark: str, source_id: str, stage: str | None = None) -> bool:
        stage = stage or self.stage
        return (stage, benchmark, str(source_id)) in self.completed_keys()

    def append_row(self, row: dict) -> None:
        payload = dict(row)
        payload.setdefault("stage", self.stage)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self.per_question_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self.rebuild()

    def load_rows(self, stage: str | None = None) -> list[dict]:
        stage = stage or self.stage
        rows: list[dict] = []
        if not self.per_question_path.exists():
            return rows
        with self.per_question_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("stage") or "base") == stage:
                    rows.append(row)
        return rows

    def _aggregate(self, rows: list[dict]) -> dict[str, dict]:
        by_bench: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get("error"):
                # Still count errored questions toward n for transparency,
                # but treat correctness as 0.
                by_bench[str(row.get("benchmark") or "")].append(row)
            else:
                by_bench[str(row.get("benchmark") or "")].append(row)

        summary: dict[str, dict] = {}
        for bench in BENCH_ORDER:
            bench_rows = by_bench.get(bench, [])
            if not bench_rows:
                continue
            n = len(bench_rows)
            acc: dict[str, float] = {}
            correct_counts: dict[str, int] = {}
            for method in METHODS:
                key = f"correct_{method}"
                successes = sum(int(r.get(key) or 0) for r in bench_rows)
                correct_counts[method] = successes
                acc[method] = successes / n if n else 0.0
            rate, low, high = binomial_ci(correct_counts["qubo"], n)
            summary[bench] = {
                "num_samples": n,
                "accuracy": acc,
                "abs_gain_vs_sc16": acc["qubo"] - acc["sc16"],
                "ci95": {"qubo": {"rate": rate, "low": low, "high": high}},
            }
        return summary

    def rebuild(self) -> None:
        rows = self.load_rows(self.stage)
        summary = self._aggregate(rows)
        self._write_csv(summary)
        self._write_md(summary)

    def _fmt_pct(self, value: float) -> str:
        return f"{100.0 * value:.2f}%"

    def _write_csv(self, summary: dict[str, dict]) -> None:
        fieldnames = [
            "benchmark",
            "n",
            "greedy",
            "cot",
            "sc16",
            "best16",
            "qubo",
            "vs_sc16",
            "ci95_low",
            "ci95_high",
        ]
        with self.table_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for bench in BENCH_ORDER:
                if bench not in summary:
                    writer.writerow(
                        {
                            "benchmark": BENCH_LABEL[bench],
                            "n": "0/8",
                            "greedy": "—",
                            "cot": "—",
                            "sc16": "—",
                            "best16": "—",
                            "qubo": "—",
                            "vs_sc16": "—",
                            "ci95_low": "—",
                            "ci95_high": "—",
                        }
                    )
                    continue
                s = summary[bench]
                a = s["accuracy"]
                ci = s["ci95"]["qubo"]
                writer.writerow(
                    {
                        "benchmark": BENCH_LABEL[bench],
                        "n": f"{s['num_samples']}/8",
                        "greedy": self._fmt_pct(a["greedy"]),
                        "cot": self._fmt_pct(a["cot"]),
                        "sc16": self._fmt_pct(a["sc16"]),
                        "best16": self._fmt_pct(a["best16"]),
                        "qubo": self._fmt_pct(a["qubo"]),
                        "vs_sc16": f"{100.0 * s['abs_gain_vs_sc16']:+.2f}pp",
                        "ci95_low": self._fmt_pct(ci["low"]),
                        "ci95_high": self._fmt_pct(ci["high"]),
                    }
                )

    def _write_md(self, summary: dict[str, dict]) -> None:
        lines = [
            f"# Matched budget outcomes ({self.stage})",
            "",
            "| Benchmark | n | Greedy | CoT | SC-16 | Best-16 | QUBO | vs SC-16 | 95% CI QUBO |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for bench in BENCH_ORDER:
            if bench not in summary:
                lines.append(
                    f"| {BENCH_LABEL[bench]} | 0/8 | — | — | — | — | — | — | — |"
                )
                continue
            s = summary[bench]
            a = s["accuracy"]
            ci = s["ci95"]["qubo"]
            ci_s = f"[{self._fmt_pct(ci['low'])}, {self._fmt_pct(ci['high'])}]"
            lines.append(
                f"| {BENCH_LABEL[bench]} | {s['num_samples']}/8 | "
                f"{self._fmt_pct(a['greedy'])} | {self._fmt_pct(a['cot'])} | "
                f"{self._fmt_pct(a['sc16'])} | {self._fmt_pct(a['best16'])} | "
                f"{self._fmt_pct(a['qubo'])} | "
                f"{100.0 * s['abs_gain_vs_sc16']:+.2f}pp | {ci_s} |"
            )
        lines.append("")
        lines.append(
            "Rates use only completed questions for that benchmark. "
            "Rebuilt after every flushed row."
        )
        lines.append("")
        self.table_md.write_text("\n".join(lines), encoding="utf-8")


def empty_error_row(
    *,
    benchmark: str,
    source_id: str,
    question: str,
    gold: str,
    stage: str,
    adapter: str,
    error: str,
) -> dict:
    return {
        "benchmark": benchmark,
        "id": source_id,
        "question": question,
        "gold": gold,
        "pred_greedy": "",
        "pred_cot": "",
        "pred_sc16": "",
        "pred_best16": "",
        "pred_qubo": "",
        "correct_greedy": 0,
        "correct_cot": 0,
        "correct_sc16": 0,
        "correct_best16": 0,
        "correct_qubo": 0,
        "runtime_greedy_s": 0.0,
        "runtime_cot_s": 0.0,
        "runtime_matched_s": 0.0,
        "error": error,
        "stage": stage,
        "adapter": adapter or "",
    }
