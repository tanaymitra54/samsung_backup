"""Read-only dataset-loader QA: label parsing, counts, topics and split disjointness."""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostic_data import BENCHMARKS, load_pool, select_examples
from evaluation.strict_answers import grade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--output", default="verification/dataset_loader_check.json")
    args = parser.parse_args()
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), scope="real dataset loading and gold parsing; no model inference", benchmarks={})
    for benchmark in args.benchmarks:
        roles, ids = {}, {}
        for role in ("fit", "tune", "eval"):
            print(f"Loading {benchmark} {role}...", flush=True)
            rows = select_examples(load_pool(benchmark, role), None, 42)
            if not rows:
                raise ValueError(f"Empty partition: {benchmark}/{role}")
            for row in rows:
                assert grade(row["gold"], row)["correct"]
            roles[role] = dict(questions=len(rows), topics=dict(Counter(r["topic"] for r in rows)),
                              source_splits=dict(Counter(r["source_split"] for r in rows)),
                              answer_types=dict(Counter(r["answer_type"] for r in rows)))
            ids[role] = {r["question_hash"] for r in rows}
        intersections = {f"{a}/{b}": len(ids[a] & ids[b]) for a, b in (("fit", "tune"), ("fit", "eval"), ("tune", "eval"))}
        if any(intersections.values()):
            raise ValueError(f"Question overlap in {benchmark}: {intersections}")
        report["benchmarks"][benchmark] = dict(roles=roles, intersections=intersections)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Dataset QA passed: {path}")


if __name__ == "__main__":
    main()
