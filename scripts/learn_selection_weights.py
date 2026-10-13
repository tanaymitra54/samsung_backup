"""Learn trace scoring and selection coefficients without using evaluation answers."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostic_data import read_jsonl
from evaluation.learned_weights import fit_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", nargs="+", required=True, help="Fit results.jsonl files, optionally across datasets")
    parser.add_argument("--tune", nargs="+", required=True, help="Disjoint tune results.jsonl files")
    parser.add_argument("--output", default="config/learned_selection_weights.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()
    if args.candidates < 2:
        parser.error("Need at least two penalty candidates")
    weights = fit_weights([r for p in args.fit for r in read_jsonl(p)], [r for p in args.tune for r in read_jsonl(p)], args.seed, args.candidates)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"Learned weights: {path}; tune selected-vote accuracy: {weights['tune_accuracy']:.3f} (not test accuracy)")


if __name__ == "__main__":
    main()
