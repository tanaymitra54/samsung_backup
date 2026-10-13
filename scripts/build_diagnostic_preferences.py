"""Export answer-verified DPO pairs from reserved fit pools."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostic_data import read_jsonl
from evaluation.strict_answers import grade
from training.dpo_data import validate_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", default="data/diagnostic_preference_pairs.jsonl")
    parser.add_argument("--topics", nargs="+", help="Use independently labeled fit questions from diagnosed failure topics")
    args = parser.parse_args()
    pairs, seen, dropped = [], set(), 0
    for path in args.input:
        for record in read_jsonl(path):
            e = record["example"]
            if e.get("role") != "fit":
                raise ValueError("Never build retraining pairs from eval/tune questions; reproduce failure areas on reserved fit questions")
            if args.topics and e.get("topic") not in args.topics:
                continue
            if record.get("error") or e["question_hash"] in seen:
                continue
            good, bad = [], []
            for sample in record["samples"]:
                label = grade(sample["raw_text"], e)
                if label["valid"]:
                    (good if label["correct"] else bad).append(sample)
            if not good or not bad:
                dropped += 1
                continue
            chosen = max(good, key=lambda s: s["confidence_score"])
            rejected = max(bad, key=lambda s: s["confidence_score"])
            from evaluation.diagnostic_runner import answer_instruction
            prompt = e["question"] + "\n" + answer_instruction(e)
            pairs.append(dict(e, prompt=prompt, chosen=chosen["raw_text"], rejected=rejected["raw_text"],
                              chosen_answer=grade(chosen["raw_text"], e)["answer"], rejected_answer=grade(rejected["raw_text"], e)["answer"],
                              seed=record["seed"], label_scope="final_answer_only; reasoning truth unverified"))
            seen.add(e["question_hash"])
    if not pairs:
        raise ValueError("No correct-vs-incorrect parseable fit pairs")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    validate_pairs(path)
    print(f"Verified {len(pairs)} training pairs; {dropped} pools lacked both labels: {path}")


if __name__ == "__main__":
    main()
