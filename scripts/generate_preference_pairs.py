"""
Build DPO pairs from the same prompt: one gold-verified correct chain and
one gold-verified incorrect chain. Self-score is not the label.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.preference_labels import chain_matches_gold, select_verified_pair


def build_content(chain: dict) -> str:
    reason = chain.get("reason", "").strip()
    answer = chain.get("answer", "").strip()
    if reason:
        return f"{reason}\n\nAnswer: {answer}"
    return f"Answer: {answer}"


def task_type_for(source: str) -> str:
    return "math" if source in {"gsm8k", "math"} else "commonsense"


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO pairs from independently verified correct vs incorrect chains."
    )
    parser.add_argument("--input", default="data/finetune_train.jsonl")
    parser.add_argument("--output", default="data/preference_pairs.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist. Run datagen first.")
        return

    pairs = []
    dropped_no_pool = 0
    dropped_no_gold = 0
    dropped_unverified = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = ""
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break
            meta = item.get("metadata", {})
            gold = meta.get("gold", "")
            pool = meta.get("full_chains_pool", [])
            if not pool:
                dropped_no_pool += 1
                continue
            if not gold:
                dropped_no_gold += 1
                continue
            task_type = task_type_for(meta.get("source", ""))
            selected = select_verified_pair(pool, gold, task_type=task_type)
            if selected is None:
                dropped_unverified += 1
                continue
            chosen, rejected = selected
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": build_content(chosen),
                    "rejected": build_content(rejected),
                    "chosen_answer": chosen.get("answer", ""),
                    "rejected_answer": rejected.get("answer", ""),
                    "gold": gold,
                    "chosen_verified_correct": chain_matches_gold(
                        chosen, gold, task_type
                    ),
                    "rejected_verified_incorrect": not chain_matches_gold(
                        rejected, gold, task_type
                    ),
                    "source": meta.get("source", ""),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("DPO Preference Pair Generation (verified correct vs incorrect)")
    print(f"Pairs generated           : {len(pairs)}")
    print(f"Dropped (no pool)         : {dropped_no_pool}")
    print(f"Dropped (no gold)         : {dropped_no_gold}")
    print(f"Dropped (no verified pair): {dropped_unverified}")
    print(f"Saved to                  : {output_path}")


if __name__ == "__main__":
    main()
