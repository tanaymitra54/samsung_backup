import argparse
import json
from pathlib import Path

def build_content(chain: dict) -> str:
    reason = chain.get("reason", "").strip()
    answer = chain.get("answer", "").strip()
    if reason:
        return f"{reason}\n\nAnswer: {answer}"
    return f"Answer: {answer}"

def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs from QUBO data.")
    parser.add_argument("--input", default="data/finetune_train.jsonl", help="Path to SFT JSONL with full_chains_pool")
    parser.add_argument("--output", default="data/preference_pairs.jsonl", help="Output path for DPO JSONL")
    parser.add_argument("--min-margin", type=float, default=0.3, help="Minimum score difference between chosen and rejected")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist. Run datagen first.")
        return

    pairs = []
    dropped_no_pool = 0
    dropped_low_margin = 0
    dropped_no_rejected = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # The prompt is the user message
            prompt = ""
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break
            
            meta = item.get("metadata", {})
            chosen_score = meta.get("correctness_score", 0.0)
            
            pool = meta.get("full_chains_pool", [])
            if not pool:
                dropped_no_pool += 1
                continue
                
            # The chosen chain's content is already in the assistant message, but we can also just build it
            # The user asked for "full reasoning + answer"
            chosen_content = ""
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    chosen_content = msg.get("content", "")
                    break

            # Find the lowest-scoring chain
            lowest_chain = min(pool, key=lambda c: c.get("correctness_score", 0.0))
            rejected_score = lowest_chain.get("correctness_score", 0.0)

            if chosen_score - rejected_score >= args.min_margin:
                rejected_content = build_content(lowest_chain)
                if rejected_content.strip() == chosen_content.strip():
                    # Same actual text despite different scores? Unlikely, but let's be safe.
                    dropped_no_rejected += 1
                    continue

                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_content,
                    "rejected": rejected_content,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "margin": chosen_score - rejected_score,
                })
            else:
                dropped_low_margin += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("DPO Preference Pair Generation")
    print("=" * 60)
    print(f"Total processed     : {len(pairs) + dropped_no_pool + dropped_low_margin + dropped_no_rejected}")
    print(f"Pairs generated     : {len(pairs)}")
    print(f"Dropped (no pool)   : {dropped_no_pool}")
    print(f"Dropped (< margin)  : {dropped_low_margin}")
    print(f"Dropped (same text) : {dropped_no_rejected}")
    print(f"Saved to            : {output_path}")
    if pairs:
        avg_margin = sum(p["margin"] for p in pairs) / len(pairs)
        print(f"Average margin      : {avg_margin:.3f}")

if __name__ == "__main__":
    main()
