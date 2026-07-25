import argparse
import json
import random
from pathlib import Path

def check_blank(text: str) -> bool:
    if not text:
        return True
    return text.strip() == ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="results/eval", type=str)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Directory {eval_dir} does not exist.")
        return

    jsonl_files = list(eval_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {eval_dir}.")
        return

    # To group by condition, we know each file gives us greedy/cot/qubo for a specific 'label' (base, sft, dpo).
    # label = base -> A (baseline), B (QUBO only)
    # label = sft -> C (SFT only), D (SFT + QUBO)
    # label = dpo -> E (DPO only), F (DPO + QUBO)
    
    mapping = {
        "base": {"greedy": "A. Baseline (Greedy)", "cot": "A. Baseline (CoT)", "qubo": "B. QUBO only"},
        "sft": {"greedy": "C. SFT only (Greedy)", "cot": "C. SFT only (CoT)", "qubo": "D. SFT + QUBO"},
        "dpo": {"greedy": "E. DPO only (Greedy)", "cot": "E. DPO only (CoT)", "qubo": "F. DPO + QUBO"},
    }

    all_stats = {}
    
    for fpath in jsonl_files:
        fname = fpath.name
        # expected format: {label}_{benchmark}_results.jsonl
        parts = fname.replace("_results.jsonl", "").split("_", 1)
        if len(parts) != 2:
            continue
        label, benchmark = parts
        
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        records = [json.loads(line) for line in lines if line.strip()]
        if not records:
            continue

        for condition_key in ["greedy", "cot", "qubo"]:
            cond_name = mapping.get(label, {}).get(condition_key, f"{label}_{condition_key}")
            stats_key = f"{cond_name} | {benchmark}"
            
            blanks = 0
            for r in records:
                ext = r.get(f"pred_{condition_key}", "")
                if check_blank(ext):
                    blanks += 1
                    
            blank_rate = blanks / len(records)
            all_stats[stats_key] = {
                "total": len(records),
                "blanks": blanks,
                "rate": blank_rate,
                "records": records,
                "condition_key": condition_key,
            }

    print("=" * 80)
    print(f"{'Condition | Benchmark':<45} | Total | Blank | Blank Rate")
    print("-" * 80)
    for k, v in all_stats.items():
        rate_str = f"{v['rate']:.1%}"
        warn = "⚠️ WARN (>10%)" if v['rate'] > 0.1 else ""
        print(f"{k:<45} | {v['total']:<5} | {v['blanks']:<5} | {rate_str:<10} {warn}")
    print("=" * 80)
    print("\n--- Random Samples (Raw vs Extracted) ---")
    
    # Print 10 random samples across all data
    sample_pool = []
    for k, v in all_stats.items():
        cond_key = v["condition_key"]
        for r in v["records"]:
            sample_pool.append({
                "source": k,
                "raw": r.get(f"pred_{cond_key}_raw", ""),
                "extracted": r.get(f"pred_{cond_key}", ""),
                "question": r.get("question", ""),
                "gold": r.get("gold", ""),
            })

    if sample_pool:
        num_samples = min(10, len(sample_pool))
        samples = random.sample(sample_pool, num_samples)
        for i, s in enumerate(samples):
            print(f"\n[Sample {i+1}] Source: {s['source']}")
            raw = str(s['raw'])
            if len(raw) > 500:
                raw = raw[:500] + " ... [TRUNCATED]"
            print(f"RAW OUTPUT:\n{raw}")
            print(f"EXTRACTED: {repr(s['extracted'])}")
            print("-" * 40)

if __name__ == "__main__":
    main()
