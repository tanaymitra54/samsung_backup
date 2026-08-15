import json
import random
from pathlib import Path
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    project_root = Path(__file__).resolve().parent.parent
    eval_dir = project_root / "results" / "eval"
    labeled_file = project_root / "data" / "validation" / "manual_label_sample_LABELED.jsonl"
    
    # Load already labeled IDs to exclude them
    labeled_ids = set()
    if labeled_file.exists():
        with open(labeled_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    labeled_ids.add(json.loads(line)['id'])
                    
    all_samples = []
    
    for fpath in eval_dir.glob("*.jsonl"):
        cond_bmark = fpath.stem.replace("_results", "")
        parts = cond_bmark.split("_")
        cond = parts[0]
        bmark = parts[1]
        
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                q_id = f"{cond}_{bmark}_{data.get('id', 'unknown')}"
                if q_id not in labeled_ids:
                    all_samples.append({
                        "id": q_id,
                        "condition": cond,
                        "benchmark": bmark,
                        "question": data.get("question", ""),
                        "raw_output": data.get("pred_cot_raw", ""),
                        "gold": data.get("gold", "")
                    })
                    
    # Sample 15
    random.seed(42)
    sample_size = min(15, len(all_samples))
    sampled = random.sample(all_samples, sample_size)
    
    out_file = project_root / "data" / "validation" / "fresh_holdout_15.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for s in sampled:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Sampled {sample_size} fresh items and saved to {out_file}")
    
    # Print them for the user to label
    print("\n--- NEW SAMPLES TO LABEL ---")
    for i, s in enumerate(sampled):
        print(f"\n[{i+1}/15] ID: {s['id']} | Benchmark: {s['benchmark']}")
        print(f"Question: {s['question'][:200]}...")
        raw = s['raw_output']
        print(f"Raw Output (tail): {raw[-300:] if len(raw) > 300 else raw}")
        print(f"Gold Answer: {s['gold']}")
        print(f"Please provide label (or 'None'): ")
        
if __name__ == "__main__":
    main()
