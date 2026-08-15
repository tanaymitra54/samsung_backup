import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path so we can import evaluation.answer_utils_v2
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.answer_utils_v2 import extract_predicted_answer_v2, strip_units

def main():
    labeled_file = project_root / "data" / "validation" / "fresh_holdout_45_v2_LABELED.jsonl"
    
    if not labeled_file.exists():
        print(f"File not found: {labeled_file}")
        return
        
    samples = []
    with open(labeled_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            samples.append(json.loads(line))
            
    print(f"Loaded {len(samples)} manually labeled samples.")
    
    correct_overall = 0
    total = len(samples)
    
    cond_stats = {}
    disagreements = []
    unparseable_counts = {}
    
    for sample in samples:
        cond = sample.get("condition", "unknown")
        bmark = sample.get("benchmark", "unknown")
        
        if cond not in cond_stats:
            cond_stats[cond] = {"correct": 0, "total": 0}
            unparseable_counts[cond] = 0
            
        raw = sample.get("raw_output", "")
        q = sample.get("question", "")
        human = sample.get("human_label")
        
        # Convert explicit "None" string to None
        if human == "None" or human == "null" or human == "":
            human = None
            
        is_mcq = bmark in ["mmlu", "bbh"]
        
        pred = extract_predicted_answer_v2(raw, is_mcq=is_mcq, question=q)
        
        if pred is None:
            unparseable_counts[cond] += 1
            
        match = False
        if human is None and pred is None:
            match = True
        elif human is not None and pred is not None:
            if is_mcq:
                import re
                h_clean = re.sub(r"[\(\)\[\]]", "", human.strip()).upper()
                p_clean = re.sub(r"[\(\)\[\]]", "", pred.strip()).upper()
                match = (h_clean == p_clean) or (p_clean in h_clean) or (h_clean in p_clean)
            else:
                h_str = strip_units(human)
                p_str = strip_units(pred)
                try:
                    match = abs(float(h_str) - float(p_str)) < 1e-6
                except ValueError:
                    match = h_str == p_str
                
        if match:
            correct_overall += 1
            cond_stats[cond]["correct"] += 1
        else:
            disagreements.append({
                "id": sample.get("id"),
                "cond": cond,
                "bmark": bmark,
                "human": human,
                "pred": pred,
                "raw_tail": raw[-300:] if len(raw) > 300 else raw
            })
            
        cond_stats[cond]["total"] += 1
        
    print("\n--- RESULTS ---")
    overall_acc = correct_overall / total * 100
    print(f"Overall Agreement: {correct_overall}/{total} ({overall_acc:.2f}%)")
    
    print("\nPer-Condition Agreement:")
    for cond, stats in cond_stats.items():
        acc = stats['correct'] / stats['total'] * 100
        unp = unparseable_counts[cond] / stats['total'] * 100
        print(f"  {cond}: {stats['correct']}/{stats['total']} ({acc:.2f}%) - Unparseable: {unp:.2f}%")
        
    if disagreements:
        print(f"\n--- {len(disagreements)} DISAGREEMENTS ---")
        for d in disagreements:
            print(f"ID: {d['id']} | BMARK: {d['bmark']}")
            print(f"HUMAN: {d['human']} | PRED: {d['pred']}")
            print(f"RAW (tail): {d['raw_tail']}")
            print("-" * 40)
            
    print("\n--- CHECKLIST ---")
    if overall_acc >= 90:
        print("[PASS] Overall agreement >= 90%")
    else:
        print("[FAIL] Overall agreement < 90%")
        
    accs = [s['correct'] / s['total'] for s in cond_stats.values()]
    if accs and max(accs) - min(accs) <= 0.05:
        print("[PASS] Per-condition variance <= 5%")
    else:
        print("[FAIL] Per-condition variance > 5%")

if __name__ == "__main__":
    main()
