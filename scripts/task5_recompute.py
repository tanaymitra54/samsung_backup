import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.answer_utils_v2 import extract_predicted_answer_v2, strip_units

def main():
    eval_dir = project_root / "results" / "eval"
    out_dir = project_root / "results" / "eval_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for fpath in eval_dir.glob("*.jsonl"):
        cond_bmark = fpath.stem.replace("_results", "")
        parts = cond_bmark.split("_")
        cond = parts[0]
        bmark = parts[1]
        
        if cond not in stats:
            stats[cond] = {}
        if bmark not in stats[cond]:
            stats[cond][bmark] = {"correct": 0, "total": 0, "unparseable": 0}
            
        out_fpath = out_dir / fpath.name
        is_mcq = bmark in ["mmlu", "bbh"]
        
        with open(fpath, 'r', encoding='utf-8') as fin, open(out_fpath, 'w', encoding='utf-8') as fout:
            for line in fin:
                if not line.strip(): continue
                data = json.loads(line)
                
                raw = data.get("pred_cot_raw", "")
                q = data.get("question", "")
                gold = str(data.get("gold", ""))
                
                pred = extract_predicted_answer_v2(raw, is_mcq=is_mcq, question=q)
                data["pred_cot_corrected"] = pred
                
                stats[cond][bmark]["total"] += 1
                
                if pred is None:
                    stats[cond][bmark]["unparseable"] += 1
                else:
                    match = False
                    if is_mcq:
                        import re
                        h_clean = re.sub(r"[\(\)\[\]]", "", gold.strip()).upper()
                        p_clean = re.sub(r"[\(\)\[\]]", "", pred.strip()).upper()
                        match = (h_clean == p_clean) or (p_clean in h_clean) or (h_clean in p_clean)
                    else:
                        if "####" in gold:
                            gold_val = gold.split("####")[-1].strip()
                        else:
                            gold_val = gold
                        h_str = strip_units(gold_val)
                        p_str = strip_units(pred)
                        try:
                            match = abs(float(h_str) - float(p_str)) < 1e-6
                        except ValueError:
                            match = h_str == p_str
                            
                    if match:
                        stats[cond][bmark]["correct"] += 1
                        data["correct_cot_corrected"] = 1
                    else:
                        data["correct_cot_corrected"] = 0
                        
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                
    print("--- RE-EXTRACTION REPORT ---")
    
    # Calculate unparseable rates per condition (aggregated across benchmarks)
    cond_unparseable_rates = {}
    cond_accuracies = {}
    
    for cond, bmarks in stats.items():
        total_q = sum(b["total"] for b in bmarks.values())
        total_u = sum(b["unparseable"] for b in bmarks.values())
        total_c = sum(b["correct"] for b in bmarks.values())
        
        rate = total_u / total_q * 100 if total_q > 0 else 0
        acc = total_c / total_q * 100 if total_q > 0 else 0
        parsed_acc = total_c / (total_q - total_u) * 100 if (total_q - total_u) > 0 else 0
        
        cond_unparseable_rates[cond] = rate
        cond_accuracies[cond] = {"overall": acc, "parsed": parsed_acc}
        
    rates = list(cond_unparseable_rates.values())
    max_diff = max(rates) - min(rates) if rates else 0
    flag_warning = max_diff > 5.0
    
    for cond in stats:
        print(f"\nCondition: {cond.upper()}")
        for bmark, bstats in stats[cond].items():
            t = bstats["total"]
            c = bstats["correct"]
            u = bstats["unparseable"]
            acc = c / t * 100 if t > 0 else 0
            u_rate = u / t * 100 if t > 0 else 0
            print(f"  - {bmark.upper()}: Acc {acc:.2f}% ({c}/{t}) | Unparseable {u_rate:.2f}% ({u})")
        print(f"  => Aggregate Acc: {cond_accuracies[cond]['overall']:.2f}% | Parsed Acc: {cond_accuracies[cond]['parsed']:.2f}% | Aggregate Unparseable: {cond_unparseable_rates[cond]:.2f}%")
        
    print("\n--- SUMMARY ---")
    if flag_warning:
        print(f"[WARNING] Unparseable rate variance is > 5% (Max difference: {max_diff:.2f}%)")
        print("          This implies asymmetrical evaluation penalizing one condition unfairly.")
    else:
        print(f"[OK] Unparseable rate variance is <= 5% (Max difference: {max_diff:.2f}%)")

if __name__ == "__main__":
    main()
