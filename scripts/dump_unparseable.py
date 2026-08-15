import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
project_root = Path('e:/Programming/SamsungPRISM')
sys.path.insert(0, str(project_root))
from evaluation.answer_utils_v2 import extract_predicted_answer_v2

base_f = project_root / 'results' / 'eval' / 'base_bbh_results.jsonl'
dpo_f = project_root / 'results' / 'eval' / 'dpo_bbh_results.jsonl'

def print_unparseable(file, limit):
    count = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            raw = data.get('pred_cot_raw', '')
            q = data.get('question', '')
            pred = extract_predicted_answer_v2(raw, True, q)
            if pred is None:
                print(f"ID: {data['id']}")
                print(f"RAW: {raw[-300:] if len(raw)>300 else raw}")
                print('-'*40)
                count += 1
                if count >= limit: break

print('=== BASE UNPARSEABLE ===')
print_unparseable(base_f, 10)
print('=== DPO UNPARSEABLE ===')
print_unparseable(dpo_f, 10)
