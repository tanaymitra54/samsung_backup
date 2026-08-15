import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_30.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        raw = data['raw_output'].replace('\n', ' ')
        print(f"{i}. {data['id']}: {raw[-200:] if len(raw)>200 else raw}")
