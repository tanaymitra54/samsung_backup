import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
project_root = Path('e:/Programming/SamsungPRISM')
sys.path.insert(0, str(project_root))

from evaluation.answer_utils_v2 import extract_predicted_answer_v2

samples = ['dpo_mmlu_58', 'dpo_gsm8k_4', 'sft_bbh_75', 'dpo_bbh_14', 'sft_bbh_9']
labeled_file = project_root / 'data' / 'validation' / 'manual_label_sample_LABELED.jsonl'

print('| ID | Human Label | Parser Extracted |')
print('|---|---|---|')

with open(labeled_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line)
        if data['id'] in samples:
            raw = data.get('raw_output', '')
            q = data.get('question', '')
            bmark = data.get('benchmark', '')
            is_mcq = bmark in ['mmlu', 'bbh']
            human = data.get('human_label')
            pred = extract_predicted_answer_v2(raw, is_mcq, q)
            print(f"| {data['id']} | {human} | {pred} |")
