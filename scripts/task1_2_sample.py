import os
import json
import random
from pathlib import Path

RESULTS_DIR = Path('e:/Programming/SamsungPRISM/results/eval')
VAL_DIR = Path('e:/Programming/SamsungPRISM/data/validation')
VAL_DIR.mkdir(parents=True, exist_ok=True)

conditions = ['base', 'sft', 'dpo']
benchmarks = ['gsm8k', 'mmlu', 'bbh']

print('=== TASK 1: INVENTORY ===')
files_inventory = []
for cond in conditions:
    for bmark in benchmarks:
        filename = f'{cond}_{bmark}_results.jsonl'
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f'File not found: {filename}')
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                continue
            num_questions = len(lines)
            first_row = json.loads(lines[0])
            raw_fields = [k for k in first_row.keys() if 'raw' in k]
            
            print(f'File: {filename}')
            print(f'Condition: {cond}, Benchmark: {bmark}')
            print(f'Number of questions: {num_questions}')
            print(f'Raw fields available: {raw_fields}')
            if not raw_fields:
                print(f'FLAG: No raw_output or equivalent field found in {filename}!')
            print('-' * 40)
            
            files_inventory.append({
                'cond': cond,
                'bmark': bmark,
                'filepath': filepath,
                'lines': lines,
                'raw_fields': raw_fields
            })

print('=== TASK 2: SAMPLING ===')
samples = []
random.seed(42) # For reproducibility
for inv in files_inventory:
    cond = inv['cond']
    bmark = inv['bmark']
    lines = inv['lines']
    raw_fields = inv['raw_fields']
    
    target_raw_field = 'pred_cot_raw' if 'pred_cot_raw' in raw_fields else (raw_fields[0] if raw_fields else None)
    
    if not target_raw_field:
        print(f'Skipping {cond} {bmark} for sampling due to missing raw field.')
        continue
        
    sampled_lines = random.sample(lines, min(15, len(lines)))
    for line_str in sampled_lines:
        row = json.loads(line_str)
        q_id = row.get('id', 'unknown')
        sample = {
            'id': f'{cond}_{bmark}_{q_id}',
            'condition': cond,
            'benchmark': bmark,
            'question': row.get('question', ''),
            'gold': row.get('gold', ''),
            'raw_output': row.get(target_raw_field, ''),
            'human_label': None
        }
        samples.append(sample)

jsonl_path = VAL_DIR / 'manual_label_sample.jsonl'
txt_path = VAL_DIR / 'manual_label_sample.txt'

with open(jsonl_path, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')

with open(txt_path, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(f'=== {s["id"]} ===\n')
        f.write(f'QUESTION: {s["question"]}\n')
        f.write(f'GOLD: {s["gold"]}\n')
        f.write('RAW OUTPUT:\n')
        f.write(f'{s["raw_output"]}\n\n')
        f.write('WHAT DID THE MODEL ACTUALLY ANSWER? (fill in below)\n')
        f.write('YOUR LABEL: \n\n')

print(f'Generated {len(samples)} samples.')
print(f'Saved JSONL to {jsonl_path}')
print(f'Saved TXT to {txt_path}')
print('Waiting for human labels...')
