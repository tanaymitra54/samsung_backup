import json
import random
from pathlib import Path

project_root = Path('e:/Programming/SamsungPRISM')
eval_dir = project_root / 'results' / 'eval'

labeled_ids = set()
for fpath in [
    project_root / 'data' / 'validation' / 'manual_label_sample_LABELED.jsonl',
    project_root / 'data' / 'validation' / 'fresh_holdout_15_LABELED.jsonl',
    project_root / 'data' / 'validation' / 'fresh_holdout_30_LABELED.jsonl'
]:
    if fpath.exists():
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    labeled_ids.add(json.loads(line)['id'])

all_samples = []
for fpath in eval_dir.glob('*.jsonl'):
    cond_bmark = fpath.stem.replace('_results', '')
    parts = cond_bmark.split('_')
    cond, bmark = parts[0], parts[1]
    
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            q_id = f"{cond}_{bmark}_{data.get('id', 'unknown')}"
            if q_id not in labeled_ids:
                all_samples.append({
                    'id': q_id,
                    'condition': cond,
                    'benchmark': bmark,
                    'question': data.get('question', ''),
                    'raw_output': data.get('pred_cot_raw', ''),
                    'gold': data.get('gold', '')
                })

random.seed(123)
sample_size = min(30, len(all_samples))
sampled = random.sample(all_samples, sample_size)

out_file = project_root / 'data' / 'validation' / 'fresh_holdout_batch3.jsonl'
with open(out_file, 'w', encoding='utf-8') as f:
    for s in sampled:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
