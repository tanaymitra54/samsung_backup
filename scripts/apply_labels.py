import json

manual_labels = {
    "sft_mmlu_67": "B",
    "base_gsm8k_78": "6",
    "base_bbh_22": "Q",
    "dpo_gsm8k_1": "3",
    "dpo_bbh_44": "3",
    "dpo_bbh_16": "valid",
    "base_mmlu_12": "165",
    "base_gsm8k_65": "12",
    "sft_gsm8k_52": "15",
    "base_gsm8k_48": "8",
    "sft_mmlu_7": "A",
    "dpo_mmlu_81": "C",
    "base_bbh_32": "B",
    "base_bbh_30": "B",
    "base_gsm8k_55": "14",
    "dpo_bbh_9": "A",
    "dpo_bbh_30": "A",
    "sft_gsm8k_2": "70000",
    "sft_mmlu_22": None,
    "base_bbh_26": "D",
    "sft_gsm8k_71": "60",
    "base_mmlu_84": "A",
    "sft_mmlu_79": None,
    "dpo_mmlu_78": "B",
    "dpo_bbh_12": None,
    "sft_bbh_11": "B",
    "sft_mmlu_6": "A",
    "dpo_gsm8k_5": "144",
    "base_bbh_7": "B",
    "base_mmlu_37": "B"
}

with open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_30.jsonl', 'r', encoding='utf-8') as fin, \
     open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_30_LABELED.jsonl', 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        qid = data['id']
        data['human_label'] = manual_labels.get(qid)
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')

with open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_45_LABELED.jsonl', 'w', encoding='utf-8') as fout:
    with open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_15_LABELED.jsonl', 'r', encoding='utf-8') as f15:
        for line in f15:
            fout.write(line)
    with open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_30_LABELED.jsonl', 'r', encoding='utf-8') as f30:
        for line in f30:
            fout.write(line)
