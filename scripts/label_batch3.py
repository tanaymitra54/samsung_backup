import json

manual_labels = {
    "base_gsm8k_15": "96",
    "dpo_gsm8k_19": "6",
    "base_gsm8k_60": "17",
    "dpo_mmlu_91": "C",
    "dpo_gsm8k_17": "57500",
    "base_gsm8k_88": "8000",
    "base_bbh_47": "A",
    "dpo_mmlu_52": "D",
    "sft_gsm8k_74": "86",
    "sft_mmlu_9": "C",
    "dpo_gsm8k_95": "40",
    "dpo_mmlu_5": "B",
    "base_mmlu_51": None,
    "base_mmlu_21": "D",
    "dpo_mmlu_2": "D",
    "sft_mmlu_4": None, # it output the polynomial text directly without a letter. Unless value match catches it.
    "dpo_gsm8k_96": "3",
    "dpo_bbh_72": "C",
    "base_mmlu_56": "D",
    "base_bbh_2": "B",
    "sft_bbh_25": None,
    "sft_mmlu_54": "D",
    "dpo_mmlu_48": None,
    "base_gsm8k_37": "2",
    "base_bbh_9": "D",
    "dpo_gsm8k_75": "7.5",
    "sft_bbh_42": "6", # might be numeric option
    "base_gsm8k_80": "10",
    "base_gsm8k_1": "3",
    "base_gsm8k_66": "24"
}

with open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_batch3.jsonl', 'r', encoding='utf-8') as fin, \
     open('e:/Programming/SamsungPRISM/data/validation/fresh_holdout_batch3_LABELED.jsonl', 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        qid = data['id']
        data['human_label'] = manual_labels.get(qid)
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
