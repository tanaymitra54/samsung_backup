#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) fine-tuning.

Uses correct reasoning traces as "chosen" and incorrect ones as "rejected"
to align the model toward better reasoning via preference learning.

Run AFTER SFT to further improve reasoning quality.
"""

import argparse
import json
import os
from importlib.metadata import version as imp_version

import torch
import yaml
from datasets import Dataset
from packaging.version import Version
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine-tuning")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--positive", default="training_data/positive_traces.json")
    parser.add_argument("--negative", default="training_data/negative_traces.json")
    parser.add_argument("--output-dir", default="checkpoints/prism-dpo")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def format_dpo_pair(question: str, chosen_trace: str, rejected_trace: str, gold_answer: str):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    chosen = f"{chosen_trace}\n\nAnswer: {gold_answer}<|im_end|>"
    rejected = f"{rejected_trace}\n\nAnswer: {gold_answer}<|im_end|>"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    model_name = args.model_name or model_cfg["name"]
    cache_dir = model_cfg.get("cache_dir")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    # Load preference pairs
    with open(args.positive) as f:
        positive = json.load(f)
    with open(args.negative) as f:
        negative = json.load(f)

    pos_by_q = {}
    for item in positive:
        pos_by_q.setdefault(item["question"], []).append(item)

    neg_by_q = {}
    for item in negative:
        neg_by_q.setdefault(item["question"], []).append(item)

    pairs = []
    for q in pos_by_q:
        if q in neg_by_q:
            pairs.append(format_dpo_pair(
                q,
                pos_by_q[q][0]["reasoning_trace"],
                neg_by_q[q][0]["reasoning_trace"],
                pos_by_q[q][0]["gold_answer"],
            ))

    if args.debug:
        pairs = pairs[:20]

    print(f"DPO preference pairs: {len(pairs)}")
    if len(pairs) == 0:
        print("No preference pairs. Need both positive and negative traces for same questions.")
        return

    dataset = Dataset.from_list(pairs)

    # ── Load model ───────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── LoRA ─────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    bf16_supported = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else "none",
        gradient_checkpointing=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )

    dpo_kwargs = {"processing_class": tokenizer} if Version(imp_version("trl")) >= Version("0.12.0") else {"tokenizer": tokenizer}

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        beta=args.beta,
        max_prompt_length=512,
        max_length=args.max_seq_length,
        **dpo_kwargs,
    )

    trainer.train()
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"DPO complete. Adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
