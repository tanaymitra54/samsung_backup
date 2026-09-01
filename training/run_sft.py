#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) with QLoRA for the PRISM reasoning pipeline.

Uses standard HuggingFace + PEFT + bitsandbytes.
Runs on multi-GPU setups. Saves LoRA adapter + merged model.
"""

import argparse
import inspect
import os
import sys

import torch
import yaml

from datasets import load_from_disk
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT fine-tuning with QLoRA")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", default="training_data/combined_dataset", help="Path to combined dataset")
    parser.add_argument("--output-dir", default="checkpoints/prism-sft", help="Output directory")
    parser.add_argument("--model-name", default=None, help="Override model name from config")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--no-merge", action="store_true", help="Skip final model merge")
    parser.add_argument("--debug", action="store_true", help="Use only 100 examples for testing")
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    model_cfg = config["model"]
    train_cfg = config["training"]

    model_name = args.model_name or model_cfg["name"]
    epochs = args.epochs or train_cfg["sft_epochs"]
    lr = args.lr or train_cfg["learning_rate"]
    per_device_batch = args.batch_size or train_cfg["batch_size"]
    lora_rank = args.lora_rank or train_cfg["lora_rank"]
    lora_alpha = train_cfg.get("lora_alpha", 32)
    lora_dropout = train_cfg.get("lora_dropout", 0.05)
    max_seq_length = args.max_seq_length or train_cfg.get("max_seq_length", 2048)
    warmup_steps = train_cfg.get("warmup_steps", 100)
    lora_targets = train_cfg.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    cache_dir = model_cfg.get("cache_dir")

    num_gpus = torch.cuda.device_count()
    print(f"=" * 60)
    print(f"PRISM SFT Fine-Tuning")
    print(f"=" * 60)
    print(f"Model: {model_name}")
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    print(f"Epochs: {epochs}, LR: {lr}, Batch/GPU: {per_device_batch}")
    print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Output: {args.output_dir}")

    if not num_gpus:
        raise RuntimeError("No CUDA GPU detected. SFT requires a GPU.")

    # ── Load dataset ──────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.dataset}")
    dataset = load_from_disk(args.dataset)
    if args.debug:
        dataset = dataset.select(range(min(100, len(dataset))))
        print(f"  DEBUG: {len(dataset)} examples")
    else:
        print(f"  Examples: {len(dataset)}")

    # ── Load model with 4-bit quantization ───────────────────────
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── LoRA config ──────────────────────────────────────────────
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training arguments ───────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    bf16_supported = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else "none",
        run_name=os.path.basename(args.output_dir),
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False if num_gpus > 1 else None,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )

    # ── Trainer ──────────────────────────────────────────────────
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters
    sft_kwargs = {}

    if "processing_class" in params:
        sft_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        sft_kwargs["tokenizer"] = tokenizer

    if "dataset_kwargs" in params:
        sft_kwargs["dataset_kwargs"] = {"text": "text", "max_seq_length": max_seq_length}
    elif "dataset_text_field" in params:
        sft_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in params:
        sft_kwargs["max_seq_length"] = max_seq_length
    else:
        tokenizer.model_max_length = max_seq_length

    if "peft_config" in params:
        sft_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        **sft_kwargs,
    )

    # ── Train ────────────────────────────────────────────────────
    print(f"\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save adapter ─────────────────────────────────────────────
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to: {adapter_path}")

    # ── Merge and save full model ────────────────────────────────
    if not args.no_merge:
        print(f"\nMerging LoRA weights with base model...")
        del model, trainer
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        merged = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = merged.merge_and_unload()

        merged_path = os.path.join(args.output_dir, "merged_model")
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged model saved to: {merged_path}")

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Adapter: {adapter_path}")
    if not args.no_merge:
        print(f"  Merged:  {merged_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
