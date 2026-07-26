import argparse
import os
import json
import torch
import yaml
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Monkey-patch FSDPModule for compatibility with newer PyTorch / trl versions
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDPModule
    import torch.distributed.fsdp
    torch.distributed.fsdp.FSDPModule = FSDPModule

from trl import DPOTrainer, DPOConfig

def main():
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning using preference pairs")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data-file", default="data/preference_pairs.jsonl", help="Path to preference data")
    parser.add_argument("--output-dir", default="checkpoints/dpo-run", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"].get("cache_dir")
    train_cfg = cfg.get("training", {})
    max_seq = train_cfg.get("max_seq_length", 2048)

    print(f"[DPO] Loading dataset from {args.data_file}...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        print("[ERROR] Preference dataset is empty.")
        return

    # DPOTrainer expects a dataset with "prompt", "chosen", "rejected"
    # The tokenizer handles formatting if prompt is text, or we can pre-format
    # Our generate_preference_pairs.py outputs pre-formatted "prompt" text, but wait...
    # Is "prompt" text? Yes, it's the raw user string.
    # We should apply chat template to the prompt.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    formatted_rows = {"prompt": [], "chosen": [], "rejected": []}
    for r in rows:
        prompt_text = r["prompt"]
        prompt_formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
        formatted_rows["prompt"].append(prompt_formatted)
        
        # We also need chosen and rejected strings. DPOTrainer will concatenate prompt + chosen.
        # But wait, our `chosen` and `rejected` fields already contain just the assistant's content.
        # So we can just pass them as strings. DPOTrainer will handle the rest.
        formatted_rows["chosen"].append(r["chosen"] + tokenizer.eos_token)
        formatted_rows["rejected"].append(r["rejected"] + tokenizer.eos_token)

    train_ds = Dataset.from_dict(formatted_rows)
    print(f"[DPO] Loaded {len(train_ds)} preference pairs.")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    print(f"[DPO] Loading model {model_name}...")
    mkw = {
        "cache_dir": cache_dir,
        "device_map": "auto" if use_cuda else None,
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
    }
    
    # Base model
    model = AutoModelForCausalLM.from_pretrained(model_name, **mkw)
    
    # We also need a reference model. PEFT handles this automatically if we pass a standard model and peft_config.
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=train_cfg.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grad_accum = max(1, 4 // args.batch_size)

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        max_prompt_length=max_seq // 2,
        max_length=max_seq,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # DPOTrainer will create reference automatically with PEFT
        args=dpo_args,
        beta=args.beta,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print(f"[DPO] Starting training for {args.epochs} epochs...")
    trainer.train()
    
    adapter_out = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"[DPO] Adapter saved to {adapter_out}")

if __name__ == "__main__":
    main()
