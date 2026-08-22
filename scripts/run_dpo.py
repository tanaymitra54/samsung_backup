"""
==============================================================================
FILE: scripts/run_dpo.py
ROLE: Direct Preference Optimization (DPO) Fine-Tuning Module
BRANCH ADDITION (abhyuday): Newly introduced trainer script using `trl` `DPOTrainer`.
Features dynamic kwarg parameter passing to maintain cross-version compatibility across
differing `trl` / `transformers` releases and includes FSDP compatibility monkey-patches.
==============================================================================
"""

import argparse
import os
import json
import sys

# Unsloth patches transformers/peft/trl at import time, so it must be imported
# before any of them -- importing it after (e.g. inside main(), once argparse
# has parsed --unsloth) is too late, since this module already imports those
# libraries below at module load time. Scanning sys.argv directly, before those
# imports, is the only way an opt-in flag can control this.
if "--unsloth" in sys.argv:
    from unsloth import FastLanguageModel
    _UNSLOTH_AVAILABLE = True
else:
    _UNSLOTH_AVAILABLE = False

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.device_utils import resolve_device

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
    parser.add_argument(
        "--device", default=None,
        help="cuda:0 / cpu. Default: auto-pick the GPU with the most free VRAM.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the newest per-epoch checkpoint in --output-dir",
    )
    parser.add_argument(
        "--sft-adapter", default=None,
        help="SFT adapter to merge into the base model before DPO. Standard DPO "
             "initialises from the SFT policy; without this the run is DPO-from-base "
             "and does not build on the SFT stage at all.",
    )
    parser.add_argument(
        "--unsloth", action="store_true",
        help="Load the model and apply LoRA via Unsloth's FastLanguageModel instead "
             "of plain transformers+peft. Free for single-GPU LoRA (this script's "
             "case). This is the less-tested of the two Unsloth paths in this repo -- "
             "the merge-SFT-adapter-then-repeft-for-DPO sequence is a more composite "
             "operation than plain SFT+Unsloth, so verify the SFT-only --unsloth path "
             "works before trusting this one for a real run.",
    )
    args = parser.parse_args()
    # Sanity check: the sys.argv pre-scan above and argparse's own parsing of
    # --unsloth must agree, or the import-order fix silently did nothing.
    assert args.unsloth == _UNSLOTH_AVAILABLE, (
        "--unsloth parsed inconsistently with the module-level sys.argv scan; "
        "this should not happen."
    )

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

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        # Also covers the Unsloth path below, which has no device_map kwarg
        # and otherwise defaults to whatever torch.cuda.current_device() is.
        torch.cuda.set_device(device.index or 0)
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    print(f"[DPO] Device: {device}")

    print(f"[DPO] Loading model {model_name}...")

    if args.unsloth:
        print(f"[DPO] Loading via Unsloth (4-bit=False, bf16={use_bf16}) ...")
        # Formatting above already used `tokenizer` to build the dataset; keep
        # that instance rather than swapping in the one returned here, so the
        # tokenizer used to format the text matches the one training sees.
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            load_in_4bit=False,
            cache_dir=cache_dir,
        )
    else:
        mkw = {
            "cache_dir": cache_dir,
            # Pinned to the one selected GPU rather than "auto": accelerate's
            # "auto" placement ignores which card we picked as most-free and
            # spreads/chooses across every visible device instead.
            "device_map": {"": device.index or 0} if use_cuda else None,
            "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        }
        model = AutoModelForCausalLM.from_pretrained(model_name, **mkw)

    # Initialise from the SFT policy when one is supplied. The adapter is merged
    # into the weights so the fresh DPO LoRA below trains on top of it, and the
    # implicit reference model DPOTrainer derives is the SFT policy rather than
    # the raw base model -- which is what DPO's objective assumes. This is a
    # standard PEFT merge and works the same whether or not Unsloth loaded the
    # base model: Unsloth's patches sit on top of an ordinary PreTrainedModel,
    # which is all PeftModel.from_pretrained / merge_and_unload need.
    if args.sft_adapter:
        print(f"[DPO] Merging SFT adapter as the starting policy: {args.sft_adapter}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=False)
        model = model.merge_and_unload()
        print("[DPO] SFT adapter merged.")
    else:
        print("[DPO] WARNING: no --sft-adapter given; training DPO from the BASE model.")

    # We also need a reference model. PEFT handles this automatically if we pass a standard model and peft_config.
    if args.unsloth:
        # Apply LoRA now, via Unsloth's own path, rather than handing DPOTrainer
        # a bare LoraConfig to wrap -- mirrors the SFT script's --unsloth branch.
        # peft_config is left None and must not be passed to DPOTrainer below,
        # or the already-wrapped model would be peft-wrapped a second time.
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=train_cfg.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=train_cfg.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    grad_accum = max(1, 4 // args.batch_size)

    import inspect
    dpo_config_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    
    kwargs_config = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "beta": args.beta,
        "bf16": use_bf16,
        "fp16": use_cuda and not use_bf16,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "remove_unused_columns": False,
        "report_to": "none",
        "optim": "paged_adamw_8bit" if use_cuda else "adamw_torch",
    }

    if "max_prompt_length" in dpo_config_params:
        kwargs_config["max_prompt_length"] = max_seq // 2
    if "max_length" in dpo_config_params:
        kwargs_config["max_length"] = max_seq
    if "max_completion_length" in dpo_config_params:
        kwargs_config["max_completion_length"] = max_seq // 2

    dpo_args = DPOConfig(**kwargs_config)

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_ds,
    }
    if peft_config is not None:
        # None on the Unsloth path: that model is already PEFT-wrapped by
        # FastLanguageModel.get_peft_model above, so DPOTrainer must not wrap
        # it again.
        trainer_kwargs["peft_config"] = peft_config

    dpo_trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters.keys())
    if "beta" in dpo_trainer_params:
        trainer_kwargs["beta"] = args.beta
    if "processing_class" in dpo_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "tokenizer" in dpo_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_prompt_length" in dpo_trainer_params:
        trainer_kwargs["max_prompt_length"] = max_seq // 2
    if "max_length" in dpo_trainer_params:
        trainer_kwargs["max_length"] = max_seq

    trainer = DPOTrainer(**trainer_kwargs)

    # Resume from the newest per-epoch checkpoint when asked. save_strategy is
    # "epoch", so the checkpoints exist; without this a crash restarts from zero.
    resume_ckpt = None
    if args.resume:
        try:
            from transformers.trainer_utils import get_last_checkpoint
            if os.path.isdir(args.output_dir):
                resume_ckpt = get_last_checkpoint(args.output_dir)
        except Exception as e:
            print(f"[DPO] Could not look for a checkpoint to resume ({e}).")
        print(
            f"[DPO] Resuming from checkpoint: {resume_ckpt}" if resume_ckpt
            else "[DPO] No existing checkpoint found; starting fresh."
        )

    print(f"[DPO] Starting training for {args.epochs} epochs...")
    trainer.train(resume_from_checkpoint=resume_ckpt)


    adapter_out = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"[DPO] Adapter saved to {adapter_out}")

if __name__ == "__main__":
    main()
