"""DPO on answer-verified fit pairs; optionally continue an SFT adapter."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.dpo_data import validate_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    parser.add_argument("--pairs", default="data/diagnostic_preference_pairs.jsonl")
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--output-dir", default="checkpoints/prism-dpo")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use 1 for a real GPU smoke test")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if any(v <= 0 or not math.isfinite(v) for v in (args.epochs, args.lr, args.beta, args.batch_size, args.lora_rank, args.max_seq_length)) or args.max_steps == 0 or args.max_steps < -1:
        parser.error("Training sizes/rates/beta must be positive; max-steps must be -1 or positive")
    pairs = validate_pairs(args.pairs)
    print(f"Verified DPO fit pairs: {len(pairs)}")
    if args.validate_only:
        return
    import torch
    import yaml
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer
    from pipeline.device_utils import hf_device_map_value, resolve_device, resolve_torch_dtype
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA DPO trainer")
    required = {"beta", "max_length", "model_adapter_name", "ref_adapter_name"}
    if not required <= set(inspect.signature(DPOConfig).parameters) or "processing_class" not in inspect.signature(DPOTrainer).parameters:
        raise RuntimeError("Install a modern TRL release with DPOConfig and processing_class; legacy TRL is unsupported")
    cfg = config["model"]
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = resolve_device(f"cuda:{local_rank}" if world_size > 1 else config.get("evaluation", {}).get("device"))
    torch.cuda.set_device(device)
    dtype = resolve_torch_dtype(config)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(cfg["name"], cache_dir=cfg.get("cache_dir"),
                                               quantization_config=bnb, device_map=hf_device_map_value(device),
                                               torch_dtype=dtype, attn_implementation=cfg.get("attn_implementation", "sdpa"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], cache_dir=cfg.get("cache_dir"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("DPO requires the model's chat template; hard-coded Qwen tags are not used")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    peft = LoraConfig(r=args.lora_rank, lora_alpha=config["training"].get("lora_alpha", 32),
                      target_modules=config["training"].get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    policy_name, reference_name = None, None
    if args.sft_adapter:
        policy_name, reference_name = "policy", "reference"
        model = PeftModel.from_pretrained(model, args.sft_adapter, adapter_name=policy_name, is_trainable=True)
        model.load_adapter(args.sft_adapter, adapter_name=reference_name, is_trainable=False)
        model.set_adapter(policy_name)
        peft = None
    dataset = Dataset.from_list([dict(prompt=[{"role": "user", "content": p["prompt"]}],
                                     chosen=[{"role": "assistant", "content": p["chosen"]}],
                                     rejected=[{"role": "assistant", "content": p["rejected"]}]) for p in pairs])
    bf16 = torch.cuda.is_bf16_supported()
    train_config = DPOConfig(output_dir=args.output_dir, beta=args.beta, max_length=args.max_seq_length,
                             model_adapter_name=policy_name, ref_adapter_name=reference_name,
                             per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=2,
                             learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
                             bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
                             gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
                             ddp_find_unused_parameters=False,
                             logging_steps=1, logging_nan_inf_filter=False, save_strategy="epoch", report_to="none", seed=args.seed,
                             optim="paged_adamw_8bit", warmup_ratio=0.05)
    trainer = DPOTrainer(model=model, ref_model=None, args=train_config, train_dataset=dataset,
                         peft_config=peft, processing_class=tokenizer)
    trainer.train()
    losses = [float(log["loss"]) for log in trainer.state.log_history if "loss" in log]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("DPO did not produce finite logged losses; no final adapter was saved")
    if any(not torch.isfinite(param).all().item() for param in trainer.model.parameters() if param.requires_grad):
        raise RuntimeError("Non-finite trained parameters; no final adapter was saved")
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        return
    output = Path(args.output_dir) / "final_adapter"
    trainer.accelerator.unwrap_model(trainer.model).save_pretrained(output, selected_adapters=[policy_name or "default"])
    adapter_path = output / policy_name if policy_name else output
    tokenizer.save_pretrained(adapter_path)
    (output / "training_provenance.json").write_text(json.dumps(dict(model=cfg["name"], sft_adapter=args.sft_adapter,
        pairs=str(Path(args.pairs).resolve()), question_hashes=[p["question_hash"] for p in pairs],
        beta=args.beta, seed=args.seed, max_steps=args.max_steps, losses=losses,
        world_size=world_size, per_device_batch_size=args.batch_size, gradient_accumulation_steps=2,
        label_scope="answer_correctness_only"), indent=2), encoding="utf-8")
    print(f"DPO adapter: {adapter_path}")


if __name__ == "__main__":
    main()
