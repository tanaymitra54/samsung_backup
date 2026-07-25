"""
scripts/fast_finetune_pipeline.py
==================================
All-in-one orchestration script: QUBO data generation -> LoRA SFT -> benchmark eval.
Designed to complete within 24 hours on a consumer CUDA GPU.

Usage (recommended - runs everything with reduced counts):
    python scripts/fast_finetune_pipeline.py --quick

Custom counts:
    python scripts/fast_finetune_pipeline.py ^
        --n-gsm8k 100 --n-arc 60 --n-logiqa 40 ^
        --n-strategyqa 40 --n-openorca 200

Skip data generation (if data/finetune_train.jsonl already exists):
    python scripts/fast_finetune_pipeline.py --quick --skip-datagen

Skip fine-tuning (only run eval on an existing adapter):
    python scripts/fast_finetune_pipeline.py --skip-finetune ^
        --adapter-path checkpoints/qubo-sft-fast/final_adapter

Stages
------
1. generate_training_data.py  -- QUBO-curated full reasoning chains
2. Inline LoRA trainer        -- QLoRA fine-tune on Llama 3.2 3B
3. run_all_benchmarks.py      -- compare base vs fine-tuned on configured benchmarks
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _run(cmd: list, stage: str, cwd: str = None):
    """Run a subprocess and stream its output. Raises SystemExit on failure."""
    cwd = cwd or str(_REPO_ROOT)
    print(f"\n{'='*65}")
    print(f"[Stage] {stage}")
    print(f"[Cmd]   {' '.join(str(c) for c in cmd)}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{stage}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[Done]  {stage}  ({_fmt_duration(elapsed)})")
    return elapsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="QUBO fast fine-tune pipeline: data gen -> SFT -> eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Dataset size shortcuts ────────────────────────────────────────────────
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced counts: gsm8k=100, arc=60, logiqa=40, strategyqa=40, openorca=200"
    )
    parser.add_argument("--n-gsm8k",      type=int, default=None, help="# GSM8K questions")
    parser.add_argument("--n-arc",        type=int, default=None, help="# ARC-Challenge questions")
    parser.add_argument("--n-logiqa",     type=int, default=None, help="# LogiQA questions")
    parser.add_argument("--n-strategyqa", type=int, default=None, help="# StrategyQA questions")
    parser.add_argument("--n-openorca",   type=int, default=None, help="# OpenOrca examples")

    # ── Stage control ─────────────────────────────────────────────────────────
    parser.add_argument("--skip-datagen",  action="store_true",
                        help="Skip data generation (use existing data/finetune_train.jsonl)")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip fine-tuning (go straight to eval)")
    parser.add_argument("--skip-eval",     action="store_true",
                        help="Skip benchmark evaluation after fine-tuning")

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument("--data-dir",    default="data",                    help="Data output directory")
    parser.add_argument("--config",      default="config/config.yaml",      help="Path to config.yaml")
    parser.add_argument("--qubo-params", default="config/best_qubo_params.yaml")
    parser.add_argument("--device",      default=None,                      help="cuda:0 / cpu")
    parser.add_argument("--adapter-path", default=None,
                        help="Existing LoRA adapter path (used when --skip-finetune)")
    parser.add_argument("--resume-datagen", action="store_true",
                        help="Pass --resume to generate_training_data.py (continue interrupted run)")

    # ── Training hyper-params ─────────────────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=5,    help="SFT epochs (default: 5)")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int,   default=4,    help="Per-device batch size (default: 4)")
    parser.add_argument("--lora-rank",  type=int,   default=32,   help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int,   default=64,   help="LoRA alpha (default: 64, = 2x rank)")
    parser.add_argument("--run-name",   default=None,
                        help="Name for this run (default: qubo-sft-fast-<timestamp>)")

    # ── Eval options ──────────────────────────────────────────────────────────
    parser.add_argument("--benchmarks",  nargs="*", default=["gsm8k", "mmlu", "bbh"],
                        help="Benchmarks to run post-finetune eval (default: gsm8k mmlu bbh)")
    parser.add_argument("--eval-subset", type=int, default=30,
                        help="Questions per benchmark for eval (default: 30)")
    parser.add_argument("--eval-output-dir", default="outputs/finetuned_eval",
                        help="Directory for evaluation results")

    return parser.parse_args()


# ── Stage 1: Data generation ─────────────────────────────────────────────────

def stage_datagen(args) -> Path:
    """Run generate_training_data.py with the appropriate size flags."""
    data_dir   = Path(args.data_dir)
    train_file = data_dir / "finetune_train.jsonl"

    if args.skip_datagen:
        if not train_file.exists():
            print(f"[ERROR] --skip-datagen requested but {train_file} does not exist.")
            sys.exit(1)
        n_lines = sum(1 for _ in open(train_file, encoding="utf-8"))
        print(f"[DataGen] Skipped. Using existing {train_file} ({n_lines} examples).")
        return data_dir

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "generate_training_data.py"),
        "--config",       str(_REPO_ROOT / args.config),
        "--qubo-params",  str(_REPO_ROOT / args.qubo_params),
        "--output-dir",   str(data_dir),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.resume_datagen:
        cmd += ["--resume"]

    # Size flags
    if args.quick:
        cmd += ["--quick"]
    size_map = {
        "n_gsm8k":      "--n-gsm8k",
        "n_mmlu":       "--n-mmlu",
        "n_arc":        "--n-arc",
        "n_logiqa":     "--n-logiqa",
        "n_strategyqa": "--n-strategyqa",
        "n_openorca":   "--n-openorca",
    }
    for attr, flag in size_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            cmd += [flag, str(val)]

    _run(cmd, "Stage 1 -- QUBO-curated data generation")
    return data_dir


# ── Stage 2: LoRA fine-tuning ────────────────────────────────────────────────

def _write_sft_script(path: Path, args, train_file: Path, val_file: Path,
                      run_name: str, adapter_out: str):
    """Write a self-contained fine-tuning script (run as subprocess to keep VRAM clean)."""
    repo = str(_REPO_ROOT)
    cfg  = str(_REPO_ROOT / args.config)
    lora_alpha = args.lora_alpha
    code = f'''# Auto-generated by fast_finetune_pipeline.py
import os, sys, json, torch
from pathlib import Path
sys.path.insert(0, {repr(repo)})

import yaml
import inspect
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
try:
    from trl import SFTConfig as _TrainCls
except ImportError:
    _TrainCls = TrainingArguments

TRAIN_FILE  = {repr(str(train_file))}
VAL_FILE    = {repr(str(val_file))}
ADAPTER_OUT = {repr(adapter_out)}
RUN_NAME    = {repr(run_name)}
CONFIG_PATH = {repr(cfg)}
EPOCHS      = {args.epochs}
LR          = {args.lr}
BATCH_SIZE  = {args.batch_size}
LORA_RANK   = {args.lora_rank}
LORA_ALPHA  = {lora_alpha}
QUALITY_THRESHOLD = 0.70   # Only train on QUBO examples with high correctness score

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

model_name  = cfg["model"]["name"]
cache_dir   = cfg["model"].get("cache_dir")
train_cfg   = cfg.get("training", {{}})
max_seq     = train_cfg.get("max_seq_length", 2048)
lora_drop   = train_cfg.get("lora_dropout", 0.05)
target_mods = train_cfg.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

# ── Step 1: Load tokenizer first (needed for correct chat template formatting) ──
print(f"[SFT] Loading tokenizer for {{model_name}} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Step 2: Load & quality-filter training data ──────────────────────────
def load_jsonl(fpath):
    rows = []
    p = Path(fpath)
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def is_quality(rec):
    """Keep only high-quality QUBO-selected reasoning examples."""
    meta = rec.get("metadata", {{}})
    if not meta.get("qubo_selected", False):
        return False   # drops OpenOrca and any non-QUBO examples
    score = meta.get("correctness_score", 0.0) or 0.0
    return score >= QUALITY_THRESHOLD

def to_text(rec):
    """Convert message list to text using the model\'s own chat template (Llama-3.2)."""
    msgs = rec.get("messages", [])
    if not msgs:
        return ""
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback: plain concatenation
        parts = []
        for m in msgs:
            role, content = m.get("role", ""), m.get("content", "")
            parts.append(f"{{role}}: {{content}}")
        return "\\n".join(parts)

print(f"[SFT] Loading {{TRAIN_FILE}} ...")
train_raw = load_jsonl(TRAIN_FILE)
val_raw   = load_jsonl(VAL_FILE)
print(f"[SFT] Raw   Train={{len(train_raw)}}  Val={{len(val_raw)}}")

# Apply quality filter
train_raw = [r for r in train_raw if is_quality(r)]
val_raw   = [r for r in val_raw   if is_quality(r)]
print(f"[SFT] After quality filter (score>={{QUALITY_THRESHOLD}}, qubo_selected=True): Train={{len(train_raw)}}  Val={{len(val_raw)}}")

train_texts = [t for t in (to_text(r) for r in train_raw) if len(t) > 20]
val_texts   = [t for t in (to_text(r) for r in val_raw)   if len(t) > 20]
print(f"[SFT] Final Train={{len(train_texts)}}  Val={{len(val_texts)}}")

if not train_texts:
    print("[ERROR] No training examples after filtering. Check data/finetune_train.jsonl.")
    sys.exit(1)

train_ds = Dataset.from_dict({{"text": train_texts}})
val_ds   = Dataset.from_dict({{"text": val_texts}}) if val_texts else None

# Keep original raw validation data for accuracy evaluation
val_raw_gsm8k = [r for r in val_raw if r.get("metadata", {}).get("source") == "gsm8k"]
if not val_raw_gsm8k:
    val_raw_gsm8k = val_raw # Fallback if no gsm8k specifically
val_raw_eval = val_raw_gsm8k[:40] # max 40 for speed

# ── Step 3: Load model ────────────────────────────────────────────────
use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_4bit = use_cuda

bnb_config = None
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

print(f"[SFT] Loading {{model_name}} (4-bit={{use_4bit}}, bf16={{use_bf16}}) ...")
mkw = {{
    "cache_dir":   cache_dir,
    "device_map":  "auto" if use_cuda else None,
    "torch_dtype": torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32),
}}
if bnb_config:
    mkw["quantization_config"] = bnb_config

model = AutoModelForCausalLM.from_pretrained(model_name, **mkw)

if use_4bit:
    model = prepare_model_for_kbit_training(model)

# ── Step 4: LoRA config ─────────────────────────────────────────────────
peft_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=target_mods,
    lora_dropout=lora_drop,
    bias="none",
    task_type="CAUSAL_LM",
)

os.makedirs(ADAPTER_OUT, exist_ok=True)
run_dir = str(Path(ADAPTER_OUT).parent)

grad_accum = max(1, 8 // BATCH_SIZE)

# Introspect _TrainCls at runtime so the script works regardless of trl version.
_sig = inspect.signature(_TrainCls.__init__).parameters

_eval_kw   = "eval_strategy" if "eval_strategy" in _sig else "evaluation_strategy"
_maxlen_kw = next((k for k in ("max_seq_length", "max_length") if k in _sig), None)
_txtfld_in_cfg = "dataset_text_field" in _sig

_cfg = {{
    "output_dir": run_dir,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": grad_accum,
    "learning_rate": LR,
    "warmup_ratio": 0.05,
    "num_train_epochs": EPOCHS,
    "bf16": use_bf16,
    "fp16": use_cuda and not use_bf16,
    "logging_steps": 10,
    "save_strategy": "epoch",
    _eval_kw: "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_gsm8k_accuracy",
    "greater_is_better": True,
    "remove_unused_columns": False,
    "report_to": "none",
    "run_name": RUN_NAME,
    "dataloader_num_workers": 0,
    "optim": "paged_adamw_8bit" if use_cuda else "adamw_torch",
}}
if _txtfld_in_cfg:
    _cfg["dataset_text_field"] = "text"
if _maxlen_kw:
    _cfg[_maxlen_kw] = max_seq

training_args = _TrainCls(**_cfg)

_sft_sig = inspect.signature(SFTTrainer.__init__).parameters
_extra = {{}}
if not _txtfld_in_cfg and "dataset_text_field" in _sft_sig:
    _extra["dataset_text_field"] = "text"
if not _maxlen_kw and "max_seq_length" in _sft_sig:
    _extra["max_seq_length"] = max_seq

# trl >= 0.12 renamed \'tokenizer\' to \'processing_class\'
if "processing_class" in _sft_sig:
    _extra["processing_class"] = tokenizer
elif "tokenizer" in _sft_sig:
    _extra["tokenizer"] = tokenizer

if "peft_config" in _sft_sig:
    _extra["peft_config"] = peft_cfg

# Subclass SFTTrainer to compute custom accuracy metric
import re
class CustomAccTrainer(SFTTrainer):
    def __init__(self, *args, val_raw_data=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_raw_data = val_raw_data
        self.gen_tokenizer = tokenizer
        self.epoch_accuracies = []

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if not self.val_raw_data:
            return metrics
            
        print("\\n[Eval] Running generation on GSM8K val subset for accuracy...")
        self.model.eval()
        correct = 0
        total = len(self.val_raw_data)
        
        for item in self.val_raw_data:
            # Reconstruct prompt (all but last assistant message)
            msgs = item.get("messages", [])
            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            prompt_str = self.gen_tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            
            inputs = self.gen_tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.gen_tokenizer.eos_token_id
                )
            gen_text = self.gen_tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            gold = str(item.get("metadata", {}).get("gold", "")).strip()
            # Extract number from gen_text
            gen_text = gen_text.replace(",", "")
            nums = re.findall(r"-?\d+(?:\.\d+)?", gen_text)
            pred_num = float(nums[-1]) if nums else None
            
            try:
                gold_num = float(gold)
            except ValueError:
                gold_num = None
                
            if pred_num is not None and gold_num is not None and abs(pred_num - gold_num) <= 1e-4:
                correct += 1
                
        acc = correct / total if total > 0 else 0.0
        metrics[f"{metric_key_prefix}_gsm8k_accuracy"] = acc
        self.epoch_accuracies.append(acc)
        print(f"[Eval] Epoch GSM8K Accuracy: {acc:.2%} ({correct}/{total})\\n")
        return metrics

print(f"[SFT] Training {len(train_texts)} examples for {EPOCHS} epoch(s) | rank={LORA_RANK} alpha={LORA_ALPHA} lr={LR} ...")
trainer = CustomAccTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    val_raw_data=val_raw_eval,
    tokenizer=tokenizer,
    **_extra,
)
trainer.train()

# Check trending upward
accs = trainer.epoch_accuracies
if len(accs) >= 2 and accs[-1] > accs[-2]:
    print("\\n" + "!" * 60)
    print("⚠️  TRENDING UPWARD: Accuracy improved on the last epoch.")
    print("⚠️  You might want to train for more epochs.")
    print("!" * 60 + "\\n")

trainer.save_model(ADAPTER_OUT)
tokenizer.save_pretrained(ADAPTER_OUT)
print(f"[SFT] Adapter saved to {{ADAPTER_OUT}}")
'''
    path.write_text(code, encoding="utf-8")


def stage_finetune(args, data_dir: Path) -> str | None:
    """Fine-tune with LoRA; returns path to saved adapter (or None)."""
    if args.skip_finetune:
        if args.adapter_path:
            print(f"[SFT] Skipped. Using adapter: {args.adapter_path}")
            return args.adapter_path
        print("[WARN] --skip-finetune set but no --adapter-path given. Eval will use base model.")
        return None

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"qubo-sft-fast-{ts}"
    adapter_out = str(_REPO_ROOT / "checkpoints" / run_name / "final_adapter")

    train_file = data_dir / "finetune_train.jsonl"
    val_file   = data_dir / "finetune_val.jsonl"
    if not train_file.exists():
        print(f"[ERROR] Training file not found: {train_file}")
        sys.exit(1)

    sft_script = _REPO_ROOT / "scripts" / "_run_sft_tmp.py"
    _write_sft_script(sft_script, args, train_file, val_file, run_name, adapter_out)

    try:
        _run([sys.executable, str(sft_script)], "Stage 2 -- LoRA SFT fine-tuning")
    finally:
        sft_script.unlink(missing_ok=True)

    print(f"[SFT] Adapter at: {adapter_out}")
    return adapter_out


# ── Stage 3: Benchmark evaluation ────────────────────────────────────────────

def stage_eval(args, adapter_path: str | None):
    """Run benchmarks on base model then fine-tuned model and print comparison."""
    if args.skip_eval:
        print("[Eval] Skipped.")
        return

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    bmark_script = str(_REPO_ROOT / "scripts" / "run_all_benchmarks.py")

    for label in ["base", "sft"]:
        if label == "sft" and not adapter_path:
            print("[Eval] No adapter -- skipping fine-tuned eval.")
            continue

        cmd = [
            sys.executable, bmark_script,
            "--output-dir",  str(out_dir),
            "--condition-label", label,
            "--subset-size", str(args.eval_subset),
            "--benchmarks",  *args.benchmarks,
            "--seed",        "42",
        ]
        if args.device:
            cmd += ["--device", args.device]
        if label == "sft" and adapter_path:
            os.environ["QUBO_ADAPTER_PATH"] = adapter_path
        elif "QUBO_ADAPTER_PATH" in os.environ:
            del os.environ["QUBO_ADAPTER_PATH"]

        _run(cmd, f"Stage 3 -- Benchmark eval ({label})")

    if "QUBO_ADAPTER_PATH" in os.environ:
        del os.environ["QUBO_ADAPTER_PATH"]

    _print_comparison(out_dir)


    # Note: _print_comparison logic is currently tailored to the old dir structure
    # and might need to be rewritten later, but the diagnostic script will
    # handle the final reporting table for this new plan.
    print("\n[Eval] Check results/eval/ for the JSONL output files.")

    print("\n" + "=" * 75)
    print(f"  {'Benchmark':<12} | {'Mode':<8} | {'Base':>8} | {'Fine-tuned':>10} | {'Delta':>8}")
    print("-" * 75)

    all_bms = sorted(set(k for k in base_data if isinstance(base_data[k], dict)) | 
                     set(k for k in ft_data if isinstance(ft_data[k], dict)))

    for bm in all_bms:
        b_item = base_data.get(bm, {})
        f_item = ft_data.get(bm, {})
        
        b_accs = b_item.get("accuracy", {}) if isinstance(b_item, dict) else {}
        f_accs = f_item.get("accuracy", {}) if isinstance(f_item, dict) else {}

        for mode in ["greedy", "cot", "qubo"]:
            b_val = b_accs.get(mode)
            f_val = f_accs.get(mode)
            b_s = f"{b_val:.2%}" if b_val is not None else "N/A"
            f_s = f"{f_val:.2%}" if f_val is not None else "N/A"
            d_s = f"{f_val - b_val:+.2%}" if (b_val is not None and f_val is not None) else "N/A"
            print(f"  {bm:<12} | {mode:<8} | {b_s:>8} | {f_s:>10} | {d_s:>8}")

    print("=" * 75)
    print(f"\nFull results saved to: {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    wall_start = time.time()
    print("\n" + "=" * 65)
    print("  QUBO Fast Fine-tune Pipeline")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stages  : datagen={not args.skip_datagen}  finetune={not args.skip_finetune}  eval={not args.skip_eval}")
    print("=" * 65)

    data_dir     = stage_datagen(args)
    adapter_path = stage_finetune(args, data_dir)
    stage_eval(args, adapter_path)

    total = time.time() - wall_start
    print("\n" + "=" * 65)
    print(f"  Pipeline complete!  Total wall time: {_fmt_duration(total)}")
    if adapter_path:
        print(f"  Adapter : {adapter_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
