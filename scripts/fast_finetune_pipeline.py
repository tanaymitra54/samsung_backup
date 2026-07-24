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
    parser.add_argument("--epochs",     type=int,   default=2,    help="SFT epochs (default: 2)")
    parser.add_argument("--lr",         type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int,   default=4,    help="Per-device batch size (default: 4)")
    parser.add_argument("--lora-rank",  type=int,   default=16,   help="LoRA rank (default: 16)")
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

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

model_name  = cfg["model"]["name"]
cache_dir   = cfg["model"].get("cache_dir")
train_cfg   = cfg.get("training", {{}})
max_seq     = train_cfg.get("max_seq_length", 1024)
lora_alpha  = train_cfg.get("lora_alpha", 32)
lora_drop   = train_cfg.get("lora_dropout", 0.05)
target_mods = train_cfg.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

# ── Load & format data ────────────────────────────────────────────────────────
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

def to_text(rec):
    msgs = rec.get("messages", [])
    parts = []
    for m in msgs:
        role, content = m.get("role",""), m.get("content","")
        if role == "system":
            parts.append(f"<<SYS>>\\n{{content}}\\n<</SYS>>")
        elif role == "user":
            parts.append(f"[INST] {{content}} [/INST]")
        elif role == "assistant":
            parts.append(content)
    return " ".join(parts).strip()

print(f"[SFT] Loading {{TRAIN_FILE}} ...")
train_raw = load_jsonl(TRAIN_FILE)
val_raw   = load_jsonl(VAL_FILE)
print(f"[SFT] Raw   Train={{len(train_raw)}}  Val={{len(val_raw)}}")

train_texts = [t for t in (to_text(r) for r in train_raw) if len(t) > 20]
val_texts   = [t for t in (to_text(r) for r in val_raw)   if len(t) > 20]
print(f"[SFT] Final Train={{len(train_texts)}}  Val={{len(val_texts)}}")

if not train_texts:
    print("[ERROR] No training examples after formatting. Check data/finetune_train.jsonl.")
    sys.exit(1)

train_ds = Dataset.from_dict({{"text": train_texts}})
val_ds   = Dataset.from_dict({{"text": val_texts}}) if val_texts else None

# ── Model & LoRA ──────────────────────────────────────────────────────────────
use_cuda = torch.cuda.is_available()
use_4bit = use_cuda

bnb_config = None
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

print(f"[SFT] Loading {{model_name}} (4-bit={{use_4bit}}) ...")
mkw = {{
    "cache_dir":   cache_dir,
    "device_map":  "auto" if use_cuda else None,
    "torch_dtype": torch.float16 if use_cuda else torch.float32,
}}
if bnb_config:
    mkw["quantization_config"] = bnb_config

model     = AutoModelForCausalLM.from_pretrained(model_name, **mkw)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if use_4bit:
    model = prepare_model_for_kbit_training(model)

peft_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=lora_alpha,
    target_modules=target_mods,
    lora_dropout=lora_drop,
    bias="none",
    task_type="CAUSAL_LM",
)

os.makedirs(ADAPTER_OUT, exist_ok=True)
run_dir = str(Path(ADAPTER_OUT).parent)

grad_accum = max(1, 8 // BATCH_SIZE)

# Introspect _TrainCls at runtime so the script works regardless of which
# trl / transformers version is installed on this machine.
_sig = inspect.signature(_TrainCls.__init__).parameters

# eval_strategy (transformers >= 4.46) vs evaluation_strategy (older)
_eval_kw = "eval_strategy" if "eval_strategy" in _sig else "evaluation_strategy"

# max_seq_length (trl < 0.16) vs max_length (trl >= 0.16) vs absent
_maxlen_kw = next((k for k in ("max_seq_length", "max_length") if k in _sig), None)

# dataset_text_field may live in SFTConfig or still in SFTTrainer
_txtfld_in_cfg = "dataset_text_field" in _sig

_cfg = {{
    "output_dir": run_dir,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": grad_accum,
    "learning_rate": LR,
    "warmup_ratio": 0.05,
    "num_train_epochs": EPOCHS,
    "fp16": use_cuda,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 1,
    _eval_kw: "epoch" if val_ds else "no",
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

# If text_field / max_seq_length / tokenizer name vary by trl version,
# inspect SFTTrainer signature dynamically.
_sft_sig = inspect.signature(SFTTrainer.__init__).parameters
_extra = {{}}
if not _txtfld_in_cfg and "dataset_text_field" in _sft_sig:
    _extra["dataset_text_field"] = "text"
if not _maxlen_kw and "max_seq_length" in _sft_sig:
    _extra["max_seq_length"] = max_seq

# trl >= 0.12 renamed 'tokenizer' to 'processing_class'
if "processing_class" in _sft_sig:
    _extra["processing_class"] = tokenizer
elif "tokenizer" in _sft_sig:
    _extra["tokenizer"] = tokenizer

if "peft_config" in _sft_sig:
    _extra["peft_config"] = peft_cfg

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    **_extra,
)

print(f"[SFT] Training {{len(train_texts)}} examples for {{EPOCHS}} epoch(s) ...")
trainer.train()
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
    out_dir = Path(args.eval_output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    bmark_script = str(_REPO_ROOT / "scripts" / "run_all_benchmarks.py")

    for label in ["base", "finetuned"]:
        if label == "finetuned" and not adapter_path:
            print("[Eval] No adapter -- skipping fine-tuned eval.")
            continue

        run_out = str(out_dir / label)
        cmd = [
            sys.executable, bmark_script,
            "--output-dir",  run_out,
            "--subset-size", str(args.eval_subset),
            "--benchmarks",  *args.benchmarks,
            "--seed",        "42",
        ]
        if args.device:
            cmd += ["--device", args.device]
        if label == "finetuned" and adapter_path:
            os.environ["QUBO_ADAPTER_PATH"] = adapter_path
        elif "QUBO_ADAPTER_PATH" in os.environ:
            del os.environ["QUBO_ADAPTER_PATH"]

        _run(cmd, f"Stage 3 -- Benchmark eval ({label})")

    if "QUBO_ADAPTER_PATH" in os.environ:
        del os.environ["QUBO_ADAPTER_PATH"]

    _print_comparison(out_dir)


def _print_comparison(out_dir: Path):
    """Load result JSONs and print a side-by-side accuracy table."""
    def _load(p: Path):
        try:
            files = sorted(p.rglob("*.json"))
            if not files:
                return {}
            with open(files[-1]) as f:
                return json.load(f)
        except Exception:
            return {}

    base_data = _load(out_dir / "base")
    ft_data   = _load(out_dir / "finetuned")

    print("\n" + "=" * 65)
    print(f"  {'Benchmark':<18} | {'Base':>8} | {'Fine-tuned':>10} | {'Delta':>8}")
    print("-" * 65)

    for bm in sorted(set(base_data) | set(ft_data)):
        if bm.startswith("_"):
            continue
        b_acc = (base_data.get(bm) or {}).get("qubo_accuracy") or (base_data.get(bm) or {}).get("accuracy")
        f_acc = (ft_data.get(bm)   or {}).get("qubo_accuracy") or (ft_data.get(bm)   or {}).get("accuracy")
        b_s   = f"{b_acc:.1%}" if b_acc is not None else "N/A"
        f_s   = f"{f_acc:.1%}" if f_acc is not None else "N/A"
        d_s   = f"{f_acc-b_acc:+.1%}" if (b_acc is not None and f_acc is not None) else "N/A"
        print(f"  {bm:<18} | {b_s:>8} | {f_s:>10} | {d_s:>8}")

    print("=" * 65)
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
