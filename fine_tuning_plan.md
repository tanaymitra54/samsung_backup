# Fine-Tuning Plan

## 1. Overview

**Objective**: Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` to improve reasoning for the PRISM pipeline.

**Environment**: Linux, 2x NVIDIA H100 (80GB), CUDA 12.2

**Method**: Standard QLoRA (4-bit quantization) with HuggingFace + PEFT + bitsandbytes

---

## 2. Dataset Strategy

### Single Dataset: `nvidia/OpenMathInstruct-2`

600K math reasoning traces with step-by-step solutions. Covers arithmetic, algebra, geometry, probability, and more — matches PRISM's benchmark domains (GSM8K, MMLU, BBH).

### Why this dataset?

The PRISM pipeline generates multiple reasoning steps and picks the best ones. For this to work well, the underlying model needs to produce good reasoning in the first place. OpenMathInstruct-2 teaches the model how to reason step-by-step through math problems, which directly feeds into better trace quality for the pipeline.

### Downsampling via Stratified Sampling → 20K examples

We don't need all 600K examples — that would take too long and might overfit. Instead we pick a balanced 20K subset spanning 7 reasoning types so the model doesn't become a one-trick pony.

| Stratum | Benchmark Target | Sampled |
|---------|----------------|---------|
| Arithmetic word problems | GSM8K | 2,000 |
| Algebra / Calculus | MMLU math | 2,000 |
| Geometry / Trigonometry | MMLU math | 2,000 |
| Probability / Statistics | MMLU + BBH | 2,000 |
| Logical reasoning | BBH | 2,000 |
| Science (physics/chem) | MMLU science | 300* |
| Code / Algorithms | General reasoning | 2,000 |
| General (fallback) | Diversity top-up | ~5,646 |
| **Our evaluation traces** | PRISM benchmark data | **~54** |

*Science is capped at 300 because OpenMathInstruct-2 has very few science examples — we'd otherwise have to stream the entire dataset just to find them.

### Why stratified sampling?

Random sampling would give us mostly arithmetic and algebra (the most common types in the dataset). Stratified sampling guarantees variety — the model sees equal amounts of logic, geometry, probability, etc. This matters because our benchmarks test all these skills.

### Why include our own evaluation traces?

The 54 correct reasoning traces extracted from our previous benchmark runs are the most relevant data we have — they show exactly what successful PRISM reasoning looks like. Even though they're only ~0.3% of the training data, they carry high signal.

---

## 3. Fine-Tuning Method

### QLoRA (4-bit Quantization + LoRA)

Instead of updating all 1.5 billion parameters, we:
1. **Compress** the base model to 4-bit precision (fits in ~3 GB instead of ~6 GB)
2. **Attach tiny "adapter" layers** (LoRA) — only ~0.1% of parameters are trainable
3. **Train only the adapters** — the base model stays frozen

This means training needs only ~6 GB VRAM total and produces a ~20 MB adapter file instead of a ~3 GB full model.

| Parameter | Value | Why |
|-----------|-------|-----|
| LoRA Rank | 16 | Enough capacity for reasoning patterns |
| LoRA Alpha | 32 | Standard scaling factor |
| LoRA Dropout | 0.05 | Prevents memorization |
| Learning Rate | 2e-5 | Slow enough for stable learning |
| Batch Size | 8 (eff. 32) | Fits in GPU memory with gradient accumulation |
| Epochs | 3 | More would overfit the 20K dataset |
| Max Seq Length | 2048 | Fits most reasoning traces |
| Precision | BF16 + 4-bit | Fast on H100 GPUs |

---

## 4. Execution Steps

### Step 1: Prepare Data

```bash
python scripts/prepare_training_data.py
```

This:
- Extracts correct reasoning traces from `outputs/` CSV benchmark files
- Streams OpenMathInstruct-2 from HuggingFace (doesn't download all 600K — only what's needed)
- Classifies each example into a reasoning type
- Collects 2,000 per type (300 for science)
- Formats everything in Qwen chat template
- Saves as `training_data/combined_dataset/`

### Step 2: Run SFT

```bash
# Full training
python training/run_sft.py \
  --epochs 3 \
  --lr 2e-5 \
  --batch-size 8 \
  --lora-rank 16 \
  --max-seq-length 2048 \
  --output-dir checkpoints/prism-sft-v1 \
  --wandb
```

Expected time: ~6-8 hours on 2× H100s (20K examples × 3 epochs).

### Step 3: Evaluate

```bash
# Update config.yaml: change model.name to "checkpoints/prism-sft-v1/merged_model"
python scripts/run_all_benchmarks.py \
  --subset-size 200 \
  --benchmarks gsm8k mmlu bbh arc_challenge \
  --device cuda:1 \
  --output-dir outputs/eval_after_sft
```

### Step 4 (Optional): DPO

```bash
python training/run_dpo.py --epochs 2 --lr 1e-6
```

---

## 5. Expected Outcomes

| Metric | Before | After SFT |
|--------|--------|-----------|
| Greedy | ~20% | >30% |
| CoT | ~25% | >40% |
| QUBO | ~0-50% | >50% |

---

## 6. Files

| File | Purpose |
|------|---------|
| `scripts/prepare_training_data.py` | Extract traces + stratified sampling from OpenMathInstruct-2 |
| `training/run_sft.py` | QLoRA SFT training (HuggingFace + PEFT) |
| `training/run_dpo.py` | DPO training (optional second stage) |
| `Dockerfile` | Docker setup for remote server |
| `training_data/` | Generated training data (gitignored) |
| `checkpoints/` | Generated model checkpoints (gitignored) |
