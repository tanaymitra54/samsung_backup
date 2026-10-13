# Team Runbook — PRISM / QUBO Pipeline (A6000 80GB + Multi-GPU)

How to set up, protect, and run the full stack on **NVIDIA RTX A6000 (80 GB)** with optional multi-GPU job sharding.

| Item | Value |
|------|--------|
| Project root | `samsung_backup/` (run all commands from here) |
| Default model | `Qwen/Qwen2.5-3B-Instruct` |
| Config | `config/config.yaml` |
| Fast / smaller eval | `config/config_fast.yaml` |
| Default question budget | **~200 per benchmark** (~1000 total); see **§8** |
| Architecture panels | **40 eval + 40 train** (`data/budget40/`) |
| Stratified sampling? | **No** on main loader (prefix subsets); budget-40 is balanced panels only — **§8.3** |
| Precision | `bfloat16` (see `hardware.dtype`) |
| Parallelism | One full model replica **per GPU** (`CUDA_VISIBLE_DEVICES` job sharding) |
| QUBO solver | **Simulated annealing** only (CPU/GPU SA). OpenJij removed. |

---

## 1) What you are running

```
Query → SLM samples 16 traces → majority answer + QUBO diversity → final prompt
      → (optional) QLoRA SFT → re-eval → outputs/
```

| Stage | Entry point | Purpose |
|-------|-------------|---------|
| Smoke / benchmarks | `scripts/run_all_benchmarks.py` | Greedy / CoT / SC-16 / Best-16 / QUBO |
| Architecture loop | `scripts/run_architecture_pipeline.py` | Matched budget → QLoRA → re-eval |
| Budget panel data | `scripts/curate_budget40.py` | Builds `data/budget40/*.jsonl` |
| Model download | `scripts/download_models.py` | Prefetch HF weights into `cache/models` |

Helper wrappers (recommended):

| Script | Role |
|--------|------|
| [`scripts/setup_venv.sh`](scripts/setup_venv.sh) | Create venv + install deps |
| [`scripts/env_check.sh`](scripts/env_check.sh) | GPU / CUDA / disk sanity check |
| [`scripts/run_smoke.sh`](scripts/run_smoke.sh) | Tiny end-to-end smoke (few questions) |
| [`scripts/run_benchmarks.sh`](scripts/run_benchmarks.sh) | Full / subset benchmark run |
| [`scripts/run_architecture.sh`](scripts/run_architecture.sh) | Architecture + QLoRA loop |
| [`docker/Dockerfile`](docker/Dockerfile) + [`docker/docker-compose.yml`](docker/docker-compose.yml) | Isolated GPU container |

---

## 2) Hardware & shared-machine rules

**Target:** NVIDIA RTX A6000, 80 GB VRAM, CUDA 12.x driver (PyTorch cu121 wheels).

```bash
nvidia-smi
# Need several free GiB per worker. Qwen2.5-3B bf16 ≈ 6–8 GB weights + KV/batch.
```

**Do not:**

- Kill other users’ CUDA processes.
- Run with `device_map="auto"` spanning all GPUs (code pins one GPU per process).
- Start a second full replica on a GPU that is already nearly full.

**If VRAM is crowded:** pick a free card and pin it:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_all_benchmarks.py --device cuda:0 --no-multi-gpu --benchmarks gsm8k --subset-size 5
```

Or use `--device cuda:N` (single-GPU; disables auto multi-GPU).

---

## 3) Protected environment (recommended)

Use **one of**:

1. **Docker GPU container** (best isolation on a shared box) — §6  
2. **Python venv + reserved GPUs** (lighter) — §4  

Common protections:

| Control | How |
|---------|-----|
| Isolated Python | `.venv` or container image |
| GPU isolation | `CUDA_VISIBLE_DEVICES` / compose `device_ids` |
| Cache isolation | Bind `./cache` only inside project (or volume) |
| Secrets | `HF_TOKEN` / `WANDB_API_KEY` via env file — never commit |
| Non-root container | `user:` in compose (optional) |
| Read-only code | Mount source read-only; write only `outputs/`, `cache/`, `checkpoints/` |

Create a private env file (gitignored):

```bash
cp docker/env.example .env
# edit HF_TOKEN=...  WANDB_API_KEY=...  CUDA_VISIBLE_DEVICES=0,1
```

---

## 4) Host setup (venv)

```bash
cd samsung_backup

# 1) Environment
bash scripts/setup_venv.sh
source .venv/bin/activate

# 2) Sanity
bash scripts/env_check.sh

# 3) Hugging Face (gated models / higher rate limits)
export HF_HOME="$(pwd)/cache/huggingface"
export TRANSFORMERS_CACHE="$(pwd)/cache/models"
# export HF_TOKEN=hf_xxx   # if needed

# 4) Prefetch model (optional but recommended)
python3 scripts/download_models.py
```

`setup_venv.sh` installs [`requirements.txt`](requirements.txt) (PyTorch **cu121** via the extra index URL).

---

## 5) Day-to-day commands

All commands assume: `cd samsung_backup && source .venv/bin/activate`.

### 5.1 Smoke test (always first)

```bash
bash scripts/run_smoke.sh
# equivalent:
python3 scripts/run_all_benchmarks.py \
  --config config/config.yaml \
  --benchmarks gsm8k \
  --subset-size 3 \
  --output-dir outputs/smoke \
  --seed 42
```

Expect `outputs/smoke/all_benchmarks_*.{json,md}` and no CUDA OOM.

### 5.2 Benchmarks (auto multi-GPU if &gt;1 GPU visible)

```bash
# Default: uses hardware.multi_gpu + all visible GPUs
bash scripts/run_benchmarks.sh

# Or explicit:
python3 scripts/run_all_benchmarks.py \
  --config config/config.yaml \
  --subset-size 200 \
  --output-dir outputs/bench_a6000 \
  --seed 42

# One benchmark, questions sharded across GPUs
python3 scripts/run_all_benchmarks.py --benchmarks gsm8k --subset-size 200

# Pin one GPU
python3 scripts/run_all_benchmarks.py --device cuda:1 --no-multi-gpu --benchmarks gsm8k --subset-size 50

# Force multi-GPU
python3 scripts/run_all_benchmarks.py --multi-gpu --benchmarks gsm8k bbh --subset-size 100

# Full datasets
python3 scripts/run_all_benchmarks.py --full --output-dir outputs/full_eval
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--device cuda:N` | Pin one GPU; disables auto multi-GPU |
| `--multi-gpu` | Force job sharding across visible GPUs |
| `--no-multi-gpu` | Force single GPU |
| `--batch-size 16` | Override config (default 16 on A6000) |
| `--no-batch` | One question at a time |
| `--use-vllm` | Optional vLLM backend (`tensor_parallel_size: 1`) |
| `--config PATH` | Alternate YAML |
| `--seed N` | Reproducibility |

### 5.3 Architecture pipeline (matched budget → QLoRA → re-eval)

```bash
# Build fixed 40Q panels once
python3 scripts/curate_budget40.py

# Full loop (eval/traces multi-GPU; QLoRA on one GPU)
bash scripts/run_architecture.sh

# Or:
python3 scripts/run_architecture_pipeline.py \
  --config config/config.yaml \
  --rounds 1 \
  --seed 42

# Eval only (skip training)
python3 scripts/run_architecture_pipeline.py --skip-qlora

# Single GPU
python3 scripts/run_architecture_pipeline.py --device cuda:0 --no-multi-gpu
```

Outputs under `outputs/architecture_40q/`:

- `budget_table.md` / `budget_table_post_qlora.md`
- `per_question.jsonl`
- `round_*/train_traces.jsonl`, adapters under `checkpoints/`

### 5.4 Training-only / data utilities

```bash
python3 scripts/generate_training_data.py --help
python3 training/sft.py --help
python3 training/run_dpo.py --config config/config.yaml   # after preference pairs exist
python3 scripts/tune_qubo_params.py --help
```

---

## 6) Docker / container (protected GPU env)

Requires: NVIDIA driver on host, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Docker Compose v2.

### 6.1 Build

```bash
cd samsung_backup
docker compose -f docker/docker-compose.yml build
```

### 6.2 Interactive shell (dev)

```bash
docker compose -f docker/docker-compose.yml run --rm prism bash
# inside container:
bash scripts/env_check.sh
bash scripts/run_smoke.sh
```

### 6.3 One-shot smoke / benchmarks

```bash
# Smoke
docker compose -f docker/docker-compose.yml run --rm prism \
  bash scripts/run_smoke.sh

# Benchmarks (set GPUs in .env: CUDA_VISIBLE_DEVICES=0,1)
docker compose -f docker/docker-compose.yml run --rm prism \
  bash scripts/run_benchmarks.sh

# Architecture
docker compose -f docker/docker-compose.yml run --rm prism \
  bash scripts/run_architecture.sh
```

### 6.4 What the compose file protects

| Mount / setting | Purpose |
|-----------------|---------|
| Project bind | Code + configs |
| `./cache` → `/workspace/cache` | Models / HF cache persist on host |
| `./outputs` → `/workspace/outputs` | Results persist |
| `./checkpoints` | Adapters persist |
| `deploy.resources.reservations.devices` | NVIDIA GPUs |
| `IPC_LOCK` / `shm_size` | Stable DataLoader / CUDA IPC |
| `.env` | Tokens not baked into image |

Edit `docker/docker-compose.yml` `device_ids: ['0','1']` to match the cards your team owns.

### 6.5 Offline / air-gapped tip

On a networked machine:

```bash
python3 scripts/download_models.py
# copy cache/models + Hugging Face hub cache into the secure host
```

Then run containers with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in `.env`.

---

## 7) Config quick reference

`config/config.yaml`:

```yaml
hardware:
  gpu: "NVIDIA RTX A6000"
  vram_gb: 80
  multi_gpu: true
  dtype: "bfloat16"

model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  load_in_4bit: false

evaluation:
  batch_size: 16
  subset_size: 200

training:
  batch_size: 16
  qlora: true
```

Use `config/config_fast.yaml` for smaller `subset_size` during iteration.

---

## 8) How many questions run? Sampling details

Controlled by `evaluation.subset_size` / `evaluation.full_eval` in config, or CLI `--subset-size` / `--full`. Loader code: `evaluation/__init__.py` (`BenchmarkRunner`). Architecture panels: `scripts/curate_budget40.py` + `data/budget40/`.

### 8.1 Default remote / A6000 benchmark run

With **`config/config.yaml`** (`subset_size: 200`, `full_eval: false`) and all 5 benchmarks:

| Benchmark | Questions (approx.) | How the subset is taken |
|-----------|--------------------:|-------------------------|
| GSM8K | **200** | First 200 of HF `openai/gsm8k` **test** (`dataset.select(range(...))`) |
| ARC-Challenge | **200** | First 200 of `allenai/ai2_arc` ARC-Challenge **test** |
| StrategyQA | **200** | First 200 rows from `voidful/StrategyQA` (train stream / list) |
| MMLU | **~200** | `subset_size // 5` ≈ **40 per STEM subject** (first 40 of each subject’s test) |
| BBH | **~200** | `subset_size // ~27` tasks ≈ **7–8 per BBH task** (first N of each task) |

**Total ≈ 1,000 questions** per full default `run_all_benchmarks` / `run_benchmarks.sh` pass (5 × 200).

Other presets:

| Mode | Config / flag | Approx. total |
|------|---------------|---------------|
| Fast iteration | `config/config_fast.yaml` → `subset_size: 50` | ~250 (5 × 50) |
| Smoke | `bash scripts/run_smoke.sh` → GSM8K × 3 | **3** |
| Custom | `--subset-size N` | N per benchmark (BBH/MMLU still split across tasks/subjects) |
| Full datasets | `--full` or `full_eval: true` | Full HF sizes (e.g. GSM8K ~1,319; ARC ~1,172; BBH/MMLU much larger) |

Per **question**, the QUBO path still draws **16 traces** (`num_answers: 4` × `num_reasons: 4`), independent of how many questions you evaluate.

### 8.1b Matched budget: who shares the 16 traces?

**Greedy and CoT do *not* use the 16 traces.** They are separate single-shot baselines (one `generate` each, different prompts). They only **share the loaded model weights** with the sampler (same GPU replica), not the sample pool.

What **does** share one pool of 16 traces per question (“matched budget”):

| Method | Uses the 16 traces? | How |
|--------|---------------------|-----|
| **Greedy** | No | One direct-answer generate |
| **CoT** | No | One “think step by step” generate |
| **SC-16** | Yes | Majority vote over answers from the same 16 samples |
| **Best-16** | Yes | Pick best sample (by verifier score) from those 16 |
| **QUBO** | Yes | QUBO select diverse rationales **inside** the majority-answer group from those 16, then final prompt → answer |

Flow in `scripts/run_all_benchmarks.py` / architecture eval:

```
per question:
  1) baseline_greedy(inference, q)     # independent
  2) baseline_cot(inference, q)        # independent
  3) run_matched_budget(...):
        samples = sampler.sample(...)  # 16 traces once
        → SC-16, Best-16, QUBO all from that same pool
```

So comparisons of SC / Best / QUBO are fair on the **same** 16; Greedy/CoT are cheaper single-call baselines, not members of that matched set.

### 8.2 Architecture / QLoRA path (budget-40)

Not driven by `subset_size`. Fixed panels under `data/budget40/`:

| Manifest | Size | Role |
|----------|-----:|------|
| `eval_manifest.jsonl` | **40** | Held-out graded eval (8 × 5 benchmarks) |
| `train_manifest.jsonl` | **40** | Disjoint train prompts for QLoRA (8 × 5) |

Seed **42**. Regenerate with `python3 scripts/curate_budget40.py`.

### 8.3 Are we using stratified sampling?

**Main benchmark loader (`BenchmarkRunner`): no — not stratified random sampling.**

- GSM8K, ARC, StrategyQA: **prefix / head** of the split (deterministic first N). No shuffle by seed inside the loader; no stratification by difficulty, topic, or answer class.
- BBH / MMLU: only a **weak quota by task or subject** (equal chunk of `subset_size`), still the **first N inside each slice** — not stratified random draws.
- So default “200 per benchmark” is a **convenience prefix subset**, not a statistically stratified sample of the full test set. Treat accuracy CIs accordingly; for claims, prefer larger N or `--full`.

**Budget-40 curator (`curate_budget40.py`): closer to balanced panel design, still not classic stratified sampling.**

- Seeded shuffle (`seed=42`) + **disjoint** train/eval `source_id`s.
- MMLU: try **≥1 question per STEM subject**, then fill.
- BBH: tasks shuffled, then samples taken with disjoint IDs.
- GSM8K / ARC / StrategyQA: random disjoint picks from pools after shuffle — **balanced across the five benchmarks (8 each)**, not stratified within each dataset’s full label space.

### 8.4 Quick mental model

```
run_benchmarks.sh  →  ~200 Q × 5 benches  →  ~1000 Q  (prefix subsets; not stratified)
run_smoke.sh       →  3 GSM8K
run_architecture   →  40 eval + 40 train   (fixed budget-40 panels)
--full             →  entire HF test (or equivalent) sets
```

---

## 9) Outputs & logging

| Path | Contents |
|------|----------|
| `outputs/` | Benchmark JSON / MD / CSV |
| `outputs/architecture_40q/` | Budget tables + traces |
| `checkpoints/` | LoRA / QLoRA adapters |
| `cache/models` | Downloaded weights |

Commit **never**: `.env`, `cache/`, `checkpoints/`, large `outputs/`.

Optional W&B:

```bash
export WANDB_API_KEY=...
python3 scripts/run_all_benchmarks.py --wandb-project prism-a6000 --subset-size 50
```

---

## 10) Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | `nvidia-smi` → free VRAM or `--device cuda:N --no-multi-gpu`; lower `--batch-size` |
| Model loads on many GPUs | Should be fixed (`hf_device_map_value`); ensure you are on current code |
| Multi-GPU not used | Check `hardware.multi_gpu: true`, no `--device` pin, `torch.cuda.device_count() > 1` |
| HF download fails | Set `HF_TOKEN`; check VPN/proxy |
| Slow first run | Weights download into `cache/` — prefetch with `download_models.py` |
| Architecture missing manifests | `python3 scripts/curate_budget40.py` |

---

## 11) Suggested team checklist

1. [ ] Driver OK: `nvidia-smi` shows A6000(s) with free memory  
2. [ ] `bash scripts/setup_venv.sh` **or** `docker compose ... build`  
3. [ ] `bash scripts/env_check.sh`  
4. [ ] `bash scripts/run_smoke.sh` (3 GSM8K questions)  
5. [ ] Agree which GPU IDs the team owns → set `CUDA_VISIBLE_DEVICES` / compose `device_ids`  
6. [ ] Prefetch models into `cache/`  
7. [ ] Confirm question budget (§8): default ~200/bench (~1000 total); architecture = 40+40; note **not** stratified for main loader  
8. [ ] Run benchmarks / architecture with a fixed `--seed`  
9. [ ] Archive `outputs/*` + config snapshot for the report  

---

## 12) One-liner cheat sheet

```bash
cd samsung_backup && source .venv/bin/activate

bash scripts/env_check.sh
bash scripts/run_smoke.sh
bash scripts/run_benchmarks.sh
python3 scripts/curate_budget40.py && bash scripts/run_architecture.sh

# Docker
docker compose -f docker/docker-compose.yml run --rm prism bash scripts/run_smoke.sh
```
