# Team Runbook — PRISM / QUBO Pipeline (A6000 80GB + Multi-GPU)

How to set up, protect, and run the full stack on **NVIDIA RTX A6000 (80 GB)** with optional multi-GPU job sharding.

| Item | Value |
|------|--------|
| Project root | `samsung_backup/` (run all commands from here) |
| Default model | `Qwen/Qwen2.5-3B-Instruct` |
| Config | `config/config.yaml` |
| Fast / smaller eval | `config/config_fast.yaml` |
| Precision | `bfloat16` (see `hardware.dtype`) |
| Parallelism | One full model replica **per GPU** (`CUDA_VISIBLE_DEVICES` job sharding) |

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

## 8) Outputs & logging

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

## 9) Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | `nvidia-smi` → free VRAM or `--device cuda:N --no-multi-gpu`; lower `--batch-size` |
| Model loads on many GPUs | Should be fixed (`hf_device_map_value`); ensure you are on current code |
| Multi-GPU not used | Check `hardware.multi_gpu: true`, no `--device` pin, `torch.cuda.device_count() > 1` |
| HF download fails | Set `HF_TOKEN`; check VPN/proxy |
| Slow first run | Weights download into `cache/` — prefetch with `download_models.py` |
| Architecture missing manifests | `python3 scripts/curate_budget40.py` |

---

## 10) Suggested team checklist

1. [ ] Driver OK: `nvidia-smi` shows A6000(s) with free memory  
2. [ ] `bash scripts/setup_venv.sh` **or** `docker compose ... build`  
3. [ ] `bash scripts/env_check.sh`  
4. [ ] `bash scripts/run_smoke.sh` (3 GSM8K questions)  
5. [ ] Agree which GPU IDs the team owns → set `CUDA_VISIBLE_DEVICES` / compose `device_ids`  
6. [ ] Prefetch models into `cache/`  
7. [ ] Run benchmarks / architecture with a fixed `--seed`  
8. [ ] Archive `outputs/*` + config snapshot for the report  

---

## 11) One-liner cheat sheet

```bash
cd samsung_backup && source .venv/bin/activate

bash scripts/env_check.sh
bash scripts/run_smoke.sh
bash scripts/run_benchmarks.sh
python3 scripts/curate_budget40.py && bash scripts/run_architecture.sh

# Docker
docker compose -f docker/docker-compose.yml run --rm prism bash scripts/run_smoke.sh
```
