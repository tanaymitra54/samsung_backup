# Quantum-Inspired AI for Reasoning: Beginner Guide

## 1) What this project is
This project improves how a small AI model solves multi-step questions (like math word problems and logic questions).

- Base model: small language model (~3B class)
- Main idea: generate many possible reasoning paths, then use optimization to pick the best subset
- Goal: beat normal prompting methods and target about 2x reasoning gain over baseline

## 2) Problem with older methods
Older methods in the papers work, but each has limits:

| Method | Strength | Limitation for our goal |
|---|---|---|
| CoT (Wei et al., 2022) | Improves step-by-step reasoning | Uses one chain; error propagation |
| Self-Consistency (Wang et al., 2023) | Multiple chains + majority vote | Expensive; voting can keep wrong but common answers |
| ToT (Yao et al., 2023) | Strong search | High compute cost |
| CR/QUBO (Esencan et al., 2024) | Optimization-based reason selection | Built for large models; large QUBO size (~900 vars) |
| QCR-LLM (Flores-Garrigos et al., 2025) | Quantum/HUBO extension | Hardware heavy; not focused on 3B SLM constraints |

## 3) Our method (simple explanation)
We use a 3-stage pipeline:

1. Sampling: model generates multiple reasoning options
2. QUBO optimization: score and select best non-redundant reason set
3. Final inference: feed selected reasons back to model for final answer

In plain words: "Generate many ideas -> filter smartly -> answer better."

## 4) What we improved vs research papers

| Improvement we added | Compared paper(s) | Shortcoming resolved | Why it matters |
|---|---|---|---|
| SLM-adapted lightweight QUBO (semantic clustering to <=200 vars) | Esencan et al. (2024) CR; Flores-Garrigos et al. (2025) QCR-LLM | Prior CR/QCR settings are not optimized for 3B SLM efficiency (often large variable spaces) | Makes optimization practical on CPU/A100 |
| Correctness-aware verifier term in QUBO | Esencan et al. (2024) CR | Similarity-only QUBO can keep fluent but wrong reasons | Reduces hallucinated/wrong chains |
| Diversity sampling with adaptive temperature + perturbation | Wang et al. (2023) Self-Consistency; Yao et al. (2023) ToT | SLM outputs can collapse to similar paths; simple voting/search may be compute-heavy | Gives optimizer real choice diversity |
| QUBO-based hyperparameter search | Wang et al. (2023) Self-Consistency; Esencan et al. (2024) CR | Hyperparameters typically tuned by expensive manual/grid search | Joint tuning improves quality/latency |
| Planned HUBO + optimized annealing | Flores-Garrigos et al. (2025) QCR-LLM; Chandarana et al. (2025) Runtime Quantum Advantage | Pairwise interactions miss higher-order logic; vanilla schedules can get stuck | Better global selection quality |
| QUBO-guided SFT loop (planned) | Margapuri et al. (2025) PEPS+PPO | Inference-only gains plateau | Converts inference gains into model gains |

### Detailed explanation: how exactly we overcome each shortcoming

1. **SLM-adapted lightweight QUBO (vs Esencan 2024 CR, Flores-Garrigos 2025 QCR-LLM)**
   - **Shortcoming in prior work:** QUBO setups were demonstrated mainly with larger-model pipelines and larger optimization spaces, which are harder to run efficiently for 3B-class SLM deployments.
   - **What we do:** Before building QUBO, we group near-duplicate reasons using semantic clustering (sentence embeddings + similarity threshold). Instead of giving one binary variable to every raw reason, we create variables for representative reason units.
   - **Why this fixes it:** The variable count drops to a tractable range (target <=200), so simulated annealing can run reliably on CPU/A100 while preserving diversity and relevance.

2. **Correctness-aware verifier in QUBO objective (vs Esencan 2024 CR)**
   - **Shortcoming in prior work:** Similarity/redundancy signals alone cannot detect whether a reason is actually correct.
   - **What we do:** We add a verifier score per reason before QUBO construction:
     - Math tasks: rule-based arithmetic consistency checks.
     - Commonsense tasks: lightweight NLI entailment scoring.
     - This score is injected into QUBO diagonal terms so factually stronger reasons receive better optimization preference.
   - **Why this fixes it:** Fluent-but-wrong chains are penalized earlier, reducing hallucination contamination in final selected subsets.

3. **Diversity-aware sampling design (vs Wang 2023 Self-Consistency, Yao 2023 ToT)**
   - **Shortcoming in prior work:** Self-consistency can be expensive and may still produce correlated outputs; SLMs especially can collapse to similar reasoning patterns.
   - **What we do:** We combine contrastive decoding, adaptive temperature, and prompt perturbation (rephrased prompts) to force structural diversity in sampled reasoning.
   - **Why this fixes it:** QUBO receives genuinely different candidates, so optimization can choose complementary reason fragments instead of repeated variants.

4. **QUBO-based hyperparameter search (vs typical manual/grid tuning in SC/CR pipelines)**
   - **Shortcoming in prior work:** Hyperparameters (temperature, top-p, sample count, subset size) are often tuned separately with expensive trial-and-error.
   - **What we do:** We encode hyperparameter choices as binary decision variables and optimize them with annealing using a proxy objective based on verifier-quality distribution and selection quality.
   - **Why this fixes it:** We move from local/manual tuning to joint optimization, improving both accuracy and runtime efficiency.

5. **Planned HUBO + improved annealing schedule (vs QCR-LLM 2025, Chandarana 2025)**
   - **Shortcoming in prior work:** Pairwise QUBO interactions miss higher-order logical dependencies; vanilla annealing can settle in local minima.
   - **What we do:**
     - Add limited higher-order (triplet) interaction modeling via HUBO/PUBO-to-QUBO reduction.
     - Introduce improved annealing schedules (momentum/counterdiabatic-inspired strategies) and compare with vanilla SA/Tabu baselines.
   - **Why this fixes it:** Better captures multi-reason coherence and improves chances of finding stronger global selections.

6. **QUBO-guided SFT feedback loop (vs Margapuri 2025 PEPS+PPO direction)**
   - **Shortcoming in prior work / baseline pipeline:** Inference-only improvements eventually saturate.
   - **What we do:** We collect high-quality QUBO-selected traces and convert them into supervised fine-tuning data (LoRA/SFT rounds), then re-run pipeline with the improved model.
   - **Why this fixes it:** The model gradually learns to generate better initial reasons, which further improves downstream QUBO selection in a positive feedback loop.

## 5) Estimated improvement vs baselines

### GSM8K projected cumulative path

| Stage | Method | Estimated Accuracy |
|---|---|---|
| Baseline | Greedy SLM | ~60% |
| +CoT | Standard prompting | ~66% |
| +Diverse sampling | Better candidate generation | ~72% |
| +QUBO selection | Combinatorial selection | ~78% |
| +Verifier in QUBO | Correctness-aware selection | ~84% |
| +SFT on selected traces | Feedback training | ~88% |
| +Optimized annealing + HUBO | Advanced optimization | ~91% |

Interpretation: baseline to final is about +31 points (60 -> 91), which exceeds the practical 2x gain target definition used in roadmap notes.

## 6) Research papers identified from notes

1. Liu et al. (2025) - Understanding LLMs
2. Wei et al. (2022) - Chain-of-Thought Prompting
3. Kojima et al. (2022) - Zero-Shot CoT
4. Yao et al. (2023) - ReAct
5. Wang et al. (2023) - Self-Consistency
6. Yao et al. (2023) - Tree of Thoughts
7. Wang et al. (2025) - Ranked Voting Self-Consistency
8. Esencan et al. (2024) - Combinatorial Reasoning (QUBO)
9. Flores-Garrigos et al. (2025) - QCR-LLM
10. Chandarana et al. (2025) - Runtime Quantum Advantage
11. Margapuri et al. (2025) - PEPS+PPO

## 6.1) Benchmarks used in these research papers

| Paper / Method | Benchmarks reported in notes |
|---|---|
| Wei et al. (2022) - Chain-of-Thought (CoT) | GSM8K, SVAMP, AQuA, CommonsenseQA |
| Kojima et al. (2022) - Zero-Shot CoT | MultiArith, GSM8K, SVAMP, AQuA, SingleEQ |
| Wang et al. (2023) - Self-Consistency | GSM8K, SVAMP, AQuA, ARC |
| Yao et al. (2023) - Tree of Thoughts (ToT) | Game of 24, Mini Crosswords, Creative Writing |
| Yao et al. (2023) - ReAct | HotpotQA, FEVER, AlfWorld, WebShop |
| Wang et al. (2025) - Ranked Voting Self-Consistency | Multiple reasoning benchmarks (paper reports broad multi-task evaluation) |
| Esencan et al. (2024) - Combinatorial Reasoning (CR) | BigBench-Hard (BBH) |
| Flores-Garrigos et al. (2025) - QCR-LLM | BIG-Bench Extra Hard (BBEH) |
| Margapuri et al. (2025) - PEPS+PPO | GSM8K, StrategyQA, EntailmentBank |
| Chandarana et al. (2025) - Runtime Quantum Advantage | Standard QUBO optimization benchmark instances (not QA/NLP benchmark datasets) |

For our PRISM comparison roadmap, the most relevant common benchmarks are GSM8K (primary), BBH/BBEH, StrategyQA, MMLU, and ARC-Challenge.

## 6.2) Current roadmap status and evaluation caution

### Current implementation status (quick view)
- Phase 1 core pipeline: mostly complete (sampling, verifier, QUBO builder, solver, inference, hyperparameter QUBO modules are implemented).
- Phase 2 optimization upgrades: partially complete (baseline simulated annealing path done; advanced schedules pending).
- Phase 3 SFT loop: scaffolded but not fully executed yet.
- Phase 4 polish/validation: pending full benchmark execution and HUBO extension.

### Latest GSM8K run interpretation
- A first run was executed with only **3 samples**.
- In that run: Greedy 66.67%, CoT 33.33%, QUBO 0.00%.
- This run is only a sanity check and is too small for performance claims.

### Important caution about the "2x target" flag
- The current script computes 2x threshold from CoT gain over Greedy.
- If CoT is lower than Greedy on a tiny sample, the threshold becomes negative and can incorrectly mark "Meets 2x target: True".
- Therefore, treat this case as **not valid for final claim**; rerun with larger samples (100/200+) before interpreting progress.

## 7) Why this is different from other methods
- Not just vote-based aggregation: uses optimization-backed subset selection.
- Not only large-model focused: explicitly engineered for smaller 3B-class models.
- Adds correctness signal directly into objective (key novelty).
- Targets deployable path now (CPU/A100), with future one-switch migration to quantum backend.

## 8) Future integration plan
- Integrate full multi-benchmark runner (GSM8K, BBH, StrategyQA, ARC, MMLU)
- Add HUBO and counterdiabatic-inspired annealing
- Run QUBO-guided SFT rounds to create self-improving loop
- Compare against all paper baselines with same evaluation protocol

## 9) Quick presentation points (non-technical)
- We do not trust one AI answer path; we generate many.
- We score quality + remove repetition mathematically.
- We keep only the best reasoning pieces.
- That selected reasoning helps the same small model answer better.
- This bridges low cost (small model) and high reasoning quality.

---

## 10) NVIDIA RTX A6000 (80 GB) + Multi-GPU Specs

Target hardware is documented in `config/config.yaml` under `hardware`:

```yaml
hardware:
  gpu: "NVIDIA RTX A6000"
  vram_gb: 80
  multi_gpu: true
  dtype: "bfloat16"
```

Default SLM: **`Qwen/Qwen2.5-3B-Instruct`** in bfloat16. A 3B model uses ~6–8 GB weights, so one full replica (model + verifier + KV cache + batch 16) fits comfortably on a single 80 GB A6000. Parallelism is **job sharding**, not tensor-parallel: each GPU runs its own complete stack via `CUDA_VISIBLE_DEVICES`.

### 10.1 Spec knobs (already in config)

| Knob | Value | Why |
|------|-------|-----|
| Model | Qwen2.5-3B-Instruct | Matches architecture slide (SLM + QLoRA) |
| `evaluation.batch_size` | 16 | A6000 headroom vs old batch 4 |
| `training.batch_size` | 16 | QLoRA SFT throughput on 80 GB |
| `pipeline.num_answers` × `num_reasons` | 4 × 4 = 16 traces | Majority answer, then diverse rationales |
| `solver.gpu.num_parallel_reads` | 1024 | GPU simulated annealing |
| `vllm.tensor_parallel_size` | 1 | Keep one model per GPU when vLLM is enabled |
| `device_map` | single index (`hf_device_map_value`) | Prevents `auto` from spanning all cards |

### 10.2 Multi-GPU run mode

When `hardware.multi_gpu: true` and more than one GPU is visible:

1. **`scripts/run_all_benchmarks.py`** — auto multi-GPU:
   - Multiple benchmarks → one (or more) benchmarks per GPU
   - Single benchmark → questions sharded across GPUs, results merged
2. **`scripts/run_architecture_pipeline.py`** — eval + train-trace collection sharded the same way; **QLoRA trains on one GPU** so adapter writes stay consistent
3. Force single GPU: `--device cuda:N` or `--no-multi-gpu`
4. Force multi-GPU: `--multi-gpu`

```bash
# Auto: use all visible A6000s
python3 scripts/run_all_benchmarks.py --benchmarks gsm8k --subset-size 200

# Pin one card
python3 scripts/run_all_benchmarks.py --device cuda:1 --benchmarks gsm8k

# Architecture loop (eval/traces multi-GPU, QLoRA single-GPU)
python3 scripts/run_architecture_pipeline.py --config config/config.yaml
```

Workers call `mask_worker_gpu(gpu_id)` from `pipeline/device_utils.py` so each process only sees one physical GPU as `cuda:0`.

### 10.3 OOM / crowded-server notes

- Do **not** use `device_map="auto"` — it shards one model across every visible GPU and fights job-parallel runs.
- If the machine is shared and VRAM is nearly full, free space or pick a free GPU with `--device cuda:N` rather than hoping fragmentation tips will help.
- The &lt;12 GiB free → force 4-bit / batch-1 heuristic in `training/sft.py` only kicks in when headroom is actually low; A6000 80 GB runs keep config batch sizes.

### 10.4 Optional next steps (same 80 GB class)

| Item | Notes |
|------|-------|
| Full-dataset eval | `full_eval: true` or `--full` |
| vLLM backend | `--use-vllm` with `tensor_parallel_size: 1` |
| Larger SLM (4B–8B) | Still fits in bf16 on 80 GB; keep one replica per GPU |
| W&B | `--wandb-project NAME` |

---

## 11) Common Q&A

### Q1: Which model are we using?
Currently **`Qwen/Qwen2.5-3B-Instruct`** (3B parameters, ~6–8 GB in bf16). This is an SLM (Small Language Model), matching the project's goal of running on small 3B-class models on **NVIDIA RTX A6000 80 GB** with multi-GPU job sharding. Config: `config/config.yaml` (`model.name` + `hardware`).

### Q2: What is the difference between Greedy, CoT, and QUBO?

| Method | What it does | How it runs |
|--------|-------------|-------------|
| **Greedy** | Direct answer, no reasoning instruction. `do_sample=False`. | 1 `model.generate()` call per question. ~0.5-2s. |
| **CoT** | Chain-of-Thought: ask model to "think step by step" before answering. Still deterministic. | 1 `model.generate()` call per question. ~1-3s. |
| **QUBO** | Full pipeline: 16 diverse samples → score by correctness → embed + cluster → build QUBO matrix → solve with simulated annealing on GPU → final answer from selected samples. | 16+ `model.generate()` calls + verifier + solver. ~15-60s per question. |

Both Greedy and CoT use the same shared model on the same GPU — only the prompt differs. `scripts/run_all_benchmarks.py:89-99`.

### Q3: What exactly is the QUBO pipeline?
```
Sampler (16 diverse samples) → Verifier (score each by correctness)
→ QUBOBuilder (embed, cluster, build Q matrix with quality + diversity terms)
→ SimulatedAnnealingSolver (solve on GPU)
→ InferencePipeline.run (generate final answer from selected subset)
```
Implemented across: `pipeline/sampling.py`, `pipeline/verifier.py`, `pipeline/qubo_builder.py`, `pipeline/solver.py`, `pipeline/inference.py`.

### Q4: How do I check if the benchmark is running on GPU?
Two ways:
1. **`nvidia-smi`** — shows GPU memory usage and utilization. If memory is allocated to a Python process, it's on GPU.
2. **Startup logs** — the script prints device info:
   ```
   Runtime device: cuda:0
   Inference model device: cuda:0
   Sampler device: cuda:0
   ```

### Q5: Why was it running out of memory (OOM)?
Root causes (all fixed):
1. **`device_map="auto"`** — was loading model across both GPUs instead of staying on one
2. **Two model copies** — `InferencePipeline` + `DiverseSampler` each loaded the full 7B model separately
3. **CPU→GPU transfer spike** — `.to(cuda)` after loading caused temporary memory doubling
4. **7B model** — too large when other processes already occupied GPU memory

Fixes applied:
- Pin `device_map` to one GPU index via `hf_device_map_value` (never `"auto"`)
- Multi-GPU job sharding with per-worker `mask_worker_gpu` / `CUDA_VISIBLE_DEVICES`
- Shared model between inference + sampler
- `low_cpu_mem_usage=True` + OOM fallback retries
- Default SLM: Qwen2.5-3B on A6000 80 GB (`config/config.yaml`)

### Q6: Why is the benchmark taking so long?
Each question runs **16 model generations** in the sampler (4 perturbations × 4 temperatures). For 50 questions that's ~800 `model.generate()` calls. Even with a 1.5B model, each generation processes up to 256 tokens. The QUBO solver and verifier add more time. A full 200-question run can take 30-60 minutes depending on GPU contention.

**Speed tips:**
- Reduce `pipeline.num_answers` in `config/config.yaml` (fewer samples = faster but potentially less diverse)
- Reduce `pipeline.max_new_tokens` (shorter outputs = faster)
- Use `--subset-size 20` for quick test runs
- Use `--benchmarks gsm8k` to run only one benchmark

### Q7: How do I run only one benchmark with fewer questions?
```bash
# GSM8K (50 questions, GPU 1)
python3 scripts/run_all_benchmarks.py --device cuda:1 --benchmarks gsm8k --subset-size 40

# MMLU (40 questions, GPU 1)
python3 scripts/run_all_benchmarks.py --device cuda:1 --benchmarks mmlu --subset-size 40
```

### Q8: How do I stop a running benchmark?
```bash
ps aux | grep run_all_benchmarks    # find PID
kill <PID>                          # graceful stop
kill -9 <PID>                       # force stop
```
Or `Ctrl+C` in the terminal where it's running.

### Q9: What does the progress bar show?
```
[    gsm8k]  ██████░░░░░░░░░░░░░░   5/50 batches  |  greedy:  62.5%  |  cot:  65.0%  |  qubo:  63.8%
```
- Bar fills as batches complete (4 questions per batch)
- Percentages show **running accuracy** so far (not final)
- Updates after each batch finishes

`scripts/run_all_benchmarks.py:588-596`.

### Q10: SLM vs LLM — why does it matter?
| | SLM (1-3B) | LLM (7B+) |
|---|---|---|
| Memory | ~3-6GB | ~15-80GB |
| Speed | Fast | Slow |
| Cost | Cheap | Expensive |
| Project fit | **Primary target** | Nice-to-have |

The project is explicitly designed for **3B-class SLMs** (small language models). The README states: "This project makes small language models (SLMs) reason better." SLMs are cheaper to deploy, faster to run, and the QUBO pipeline compensates for their weaker reasoning ability.

### Q11: What benchmarks are included and what do they test?
| Benchmark | Type | Questions | What it tests |
|-----------|------|-----------|---------------|
| GSM8K | Math word problems | 1,319 | Multi-step arithmetic reasoning |
| BBH | BigBench Hard | ~6,500 | Logical reasoning, classification |
| StrategyQA | Yes/no strategy | ~2,290 | Commonsense strategic reasoning |
| ARC-Challenge | Science MCQs | 1,172 | Grade-school science knowledge |
| MMLU | Multi-subject MCQs | ~14,000 | Knowledge across 57 subjects |

Config: `config/config.yaml:66-71`.

### Q12: How do I interpret the QUBO accuracy versus Greedy/CoT?
The QUBO accuracy should be **higher** than both Greedy and CoT if the pipeline working correctly. The key metric is `abs_gain_vs_greedy` — the absolute improvement over the simplest baseline. An improvement of +5-15 percentage points is a good result for an SLM.

### Q13: What flags are available for the benchmark script?
```
--device cuda:N       Pin to one GPU (disables auto multi-GPU)
--benchmarks gsm8k    Run specific benchmark(s) only
--subset-size N       Use N questions instead of full dataset
--full                Run full dataset (ignores subset-size)
--batch-size N        Batch size for inference (default: 16 on A6000)
--no-batch            Disable batched inference
--use-vllm            Use vLLM backend instead of transformers
--multi-gpu           Force multi-GPU job sharding
--no-multi-gpu        Force single-GPU even if many cards are visible
--config PATH         Config YAML (default: config/config.yaml)
--output-dir DIR      Output directory for results
--seed N              Random seed (default: 42)
```
