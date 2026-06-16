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

## 10) H100 GPU Optimization Roadmap (Jun–Sep 2026)

With **remote access to NVIDIA H100 GPUs (80 GB HBM3, FP8 Tensor Cores, 3.35 TB/s bandwidth)** via Fortinet VPN, the following advanced changes unlock the project's full potential across five phases.

### 10.1 Phase A — Quick Wins: FP8, Larger Models, FlashAttention

**Goal:** Immediately leverage H100 hardware without architectural changes.

| Change | Files | What & Why |
|--------|-------|------------|
| **A1. 4-bit quantization + FlashAttention** | `config/config.yaml`, `pipeline/sampling.py`, `pipeline/inference.py` | Load models in 4-bit NF4 (`BitsAndBytes`) + `attn_implementation="flash_attention_2"`. Reduces VRAM ~4x, enables 70B models on 80 GB; FlashAttention-2 gives 2-4x faster attention on H100. |
| **A2. Upgrade default model** | `config/config.yaml` | Switch from `deepseek-coder-1.5b-instruct` (1.5B) to `Qwen/Qwen2.5-7B-Instruct` (7B) or `meta-llama/Llama-3.1-8B-Instruct` (8B). 4-bit quantized 8B model uses ~5 GB VRAM—leaves plenty for KV cache and batching. |
| **A3. Scale sampling parameters** | `config/config.yaml` | Increase `num_answers: 4`, `num_reasons: 4` (from 2 each) → 16×4 = 64 diverse samples per question (vs 8 currently). H100's 3.35 TB/s bandwidth makes this fast. |
| **A4. New dependencies** | `requirements.txt` | Add `bitsandbytes`, `flash-attn`, `peft`, `trl`, `vllm`, `wandb`, `accelerate` (already present). |

**Technical detail — FP8 note:** H100 HBM3 GPUs have dedicated FP8 Tensor Cores. `transformers` `model.generate()` auto-accelerates float16 matmuls through Transformer Engine. For explicit FP8 KV cache, use `vLLM` with `quantization="fp8"` (Phase E).

---

### 10.2 Phase B — Full-Scale Evaluation

**Goal:** Run all 5 benchmarks at full dataset size with batched GPU inference.

| Change | Files | What & Why |
|--------|-------|------------|
| **B1. Full-dataset benchmarks** | `config/config.yaml`, `scripts/run_all_benchmarks.py`, `evaluation/__init__.py` | Set `full_eval: true`. Run full GSM8K (1,319), MMLU (5 subjects ~3,000), StrategyQA (~2,290), ARC-Challenge (1,172), BBH (~6,500). H100 completes a 7B model inference at ~50-100 tok/s. |
| **B2. Batch inference** | `scripts/run_all_benchmarks.py` | Group N questions, pad to max length, run one `model.generate(batch)` call. Reduces GPU kernel launch overhead 10-20x. |
| **B3. Multi-GPU parallel benchmarks** | `scripts/run_all_benchmarks.py` | If multi-GPU: dispatch each benchmark to a different GPU via `ProcessPoolExecutor`. Each benchmark runs independently on its own GPU. |

**Technical detail — batched generation:**
```python
# Tokenizer pads all inputs to same length
inputs = tokenizer(questions, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)
```

---

### 10.3 Phase C — Training Pipeline (SFT Feedback Loop)

**Goal:** Convert inference-only gains into model parameter improvements via iterative fine-tuning. **Phase 3 from roadmap — currently a stub.**

| Change | Files | What & Why |
|--------|-------|------------|
| **C1. QUBO trace collection** | `training/sft.py` | Run QUBO pipeline on training splits; collect `(question, selected_traces, correct_answer)` tuples. Filter: keep only samples where pipeline produced correct answer. Save as HF Dataset (JSONL). |
| **C2. LoRA fine-tuning** | `training/sft.py` | Fine-tune Llama-3.1-8B with LoRA (`r=16`, `alpha=32`) via `peft` + `trl.SFTTrainer`. H100's 80 GB: batch size 8-16, 2-3 hours for 3 epochs on full GSM8K training set (~7.5k samples). |
| **C3. Iterative self-improvement loop** | `training/sft.py` | Round 1: train on QUBO traces → improved model → run QUBO pipeline → better traces → Round 2: train on improved traces → repeat. Each iteration should improve accuracy. |

**Technical detail — SFT with LoRA:**
```python
model = AutoModelForCausalLM.from_pretrained("model", quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"])
trainer = SFTTrainer(model=model, train_dataset=dataset, args=TrainingArguments(...), peft_config=peft_config)
trainer.train()
```

---

### 10.4 Phase D — GPU-Accelerated Optimization

**Goal:** Replace CPU solver with massively parallel GPU annealing; extend to HUBO.

| Change | Files | What & Why |
|--------|-------|------------|
| **D1. GPU Simulated Annealing** | `pipeline/solver.py` | Port SA solver to GPU with 1024+ parallel reads (vs 2 on CPU). Each "chain" = independent state evolving in parallel via `torch` tensor ops. Expected: ~50 ms per solve (vs ~5 s CPU) = **100x speedup**. |
| **D2. Parallel tempering** | `pipeline/solver.py` | N replicas at different temperatures exchange states periodically. Better at escaping local minima than single-temperature SA. Trivially parallel on GPU. |
| **D3. Counterdiabatic annealing** | `pipeline/solver.py` | Add momentum term `λ(t)·dT/dt·Σ(∂E/∂x)²` to suppress freeze-out at phase transitions. Gradient computed via `torch.autograd`. |
| **D4. HUBO extension** | `pipeline/qubo_builder.py` | Add triplet interaction terms (cubic): `E = xᵀQx + ΣTᵢⱼₖxᵢxⱼxₖ`. Triplet penalizes triple redundancy (e.g., three near-identical reasons). O(n³) matrix → H100 Tensor Cores essential. PyQUBO supports HUBO natively. |

**Technical detail — GPU SA (simplified):**
```python
states = torch.randint(0, 2, (num_reads, n), device="cuda")
for t in cooling_schedule:
    flip_idx = torch.randint(0, n, (num_reads,), device="cuda")
    states[torch.arange(num_reads), flip_idx] ^= 1
    new_energy = torch.einsum('ri,ij,rj->r', states, Q, states)
    accept = (delta < 0) | (torch.rand(num_reads, device="cuda") < torch.exp(-delta / t))
    states[~accept, flip_idx[~accept]] ^= 1
```

---

### 10.5 Phase E — Infrastructure & Experiment Tracking

**Goal:** Production-quality throughput, reproducibility, and tracking.

| Change | Files | What & Why |
|--------|-------|------------|
| **E1. vLLM integration** | `pipeline/inference.py` (optional backend) | Replace `transformers.generate()` with `vLLM.LLM.generate()` for benchmark inference. Supports FP8 KV cache, PagedAttention, continuous batching. 10-20x throughput improvement. |
| **E2. Experiment tracking (W&B)** | `scripts/*.py`, `training/sft.py` | Log all benchmark accuracy, training loss, QUBO energy landscapes, hyperparameters to Weights & Biases. Enables systematic hyperparameter sweeps (LoRA rank, learning rate, penalty weights, etc.). |
| **E3. VPN compatibility** | `config/config.yaml` | Ensure HF Hub downloads work through Fortinet VPN. Models/datasets use HTTPS → work through VPN. Add `HF_ENDPOINT` config option for corporate proxy workarounds. |

---

### 10.6 Implementation Order & Timeline

```
Week 1 (Jun 15-21):  Phase A — FP8, larger model, FlashAttention, more samples
                     → Verify: GSM8K @ subset-size 200 on H100

Week 2 (Jun 22-28):  Phase B — Full-dataset benchmarks, batch inference, GPU dispatch
                     → Verify: All 5 benchmarks @ full size, results CSV/JSON/MD

Week 3 (Jun 29-Jul 5): Phase C — SFT training pipeline, LoRA fine-tuning
                     → Verify: Accuracy improvement vs pre-training baseline

Week 4 (Jul 6-12):   Phase D — GPU solver, parallel tempering, counterdiabatic, HUBO
                     → Verify: Solver energy/time vs CPU baseline; HUBO on BBH

Week 5 (Jul 13-19):  Phase E — vLLM, W&B tracking, final benchmark sweep
                     → Verify: Complete benchmark dashboard with all methods
```

### 10.7 Expected Performance Improvement (GSM8K)

| Stage | Method | Config | Est. Accuracy | Time per 200 Qs |
|-------|--------|--------|:-------------:|:---------------:|
| Current | deepseek-coder-1.5B + QUBO | CPU/FP32 | ~60-66% | ~30 min |
| + Phase A | Qwen2.5-7B + QUBO | H100 4-bit | ~72-76% | ~5 min |
 | + Phase B | Full GSM8K eval | H100 batch | ~74-78% | ~8 min |
| + Phase C | LoRA SFT on QUBO traces | H100 train | ~84-88% | — |
| + Phase D | GPU SA + HUBO | H100 GPU | ~88-91% | ~3 min |
| + Phase E | vLLM + W&B sweep | H100 tuned | **~91%** | ~2 min |

---

## 11) Common Q&A

### Q1: Which model are we using?
Currently **`Qwen/Qwen2.5-1.5B-Instruct`** (1.5B parameters, ~3GB). This is an SLM (Small Language Model), matching the project's goal of running on small 3B-class models. Earlier we used `Qwen/Qwen2.5-7B-Instruct` (7B, an LLM) but it caused GPU OOM on crowded servers and was slower. Config: `config/config.yaml:2`.

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
- GPU masking via `CUDA_VISIBLE_DEVICES` before `torch` import (`scripts/run_all_benchmarks.py:14-25`)
- Single device map `{"": device_index}` for 4-bit loads (`pipeline/device_utils.py:18`)
- Shared model between inference + sampler (`scripts/run_all_benchmarks.py:492-496`)
- `low_cpu_mem_usage=True` + direct GPU load (`pipeline/inference.py:54-55`)
- OOM fallback retries with smaller input/output limits + `use_cache=False`
- Switched to 1.5B SLM (`config/config.yaml:2`)

### Q6: Why is the benchmark taking so long?
Each question runs **16 model generations** in the sampler (4 perturbations × 4 temperatures). For 50 questions that's ~800 `model.generate()` calls. Even with a 1.5B model, each generation processes up to 256 tokens. The QUBO solver and verifier add more time. A full 200-question run can take 30-60 minutes depending on GPU contention.

**Speed tips:**
- Reduce `pipeline.num_answers` in `config/config.yaml` (fewer samples = faster but potentially less diverse)
- Reduce `pipeline.max_new_tokens` (shorter outputs = faster)
- Use `--subset-size 20` for quick test runs
- Use `--benchmarks gsm8k` to run only one benchmark

### Q7: How do I run only one benchmark with fewer questions?
```bash
python3 scripts/run_all_benchmarks.py --device cuda:1 --benchmarks gsm8k --subset-size 30
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
--device cuda:N       Run on specific GPU (default: cuda:0)
--benchmarks gsm8k    Run specific benchmark(s) only
--subset-size N       Use N questions instead of full dataset
--full                Run full dataset (ignores subset-size)
--batch-size N        Batch size for inference (default: 4)
--no-batch            Disable batched inference
--use-vllm            Use vLLM backend instead of transformers
--multi-gpu           Distribute benchmarks across GPUs
--output-dir DIR      Output directory for results
--seed N              Random seed (default: 42)
```
