# Implementation Roadmap: Quantum-Inspired AI for Multi-Stage Reasoning

## Goal
Achieve **2× performance gain** over Llama-3.2-3B baseline on GSM8K (~60% → ~91%) using QUBO-optimized SLM reasoning pipeline.

## Current Status (as of May 21, 2026)

### Phase-wise progress snapshot
| Phase | Planned Scope | Current Status |
|---|---|---|
| Phase 1 — Core Pipeline | Sampling, verifier, QUBO builder, solver, inference, hyperparameter QUBO | **Mostly complete** (core modules implemented) |
| Phase 2 — QUBO Solver Optimization | Lightweight QUBO + optimized annealing variants | **Partially complete** (lightweight QUBO + baseline SA path done; advanced schedule pending) |
| Phase 3 — SFT Feedback Loop | SFT on QUBO-selected traces | **Stub stage** (file scaffolded, full training pending GPU cycle) |
| Phase 4 — Polish/Validation | HUBO extension + multi-benchmark validation | **Partially complete** (MMLU + ARC-Challenge integrated; full-run validation pending) |

### What is done in implementation
- End-to-end core files implemented: `pipeline/sampling.py`, `pipeline/verifier.py`, `pipeline/qubo_builder.py`, `pipeline/solver.py`, `pipeline/inference.py`, `pipeline/hyperparam_qubo.py`.
- Method-comparison pipeline script implemented: `scripts/generate_comparison.py`.
- GSM8K baseline-vs-QUBO evaluation runner created: `evaluation/run_gsm8k_comparison.py` with answer normalization in `evaluation/answer_utils.py`.
- Multi-benchmark evaluation runner (`evaluation/__init__.py`) extended with production-ready multi-choice support:
  - MMLU integrated with 5 STEM subjects (`abstract_algebra`, `college_computer_science`, `college_physics`, `electrical_engineering`, `machine_learning`) from HuggingFace `cais/mmlu` using `test` split.
  - ARC-Challenge integrated from HuggingFace `ai2_arc` (`ARC-Challenge`, `test` split).
  - Unified MCQ prompt formatting added for both benchmarks (question + labeled choices + `Answer:` suffix).
  - Strict MCQ scoring path added (extract A/B/C/D from model output before comparison), to avoid inflated accuracy from loose substring matching.

### Latest evaluation note
- A first GSM8K run completed with **3 samples only** (very small sanity check).
- Reported output showed Greedy > CoT > QUBO for that tiny sample; this is **not statistically reliable** and must not be treated as final benchmark performance.
- The `2x target` flag can look misleading when CoT underperforms Greedy on tiny samples; this requires larger-sample validation.

### Next milestone actions
1. Run GSM8K comparison at meaningful size (`--subset-size 100` then `200`).
2. Audit per-question prediction CSV for extraction/format mismatches.
3. Confirm stable baseline metrics (Greedy/CoT) before claiming gains.
4. Run `BenchmarkRunner.run_all()` for BBH, StrategyQA, MMLU, and ARC-Challenge and save a timestamped metrics report.
5. Start Phase 2 advanced annealing experiments and Phase 3 SFT execution once compute window is allocated.

## Detailed Change Log (Beginner-Friendly)

This section explains exactly what was changed in `evaluation/__init__.py` and why.

### 1) Added MMLU test-split loading (instead of validation)
- **What changed:** MMLU loader now pulls `split="test"` from `cais/mmlu`.
- **Why:** Using test split is standard for benchmark reporting and makes results easier to compare with papers and public baselines.
- **How it works:**
  1. Pick 5 STEM subjects.
  2. Load each subject separately.
  3. Take a small subset per subject when `full_eval: false`.
  4. Convert each row to a prompt with A/B/C/D options.
  5. Store ground truth as a single letter (`A`, `B`, `C`, or `D`).

### 2) Added ARC-Challenge benchmark loader
- **What changed:** New `load_arc_challenge()` was added and registered in `load_benchmark()`.
- **Why:** `arc_challenge` already existed in config but had no loader, which caused runtime failure (`Unknown benchmark`).
- **How it works:**
  1. Load `ai2_arc`, config `ARC-Challenge`, `split="test"`.
  2. Read each question and all choice labels/texts.
  3. Build a multiple-choice prompt in the same style as MMLU.
  4. Save `answerKey` as the target label.

### 3) Added strict multiple-choice scoring
- **What changed:** New MCQ scoring path extracts one final letter from model output and compares exactly to ground truth.
- **Why:** Previous generic scoring counted partial substring matches; for letter answers this can overestimate accuracy.
- **How it works:**
  1. Parse model text using regex to find a final choice (`A-D`).
  2. Normalize to uppercase.
  3. Compare exact label equality against the gold label.
  4. Compute accuracy from exact matches only.

### 4) Routed scoring by benchmark type
- **What changed:** `run_all()` now uses MCQ scoring for `mmlu` and `arc_challenge`, while keeping previous scoring for open-text tasks.
- **Why:** Different benchmarks require different evaluation logic; one-size-fits-all scoring is unreliable.

### 5) Basic verification completed
- **What changed:** Ran Python compile check for `evaluation/__init__.py`.
- **Why:** Confirms no syntax errors before running long benchmark jobs.
- **Command used:** `python3 -m py_compile evaluation/__init__.py`

## Phased Strategy

### Phase 1 — Core Pipeline (CPU-friendly, Jun–Jul)
| Priority | Strategy | Impact | Deps |
|----------|----------|--------|------|
| **P2** | Diverse Sampling — contrastive decoding + adaptive temperature + prompt perturbation | +8–12% | `transformers`, `datasets` |
| **P1** | Reason Verifier — lightweight NLI/rule-based scorer as QUBO diagonal term | +15–20% | `transformers`✅ |
| **P4** | Hyperparameter QUBO Search — encode inference params as binary QUBO vars | +5–8% | `pyqubo`, `dimod`, `openjij` |

### Phase 2 — QUBO Solver (Jul)
| Strategy | Detail |
|----------|--------|
| Lightweight QUBO | Semantic clustering (TF-IDF/sentence embeddings) → ≤200 vars, CPU-tractable |
| Optimized Annealing | Start with vanilla SA → counterdiabatic-inspired momentum SA on GPU |

### Phase 3 — SFT Feedback Loop (Jul–Aug, requires A100 GPU)
| Strategy | Impact |
|----------|--------|
| **P3** SFT on QUBO-Selected Traces — 2–3 epochs fine-tuning on QUBO-selected reasoning traces | +10–15% |

### Phase 4 — Polish (Aug–Sep)
| Strategy | Impact |
|----------|--------|
| **P5** HUBO Extension — triple-wise reason interactions via cubic QUBO | +5–10% on BBH |
| **P7** Multi-Benchmark Evaluation — GSM8K, BBH, StrategyQA, ARC-Challenge, MMLU | Validation |

## Benchmark Datasets & 2× Definition

The **2× target** means doubling the accuracy gain over the baseline greedy decoding:
- Baseline: Llama-3.2-3B greedy = ~62% GSM8K
- +CoT = ~68% → gain = +6%
- **Minimum 2×**: 62% + (2 × 6%) = **≥74%**
- **Ambitious target**: **>90%** (projected cumulative)

| Benchmark | Type | Baseline (greedy) | +CoT Baseline | 2× Target | SOTA Reference |
|-----------|------|-------------------|---------------|-----------|----------------|
| **GSM8K** (primary) | Grade-school math | ~60–62% | ~66–68% | **>90%** | Phi-3.5-mini: 86.2% |
| **BBH / BBEH** | Complex reasoning | ~42–45% | ~48% | **>80%** | Llama-3.1-8B: 57% |
| **StrategyQA** | Commonsense QA | ~62% | ~65% | **>80%** | Phi-3.5-mini: 74% |
| **MMLU** | General knowledge | ~63% | ~64% | **>70%** | Llama-3.1-8B: 84.6% |
| **ARC-Challenge** | Science reasoning | — | — | TBD | — |

## June 2026 — Month 1 Targets

### Focus: Core Pipeline Foundation (CPU-friendly)

| Item | Est. Effort | Impact | Why This Month |
|------|-------------|--------|----------------|
| **P2: Diverse Sampling** (`sampling.py`) | ~1 week | +8–12% | No GPU needed, highest leverage before QUBO |
| **P1: Reason Verifier** (`verifier.py`) | ~1 week | +15–20% | Biggest accuracy impact, RoBERTa runs on CPU |
| **P4: Hyperparameter QUBO** (`hyperparam_qubo.py`) | ~1 week | +5–8% | PyQUBO works on CPU, teaches QUBO mechanics |
| **Core QUBO Builder** (`qubo_builder.py`, `solver.py`) | ~1.5 weeks | Foundation | Semantic clustering + vanilla SA solver |
| **Inference Pipeline** (`inference.py`) | ~0.5 week | Integration | Ties everything together |

### End-of-June Milestone
A working end-to-end pipeline on **GSM8K** that:
1. Takes a math question → samples diverse reasons → scores with verifier → builds QUBO → selects best subset → final answer
2. Validated on a small subset (e.g., 200 GSM8K samples) on CPU
3. Baseline comparison script ready

**Key metric:** Run full pipeline on CPU with ≤200 QUBO variables and measure accuracy vs. baseline CoT.

### What we skip in June
| Skip | Needs |
|------|-------|
| **P3: SFT** | A100 GPU |
| **P5: HUBO Extension** | Complex, wait until QUBO works |
| **P6: Optimized Annealing** | GPU-accelerated |
| **P7: Full Multi-Benchmark** | Validation, not build |

## Files to Create
```
pipeline/
├── sampling.py          # Diverse sampling (contrastive decoding, adaptive temp)
├── verifier.py          # Lightweight reason correctness scorer
├── qubo_builder.py      # QUBO matrix construction + semantic clustering
├── solver.py            # Simulated annealing (vanilla + counterdiabatic)
├── inference.py         # Final prompt assembly + answer generation
├── hyperparam_qubo.py   # Hyperparameter encoding in QUBO
training/
├── sft.py              # SFT on QUBO-selected traces
evaluation/
├── benchmark.py        # Multi-benchmark evaluation runner
```

## Dependencies to Install
```
pyqubo dimod openjij datasets accelerate sentence-transformers scikit-learn
```

## Hardware Requirements

### Storage & Memory (Llama-3.2-3B)
| Loading Mode | RAM/VRAM | Disk Cache (one-time) | Total |
|-------------|----------|----------------------|-------|
| fp32 (CPU default) | ~12 GB RAM | ~6 GB | ~18 GB |
| fp16 (GPU) | ~6 GB VRAM | ~6 GB | ~12 GB |
| 8-bit | ~3.5 GB | ~6 GB | ~9.5 GB |
| 4-bit | ~2 GB | ~6 GB | ~8 GB |

Model is downloaded from HuggingFace on first run (~6GB cached in `~/.cache/huggingface/`).
Requires `huggingface-cli login` with approved HuggingFace account (Llama-3.2-3B is gated).
Swap model name in `config.yaml` for open models (Phi-3.5-mini, Qwen2.5) to skip gating.

## Cumulative Projection (GSM8K)
| Stage | Method | Accuracy | Gain |
|-------|--------|----------|------|
| 0 | Baseline greedy | ~60% | — |
| 1 | + CoT | ~66% | +6% |
| 2 | + Diverse Sampling | ~72% | +6% |
| 3 | + QUBO Reason Selection | ~78% | +6% |
| 4 | + Reason Verifier | ~84% | +6% |
| 5 | + SFT on QUBO Traces | ~88% | +4% |
| 6 | + Optimized Annealing + HUBO | ~91% | **2× achieved** |
