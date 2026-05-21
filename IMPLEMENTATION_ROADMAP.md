# Implementation Roadmap: Quantum-Inspired AI for Multi-Stage Reasoning

## Goal
Achieve **2× performance gain** over greedy decoding baseline on GSM8K (~60% → ~91%) using a QUBO-optimized SLM reasoning pipeline.

---

## System Architecture Overview

```mermaid
graph TD
    Q[Question] --> S[DiverseSampler]
    S --> |"N diverse reasoning paths"| V[ReasonVerifier]
    V --> |"Correctness scores"| QB[QUBOBuilder]
    
    subgraph "QUBO Pipeline Core"
        QB --> |"Q matrix + selected indices"| SA[SimulatedAnnealingSolver]
        SA --> |"Optimal binary state"| IP[InferencePipeline]
    end
    
    IP --> A[Final Answer]
    
    subgraph "SLM Backend"
        M1[deepseek-coder-1.5b-instruct]
    end
    
    S -.-> M1
    IP -.-> M1
    
    style Q fill:#e1f5fe
    style A fill:#c8e6c9
    style M1 fill:#fff3e0
```

---

## Current Status (as of May 22, 2026)

### Phase-wise Progress Snapshot

| Phase | Planned Scope | Current Status |
|-------|---------------|----------------|
| Phase 1 — Core Pipeline | Sampling, verifier, QUBO builder, solver, inference, hyperparameter QUBO | **Complete** |
| Phase 2 — QUBO Solver Optimization | Lightweight QUBO + optimized annealing variants | **Partially complete** (baseline SA done; advanced schedule pending) |
| Phase 3 — SFT Feedback Loop | SFT on QUBO-selected traces | **Stub stage** (scaffolded, pending GPU) |
| Phase 4 — Polish/Validation | HUBO extension + multi-benchmark validation | **Partially complete** (all 5 benchmark loaders integrated; evaluation script ready; results pending) |

### Multi-Benchmark Evaluation Pipeline

```mermaid
flowchart LR
    C[config.yaml] --> BR[BenchmarkRunner]
    BR --> G[GSM8K]
    BR --> B[BBH]
    BR --> S[StrategyQA]
    BR --> M[MMLU]
    BR --> A[ARC-Challenge]
    
    G --> R1["run_all_benchmarks.py"]
    B --> R1
    S --> R1
    M --> R1
    A --> R1
    
    R1 --> CSV[all_benchmarks_*.csv]
    R1 --> JSON[all_benchmarks_*.json]
    R1 --> MD[all_benchmarks_*.md]
    
    style C fill:#fff3e0
    style CSV fill:#e8f5e9
    style JSON fill:#e8f5e9
    style MD fill:#e8f5e9
```

### What is done in implementation

#### Pipeline Core (Phase 1 — Complete)
- **`pipeline/sampling.py`** — `DiverseSampler`: 4 prompt perturbations × random temperature (0.3–0.9) × contrastive decoding
- **`pipeline/verifier.py`** — `ReasonVerifier`: arithmetic consistency for math, NLI entailment (cross-encoder/nli-distilroberta-base) for commonsense
- **`pipeline/qubo_builder.py`** — `QUBOBuilder`: semantic clustering (all-MiniLM-L6-v2 + KMeans) → QUBO matrix with correctness diagonal + redundancy penalties
- **`pipeline/solver.py`** — `SimulatedAnnealingSolver`: multi-read SA with exponential cooling (500 iters, 100→0.01, α=0.99)
- **`pipeline/inference.py`** — `InferencePipeline`: relevance re-ranking + final answer generation
- **`pipeline/hyperparam_qubo.py`** — `HyperparameterQUBO`: one-hot encoded QUBO over parameter grids

#### Evaluation Suite (Phase 4 — Extended)
- **`evaluation/__init__.py`** — `BenchmarkRunner` with 5 production-ready benchmark loaders:

| Benchmark | Dataset | Split | Format | Accuracy |
|-----------|---------|-------|--------|----------|
| GSM8K | `gsm8k` (main) | test | Free-text math | Numeric exact match |
| BBH | `lukaemon/bbh` | test | Free-text reasoning | Substring match |
| StrategyQA | `taesiri/strategy_qa` | test | Yes/No | Boolean match |
| MMLU | `cais/mmlu` (5 STEM subjects) | test | A/B/C/D MCQ | Letter extraction |
| ARC-Challenge | `ai2_arc` (ARC-Challenge) | test | A/B/C/D MCQ | Letter extraction |

- **`evaluation/run_gsm8k_comparison.py`** — GSM8K-specific runner: greedy vs CoT vs QUBO (per-question CSV + summary JSON + Markdown report)
- **`evaluation/answer_utils.py`** — Numeric answer extraction + normalization for GSM8K gold/predicted answers
- **`scripts/run_all_benchmarks.py`** — **NEW**: unified multi-benchmark evaluation entry point. Runs all 5 configured benchmarks in sequence, produces per-question CSV + summary JSON + Markdown report with greedy/CoT/QUBO accuracy per benchmark. Supports `--subset-size`, `--full`, `--benchmarks` filters.

#### Model Configuration
- **`config/config.yaml`** — SLM switched to `deepseek-ai/deepseek-coder-1.5b-instruct` (open, ~3 GB, CPU-friendly fp32 or GPU fp16)

### Scoring Decision Logic

```mermaid
flowchart TD
    Q[Question + Prediction] --> BM{Benchmark Type?}
    BM -->|"mmlu / arc_challenge"| MCQ[MCQ Scoring Path]
    BM -->|"gsm8k / bbh / strategyqa"| OPEN[Open-Text Scoring Path]
    
    MCQ --> EX[Extract A/B/C/D via regex]
    EX --> CMP{Letter matches gold?}
    CMP -->|Yes| CORR[Correct ✅]
    CMP -->|No| INC[Incorrect ❌]
    
    OPEN --> GSM{gsm8k?}
    GSM -->|Yes| NUM[Numeric extraction + normalize]
    GSM -->|No| SUB[Case-insensitive substring match]
    NUM --> CMP2{Equal to gold?}
    SUB --> CMP2
    CMP2 -->|Yes| CORR
    CMP2 -->|No| INC
```

### Pipeline Execution Sequence per Question

```mermaid
sequenceDiagram
    participant S as DiverseSampler
    participant V as ReasonVerifier
    participant Q as QUBOBuilder
    participant Sol as SimulatedAnnealingSolver
    participant I as InferencePipeline
    participant M as SLM (deepseek-coder-1.5b)
    
    Note over S,M: Step 1: Diverse Sampling
    S->>M: Generate with perturbation 1 (temp=0.3-0.9)
    M-->>S: Reason + Answer
    S->>M: Generate with perturbation 2
    M-->>S: Reason + Answer
    S->>M: Generate with perturbation N
    M-->>S: Reason + Answer
    S-->>V: N diverse samples
    
    Note over V,Q: Step 2: Scoring
    V->>V: Extract arithmetic / NLI entailment
    V-->>Q: Samples with correctness_score
    
    Note over Q,Sol: Step 3: QUBO Construction
    Q->>Q: Embed reasons (all-MiniLM-L6-v2)
    Q->>Q: KMeans clustering (k≤50)
    Q->>Q: Pick best per cluster
    Q->>Q: Build Q matrix (≤200 vars)
    Q-->>Sol: Q matrix + selected indices
    
    Note over Sol,I: Step 4: Optimization
    Sol->>Sol: Simulated Annealing (500 iters × 2 reads)
    Sol-->>I: Optimal binary state vector
    
    Note over I,M: Step 5: Final Answer
    I->>I: Re-rank selected reasons by relevance
    I->>M: Structured prompt with top-K reasons
    M-->>I: Final answer
    I-->>I: Return answer string
```

### Latest Evaluation Note
- A first GSM8K run completed with **3 samples only** (very small sanity check). Not statistically reliable.
- Full multi-benchmark evaluation is now scripted and ready in `scripts/run_all_benchmarks.py`.
- **All 5 benchmarks** (GSM8K, BBH, StrategyQA, MMLU, ARC-Challenge) will run in a single invocation, each with greedy, CoT, and QUBO pipeline passes.

### Development Timeline

```mermaid
gantt
    title Project Development Phases
    dateFormat  YYYY-MM
    axisFormat  %Y-%m
    
    section Phase 1 — Core Pipeline
    Diverse Sampling           :done, 2026-05, 2026-06
    Reason Verifier            :done, 2026-05, 2026-06
    QUBO Builder               :done, 2026-05, 2026-06
    SA Solver                  :done, 2026-05, 2026-06
    Inference Pipeline         :done, 2026-05, 2026-06
    
    section Phase 2 — Solver Optimization
    Advanced Annealing         :active, 2026-06, 2026-07
    
    section Phase 3 — SFT Feedback
    SFT Training               :2026-07, 2026-08
    
    section Phase 4 — Validation
    Multi-Benchmark Eval       :active, 2026-05, 2026-06
    HUBO Extension             :2026-08, 2026-09
    Full Benchmark Run         :2026-06, 2026-06
```

### Next Milestone Actions
1. Run `python3 scripts/run_all_benchmarks.py --subset-size 100` to produce first multi-benchmark CSV report.
2. Run at `--subset-size 200` for more stable estimates.
3. Audit per-question CSV for extraction/format mismatches, especially MMLU and ARC-Challenge MCQ parsing.
4. Confirm stable baseline metrics (Greedy/CoT) across all 5 benchmarks before claiming QUBO gains.
5. Start Phase 2 advanced annealing experiments and Phase 3 SFT execution once compute window is allocated.

---

## Benchmark Datasets & 2× Definition

### Target Benchmarks

```mermaid
quadrantChart
    title Benchmark Landscape
    x-axis "Easy" --> "Hard"
    y-axis "Narrow" --> "Broad"
    quadrant-1 "High-Impact Targets"
    quadrant-2 "General Knowledge"
    quadrant-3 "Foundation"
    quadrant-4 "Domain-Specific"
    GSM8K: [0.3, 0.6]
    BBH: [0.7, 0.5]
    StrategyQA: [0.4, 0.7]
    MMLU: [0.5, 0.9]
    ARC-Challenge: [0.6, 0.4]
```

The **2× target** means doubling the accuracy gain over the baseline greedy decoding:
- Baseline: ~60% GSM8K greedy
- +CoT = ~66% → gain = +6%
- **Minimum 2×**: 60% + (2 × 6%) = **≥72%**
- **Ambitious target**: **>90%** (projected cumulative)

| Benchmark | Type | Split | Format | Baseline (greedy) | +CoT Baseline | 2× Target | SOTA Reference |
|-----------|------|-------|--------|-------------------|---------------|-----------|----------------|
| **GSM8K** | Grade-school math | test | Free-text numeric | ~60–62% | ~66–68% | **>90%** | Phi-3.5-mini: 86.2% |
| **BBH** | Complex reasoning | test | Free-text | ~42–45% | ~48% | **>80%** | Llama-3.1-8B: 57% |
| **StrategyQA** | Commonsense QA | test | Yes/No | ~62% | ~65% | **>80%** | Phi-3.5-mini: 74% |
| **MMLU** (5 STEM subjects) | General knowledge | test | 4-way MCQ | ~40–45% | ~42–46% | **>55%** | Llama-3.1-8B: 84.6% |
| **ARC-Challenge** | Science reasoning | test | 4-way MCQ | ~35–40% | ~38–42% | **>50%** | GPT-3.5: 85% |

> **Note:** MMLU and ARC-Challenge baselines shown are for 1.5B-class models on the 5-subject subset only. Full 57-subject MMLU typically reports higher baselines for larger models.

### Benchmark Dataset Details

```mermaid
erDiagram
    GSM8K {
        string question "Grade-school math word problem"
        string answer "Numeric answer after ####"
    }
    BBH {
        string input "Complex reasoning task"
        string target "Expected output"
    }
    StrategyQA {
        string question "Yes/no commonsense question"
        bool answer "True or False"
    }
    MMLU {
        string question "STEM knowledge question"
        list choices "4 options A-D"
        int answer "Correct index 0-3"
    }
    ARC-Challenge {
        string question "Science question"
        dict choices "Labels + texts"
        string answerKey "Correct label A-D"
    }
```

### MMLU Subject Coverage (5 STEM subjects)

| Subject | Category | Questions (test) | Topics |
|---------|----------|-----------------|--------|
| `abstract_algebra` | STEM - Math | ~100 | Groups, rings, fields, linear algebra |
| `college_computer_science` | STEM - CS | ~100 | Algorithms, data structures, theory |
| `college_physics` | STEM - Physics | ~100 | Mechanics, electromagnetism, thermodynamics |
| `electrical_engineering` | STEM - Engineering | ~100 | Circuits, signals, systems, electronics |
| `machine_learning` | STEM - AI/ML | ~112 | Supervised, unsupervised, neural nets, probability |

```mermaid
pie title MMLU Subject Composition
    "abstract_algebra" : 20
    "college_computer_science" : 20
    "college_physics" : 20
    "electrical_engineering" : 20
    "machine_learning" : 22
```

---

## Run Script: Multi-Benchmark Evaluation

### How to Run

```bash
# Quick test (50 samples per benchmark)
python3 scripts/run_all_benchmarks.py --subset-size 50

# Full run (200 samples per benchmark, default)
python3 scripts/run_all_benchmarks.py

# Run all benchmarks on full datasets
python3 scripts/run_all_benchmarks.py --full

# Run specific benchmarks only
python3 scripts/run_all_benchmarks.py --benchmarks gsm8k mmlu

# Custom output directory
python3 scripts/run_all_benchmarks.py --output-dir ./results
```

### Output Files

```text
outputs/
├── all_benchmarks_{timestamp}.csv      # Per-question predictions (all methods)
├── all_benchmarks_{timestamp}.json     # Summary accuracy per benchmark
└── all_benchmarks_{timestamp}.md       # Human-readable Markdown report
```

### CSV Column Structure

```mermaid
flowchart LR
    subgraph CSV Columns
        BM[benchmark] --> ID[id]
        ID --> Q[question]
        Q --> G[gold]
        G --> PG[pred_greedy]
        PG --> PC[pred_cot]
        PC --> PQ[pred_qubo]
        PQ --> CG[correct_greedy]
        CG --> CC[correct_cot]
        CC --> CQ[correct_qubo]
        CQ --> RG[runtime_greedy_s]
        RG --> RC[runtime_cot_s]
        RC --> RQ[runtime_qubo_s]
    end
```

### Per-Question Pipeline Flow (run_all_benchmarks.py)

```mermaid
flowchart TD
    Q[Question] --> Greedy[Greedy Baseline]
    Q --> CoT[CoT Baseline]
    Q --> QUBO[QUBO Pipeline]
    
    Greedy --> GE["generate_answer(prompt='Question: ...\\nAnswer:')"]
    CoT --> CE["generate_answer(prompt='Let\\'s think step by step...')"]
    
    QUBO --> SAMP[DiverseSampler.sample<br/>4 perturbations × 2 answers = 8 samples]
    SAMP --> VER[ReasonVerifier.score_batch<br/>Arithmetic consistency / NLI]
    VER --> QBLD[QUBOBuilder.build_qubo<br/>Embed → Cluster → Q matrix]
    QBLD --> SOLV[SimulatedAnnealingSolver.solve<br/>500 iters × 2 reads]
    SOLV --> INF[InferencePipeline.run<br/>Re-rank → Build prompt → Generate]
    
    GE --> EX[extract_answer]
    CE --> EX
    INF --> EX
    EX --> CMP2{is_correct?}
    CMP2 --> CSV[Write to CSV row]
    
    style Q fill:#e1f5fe
    style CSV fill:#c8e6c9
```

---

## Phased Strategy

### Phase 1 — Core Pipeline (Complete)
| Priority | Strategy | Impact | Deps |
|----------|----------|--------|------|
| **P2** | Diverse Sampling — contrastive decoding + adaptive temperature + prompt perturbation | +8–12% | `transformers`, `datasets` ✅ |
| **P1** | Reason Verifier — lightweight NLI/rule-based scorer as QUBO diagonal term | +15–20% | `transformers` ✅ |
| **P4** | Hyperparameter QUBO Search — encode inference params as binary QUBO vars | +5–8% | `pyqubo`, `dimod`, `openjij` |

### Phase 2 — QUBO Solver (In Progress)
| Strategy | Detail |
|----------|--------|
| Lightweight QUBO | Semantic clustering (TF-IDF/sentence embeddings) → ≤200 vars, CPU-tractable ✅ |
| Optimized Annealing | Vanilla SA → counterdiabatic-inspired momentum SA on GPU |

### Phase 3 — SFT Feedback Loop (Pending GPU)
| Strategy | Impact |
|----------|--------|
| SFT on QUBO-Selected Traces — 2–3 epochs fine-tuning on QUBO-selected reasoning traces | +10–15% |

### Phase 4 — Polish & Validation (In Progress)
| Strategy | Impact | Status |
|----------|--------|--------|
| Multi-Benchmark Evaluation — GSM8K, BBH, StrategyQA, ARC-Challenge, MMLU | Validation | ✅ Scripted, ready to run |
| MMLU Loader (5 STEM subjects) | Breadth | ✅ Integrated |
| ARC-Challenge Loader | Breadth | ✅ Integrated |
| MCQ-Specific Scoring (letter extraction via regex) | Accuracy | ✅ Implemented |
| HUBO Extension — triple-wise reason interactions via cubic QUBO | +5–10% on BBH | ⏳ Pending |

---

## Detailed Change Log

### 1) Added MMLU Benchmark Loader
- **File:** `evaluation/__init__.py` — `load_mmlu()`
- **Dataset:** `cais/mmlu`, test split, 5 STEM subjects
- **Format:** Each question converted to `Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:`
- **Answers:** Mapped from index (0-3) to letter (A-D)

### 2) Added ARC-Challenge Benchmark Loader
- **File:** `evaluation/__init__.py` — `load_arc_challenge()`
- **Dataset:** `ai2_arc`, `ARC-Challenge` config, test split
- **Format:** `Question: ...\nA. {text}\nB. {text}\n...\nAnswer:`
- **Answers:** `answerKey` field (A/B/C/D)

### 3) Added MCQ-Specific Scoring
- **File:** `evaluation/__init__.py` — `compute_accuracy_mcq()`, `_extract_mcq_choice()`
- **Regex extraction:** Parses `ANSWER: A` pattern or standalone `A-D` from model output
- **Routing:** `run_all()` uses MCQ path for `mmlu` and `arc_challenge`, standard path for others

### 4) Created Unified Multi-Benchmark Entry Point
- **File:** `scripts/run_all_benchmarks.py`
- **Behavior:** Loads all 5 benchmarks from config, runs greedy/CoT/QUBO per question, outputs CSV+JSON+MD
- **CLI:** `--subset-size`, `--full`, `--benchmarks`, `--output-dir`

### 5) Switched SLM to deepseek-coder-1.5b-instruct
- **File:** `config/config.yaml`
- **Change:** `"Qwen/Qwen2.5-1.5B-Instruct"` → `"deepseek-ai/deepseek-coder-1.5b-instruct"`
- **Why:** Open model (no gating), strong reasoning capabilities, comparable size (~3 GB)

---

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

```mermaid
journey
    title GSM8K Accuracy Progression
    section Current
      Baseline Greedy: 3: Current
      +CoT: 3: Current
    section Pipeline
      +Diverse Sampling: 4: Projected
      +QUBO Selection: 5: Projected
      +Reason Verifier: 6: Projected
    section Future
      +SFT: 7: Projected
      +Optimized Annealing + HUBO: 9: Target
```

---

## Dependencies

```
torch transformers datasets sentence-transformers scikit-learn
pyqubo dimod openjij accelerate pyyaml
```

## Hardware Requirements (deepseek-coder-1.5b-instruct)

| Loading Mode | RAM/VRAM | Disk Cache | Total |
|-------------|----------|-----------|-------|
| fp32 (CPU) | ~6 GB RAM | ~3 GB | ~9 GB |
| fp16 (GPU) | ~3 GB VRAM | ~3 GB | ~6 GB |

- Fully open model — no HuggingFace gating or login required
- Downloads automatically on first run to `~/.cache/huggingface/`
- Auto-detects CUDA GPU at runtime; falls back to CPU gracefully
