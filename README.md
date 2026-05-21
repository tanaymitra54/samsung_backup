# Quantum-Inspired Annealing for Multi-Stage Reasoning

> A modular reasoning framework that generates diverse chains-of-thought, scores them, encodes selection as a QUBO optimization problem, and composes final answers using selected high-value reasoning traces.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active_Research-blue)
![Optimization](https://img.shields.io/badge/Optimization-QUBO-purple)

---

## 1) Executive Summary

This project implements a **quantum-inspired inference-time reasoning system** for small language models (SLMs). Instead of trusting a single generated chain of thought, the pipeline:

1. samples many diverse candidate reasoning paths,
2. verifies and scores candidate quality,
3. builds a QUBO objective balancing correctness and diversity,
4. solves the optimization via simulated annealing,
5. synthesizes a final answer from selected traces.

### Why this matters
- Standard decoding is brittle for multi-step reasoning.
- Majority-vote style methods improve robustness but can remain redundant.
- Optimization-aware trace selection introduces a principled way to trade off **quality vs diversity**.

### Core novelty
- Treats reasoning-trace selection as an explicit combinatorial optimization problem.
- Integrates lightweight verifier scores and semantic similarity penalties into a unified QUBO matrix.
- Supports benchmark-oriented evaluation workflows (GSM8K, BBH, StrategyQA, ARC-Challenge, MMLU).

---

## 2) Motivation and Background

Modern SLM reasoning often fails due to local decoding errors, shallow heuristics, or brittle intermediate steps. Inference-time ensembling helps, but naively aggregating many traces can over-index on similar errors. This project is motivated by a simple observation:

> The best final answer often emerges from a **small, diverse, high-quality subset** of reasoning traces.

This repository explores a practical mechanism for subset selection using QUBO-style objectives, then uses selected traces to guide final generation.

```mermaid
journey
    title Why the Pipeline Exists
    section Typical LLM Flow
      One-pass decoding: 2: User
      Fragile intermediate steps: 2: Researcher
      Inconsistent final accuracy: 2: Engineer
    section This Project's Flow
      Multi-sample reasoning generation: 4: Pipeline
      Verifier-aware scoring: 4: Pipeline
      QUBO-based subset optimization: 5: Pipeline
      Better trace composition for final answer: 4: Pipeline
```

---

## 3) Feature Set

### Functional features
- Multi-template, multi-temperature reasoning sample generation.
- Heuristic + NLI-based reasoning verification.
- Semantic embedding and clustering for trace compression.
- QUBO matrix construction with correctness/diversity terms.
- Simulated annealing solver with configurable schedule.
- Final-answer generation from selected traces.
- Benchmark runners for GSM8K comparisons and multi-benchmark evaluation.

### Technical features
- YAML-driven configuration (`config/config.yaml`).
- GPU auto-detection for model modules (`cuda` when available).
- Modular code design by subsystem (`pipeline`, `evaluation`, `scripts`, `training`).
- Export paths for CSV/JSON/Markdown benchmark reports.

### Capability comparison

| Capability | Baseline Greedy | Plain CoT | This Pipeline |
|---|---:|---:|---:|
| Multiple reasoning traces | No | Limited | Yes |
| Verification step | No | No | Yes |
| Diversity-aware selection | No | No | Yes (QUBO) |
| Optimization objective | No | No | Explicit |
| Benchmark reporting | Basic | Basic | Structured CSV/JSON/MD |

---

## 4) System Architecture

### 4.1 High-level architecture

```mermaid
flowchart LR
    U[User Question] --> S[DiverseSampler]
    S --> V[ReasonVerifier]
    V --> Q[QUBOBuilder]
    Q --> A[SimulatedAnnealingSolver]
    A --> I[InferencePipeline]
    I --> O[Final Answer]
```

### 4.2 Module interaction graph

```mermaid
graph TD
    CFG[config/config.yaml] --> SM[pipeline/sampling.py]
    CFG --> VR[pipeline/verifier.py]
    CFG --> QB[pipeline/qubo_builder.py]
    CFG --> SV[pipeline/solver.py]
    CFG --> IN[pipeline/inference.py]
    SM --> VR --> QB --> SV --> IN
    EV1[evaluation/run_gsm8k_comparison.py] --> SM
    EV1 --> VR
    EV1 --> QB
    EV1 --> SV
    EV1 --> IN
    EV2[evaluation/__init__.py] --> OUT[Benchmark Metrics]
```

### 4.3 Runtime sequence

```mermaid
sequenceDiagram
    participant User
    participant Sampler
    participant Verifier
    participant QUBO
    participant Solver
    participant Inference

    User->>Sampler: question
    Sampler->>Sampler: generate candidate traces
    Sampler->>Verifier: samples
    Verifier->>Verifier: score correctness
    Verifier->>QUBO: scored traces
    QUBO->>QUBO: build Q matrix + var mapping
    QUBO->>Solver: Q
    Solver->>Solver: simulated annealing
    Solver->>Inference: selected indices
    Inference->>Inference: rank selected reasons by relevance
    Inference-->>User: final answer
```

### 4.4 Data-flow model

```mermaid
flowchart TD
    QN[Question text] --> P1[Prompt perturbations]
    P1 --> P2[Generated samples]
    P2 --> P3[Reason + answer parsing]
    P3 --> P4[Verifier scoring]
    P4 --> P5[Embeddings]
    P5 --> P6[Cluster representatives]
    P6 --> P7[QUBO matrix]
    P7 --> P8[Binary state solution]
    P8 --> P9[Selected reasons]
    P9 --> P10[Final prompt synthesis]
    P10 --> ANS[Generated answer]
```

### 4.5 Dependency graph

```mermaid
classDiagram
    class DiverseSampler {
      +sample(question)
      +generate_with_contrastive_decode()
      +perturb_prompt()
    }
    class ReasonVerifier {
      +score_batch(samples)
      +verify_math()
      +verify_commonsense()
    }
    class QUBOBuilder {
      +build_qubo(samples)
      +_embed_reasons()
      +_cluster_reasons()
    }
    class SimulatedAnnealingSolver {
      +solve(Q)
      +_compute_energy(state,Q)
    }
    class InferencePipeline {
      +run(question,indices,samples)
      +build_final_prompt()
      +generate_answer()
    }

    DiverseSampler --> ReasonVerifier
    ReasonVerifier --> QUBOBuilder
    QUBOBuilder --> SimulatedAnnealingSolver
    SimulatedAnnealingSolver --> InferencePipeline
```

---

## 5) End-to-End Workflow

### Step-by-step processing
1. **Input question** enters sampling stage.
2. **Prompt perturbations** create diverse generation contexts.
3. **Candidate reasons/answers** are parsed from model output.
4. **Verifier scoring** estimates correctness confidence.
5. **Embedding + clustering** reduce redundancy and control variable count.
6. **QUBO matrix** encodes quality/diversity tradeoff.
7. **Annealing solver** finds low-energy binary selection state.
8. **Reason ranking** by semantic relevance to question.
9. **Final prompt composition** from selected traces.
10. **Final answer generation** returned.

### Decision logic

```mermaid
flowchart TD
    A[Generate samples] --> B{Any samples?}
    B -- No --> C[Return empty answer fallback]
    B -- Yes --> D[Score + build QUBO]
    D --> E[Solve QUBO]
    E --> F{Selected indices empty?}
    F -- Yes --> G[Fallback to first K samples]
    F -- No --> H[Use selected indices]
    G --> I[Final inference]
    H --> I
```

### Pipeline state transitions

```mermaid
stateDiagram-v2
    [*] --> Sampling
    Sampling --> Verifying
    Verifying --> BuildingQUBO
    BuildingQUBO --> Solving
    Solving --> ComposingAnswer
    ComposingAnswer --> Completed
    Sampling --> Failed: no samples
    Solving --> Fallback: empty selection
    Fallback --> ComposingAnswer
```

---

## 6) Technical Deep Dive

### `pipeline/sampling.py` - DiverseSampler
- **Purpose:** Generate varied reasoning candidates.
- **Design choices:** multiple prompt templates + random temperatures.
- **Output schema:** `{reason, answer, diversity_score, temperature, prompt_template}`.
- **Tradeoff:** diversity improves coverage but increases inference cost.

### `pipeline/verifier.py` - ReasonVerifier
- **Purpose:** Assign quality estimates to generated traces.
- **Math mode:** extracts arithmetic expressions and checks consistency.
- **Commonsense mode:** NLI entailment score via cross-encoder.
- **Tradeoff:** lightweight heuristics are fast, but not equivalent to symbolic proof checking.

### `pipeline/qubo_builder.py` - QUBOBuilder
- **Purpose:** Convert scored traces into optimization objective.
- **Diagonal terms:** favor high correctness (negative energy reward).
- **Off-diagonal terms:** penalize semantic similarity to encourage diversity.
- **Complexity:** similarity matrix is quadratic in selected variable count.

### `pipeline/solver.py` - SimulatedAnnealingSolver
- **Purpose:** Approximate low-energy binary assignment.
- **Method:** iterative bit flips with Metropolis acceptance.
- **Configuration:** initial/final temperature, cooling rate, iterations, num reads.
- **Tradeoff:** no global optimality guarantee; practical and hardware-light.

### `pipeline/inference.py` - InferencePipeline
- **Purpose:** turn selected reasons into final answer.
- **Logic:** rank selected reasons by embedding relevance, build final prompt, decode deterministically.
- **Tradeoff:** prompt length vs context quality.

### `evaluation/__init__.py` - BenchmarkRunner
- **Purpose:** unified benchmark loading and scoring.
- **Benchmarks:** GSM8K, BBH, StrategyQA, ARC-Challenge, MMLU.
- **Scoring modes:** relaxed text-match for open responses; strict MCQ extraction for A/B/C/D tasks.

---

## 7) Algorithms and Methodology

### 7.1 Objective formulation
Given binary selection vector `x in {0,1}^n` and QUBO matrix `Q`, optimize:

`min E(x) = x^T Q x`

Where:
- `Q_ii` captures quality reward (higher correctness -> lower diagonal energy).
- `Q_ij` captures pairwise redundancy penalty via cosine similarity.

### 7.2 Practical decomposition

```mermaid
flowchart LR
    C[Correctness score] --> D[Diagonal terms Qii]
    S[Semantic similarity] --> O[Off-diagonal Qij]
    D --> Q[QUBO Matrix Q]
    O --> Q
    Q --> X[Annealing solution x]
```

### 7.3 Simulated annealing acceptance rule
For candidate state transition with energy change `DeltaE` and temperature `T`:

- always accept if `DeltaE < 0`
- otherwise accept with probability `exp(-DeltaE / T)`

Cooling schedule (current implementation):

`T_{k+1} = max(T_final, T_k * cooling_rate)`

### 7.4 Method comparison axis

```mermaid
quadrantChart
    title Reason Selection Methods (Conceptual)
    x-axis Low Redundancy --> High Redundancy
    y-axis Low Accuracy Signal --> High Accuracy Signal
    quadrant-1 High signal, high redundancy
    quadrant-2 High signal, low redundancy
    quadrant-3 Low signal, low redundancy
    quadrant-4 Low signal, high redundancy
    QUBO: [0.35, 0.8]
    Self-Consistency: [0.75, 0.55]
    Random Subset: [0.6, 0.3]
```

---

## 8) Folder Structure

```text
.
├── config/
│   └── config.yaml
├── evaluation/
│   ├── __init__.py
│   ├── answer_utils.py
│   └── run_gsm8k_comparison.py
├── pipeline/
│   ├── __init__.py
│   ├── sampling.py
│   ├── verifier.py
│   ├── qubo_builder.py
│   ├── solver.py
│   ├── inference.py
│   └── hyperparam_qubo.py
├── scripts/
│   ├── generate_comparison.py
│   └── evaluate_accuracy.py
├── training/
│   ├── __init__.py
│   └── sft.py
├── outputs/
├── cache/
├── requirements.txt
├── IMPLEMENTATION_ROADMAP.md
└── README.md
```

---

## 9) Installation and Setup

### Prerequisites
- Python 3.10+
- Optional GPU for faster model inference
- `pip` or equivalent environment manager

### Quick start

```bash
git clone <your-repo-url>
cd Quantum-Annealing-SLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration
Edit `config/config.yaml` for model, sampling, solver, and evaluation parameters.

---

## 10) Usage

### Run GSM8K baseline vs QUBO comparison

```bash
python3 evaluation/run_gsm8k_comparison.py --subset-size 100 --output-dir outputs
```

### Run method comparison utility

```bash
python3 scripts/generate_comparison.py
```

### Run small local accuracy harness

```bash
python3 scripts/evaluate_accuracy.py
```

### Programmatic benchmark runner

```python
from evaluation import BenchmarkRunner

runner = BenchmarkRunner(config_path="config/config.yaml")
results = runner.run_all(pipeline_fn=lambda q: "A")
print(results)
```

---

## 11) Reproducibility Notes

- Keep model checkpoints and tokenizer versions fixed.
- Record `config/config.yaml` snapshot for each run.
- Save outputs (`csv`, `json`, `md`) with timestamps.
- Compare runs at consistent subset sizes before interpreting trends.

```mermaid
gitGraph
   commit id: "baseline config"
   branch experiments
   commit id: "sampler tuning"
   commit id: "solver tuning"
   commit id: "benchmark run"
   checkout main
   merge experiments
   commit id: "report"
```

---

## 12) Performance and Evaluation Design

### Current benchmark targets

| Benchmark | Task Type | Status |
|---|---|---|
| GSM8K | math reasoning | Implemented |
| BBH | broad reasoning | Implemented |
| StrategyQA | commonsense QA | Implemented |
| ARC-Challenge | science MCQ | Implemented |
| MMLU (STEM subset) | MCQ reasoning | Implemented |

### Evaluation output artifacts
- Per-sample predictions CSV
- Aggregate summary JSON
- Human-readable report Markdown

```mermaid
pie title Evaluation Artifact Mix
    "Per-sample CSV" : 50
    "Summary JSON" : 25
    "Readable report MD" : 25
```

---

## 13) Roadmap

```mermaid
gantt
    title Project Roadmap (Conceptual)
    dateFormat  YYYY-MM-DD
    section Core Pipeline
    Sampling and Verifier          :done,    a1, 2026-05-01, 20d
    QUBO Builder + SA Solver       :done,    a2, 2026-05-10, 18d
    section Evaluation
    GSM8K Comparison Runner        :done,    b1, 2026-05-15, 10d
    Multi-benchmark Integration    :active,  b2, 2026-05-18, 15d
    section Future
    Advanced annealing schedules   :         c1, 2026-06-10, 20d
    SFT feedback loop execution    :         c2, 2026-07-01, 30d
```

---

## 14) Risk and Tradeoff Analysis

| Area | Benefit | Risk | Mitigation |
|---|---|---|---|
| Diverse sampling | Better search coverage | Higher latency | tune sample count |
| Verifier scoring | Better trace quality signal | Score noise | combine math + NLI signals |
| QUBO selection | principled optimization | quadratic pairwise costs | cap variables via clustering |
| SA optimization | fast approximate solution | local minima | multi-read runs, schedule tuning |

---

## 15) Contributor Guide

### Recommended extension points
- New verifier signals (symbolic math checks, tool calls).
- Alternative QUBO/HUBO formulations.
- Better solver backends (tabu, hybrid, annealer APIs).
- Dataset adapters + benchmark-specific answer normalization.

### Contribution workflow
1. Create feature branch.
2. Keep config diffs explicit.
3. Add experiment script and reproducibility notes.
4. Include output artifact samples where applicable.

---

## 16) Project Maturity Snapshot

```mermaid
flowchart LR
    R1[Prototype] --> R2[Research Validation]
    R2 --> R3[Benchmark Stabilization]
    R3 --> R4[Training Feedback Loop]
    R4 --> R5[Production Hardening]
```

```mermaid
pie title Maturity Distribution
    "Core pipeline" : 40
    "Evaluation stack" : 30
    "Training loop" : 10
    "Optimization research" : 20
```

---

## 17) Acknowledgements

- Hugging Face ecosystem (`transformers`, `datasets`)
- Sentence-Transformers for semantic embeddings
- Open-source optimization and scientific Python stack

---

## 18) Citation

If you use this project in reports or demos, cite as:

```bibtex
@misc{quantum_annealing_slm_2026,
  title        = {Quantum-Inspired Annealing for Multi-Stage Reasoning},
  author       = {Project Contributors},
  year         = {2026},
  note         = {Research engineering project repository}
}
```

---

## 19) License

This repository currently uses project-specific internal governance. Add a `LICENSE` file (for example MIT/Apache-2.0) before public open-source release.
