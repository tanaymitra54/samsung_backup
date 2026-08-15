# Quantum-Inspired Annealing for Multi-Stage Reasoning
## Codebase Architecture & Teammate Documentation Guide (`abhyuday` branch)

This document provides a comprehensive technical overview of the codebase architecture, system components, data flow, hyperparameter tuning, training pipelines, and specific additions introduced in the `abhyuday` branch.

---

## 1. High-Level Architecture Overview

The system addresses multi-stage mathematical and logical reasoning by combining **Diverse CoT Sampling**, **NLI/Process Verification**, **QUBO Matrix Formulation**, **Quantum-Inspired Simulated Annealing Selection**, and **Preference Optimization (SFT + DPO)**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DIVERSE SAMPLING (pipeline/sampling.py)                                 │
│    Prompt Perturbations (4 framings) × Temp Randomization [0.3, 0.9]        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. REASON VERIFICATION (pipeline/verifier.py)                               │
│    Scoring: α·Answer_Match + β·Arithmetic_Consistency + γ·Cross_Consensus   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. QUBO FORMULATION (pipeline/qubo_builder.py)                              │
│    Diagonal: Quality Scores | Off-Diagonal: Cosine Sim + Answer Agreement  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. SIMULATED ANNEALING SOLVER (pipeline/solver.py)                          │
│    Solves x* = argmax (x^T Q x) -> Selects non-redundant optimal subset    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. FINAL ANSWER INFERENCE / FINE-TUNING PIPELINE                            │
│    • CoT Scaffolding Inference (pipeline/inference.py)                     │
│    • QUBO Data Generation (scripts/generate_training_data.py)               │
│    • Fast LoRA SFT Training (scripts/fast_finetune_pipeline.py)             │
│    • Preference Pair Extraction (scripts/generate_preference_pairs.py)      │
│    • Direct Preference Optimization DPO (scripts/run_dpo.py)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & Module Breakdown

```
.
├── config/
│   ├── config.yaml               # Active production system configuration
│   ├── config_fast.yaml          # Lightweight profile for fast iteration & subset eval
│   ├── config_original.yaml      # Reference baseline configuration profile
│   └── best_qubo_params.yaml     # Best QUBO objective parameters from grid search
├── evaluation/
│   ├── __init__.py               # Benchmark dataset loaders & execution runner
│   ├── answer_utils.py           # MCQ & numerical answer extraction heuristics
│   └── run_gsm8k_comparison.py   # Baseline vs QUBO comparison evaluator
├── pipeline/
│   ├── __init__.py
│   ├── device_utils.py           # PyTorch CUDA device resolution & GPU boundary guards
│   ├── hyperparam_qubo.py        # QUBO hyperparameter abstraction layers
│   ├── inference.py              # Final answer CoT generation + LoRA adapter merging
│   ├── qubo_builder.py           # QUBO Q-matrix formulator (quality + diversity penalties)
│   ├── sampling.py               # Diverse trace sampler with prompt perturbations
│   ├── solver.py                 # Quantum-inspired Simulated Annealing solver
│   └── verifier.py               # NLI entailment & mathematical consistency verifier
├── scripts/
│   ├── diagnose_eval.py          # Evaluation log diagnostic tool & error sampler
│   ├── download_models.py        # Offline model weights & tokenizer caching utility
│   ├── evaluate_all.py           # Automated evaluation driver (Base vs SFT vs DPO)
│   ├── fast_finetune_pipeline.py # End-to-end fast QLoRA SFT training pipeline
│   ├── generate_preference_pairs.py # DPO chosen/rejected preference pair generator
│   ├── generate_training_data.py # QUBO-curated training dataset generator
│   ├── run_all_benchmarks.py     # Resumable evaluation suite with question caching
│   ├── run_dpo.py                # Direct Preference Optimization (DPO) trainer
│   └── tune_qubo_params.py       # 81-combination QUBO hyperparameter grid search
├── training/
│   └── sft.py                    # Standard SFT training loop abstractions
├── CODEBASE_DOCUMENTATION.md     # Teammate codebase reference (this document)
└── test_hf.py                    # Hugging Face environment authentication test
```

---

## 3. Summary of Key Additions in `abhyuday` Branch

| Component / File | Description of Additions in `abhyuday` Branch |
| :--- | :--- |
| **QUBO Grid Search** (`scripts/tune_qubo_params.py`) | Evaluates 81 combinations of QUBO objective weights (`penalty_weight`, `diversity_bonus`, `cardinality_penalty`, `answer_agree_weight`) over cached GSM8K reasoning pools to discover optimal coefficients (`config/best_qubo_params.yaml`). |
| **LoRA Adapter Fusion** (`pipeline/inference.py`) | Added dynamic LoRA adapter loading and weight fusion (`PeftModel.from_pretrained` + `merge_and_unload`) via `adapter_path` parameter or `QUBO_ADAPTER_PATH` env variable. Enabled KV caching (`use_cache=True`). |
| **Fast SFT Pipeline** (`scripts/fast_finetune_pipeline.py`) | Integrated QLoRA Supervised Fine-Tuning pipeline leveraging Hugging Face `trl` (`SFTTrainer`) with 4-bit quantization support and automatic benchmark evaluation dispatch. |
| **DPO Training Pipeline** (`scripts/run_dpo.py`) | Implemented Direct Preference Optimization using `DPOTrainer` with dynamic version-compatible kwargs across `trl` releases and FSDP compatibility patches. |
| **Preference Pair Generator** (`scripts/generate_preference_pairs.py`) | Extracts chosen vs rejected reasoning paths from SFT candidate pools (`full_chains_pool`) using a configurable verifier score margin (default `0.3`). |
| **Resumable Benchmarking** (`scripts/run_all_benchmarks.py`) | Added question-level partial caching and resume capability to benchmark evaluation scripts, preventing loss of progress on long runs. |
| **Robust MCQ Extraction** (`evaluation/answer_utils.py`) | Introduced 5-stage heuristic regex parser for MCQ answers (explicit tags, choice phrase matching, 300-char conclusion scan, standalone letter lookback, fallback). |
| **CUDA Device Boundary Guard** (`pipeline/device_utils.py`) | Added GPU index validation (`idx >= count` fallback to `cuda:0`) for single and multi-GPU safety. |

---

## 4. End-to-End Execution Workflow for Teammates

### Step 1: Download Model Weights (Offline Support)
```bash
python scripts/download_models.py --config config/config.yaml
```

### Step 2: Tune QUBO Parameters (Grid Search)
```bash
python scripts/tune_qubo_params.py --num-questions 300 --output-dir results/
```
*Saves optimal parameters to `config/best_qubo_params.yaml`.*

### Step 3: Generate Training Data with QUBO Curation
```bash
python scripts/generate_training_data.py --config config/config.yaml --qubo-params config/best_qubo_params.yaml
```
*Outputs `data/finetune_train.jsonl` and `data/finetune_val.jsonl`.*

### Step 4: Run Supervised Fine-Tuning (SFT)
```bash
python scripts/fast_finetune_pipeline.py --quick
```
*Saves trained LoRA adapter to `checkpoints/qubo-sft-fast/final_adapter`.*

### Step 5: Generate DPO Preference Pairs
```bash
python scripts/generate_preference_pairs.py --input data/finetune_train.jsonl --output data/preference_pairs.jsonl --min-margin 0.3
```

### Step 6: Run Direct Preference Optimization (DPO)
```bash
python scripts/run_dpo.py --config config/config.yaml --data-file data/preference_pairs.jsonl --output-dir checkpoints/dpo-run --epochs 3
```

### Step 7: Run Comprehensive Benchmark Evaluation
```bash
python scripts/evaluate_all.py --sft-adapter checkpoints/qubo-sft-fast/final_adapter --dpo-adapter checkpoints/dpo-run/final_adapter --eval-subset 100 --benchmarks gsm8k mmlu bbh
```

---

## 5. Contact & Maintainers

- **Primary Contributor**: Abhyuday (`abhyuday` branch)
- **Target Remote Repository**: `samsung` (`https://github.ecodesamsung.com/SRIB-PRISM/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning.git`)
