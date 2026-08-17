# Quantum-Inspired Annealing for Multi-Stage Reasoning
## Codebase Architecture & Teammate Documentation Guide (`abhyuday` branch)

This document provides a comprehensive technical overview of the codebase architecture, system components, data flow, hyperparameter tuning, training pipelines, and specific additions introduced in the `abhyuday` branch.

---

## 1. High-Level Architecture Overview

This section documents the **architecture finalised with the Samsung mentors**. All
work from this point forward follows this diagram. The canonical implementation of
the full cycle is `scripts/run_architecture_loop.py`.

The system addresses multi-stage mathematical and logical reasoning by combining
**Diverse CoT Sampling**, **NLI/Process Verification**, **QUBO Matrix Formulation**,
**Quantum-Inspired Simulated Annealing Selection**, and **QLoRA SFT**, wired as a
closed loop that feeds each round's fine-tuned model back into the next round.

```
                            query + initial prompt
                                      │
                                      ▼
                       ┌──────────────────────────────┐
              ┌───────▶│  SLM generates multiple paths│◀────────────┐
              │        │      pipeline/sampling.py    │             │
              │        └──────────────┬───────────────┘             │
              │                       ▼                             │
              │        ┌──────────────────────────────┐             │
              │        │  QUBO mapping + Quantum      │             │
              │        │  solver                      │             │  "re run
              │        │  pipeline/qubo_builder.py    │             │  benchmark"
              │        │  pipeline/solver.py          │             │
              │        └──────────────┬───────────────┘             │
              │                       ▼                             │
              │        ┌──────────────────────────────┐             │
              │        │  Select the best reasoning   │             │
              │        └──────────────┬───────────────┘             │
              │                       ▼                             │
              │        ┌──────────────────────────────┐             │
              │        │  generate final prompt       │             │
              │        │  inference.build_final_prompt│             │
              │        └──────────────┬───────────────┘             │
              │                       ▼                             │
              │        ┌──────────────────────────────┐             │
              │        │        QLoRA SFT             │◀──────┐     │
              │        │ scripts/fast_finetune_*.py   │       │     │
              │        └──────────────┬───────────────┘       │     │
              │                       ▼                       │     │
              │        ┌──────────────────────────────┐   "Re-train │
              │        │ SLM Qwen 3B + LoRA Adapters  │    QLoRA"   │
              │        │  pipeline/inference.py       │       │     │
              │        └──────────────┬───────────────┘       │     │
              │                       ▼                       │     │
              │        ┌──────────────────────────────┐       │     │
              │        │        final answer          │       │     │
              │        └──────────────┬───────────────┘       │     │
              │                       ▼                       │     │
              │        ┌──────────────────────────────┐       │     │
              └────────┤      Outputs directory       ├───────┘─────┘
                       └──────────────────────────────┘
```

**Note on the verifier.** `pipeline/verifier.py` is not drawn as its own box in the
mentor diagram because it sits inside "QUBO mapping": it produces the per-chain
quality scores that become the diagonal of the QUBO matrix. It runs between sampling
and QUBO construction.

### The two feedback arrows

These are what make the architecture a loop rather than a single pass, and they are
implemented in `scripts/run_architecture_loop.py`:

| Arrow | Meaning | Implementation |
| :--- | :--- | :--- |
| **"re run benchmark"** | Round N+1 generates its candidate reasoning paths using the model produced by round N, not the base SLM. | `generate_training_data.py --adapter-path <round N adapter>` merges the LoRA before sampling. |
| **"Re-train QLoRA"** | Each round trains a new adapter on the freshly curated QUBO data. | `fast_finetune_pipeline.py --skip-datagen --skip-eval` per round. |

A fresh LoRA is trained from the base SLM each round rather than stacking adapters.
Improvement carries across rounds through the **data** (each round's QUBO-curated
traces come from a stronger model), which avoids compounding adapter drift.

### Base model

The diagram mandates **Qwen 3B**. `config/config.yaml` is set to
`Qwen/Qwen2.5-3B-Instruct`. Weights are not in `./cache/models` yet — run
`python scripts/download_models.py` before the first loop.

### Relationship to DPO

`scripts/run_dpo.py` and `scripts/generate_preference_pairs.py` remain in the
repository but sit **outside** the finalised diagram, which specifies QLoRA SFT only.
Treat them as an optional experimental branch, not part of the mandated pipeline.

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
│   ├── run_architecture_loop.py  # CLOSED-LOOP orchestrator for the finalised architecture
│   ├── run_dpo.py                # DPO trainer (outside the finalised diagram)
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

## 3.1 Corrections Applied While Aligning to the Finalised Architecture

| File | Problem found | Fix |
| :--- | :--- | :--- |
| `pipeline/sampling.py` | A stray `"""` introduced by commit `02ac939` closed the module docstring early, so lines 37-56 of prose parsed as code. The file raised `IndentationError` on import, which broke `import pipeline` and therefore **every** script in the repo. | Removed the stray delimiter; the whole package imports again. |
| `scripts/fast_finetune_pipeline.py` | `stage_eval()` called `_print_comparison()`, which was never defined, then ran ~25 lines of orphaned code referencing undefined `base_data` / `ft_data`. The eval stage always crashed with `NameError`. | Implemented `_print_comparison()` to read the per-question `<label>_<benchmark>_results.jsonl` files; deleted the orphaned block. |
| `scripts/run_all_benchmarks.py` | `run_qubo_pipeline()` computed `task_type` and passed it to the verifier but called `sampler.sample(question)` without it, so non-math benchmarks (MMLU, BBH, StrategyQA, ARC) were sampled with the math-specific "identify the units and quantities" prompt. | Pass `task_type=task_type` through to the sampler. |
| `scripts/generate_training_data.py` | No way to load a trained adapter, so data generation always sampled from the base SLM. This made the architecture's retrain loop impossible to close. | Added `--adapter-path` (and `QUBO_ADAPTER_PATH` fallback) which merges the LoRA before sampling. |
| `config/config.yaml` | Active model was `meta-llama/Llama-3.2-3B-Instruct`, contradicting the diagram's "SLM Qwen 3B". | Set to `Qwen/Qwen2.5-3B-Instruct` and added it to `candidates` so `download_models.py` fetches it. |

**Note:** `training/sft.py` (`QUBOSFTTrainer`, including its unused `iterative_train`)
is dead code — nothing imports it. The live training path is the self-contained
trainer generated by `fast_finetune_pipeline.py::_write_sft_script`. The closed loop
is implemented in `run_architecture_loop.py`, not in `training/sft.py`.

---

## 3.2 Evaluation Integrity Fixes

These change what the reported numbers *mean*. Results produced before these fixes
are not comparable with results produced after them.

### Answer-key leakage in QUBO selection (most important)

`run_all_benchmarks.py` passed the gold answer into `verifier.score_batch()` at
**evaluation** time. `verify_math()` weights `answer_match` at 0.63, so the QUBO
diagonal effectively encoded "this chain already has the right answer" and the
solver selected chains using the answer key. Measured on the real code path:

| chain | gold passed | gold withheld |
| :--- | ---: | ---: |
| wrong-answer chain | 0.135 | 0.500 |
| right-answer chain | 0.765 | 0.500 |

The greedy and CoT baselines never saw gold, so the comparison was also unfair.
Evaluation now withholds gold and scores chains with **gold-free** signals:

```
correctness_score = 0.35 × process_quality + 0.65 × cross_chain_consensus
```

where `process_quality` is arithmetic consistency (math) or the NLI/coverage/
structure composite (language), and consensus is self-consistency (Wang et al.
2023). Weights live in `config.yaml` under `verifier.scoring.gold_free_*`.

Verified: on a pool of five chains where three agree on the correct answer, the
majority chains score 0.675 and the minority 0.350 — correct ranking, no gold.

Using gold to curate **training** data is still correct and unchanged; that is
rejection sampling, not leakage. Only the eval path changed.

`--oracle-selection` restores the old behaviour for ablation, so the oracle
ceiling can be reported *alongside* the deployable number. Never report it alone.

### Answer extraction and grading

| Problem | Fix |
| :--- | :--- |
| `extract_mcq_choice` ended with "return the last A–J letter found anywhere", which fired on the article "a" and the pronoun "I" — any prose answer without an explicit tag scored as choice A. | Rewritten to try delimiter- or keyword-anchored patterns strongest-first, treat a whole-response letter as unambiguous, exclude bare `A`/`I` from the prose fallback, and **abstain** (return `""`) rather than guess. 17/17 cases pass. |
| `is_correct` fell back to `gold in pred`, so `"...= False, therefore the answer is True"` scored **correct** for gold `False`, and gold `"no"` matched inside `"I do not know"`. | Grading now dispatches on the **gold's format** (parenthesised MCQ letter / boolean / numeric / free-form) and compares only the *conclusion span* — text after the last final-answer marker. 16/16 cases pass. |
| BBH was typed `"math"`, but its golds are ~63% parenthesised MCQ letters plus booleans and word-sorting strings; only a couple of its 27 subtasks are arithmetic. | `TASK_TYPE["bbh"] = "commonsense"`, routing it to the language scorer and string-based consensus. |

### Data pipeline

| Problem | Fix |
| :--- | :--- |
| `--quick` was a no-op: it only applied reduced sizes when an `--n-*` arg was `None`, but all of them had numeric defaults. It printed "Using reduced dataset sizes" and ran 2100 questions instead of 850. | `--n-*` now default to `None`; `FULL_RUN_DEFAULTS` supplies real defaults after `--quick` resolves. Verified: default 2100, `--quick` 850, explicit overrides still win. |
| `DATASET_CONFIGS["mmlu"]` used `split="test"` with `n: 800` — the same split the evaluation suite scores on. Inert only because the CLI default was 0. | Switched to `auxiliary_train` (MMLU's designated training pool), and `load_mmlu` now raises if the split is ever `test`. |
| `run_dpo.py` loaded the **base** model and trained a fresh LoRA, so the "DPO" arm never built on SFT. | Added `--sft-adapter`, merged before training, making it a proper SFT → DPO pipeline. |

---

## 3.2b Config and tuned-parameter handling

`config/config.yaml` carries every key the pipeline reads. Where a key is absent
the code falls back to a hard-coded default, which was silently harmful in one
case: `qubo.cardinality_penalty` defaulted to `0.1`, and at that value against
`penalty_weight: 2.0` the solver selects exactly **one** chain, so the
multi-chain CoT scaffold never engages. It is now set to `0.8`, chosen by
measurement (0.1 → 1 chain, 0.8 → 3 chains on a 12-chain pool).

`model.name` is the Hub repo id `Qwen/Qwen2.5-3B-Instruct`, resolved from
`cache_dir: ./cache/models`. The weights live there as a Hub snapshot, not as a
loose directory. Set `HF_HUB_OFFLINE=1` to force use of the cached copy.

### Stale tuned parameters

`config/best_qubo_params.yaml` now carries a `stale:` flag. `load_config()`
refuses to apply a params file marked stale and prints why, so a superseded
search cannot silently overwrite `config.yaml`.

The shipped file is stale on three counts: it was tuned with the gold answer
visible; its grid searched `cardinality_penalty` over `[0.05, 0.1, 0.2]`, a range
in which no combination can select more than one chain; and its
`validation_accuracy: 0.7778` came from a 10-question set (SE ≈ ±13 points) with
only 52 of 81 combinations evaluated — its own notes record that all `pw=1.0`
combinations scored identically, i.e. the search could not discriminate at all.

`tune_qubo_params.py` writes `stale: false` on a normal gold-free run, and
`stale: true` automatically when `--oracle-scoring` was used, so an ablation run
can never become the deployed configuration.

---

## 3.3 Datasets

Run `python scripts/prepare_datasets.py` to fetch everything and audit the
train/eval boundary. `--audit-only` skips fetching; `--eval-only` does evaluation
sets only.

### Fine-tuning sources

| Source | Split | Available | Suggested | Why this dataset |
| :--- | :--- | ---: | ---: | :--- |
| GSM8K | `train` | 7,473 | 3,000 | Grade-school multi-step arithmetic; the math backbone |
| **MATH (Hendrycks)** | `train` | 7,687 usable | 2,500 | **Added.** Competition math — you evaluate on MATH-500 and AIME but trained only on grade-school arithmetic. Largest train/eval capability gap in the suite |
| MMLU | `auxiliary_train` | 99,842 | 2,000 | Academic breadth. MMLU has no train split; this is its designated training pool |
| ARC-Challenge | `train` | 1,117 | 1,117 | Science MCQ needing retrieval plus reasoning |
| StrategyQA | `train` | 1,603 | 1,600 | Implicit multi-hop yes/no; trains decomposition |
| LogiQA | `train` | — | 1,200 | Formal deduction; closest proxy for BBH-style tasks |

~11,400 questions in, roughly 9,700 curated examples out — about 5× the current
1,774. BBH is deliberately **not** trained on: it has no training split, so it
stays a pure out-of-distribution probe.

`load_math` keeps only problems with plain-numeric answers (7,687 of 11,995).
MATH answers are frequently symbolic (`\frac{\pi}{2}`), and training-data curation
scores chains against gold numerically — symbolic items would be scored wrong
regardless of the reasoning and would poison the curated set.

### Contamination audit

Identity is question text **plus answer options**. Comparing stems alone is
misleading in both directions: MMLU has distinct items whose stem is literally
"Which of the following is true?" (5 false positives, 0 real), while ARC has 7
stem matches of which 2 are genuine duplicates.

```
[PASS] gsm8k        -> gsm8k          train= 7,473  eval=1,319  raw overlap=0
[held] math         -> math 500       train=11,961  eval=  500  raw overlap=4
[held] arc          -> arc_challenge  train= 1,117  eval=1,171  raw overlap=2
[held] strategyqa   -> strategyqa     train= 1,603  eval=  687  raw overlap=1
[PASS] mmlu         -> mmlu           train=34,674  eval=13,308  raw overlap=0
```

`[held]` means the published splits do overlap but the training loader removes
those items before use. MATH, ARC and StrategyQA each drop their overlapping
items at load time, so nothing contaminated reaches training.

### Evaluation sizing

Accuracy standard error is `sqrt(0.25/n)`:

| n | SE | verdict |
| ---: | ---: | :--- |
| 30 | ±9.1 | a 5-point change is invisible |
| 100 | ±5.0 | still too noisy |
| 300 | ±2.9 | workable minimum |
| 1000 | ±1.6 | report-grade |

The `--eval-subset 30` default is far too small to detect the improvements you
are trying to measure. Use 300 minimum, 500–1000 for anything you report.

### StrategyQA source

`load_strategyqa` previously raised `NotImplementedError`. Both prior sources are
broken under `datasets>=3` — `wics/strategy-qa` is a loading script (no longer
supported) and `voidful/StrategyQA` fails schema validation mid-generation. Now
uses `ChilleD/StrategyQA`, which ships disjoint `train` (1,603) / `test` (687)
splits. Gold is normalised to `yes`/`no`, and grading treats `true`/`false` as the
same polarity so either phrasing scores correctly.

---

## 4. End-to-End Execution Workflow for Teammates

### Preferred path: run the finalised architecture as a closed loop

This single command implements the whole mentor-approved diagram, including both
feedback arrows. Use this for all new work.

```bash
# One-time: fetch the Qwen 3B weights the architecture mandates
python scripts/download_models.py --config config/config.yaml

# Three closed-loop rounds, with a base-model baseline for comparison
python scripts/run_architecture_loop.py --rounds 3 --quick --baseline

# Everything on: SFT + DPO arms per round, failure-targeted curriculum
python scripts/run_architecture_loop.py --rounds 3 --baseline --dpo --focus
```

Per round it runs: sample multiple paths → verifier scoring → QUBO build → annealing
solve → select best reasoning → build final prompt → QLoRA SFT → (optional DPO from
the SFT policy) → evaluate → feed the stronger adapter into the next round's sampling.

**`--dpo`** adds a second arm each round: preference pairs are extracted from the
round's curated data and DPO trains from the SFT policy. Both arms are evaluated and
the better one (mean QUBO accuracy across benchmarks) is carried forward — the loop
does not assume DPO always wins.

**`--focus`** turns the loop into a curriculum. Each datagen run writes
`hard_questions.json` (train-split questions that were filtered out or whose best
chain scored below 0.70); the next round prepends them, repeated `--focus-repeat`
times.

> **Curriculum safety note.** The focus set is drawn from **training** splits only.
> Each round also writes `failed_questions.json` from the benchmark results, but that
> is **diagnostics only and must never be fed into training** — those are test-split
> items, and training on them would contaminate the benchmark. The loop deliberately
> keeps the two separate.

Artefacts land under `outputs/architecture_loop/`:

```
outputs/architecture_loop/
├── loop_manifest.json          # per-round adapter paths + accuracies
├── round0_baseline/results/    # base model, no adapter
├── round1/{data,results}/
└── round2/{data,results}/
```

Adapters are written to `checkpoints/qubo-loop-round<N>/final_adapter`.

If a run is interrupted, `--resume` skips every stage whose output already exists:

```bash
python scripts/run_architecture_loop.py --rounds 3 --quick --baseline --resume
```

---

### Manual stage-by-stage path (debugging individual components)

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
