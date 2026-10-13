# Samsung PRISM: QUBO reasoning diagnostics

Compare QUBO selection against CoT and Greedy using the **same 16 generated reasoning traces**. Diagnose which questions and dataset topics improve or regress, with saved evidence, repeated seeds and paired statistical tests. The default model is `Qwen/Qwen2.5-3B-Instruct`.

Start with the [A6000 cluster guide](docs/cluster/README.md). It covers installation, GPU allocation, calibration, evaluation, Slurm, resuming, DPO and troubleshooting. The [diagnostic protocol](docs/diagnostic_evaluation.md) explains grading/statistics; the [mentor audit](docs/mentor_audit.md) records the previous implementation's issues and historical question counts.

> Looking for the original end-to-end pipeline documentation (architecture, QUBO math, sampling/verifier/solver modules, benchmark design)? See [docs/pipeline_overview.md](docs/pipeline_overview.md).

## What is compared

| Method | Shared candidate pool | Selection / final answer |
|---|---:|---|
| CoT-16 (`cot16`) | 16 | Majority vote; no extra generation |
| QUBO (`qubo6`) | 16 | Exact minimum-energy six-trace subset, then deterministic synthesis |
| Greedy (`greedy6`) | 16 | Six highest learned scores, then the same synthesis formatting |
| Random-six | 16 | Seeded random six, then synthesis |
| CoT-16 synthesis | 16 | All 16 traces, then synthesis |
| Selection/voting controls | 16 | QUBO-six, Greedy-six, random-six votes and confidence-weighted voting |

QUBO solves all 8,008 size-six subsets of a 16-variable objective. This evaluates the selection objective; it is not a quantum-computing or annealing-speed claim. CoT voting and synthesis have different generation/token costs, so the report includes additional controls and actual token counts.

Trace correctness scores are **learned** from reserved fit questions. The redundancy penalty is calibrated on disjoint tune questions using selected-answer vote accuracy, a proxy for synthesis performance. Evaluation uses frozen coefficients and never passes gold answers into generation or selection. No trained/calibrated artifact or benchmark accuracy is bundled as a claimed result.

## Reference environment

- Linux cluster, Python **3.11**, NVIDIA driver supporting CUDA 12.1.
- RTX **A6000: 48 GB per GPU**, as specified by [NVIDIA](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/).
- PyTorch **2.5.1 + cu121**, transformers **4.51.3**, TRL **0.19.1**, PEFT **0.15.2**; remaining versions are pinned in [requirements.txt](requirements.txt).
- Single-node multi-GPU evaluation uses independent model replicas and spawned workers. DPO supports distributed training with `torchrun`.

```bash
git clone https://github.ecodesamsung.com/SRIB-PRISM/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning.git
cd 26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning
git checkout main
git pull --ff-only
PYTHON=python3.11 bash scripts/setup_venv.sh
source .venv/bin/activate
python -m pip check
python scripts/cluster_preflight.py
python -m pytest -q tests
```

Run the GPU preflight and model jobs **inside a scheduler allocation**, not on a login node. The setup script explicitly installs PyTorch from the [official cu121 index](https://pytorch.org/get-started/previous-versions/), then the pinned packages. The cluster guide explains module loading and cache placement without assuming the Samsung scheduler's partition/account names.

## Run the complete diagnostic workflow

Allocate GPUs first. One dispatcher collects fit/tune pools, learns weights, and evaluates each benchmark sequentially while using all allocated GPUs:

```bash
python scripts/run_diagnostics.py --multi-gpu --output-dir outputs/a6000_run
```

For one GPU or one benchmark:

```bash
python scripts/run_diagnostics.py --benchmarks gsm8k --device cuda:0 --output-dir outputs/gsm8k_run
```

Preview commands without loading models:

```bash
python scripts/run_diagnostics.py --benchmarks gsm8k --multi-gpu --plan
```

Use `--resume` with the same output directory to reuse completed question/seed records. A changed model, adapter, scientific config, code, seeds or panel requires a fresh run/calibration. Output directories must belong to one active dispatcher. Inference is unbatched within each GPU; GPUs process different complete question/seed comparisons.

## Individual dataset entrypoints

| Dataset | Entry point | Reserved evaluation pool | Topic labels |
|---|---|---|---|
| GSM8K | `scripts/evaluate_gsm8k.py` | Official test minus calibration-source duplicates | Human annotations required |
| MMLU | `scripts/evaluate_mmlu.py` | Official test minus calibration-source duplicates | All 57 subjects |
| BBH | `scripts/evaluate_bbh.py` | Reserved 60% of public labeled test data | All 27 tasks |
| ARC-Challenge | `scripts/evaluate_arc.py` | Official test minus calibration-source duplicates | Human annotations required |
| StrategyQA | `scripts/evaluate_strategyqa.py` | Reserved 60% of public labeled training JSON | Human annotations required |

Each supports `--role fit/tune/eval`, `--collect-only`, `--questions`, `--seeds`, `--full`, `--weights`, `--adapter`, `--manifest`, `--exclude-manifest`, `--multi-gpu`, `--gpu-ids`, `--no-multi-gpu`, `--device`, `--resume`, `--plan` and `--analyze-only`. See `--help` and the [step-by-step guide](docs/cluster/README.md).

Main-config defaults: **200 fit questions × one seed**, **200 tune questions × one seed**, then **200 evaluation questions × five seeds** per dataset. Every question/seed generates 16 candidates; evaluation adds four synthesis generations for its controls. Requested counts are capped by available distinct data and the actual panel is saved. `config_fast.yaml` uses 20 questions and three evaluation seeds with the same generation budgets; it is a functional check, not adequate per-topic evidence.

BBH/StrategyQA partition by stable question hashes: 20% fit, 20% tune, 60% evaluation. GSM8K/ARC calibration uses train; MMLU calibration uses validation partitioned 80/20 fit/tune. Topic sampling is stratified using actual metadata. GSM8K/ARC/StrategyQA remain unlabelled unless a reviewed manifest supplies topics. Real dataset QA identified MMLU validation/test overlaps, which are excluded automatically.

## Outputs and interpretation

`outputs/a6000_run/<benchmark>/` contains `fit/`, `tune/`, `learned_weights.json`, and `eval/`. Within evaluation:

- `run.json`, `effective_config.yaml`, `manifest.jsonl`, `weights.json`: reproducibility/provenance.
- `shards/worker-*.jsonl`: durable per-worker records, recoverable after interruption.
- `results.jsonl`: merged records in panel/seed order, including raw chains, scores, similarities, selected indices, energy, final answers and token counts.
- `execution.json`: worker devices, completion counts and resolved model revisions/precision.
- `diagnosis.md`, `diagnosis.json`: per-topic/per-question comparisons and observed failure categories.
- `human_review.csv`: raw evidence plus unfilled reasoning/hallucination review fields.

Reports regrade raw answers and replay selection. They use paired bootstrap intervals, question-clustered sign-flip tests, per-seed exact McNemar tests and Holm adjustment. Fewer than 30 distinct questions per group, fewer than three seeds, inconsistent signs, corrected p > 0.05, or execution errors prevent a repeatable-advantage label. This is exploratory evidence on that panel; improvements need confirmation on a fresh reserved panel. **Correct answers and NLI scores do not prove truthful reasoning.**

The orchestrator also writes `combined_diagnosis.md`/`.json` at its output root, correcting across all selected datasets. Use that combined report when searching for advantages across datasets; individual reports correct only within their own comparisons.

```bash
python scripts/evaluate_gsm8k.py --analyze-only --output-dir outputs/a6000_run/gsm8k/eval
```

## Retraining

Identify failure topics using evaluation, then generate answer-verified positive/negative pairs from independent **fit** questions in those areas. Never train on evaluated questions and claim them as unseen again.

```bash
python scripts/build_diagnostic_preferences.py --input outputs/a6000_run/gsm8k/fit/results.jsonl --output data/dpo_pairs.jsonl
python training/run_dpo.py --pairs data/dpo_pairs.jsonl --validate-only
python training/run_dpo.py --pairs data/dpo_pairs.jsonl --sft-adapter checkpoints/YOUR_SFT_ADAPTER --max-steps 1 --output-dir checkpoints/dpo_smoke
```

`--topics NAME ...` filters independently annotated fit examples. See the cluster guide for full single-/multi-GPU DPO commands, finite-loss checks, adapter reload, exclusions and recalibration. The shipped QLoRA architecture loop remains a legacy experiment; it is separate from the new diagnostic protocol.

## Repository map

| Path | Purpose |
|---|---|
| `evaluation/diagnostic_runner.py` | Shared generation, strict 16-trace protocol and provenance |
| `evaluation/parallel_execution.py` | GPU sharding, durable writes, recovery and merge |
| `evaluation/diagnostic_data.py` | Dataset metadata, partitions and duplicate exclusion |
| `evaluation/selection_protocol.py` | Learned score application and exact-six selection |
| `evaluation/learned_weights.py` | Fit/tune learning with leakage rejection |
| `evaluation/diagnostic_report.py` | Regrading, statistics and failure reports |
| `scripts/run_diagnostics.py` | Complete five-dataset orchestration |
| `scripts/cluster_preflight.py` | Package/API/CUDA checks |
| `scripts/check_diagnostic_datasets.py` | Real dataset/gold-label/disjointness QA |
| `training/run_dpo.py`, `training/dpo_data.py` | Validated DPO and SFT-reference continuation |
| `scripts/slurm_diagnostics.sbatch` | Single-node Slurm GPU job example |
| `docs/cluster/README.md` | Canonical cluster setup/run instructions |
| `tests/` | Pure protocol tests and actual PyTorch tiny-model tests |
| `verification/` | Recorded checks, with scope stated in each artifact |

Legacy `run_all_benchmarks.py`, `run_architecture_pipeline.py`, `training/sft.py`, and their historical outputs retain different protocols. They must not be used to claim the new CoT-16/Greedy-top-six comparison. `prepare_training_data.py` is deprecated because it mixes evaluation traces into training. See [docs/legacy_workflows.md](docs/legacy_workflows.md).
