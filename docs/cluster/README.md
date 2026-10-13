# Samsung PRISM A6000 cluster guide

Follow these steps from a Linux/Bash checkout. Use Python **3.11**, a compatible NVIDIA driver and allocated RTX A6000 GPUs (**48 GB per card**, [NVIDIA specifications](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/)). Partition/account/module names depend on the Samsung cluster; obtain them from its administrator.

## 1. Obtain and record the code version

```bash
git clone https://github.ecodesamsung.com/SRIB-PRISM/26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning.git
cd 26ARS05VITC_Quantum-inspired_Annealing_for_Multi-stage_Reasoning
git checkout main
git pull --ff-only
git rev-parse HEAD
```

For existing checkouts, preserve local work before pulling. Record the tested commit. Changed code after calibration requires new calibration and a fresh run directory.

## 2. Create the reference environment and caches

Load the cluster's actual Python 3.11 module if needed. The pinned reference stack is PyTorch 2.5.1/cu121, transformers 4.51.3, TRL 0.19.1 and PEFT 0.15.2; see `requirements.txt`. The [official PyTorch index](https://pytorch.org/get-started/previous-versions/) supplies the CUDA wheel; a separate CUDA compiler is not required for the supported SDPA backend.

```bash
PYTHON=python3.11 bash scripts/setup_venv.sh
source .venv/bin/activate
python -m pip check
mkdir -p cache/models cache/huggingface outputs checkpoints logs verification
export HF_HOME="$PWD/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python scripts/download_models.py --config config/config.yaml
```

The downloader fetches the configured Qwen model, NLI verifier and MiniLM embedder into `model.cache_dir`. Change that YAML path to persistent scratch storage if necessary, then keep it consistent throughout calibration/evaluation. Prefetch using approved network access if compute nodes are offline. Use the cluster's secret mechanism for `HF_TOKEN`; do not commit tokens.

Prepared dataset caches are isolated by the installed `datasets` version under `HF_HOME` (or the repository cache if unset). `PRISM_DATASETS_CACHE` overrides that base directory. This prevents incompatible Arrow feature schemas from an older/newer shared environment from being reused; existing caches are preserved.

## 3. Allocate GPUs and validate the environment

Run CUDA commands on an allocated GPU node, not a login node. A Slurm interactive example is `salloc --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=16 --mem=64G`; add the site's real partition/account options. Inside the allocation:

```bash
source .venv/bin/activate
nvidia-smi
python scripts/cluster_preflight.py --min-gpus 4 --output verification/a6000_preflight.json
python -m pytest -q tests
python scripts/check_diagnostic_datasets.py
```

Preflight checks real package imports, DPO/SFT API parameters, config invariants, allocated CUDA devices, BF16 matrix operations, bitsandbytes NF4 forward/backward and memory. Runtime tests use tiny random models to test actual PyTorch generation, DPO updates/frozen reference/reload, and two spawned CPU workers. These checks do not establish model quality or successful full A6000 QLoRA/NCCL training.

Honor the scheduler's `CUDA_VISIBLE_DEVICES`. If it grants physical cards 3 and 7, they appear to this job as logical `cuda:0` and `cuda:1`; select them with `--gpu-ids 0 1`. GPU UUID allocations also work. Do not overwrite visibility with physical indices inside a scheduled job.

## 4. Check real model generation

```bash
OUT=outputs/generation_smoke bash scripts/run_smoke.sh
python scripts/evaluate_gsm8k.py --role fit --collect-only --questions 4 --seeds 42 --multi-gpu --output-dir outputs/multigpu_smoke
```

The first command generates 16 real traces for each of two reserved fit questions on one GPU. The second checks sharding. Inspect `results.jsonl`, `execution.json` and `shards/worker-*.jsonl` for errors, final-answer formatting, output-budget exhaustion and disjoint completed tasks. These tiny runs do not claim accuracy or sufficient calibration data.

## 5. Run fit → tune → learn → evaluate

All five datasets, sequential stages, all allocated GPUs:

```bash
python scripts/run_diagnostics.py --multi-gpu --output-dir outputs/a6000_run
```

One dataset on selected logical GPUs, or on one GPU:

```bash
python scripts/run_diagnostics.py --benchmarks gsm8k --gpu-ids 0 1 --output-dir outputs/gsm8k_two_gpu
python scripts/run_diagnostics.py --benchmarks gsm8k --device cuda:0 --output-dir outputs/gsm8k_single_gpu
python scripts/run_diagnostics.py --multi-gpu --plan
```

Main config: 200 distinct fit questions × one seed, 200 tune questions × one seed, and 200 evaluation questions × five seeds per dataset. Every task samples 16 chains; evaluation adds four synthesis generations for its controls. Output caps are 512 tokens. No input truncation or reduced-budget OOM fallback occurs in this diagnostic backend. Actual counts and token costs are saved. `--questions`, `--calibration-questions` and `--seeds` override orchestration budgets.

`config_fast.yaml` uses 20 questions per stage and three evaluation seeds with the same generation settings. It is a functional check; it can fail calibration if all fit traces have the same correctness label. Gather more questions rather than inventing coefficients. For MMLU/BBH topic evidence, 200 total questions usually leaves too few per topic: use a larger prespecified panel or reviewed manifests with at least 30 distinct questions per target topic and preferably more. Significant exploratory areas require confirmation on a fresh reserved panel.

### Individual dataset execution

```bash
python scripts/evaluate_gsm8k.py --role fit --collect-only --multi-gpu --output-dir outputs/manual/gsm8k/fit
python scripts/evaluate_gsm8k.py --role tune --collect-only --multi-gpu --output-dir outputs/manual/gsm8k/tune
python scripts/learn_selection_weights.py --fit outputs/manual/gsm8k/fit/results.jsonl --tune outputs/manual/gsm8k/tune/results.jsonl --output outputs/manual/gsm8k/learned_weights.json
python scripts/evaluate_gsm8k.py --weights outputs/manual/gsm8k/learned_weights.json --multi-gpu --output-dir outputs/manual/gsm8k/eval
```

Substitute `evaluate_mmlu.py`, `evaluate_bbh.py`, `evaluate_arc.py` or `evaluate_strategyqa.py` and corresponding directories. Each supports `--questions`, `--seeds`, `--full`, reviewed `--manifest`, `--adapter`, and repeated `--exclude-manifest` options. Fit/tune/evaluation roles are disjoint; the [protocol guide](../diagnostic_evaluation.md) explains public-split limitations and exact grading.

## 6. Resume and inspect evidence

```bash
python scripts/run_diagnostics.py --multi-gpu --output-dir outputs/a6000_run --resume
python scripts/evaluate_gsm8k.py --analyze-only --output-dir outputs/a6000_run/gsm8k/eval
```

Workers append to separate durable shard files. Resume reuses completed question/seed pairs, preserves an interrupted last write, runs missing tasks and merges in panel order. Complete corrupt lines, conflicting duplicates, unexpected questions, mixed revisions or incompatible provenance fail. Existing execution-error records remain errors; fix their cause and use a fresh directory to rerun. Do not delete difficult examples to improve a denominator.

Use one active dispatcher per output directory. Worker/GPU count may change on resume; scientific config, code, panel, seeds, model/adapter and precision must stay compatible. Evaluation uses spawned workers on **one node**; do not launch it with `torchrun` or multiple `srun` tasks. Use separate dataset jobs/directories across nodes.

Review `diagnosis.md`/`diagnosis.json`, raw `results.jsonl`, and `human_review.csv`. Counts distinguish missing correct candidates, selection misses, synthesis failures, invalid answers, confidence errors and token limits. They do not establish hallucinations without independent evidence review. Topic metadata and observed question-length bins are separate. Keep manifests, weights, logs and provenance with results.

For searches across datasets, use the orchestrator's root-level `combined_diagnosis.md`/`.json`, whose correction includes every comparison across the selected dataset reports. Individual reports apply within-report correction only.

## 7. Slurm batch execution

```bash
mkdir -p logs
sbatch scripts/slurm_diagnostics.sbatch
```

The example requests one node, four GPUs, one parent task, 16 CPU cores, 64 GB host RAM and a 24-hour allocation. Adjust resources, time, account and partition to the actual cluster; this is not a runtime estimate. Submit from the repository root. To resume a stable directory in a later job:

```bash
OUT=outputs/team_run sbatch --export=ALL scripts/slurm_diagnostics.sbatch
```

The batch script uses `--resume`, which also starts a fresh run if the directory is absent. Check scheduler exit status and logs. Do not run two jobs concurrently with the same `OUT`.

## 8. DPO, single-/multi-GPU

Reproduce diagnosed failure areas using independent **fit** examples; never export evaluation/tune failures into training and report them as unseen. Optional `--topics NAME ...` filters reviewed fit topics.

```bash
python scripts/build_diagnostic_preferences.py --input outputs/a6000_run/gsm8k/fit/results.jsonl --output data/dpo_pairs.jsonl
python training/run_dpo.py --pairs data/dpo_pairs.jsonl --validate-only
python training/run_dpo.py --pairs data/dpo_pairs.jsonl --sft-adapter checkpoints/YOUR_SFT_ADAPTER --max-steps 1 --batch-size 1 --output-dir checkpoints/dpo_smoke
```

Inspect a real GPU smoke for finite loss and a reloadable adapter before full training. Omit `--max-steps 1` for full training and use a fresh output directory. Without `--sft-adapter`, DPO starts a fresh adapter against the base reference. With it, policy/reference both start from SFT and the reference is frozen. Non-finite logged losses or trained parameters stop final export.

Distributed DPO in a four-GPU single-node allocation:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 training/run_dpo.py --pairs data/dpo_pairs.jsonl --sft-adapter checkpoints/YOUR_SFT_ADAPTER --batch-size 1 --output-dir checkpoints/dpo_four_gpu
```

`LOCAL_RANK` pins each process; the HF trainer distributes data; rank zero exports. Effective batch = per-device batch × two accumulation steps × world size. This is training data parallelism. Actual CUDA/NCCL/4-bit behavior must pass the team GPU smoke.

The trainer prints the adapter path: `final_adapter/` for a fresh default adapter or `final_adapter/policy/` when continuing SFT. Check reload with an individual runner's `--adapter` during a small fit collection. After retraining, recollect calibration pools for that adapter, refit coefficients, then evaluate a fresh confirmation panel. Pass `--exclude-manifest data/dpo_pairs.jsonl` and all known prior training manifests. Unknown historical training data cannot be reconstructed automatically.

## 9. Docker alternative

Use the administrator's supported NVIDIA container runtime inside allocations. The Dockerfile uses the same Python/cu121/pinned stack as the venv; bind mounts persist caches/results/checkpoints.

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm prism python scripts/cluster_preflight.py
docker compose -f docker/docker-compose.yml run --rm prism python scripts/run_diagnostics.py --multi-gpu --output-dir outputs/docker_run
```

For unscheduled containers `NVIDIA_VISIBLE_DEVICES` can constrain host GPUs. Inside containers, `--gpu-ids` uses logical indices. On scheduled clusters, use the supported launch that preserves the allocation, rather than exposing unallocated devices.

## 10. Troubleshooting

| Failure | Action |
|---|---|
| CUDA unavailable / insufficient GPUs | Inspect allocation, visibility and preflight on the GPU node |
| Invalid GPU indices | Use logical indices within the allocation |
| Incompatible DPO/SFT API | Rebuild pinned Python 3.11 venv; run pip check/preflight |
| OOM | Inspect free memory; reduce DPO batch size; for evaluation choose consistent 4-bit settings before collecting new calibration |
| Many chains hit output limits | Increase budgets before recollection/recalibration; inspect raw chains |
| No mixed calibration labels | Gather more fit data; do not fabricate coefficients |
| Resume provenance mismatch | Use a fresh output root and recalibrate |
| Download blocked | Prefetch through approved network access into persistent caches |
| Inconclusive topic result | Increase prespecified distinct-question counts; confirm on fresh data |

Verification artifacts describe the exact scope of completed checks. No package/API/tiny-model test alone establishes benchmark improvement or qualifies the team's full A6000 cluster run.
