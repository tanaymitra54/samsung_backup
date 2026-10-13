# Test suites

Run from the repository root in the pinned environment:

```bash
python -m pytest -q tests
```

| Suite | What it verifies |
|---|---|
| `test_protocol.py` | Existing exact-K, objective energy, grouping, CI and preference-label checks |
| `test_diagnostic_evaluation.py` | Strict grading, shared-16/exact-six selection, independent energy enumeration, gold exclusion, fit/tune separation, learned coefficients, statistical nulls, source schemas, MMLU duplicate exclusion, CLI resume and offline verification |
| `test_parallel_execution.py` | Durable shard merge, interrupted-write recovery, corruption/conflict rejection, scheduler logical GPU selection, cross-dataset statistical correction |
| `test_torch_runtime.py` | Real tiny-Qwen generation/confidence/context limits, tensor QUBO energy, scheduler mapping, actual TRL DPO updates/frozen reference/adapter reload, and two spawned CPU workers doing PyTorch operations |

The PyTorch runtime suite skips when PyTorch is unavailable. A skipped suite is **not** an ML runtime pass. Tiny random-model tests establish executable behavior, not pretrained benchmark accuracy or a successful CUDA/4-bit/distributed training run.

On the team's GPU allocation, also run:

```bash
python scripts/cluster_preflight.py --min-gpus 2
OUT=outputs/generation_smoke bash scripts/run_smoke.sh
python scripts/evaluate_gsm8k.py --role fit --collect-only --questions 4 --seeds 42 --multi-gpu --output-dir outputs/multigpu_smoke
```

Preflight exercises actual CUDA BF16 and bitsandbytes NF4 forward/backward on each visible GPU. Inspect real model generation/shards, then run the documented one-step QLoRA DPO smoke before full training. Use fresh directories or a compatible `--resume`; keep exit statuses and logs with the run.
