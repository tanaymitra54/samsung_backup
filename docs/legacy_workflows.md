# Legacy workflows

Use [`scripts/run_diagnostics.py`](../scripts/run_diagnostics.py) and the [cluster guide](cluster/README.md) for the maintained CoT-16 / QUBO-six / Greedy-six comparison. The programs below retain earlier method definitions.

| Program | Scope / limitation |
|---|---|
| `scripts/run_all_benchmarks.py` | Named CoT/Greedy columns use one generation; separate SC/best-of controls exist; QUBO prunes answer groups; grading differs |
| `scripts/run_architecture_pipeline.py` | 40-question base → QLoRA → post-training loop. Evaluation/trace workers can shard GPUs; SFT stays on one GPU |
| `training/sft.py` | Iterative QLoRA. Non-GSM8K collection now uses disjoint diagnostic fit/eval partitions, but selection/prompts remain legacy |
| `scripts/generate_training_data.py` | Source caps total 5,000 before filtering despite a 6,000 target |
| `scripts/tune_qubo_params.py` | Hand-set signal weights and low-score exclusions; use the new fit/tune learner |
| `scripts/generate_preference_pairs.py` | Older pairs lack reserved-role provenance; strict DPO expects the diagnostic exporter |
| `scripts/prepare_training_data.py` | Deprecated: leaks evaluation CSV traces into training |

The 40-question manifests are functional panels, not broad per-topic evidence. BBH train rows use public test data; StrategyQA uses public labeled train; MMLU covers only five STEM subjects. Independent IDs alone do not prove unseen questions. See [the historical audit](mentor_audit.md).

For reproduction only:

```bash
python scripts/run_all_benchmarks.py --config config/config.yaml --benchmarks gsm8k --subset-size 3 --no-multi-gpu
bash scripts/run_architecture.sh --no-multi-gpu
```

`run_benchmarks.sh` now starts the diagnostic workflow and `run_smoke.sh` starts its generation smoke. Legacy worker pools use spawn and scheduler-aware device mapping. Use the pinned environment/preflight for both paths. Historical output folders do not describe the current runner protocol.
