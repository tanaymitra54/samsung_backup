# Team runbook

The maintained A6000 setup and execution instructions are in
[docs/cluster/README.md](docs/cluster/README.md).

Use `scripts/run_diagnostics.py` for the mentor-requested comparison. It runs
fit-pool collection, tune-pool collection, learned coefficient calibration,
and repeated-seed evaluation for each selected dataset. Its dataset runners
support one worker per allocated GPU.

```bash
PYTHON=python3.11 bash scripts/setup_venv.sh
source .venv/bin/activate
python scripts/cluster_preflight.py --min-gpus 2
python -m pytest -q tests
python scripts/run_diagnostics.py --multi-gpu --output-dir outputs/a6000_run
```

Run CUDA commands inside a GPU allocation. RTX A6000 has **48 GB per GPU**.
Honor the scheduler's `CUDA_VISIBLE_DEVICES`; `--gpu-ids` uses logical indices
within that allocation. Evaluation is a single dispatcher with spawned
workers: do not launch it with `torchrun` or multiple `srun` tasks. Distributed
DPO training uses `torchrun`, as documented in the cluster guide.

For scientific definitions and individual dataset commands, see
[docs/diagnostic_evaluation.md](docs/diagnostic_evaluation.md). For the earlier
implementation audit and historical counts, see
[docs/mentor_audit.md](docs/mentor_audit.md). Legacy workflows are documented
separately in [docs/legacy_workflows.md](docs/legacy_workflows.md).
