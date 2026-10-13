#!/usr/bin/env bash
# Add --partition/--account at submission using Samsung's actual scheduler names.
# Run from the repository root, after `mkdir -p logs`.
#SBATCH --job-name=prism-diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/prism-%j.out
#SBATCH --error=logs/prism-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
OUT="${OUT:-outputs/slurm-${SLURM_JOB_ID}}"
python scripts/cluster_preflight.py --min-gpus 4 --output "$OUT/preflight.json"
python scripts/run_diagnostics.py --multi-gpu --output-dir "$OUT" --resume
