#!/bin/bash
#SBATCH --job-name=ttp_eval_api
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ttp_eval_api_%j.out
#SBATCH --error=logs/ttp_eval_api_%j.err

set -euo pipefail

# Create logs directory
mkdir -p logs

module purge
module load 2023 || {
    echo "Error: Failed to load module 2023" >&2
    exit 1
}
module load Python/3.11.3-GCCcore-12.3.0 || {
    echo "Error: Failed to load module Python/3.11.3-GCCcore-12.3.0" >&2
    exit 1
}

# Change to project directory
cd "$HOME/TowardsSaferPretraining" || {
    echo "Error: Failed to change to project directory" >&2
    exit 1
}

# Activate virtual environment with error checking
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Load API keys from .env if present, otherwise from example.env.
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
elif [ -f "example.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "example.env"
  set +a
fi

# Create output directories
mkdir -p results/ttp_eval_baselines
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# This job previously ran Perspective API for Table 4. Perspective is omitted by default.
echo "Deprecated: jobs/run_ttp_eval_api.sh relied on Perspective API, which is omitted by default now."
echo "Use jobs/run_ttp_eval.sh for TTP on TTP-Eval, or scripts/evaluate_ttp_eval.py with explicit --setups."
exit 1

# (Intentionally no-op)
