#!/bin/bash
#SBATCH --job-name=ttp_eval
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Enable strict error handling
set -euo pipefail

# Ensure logs directory exists
mkdir -p "$SLURM_SUBMIT_DIR/logs"

# Move SLURM output files to logs directory
if [ -f "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" ]; then
  mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SLURM_SUBMIT_DIR/logs/"
fi
if [ -f "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" ]; then
  mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$SLURM_SUBMIT_DIR/logs/"
fi

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Set project directory.
# In Slurm, $0 can point to the copied slurm_script under /var/spool, so deriving
# PROJECT_DIR from ${0%/*} is unreliable. Prefer SLURM_SUBMIT_DIR.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/TowardsSaferPretraining}}"

# Change to project directory with error checking
cd "$PROJECT_DIR" || {
    echo "Error: Failed to change to project directory: $PROJECT_DIR" >&2
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

# Create output directory
mkdir -p results/ttp_eval
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Ensure required keys exist
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY is required for openai_ttp setup" >&2
  exit 1
fi

# Run TTP evaluation on full TTP-Eval dataset (Table 3)
if python scripts/evaluate_ttp_eval.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --setups openai_ttp \
  --openai-model gpt-4o \
  --dimension toxic \
  --output results/ttp_eval/ttp_results.json; then
    echo "TTP Evaluation Complete!"
    echo "Results saved to: results/ttp_eval/ttp_results.json"
else
    echo "Error: TTP evaluation failed" >&2
    exit 1
fi
