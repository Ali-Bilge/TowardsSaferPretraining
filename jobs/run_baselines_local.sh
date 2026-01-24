#!/bin/bash
#SBATCH --job-name=baselines_local
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/baselines_local_%j.out
#SBATCH --error=logs/baselines_local_%j.err

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
module load CUDA/12.1.1 || {
    echo "Error: Failed to load module CUDA/12.1.1" >&2
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

# Load environment variables from .env file
set -a; source .env 2>/dev/null || true; set +a

# Create output directories
mkdir -p results/moderation
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Run ONLY GPU-heavy local baselines (Table 7 local rows)
if python scripts/evaluate_openai_moderation.py \
  --baselines llama_guard llama_guard_zero_shot llama_guard_few_shot harmformer \
  --device cuda \
  --output results/moderation/table7_local_results.json; then
    echo "Baselines (local) complete!"
    echo "Results saved to: results/moderation/table7_local_results.json"
else
    echo "Error: Baselines (local) failed" >&2
    exit 1
fi
