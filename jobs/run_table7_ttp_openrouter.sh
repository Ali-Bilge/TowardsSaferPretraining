#!/bin/bash
#SBATCH --job-name=table7_ttp_or
#SBATCH --partition=rome
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/table7_ttp_openrouter_%j.out
#SBATCH --error=logs/table7_ttp_openrouter_%j.err

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

# Load API keys from .env
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# Create output directories
mkdir -p results/moderation
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Ensure OpenRouter key exists
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Error: OPENROUTER_API_KEY is required" >&2
  exit 1
fi

# Run TTP via OpenRouter on OpenAI Moderation dataset (Table 7)
if python scripts/evaluate_openai_moderation.py \
  --baselines ttp_openrouter \
  --openrouter-model "openai/gpt-4o" \
  --device cpu \
  --output results/moderation/table7_ttp_openrouter.json; then
    echo "Table 7 TTP (OpenRouter) complete!"
    echo "Results saved to: results/moderation/table7_ttp_openrouter.json"
else
    echo "Error: Table 7 TTP (OpenRouter) failed" >&2
    exit 1
fi
