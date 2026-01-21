#!/bin/bash
#SBATCH --job-name=havoc_all
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Enable strict error handling
set -euo pipefail

# Create logs directory
mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Change to project directory with error checking
cd "$HOME/TowardsSaferPretraining" || {
    echo "Error: Failed to change to project directory" >&2
    exit 1
}

# Activate virtual environment with error checking
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Optional CodeCarbon tracking
mkdir -p results/codecarbon
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Verify script exists
if [ ! -f "scripts/evaluate_havoc_modeleval.py" ]; then
    echo "Error: evaluate_havoc_modeleval.py script not found" >&2
    exit 1
fi

# Create output directory
mkdir -p results/havoc

# Model keys present in data/HAVOC/havoc_modeleval.tsv
MODEL_KEYS=(
  "gemma_9b"
  "llama_3b"
  "mistral_7b"
)

# Run evaluation for each model
FAILED_MODELS=()
for model_key in "${MODEL_KEYS[@]}"; do
  echo "Evaluating modeleval leakage for $model_key..."

  if python scripts/evaluate_havoc_modeleval.py \
    --data-path data/HAVOC/havoc.tsv \
    --modeleval-path data/HAVOC/havoc_modeleval.tsv \
    --model-key "$model_key" \
    --output "results/havoc/${model_key}_results.json"; then
    echo "Completed: $model_key"
  else
    echo "Failed: $model_key" >&2
    FAILED_MODELS+=("$model_key")
  fi
done

# Report failures
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models: ${FAILED_MODELS[*]}" >&2
    exit 1
fi

echo "All HAVOC evaluations complete!"
