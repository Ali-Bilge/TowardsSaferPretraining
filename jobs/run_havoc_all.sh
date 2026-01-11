#!/bin/bash
#SBATCH --job-name=havoc_all
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/havoc_all_%j.out
#SBATCH --error=logs/havoc_all_%j.err

# Enable strict error handling
set -euo pipefail

# Create logs directory
mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Change to project directory with error checking
cd $HOME/TowardsSaferPretraining || {
    echo "Error: Failed to change to project directory" >&2
    exit 1
}

# Activate virtual environment with error checking
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Verify script exists
if [ ! -f "scripts/evaluate_havoc.py" ]; then
    echo "Error: evaluate_havoc.py script not found" >&2
    exit 1
fi

# Create output directory
mkdir -p results/havoc

# Models from the paper
MODELS=(
  "google/gemma-2-2b"
  "google/gemma-2-9b"
  "google/gemma-2-27b"
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-3B"
  "mistralai/Mistral-7B-v0.3"
)

# Run evaluation for each model
FAILED_MODELS=()
for model in "${MODELS[@]}"; do
  model_name=$(echo "$model" | cut -d'/' -f2)
  echo "Evaluating $model_name..."

  if python scripts/evaluate_havoc.py \
    --model "$model" \
    --backend transformers \
    --device cuda \
    --output "results/havoc/${model_name}_results.json"; then
    echo "Completed: $model_name"
  else
    echo "Failed: $model_name" >&2
    FAILED_MODELS+=("$model_name")
  fi
done

# Report failures
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models: ${FAILED_MODELS[*]}" >&2
    exit 1
fi

echo "All HAVOC evaluations complete!"
