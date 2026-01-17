#!/bin/bash
#SBATCH --job-name=ttp_eval_local
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ttp_eval_local_%j.out
#SBATCH --error=logs/ttp_eval_local_%j.err

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

# Create output directories
mkdir -p results/ttp_eval_baselines
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Model configuration (can be overridden via environment)
GEMMA_MODEL="${GEMMA_2_27B_MODEL_ID:-google/gemma-2-27b-it}"
QUANTIZATION="${TTP_LOCAL_QUANTIZATION:-none}"

# Build model args
MODEL_ARGS="--local-model $GEMMA_MODEL"

# Optionally add R1 model if specified
if [ -n "${R1_MODEL_ID:-}" ]; then
  MODEL_ARGS="$MODEL_ARGS --local-model $R1_MODEL_ID"
fi

# Table 4 local-model rows: TTP with local LLMs (Gemma 2 27B, optional R1)
if python scripts/evaluate_ttp_eval.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --setups local_ttp \
  $MODEL_ARGS \
  --device cuda \
  --quantization "$QUANTIZATION" \
  --dimension toxic \
  --output results/ttp_eval_baselines/table4_local_ttp.json; then
    echo "Table 4 local LLM evaluation complete!"
    echo "Results saved to: results/ttp_eval_baselines/table4_local_ttp.json"
else
    echo "Error: Table 4 local LLM evaluation failed" >&2
    exit 1
fi
