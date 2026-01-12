#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

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

# Activate virtual environment
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment" >&2
    exit 1
}

# Load API keys from .env if present, otherwise from example.env.
# (Some environments block writing dotfiles; scripts also support example.env fallback.)
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
mkdir -p results/baselines
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Select baselines based on available credentials.
BASELINES=("harmformer")
EXTRA_ARGS=()

if [ -n "${OPENAI_API_KEY:-}" ]; then
  BASELINES+=("ttp")
  EXTRA_ARGS+=(--openai-key "$OPENAI_API_KEY")
fi

# Llama Guard is often gated (Meta terms acceptance required).
# Under repo-parity runs, enable it automatically when an HF token is present,
# but allow opt-out.
if [ "${DISABLE_LLAMA_GUARD:-0}" = "1" ]; then
  echo "Info: skipping llama_guard baseline (DISABLE_LLAMA_GUARD=1)." >&2
elif [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] || [ -n "${HF_TOKEN:-}" ]; then
  BASELINES+=("llama_guard")
else
  echo "Info: skipping llama_guard baseline (no HF token found)." >&2
fi

if [ -n "${PERSPECTIVE_API_KEY:-}" ]; then
  BASELINES+=("perspective")
  EXTRA_ARGS+=(--perspective-key "$PERSPECTIVE_API_KEY")
fi

# Run baseline comparison
# This reproduces Table 4 (TTP vs Perspective API) and Table 7 (OpenAI Moderation dataset)
if python scripts/compare_baselines.py \
  --baselines "${BASELINES[@]}" \
  --device cuda \
  "${EXTRA_ARGS[@]}" \
  --output results/baselines/baseline_comparison.json; then
    echo "Baseline comparison complete!"
    echo "Results saved to: results/baselines/baseline_comparison.json"
else
    echo "Error: Baseline comparison failed" >&2
    exit 1
fi
