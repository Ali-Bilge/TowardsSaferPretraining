#!/bin/bash
#SBATCH --job-name=ttp_eval_local
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=2
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
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

# Reduce fragmentation risk for large-model inference
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

# Load env keys if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# Create output directories
mkdir -p results/ttp_eval_baselines
mkdir -p results/codecarbon

# Optional CodeCarbon tracking
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$HOME/TowardsSaferPretraining/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Model configuration (can be overridden via environment)
GEMMA_MODEL="${GEMMA_2_27B_MODEL_ID:-google/gemma-2-27b-it}"
QUANTIZATION="${TTP_LOCAL_QUANTIZATION:-none}"
# Default to no quantization when requesting 2 GPUs. If you want 4bit/8bit, install bitsandbytes and set this env var.
QUANTIZATION_LARGE="${TTP_LOCAL_QUANTIZATION_LARGE:-none}"

# Resolve LLaMa-32B (R1 distill) model id
LLAMA_MODEL="${LLAMA_32B_MODEL_ID:-}"
if [ -z "$LLAMA_MODEL" ] && [ -n "${R1_MODEL_ID:-}" ]; then
  LLAMA_MODEL="$R1_MODEL_ID"
fi

# Table 4 local-model rows: run models separately to avoid loading multiple large models at once.

# 1) Gemma 2 27B
if python scripts/evaluate_ttp_eval.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --setups local_ttp \
  --local-model "$GEMMA_MODEL" \
  --device cuda \
  --dtype float16 \
  --quantization "$QUANTIZATION" \
  --dimension toxic \
  --output results/ttp_eval_baselines/table4_local_ttp_gemma.json; then
    echo "Table 4 local LLM evaluation complete (Gemma)!"
    echo "Results saved to: results/ttp_eval_baselines/table4_local_ttp_gemma.json"
else
    echo "Error: Table 4 local LLM evaluation failed (Gemma)" >&2
    exit 1
fi

# 2) DeepSeek R1 Distill LLaMa 32B (optional)
if [ -n "$LLAMA_MODEL" ]; then
  if python scripts/evaluate_ttp_eval.py \
    --data-path data/TTP-Eval/TTPEval.tsv \
    --setups local_ttp \
    --local-model "$LLAMA_MODEL" \
    --device cuda \
    --dtype float16 \
    --quantization "$QUANTIZATION_LARGE" \
    --dimension toxic \
    --output results/ttp_eval_baselines/table4_local_ttp_llama32b.json; then
      echo "Table 4 local LLM evaluation complete (LLaMa 32B)!"
      echo "Results saved to: results/ttp_eval_baselines/table4_local_ttp_llama32b.json"
  else
      echo "Error: Table 4 local LLM evaluation failed (LLaMa 32B)" >&2
      exit 1
  fi
fi
