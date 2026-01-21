#!/bin/bash
#SBATCH --job-name=ttp_eval_local_llms
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/TowardsSaferPretraining}}"
cd "$PROJECT_DIR"
source venv/bin/activate

# Load env keys if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

mkdir -p results/ttp_eval_baselines results/codecarbon
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-$PROJECT_DIR/results/codecarbon}"
export CODECARBON_EXPERIMENT_ID="${CODECARBON_EXPERIMENT_ID:-${SLURM_JOB_ID:-}}"

# Default local models for paper Table 4 rows (local LLMs).
LOCAL_MODELS=()
if [ -n "${GEMMA_2_27B_MODEL_ID:-}" ]; then
  LOCAL_MODELS+=("$GEMMA_2_27B_MODEL_ID")
else
  LOCAL_MODELS+=("google/gemma-2-27b-it")
fi
if [ -n "${R1_MODEL_ID:-}" ]; then
  LOCAL_MODELS+=("$R1_MODEL_ID")
fi

EXTRA_ARGS=()
for m in "${LOCAL_MODELS[@]}"; do
  EXTRA_ARGS+=(--local-model "$m")
done

# Optional: quantize to fit large models on a single GPU.
QUANT="${TTP_LOCAL_QUANTIZATION:-none}"

python scripts/evaluate_ttp_eval.py \
  --setups local_ttp \
  --output "results/ttp_eval_baselines/results.json" \
  --device cuda \
  --quantization "$QUANT" \
  "${EXTRA_ARGS[@]}" \
  --dimension toxic

