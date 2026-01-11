#!/bin/bash
set -euo pipefail

#SBATCH --job-name=harmformer_eval
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/harmformer_eval_%j.out
#SBATCH --error=logs/harmformer_eval_%j.err

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

# Run HarmFormer test with error checking
if python scripts/test_harmformer.py; then
    echo "HarmFormer Evaluation Complete!"
else
    echo "Error: HarmFormer evaluation failed" >&2
    exit 1
fi
