#!/bin/bash
set -e
#SBATCH --job-name=baselines
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/baselines_%j.out

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Verify target directory exists before cd
if [ ! -d "$HOME/TowardsSaferPretraining" ]; then
    echo "Error: Directory $HOME/TowardsSaferPretraining does not exist" >&2
    exit 1
fi
cd $HOME/TowardsSaferPretraining

# Verify virtualenv activation script exists before sourcing
if [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment activation script venv/bin/activate not found" >&2
    exit 1
fi
source venv/bin/activate

# Run baseline comparison
mkdir -p results
python scripts/compare_baselines.py \
  --output results/baseline_comparison.json

# Check if the Python command succeeded
if [ $? -eq 0 ]; then
    echo "Baseline Comparison Complete!"
else
    echo "Error: Baseline comparison failed!" >&2
    exit 1
fi
