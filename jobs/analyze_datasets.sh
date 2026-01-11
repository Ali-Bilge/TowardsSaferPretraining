#!/bin/bash
#SBATCH --job-name=analyze_datasets
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/analyze_datasets_%j.out
#SBATCH --error=logs/analyze_datasets_%j.err

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Enable strict mode for fail-fast error handling
set -euo pipefail

cd $HOME/TowardsSaferPretraining
source venv/bin/activate

# Create output directory
mkdir -p results/dataset_analysis

# Analyze C4 (100K samples)
echo "Analyzing C4 dataset..."
python scripts/analyze_dataset.py \
  --dataset "C4" \
  --jsonl datasets/samples/c4_100k.jsonl \
  --device cuda \
  --output results/dataset_analysis/c4_100k_analysis.json

# Analyze FineWeb (100K samples)
echo "Analyzing FineWeb dataset..."
python scripts/analyze_dataset.py \
  --dataset "FineWeb" \
  --jsonl datasets/samples/fineweb_100k.jsonl \
  --device cuda \
  --output results/dataset_analysis/fineweb_100k_analysis.json

echo "Dataset analysis complete!"
echo "Results saved to: results/dataset_analysis/"
