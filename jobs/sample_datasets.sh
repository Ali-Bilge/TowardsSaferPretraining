#!/bin/bash
set -euo pipefail
#SBATCH --job-name=sample_datasets
#SBATCH --partition=thin
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/sample_datasets_%j.out

mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $HOME/TowardsSaferPretraining
source venv/bin/activate || exit 1

python scripts/sample_datasets.py || exit 1

echo "Dataset sampling complete!"
