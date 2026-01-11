#!/bin/bash
# Enable strict error handling
set -euo pipefail

#SBATCH --job-name=setup_env
#SBATCH --partition=thin
#SBATCH --time=00:30:00
#SBATCH --mem=8G

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Use environment variable for virtualenv path with sensible default
VENV_DIR="${VENV_DIR:-./venv}"

# Check if virtualenv already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Environment setup complete!"
