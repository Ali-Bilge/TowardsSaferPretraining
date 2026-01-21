#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=rome
#SBATCH --time=00:30:00
#SBATCH --mem=8G

# Enable strict error handling
set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Use environment variable for virtualenv path with sensible default
VENV_DIR="${VENV_DIR:-./venv}"

# Use environment variable for requirements file path with sensible default
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

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
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Error: Requirements file '$REQUIREMENTS_FILE' not found in $(pwd)" >&2
    exit 1
fi

echo "Environment setup complete!"
