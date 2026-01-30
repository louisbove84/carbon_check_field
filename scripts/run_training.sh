#!/bin/bash
# Run test training with virtual environment
# This script activates the venv and runs the training script

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the training script with arguments passed through
python ml_pipeline/tools/test_training.py "$@"
