#!/bin/bash
# Script to download model checkpoints and set environment variables
# Located in experiments/scripts/download_checkpoints.sh

# Default directory
DEFAULT_DIR="./pretrained_models/"
OUTPUT_DIR="${1:-$DEFAULT_DIR}"

# Make script executable
chmod +x $(dirname "$0")/download_checkpoints.py

# Run the Python script from the same directory as this script
python $(dirname "$0")/download_checkpoints.py --output-dir "$OUTPUT_DIR"

# Export the environment variable for this shell session
if [ $? -eq 0 ]; then
    export MODEL_CHECKPOINTS_DIR=$(realpath "$OUTPUT_DIR")
    echo "Exported MODEL_CHECKPOINTS_DIR=$MODEL_CHECKPOINTS_DIR for current session"
    echo "To use the models in your code, access them at \$MODEL_CHECKPOINTS_DIR"
fi