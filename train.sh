#!/bin/bash

# Generic training script for YOLO Classification Pipeline
# Usage: ./train.sh [optional: --config path/to/config.yaml]

# Activate your environment here if needed
# source activate yolo_cls_env

# Default config file
CONFIG_FILE="config_template.yaml"

# Check if a custom config is provided
if [ "$1" == "--config" ] && [ -n "$2" ]; then
    CONFIG_FILE="$2"
elif [ -f "config.yaml" ]; then
    # Look for a user-created config.yaml in the current directory
    CONFIG_FILE="config.yaml"
fi

echo "Using configuration: $CONFIG_FILE"

# Run the pipeline
python run_pipeline.py --config "$CONFIG_FILE"
