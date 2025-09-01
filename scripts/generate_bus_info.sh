#!/bin/bash

# Script to generate info.json files for bus dataset train and test splits

set -e  # Exit on any error

echo "Generating info.json files for bus dataset..."

# Generate train split
echo "Processing train split..."
python3 scripts/generate_bus_info.py \
    --data_dir datasamples/bus \
    --output_dir datasamples/bus_processed_train \
    --split train

echo "Train split completed!"

# Generate test split
echo "Processing test split..."
python3 scripts/generate_bus_info.py \
    --data_dir datasamples/bus \
    --output_dir datasamples/bus_processed_test \
    --split test

echo "Test split completed!"

echo "All done! Generated info.json files for both train and test splits."
echo "Train: datasamples/bus_processed_train/info.json"
echo "Test: datasamples/bus_processed_test/info.json"
