#!/bin/bash

# # Preprocessing script for ship_latents
# # This script runs the Python preprocessing script to generate info.json

# echo "Starting ship_latents preprocessing..."

# # Check if Python script exists
# if [ ! -f "scripts/preprocess_ship_latents.py" ]; then
#     echo "Error: scripts/preprocess_ship_latents.py not found!"
#     exit 1
# fi

# # Check if ship_latents directory exists (relative to parent directory)
# if [ ! -d "../ship_latents" ]; then
#     echo "Error: ship_latents directory not found in parent directory!"
#     exit 1
# fi

# Run the preprocessing script
python preprocess_ship_latents.py \
    --ship_latents_dir ../../ship_latents \
    --output_dir ../../ship_latents_processed_test \
    --transforms_file transforms_test.json

python preprocess_ship_latents.py \
    --ship_latents_dir ../../ship_latents \
    --output_dir ../../ship_latents_processed_train \
    --transforms_file transforms_train.json

echo "Preprocessing completed!"
echo "Check the ship_latents_processed directory for the generated files."
