#!/bin/bash

# Script to generate info.json files for bus dataset train and test splits

set -e  # Exit on any error

echo "Generating info.json files for bus dataset..."
cd ../data-preprocess

# Find all folders in datasamples/objaverse_synthetic that are not marked *_processed_*
for target_folder in ../../datasamples/objaverse_synthetic/*/; do
    # Extract just the folder name
    folder_name=$(basename "$target_folder")
    
    # Skip if it's a processed folder
    if [[ "$folder_name" == *"_processed_"* ]]; then
        echo "Skipping processed folder: $folder_name"
        continue
    fi
    
    # Skip if the folder doesn't exist or is not a directory
    if [[ ! -d "$target_folder" ]]; then
        continue
    fi
    
    echo "Processing folder: $folder_name"
    
    # Generate train split
    echo "Processing train split for $folder_name..."
    python3 preprocess_objaverse_data.py \
        --data_dir ../../datasamples/objaverse_synthetic/${folder_name} \
        --output_dir ../../datasamples/objaverse_synthetic/${folder_name}_processed_train \
        --split train --cleanup_input 

    echo "Train split completed for $folder_name!"

    # Generate test split
    echo "Processing test split for $folder_name..."
    python3 preprocess_objaverse_data.py \
        --data_dir ../../datasamples/objaverse_synthetic/${folder_name} \
        --output_dir ../../datasamples/objaverse_synthetic/${folder_name}_processed_test \
        --split test --cleanup_input

    echo "Test split completed for $folder_name!"
    
    echo "Completed processing $folder_name"
    echo "Train: ../../datasamples/objaverse_synthetic/${folder_name}_processed_train/info.json"
    echo "Test: ../../datasamples/objaverse_synthetic/${folder_name}_processed_test/info.json"
    echo "----------------------------------------"
done

echo "All done! Generated info.json files for all folders in objaverse_synthetic."
