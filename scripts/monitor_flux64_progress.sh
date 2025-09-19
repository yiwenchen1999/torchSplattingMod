#!/bin/bash

# Script to monitor and manage FLUX64 processing progress

PROGRESS_FILE="scripts/completed_flux64_scenes.txt"
DATA_ROOT="/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense"

echo "=== FLUX64 Processing Progress Monitor ==="
echo ""

# Check if progress file exists
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "Progress file not found: $PROGRESS_FILE"
    echo "Run GSconstruct_flux64.sh first to start processing"
    exit 1
fi

# Count completed scenes
completed_count=$(wc -l < "$PROGRESS_FILE")
echo "Completed scenes: $completed_count"

# Count total folders with GSTrain and GSTest
total_folders=$(find "$DATA_ROOT" -maxdepth 1 -type d -exec test -d {}/GSTrain \; -exec test -d {}/GSTest \; -print | wc -l)
echo "Total folders with GSTrain and GSTest: $total_folders"

# Calculate progress percentage
if [ "$total_folders" -gt 0 ]; then
    progress_percent=$((completed_count * 100 / total_folders))
    echo "Progress: $progress_percent% ($completed_count/$total_folders)"
else
    echo "Progress: 0% (no eligible folders found)"
fi

echo ""

# Show completed scenes
if [ "$completed_count" -gt 0 ]; then
    echo "Completed scenes:"
    cat "$PROGRESS_FILE" | sed 's/^/  /'
else
    echo "No scenes completed yet"
fi

echo ""

# Show next few folders to be processed
echo "Next folders to be processed:"
find "$DATA_ROOT" -maxdepth 1 -type d | grep -v "^$DATA_ROOT$" | sort | while read folder; do
    folder_name=$(basename "$folder")
    scene_name="${folder_name}_flux"
    
    # Check if this scene is already completed
    if grep -q "^$scene_name$" "$PROGRESS_FILE"; then
        continue
    fi
    
    # Check if GSTrain and GSTest exist
    if [ -d "$folder/GSTrain" ] && [ -d "$folder/GSTest" ]; then
        train_count=$(find "$folder/GSTrain" -name "gt_*.png" | wc -l)
        test_count=$(find "$folder/GSTest" -name "gt_*.png" | wc -l)
        
        if [ "$train_count" -ge 200 ] && [ "$test_count" -ge 100 ]; then
            echo "  $folder_name (train: $train_count, test: $test_count)"
        fi
    fi
done | head -10

echo ""
echo "To resume processing, run:"
echo "  sbatch scripts/GSconstruct_flux64.sh"
