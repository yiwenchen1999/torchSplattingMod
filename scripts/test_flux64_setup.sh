#!/bin/bash

# Test script to verify FLUX64 setup before running the full pipeline

echo "=== Testing FLUX64 Setup ==="
echo ""

# Test 1: Check if data directory exists
DATA_ROOT="/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense"
if [ -d "$DATA_ROOT" ]; then
    echo "✓ Data root directory exists: $DATA_ROOT"
else
    echo "❌ Data root directory not found: $DATA_ROOT"
    exit 1
fi

# Test 2: Count folders with GSTrain and GSTest
eligible_folders=$(find "$DATA_ROOT" -maxdepth 1 -type d -exec test -d {}/GSTrain \; -exec test -d {}/GSTest \; -print | wc -l)
echo "✓ Found $eligible_folders folders with GSTrain and GSTest"

# Test 3: Check a sample folder
sample_folder=$(find "$DATA_ROOT" -maxdepth 1 -type d -exec test -d {}/GSTrain \; -exec test -d {}/GSTest \; -print | head -1)
if [ -n "$sample_folder" ]; then
    folder_name=$(basename "$sample_folder")
    echo "✓ Sample folder: $folder_name"
    
    # Check image counts
    train_count=$(find "$sample_folder/GSTrain" -name "gt_*.png" | wc -l)
    test_count=$(find "$sample_folder/GSTest" -name "gt_*.png" | wc -l)
    echo "  - GSTrain images: $train_count"
    echo "  - GSTest images: $test_count"
    
    if [ "$train_count" -ge 200 ] && [ "$test_count" -ge 100 ]; then
        echo "  ✓ Sample folder meets requirements"
    else
        echo "  ⚠ Sample folder doesn't meet minimum requirements"
    fi
else
    echo "❌ No eligible folders found"
    exit 1
fi

# Test 4: Check if encode_with_flux.py exists
if [ -f "data-preprocess/encode_with_flux.py" ]; then
    echo "✓ FLUX encoding script exists"
else
    echo "❌ FLUX encoding script not found: data-preprocess/encode_with_flux.py"
    exit 1
fi

# Test 5: Check if train_latents.py exists
if [ -f "train_latents.py" ]; then
    echo "✓ Training script exists"
else
    echo "❌ Training script not found: train_latents.py"
    exit 1
fi

# Test 6: Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "⚠ nvidia-smi not available (may be normal in some environments)"
fi

echo ""
echo "=== Setup Test Complete ==="
echo "If all tests passed, you can run:"
echo "  sbatch scripts/GSconstruct_flux64.sh"
echo ""
echo "To monitor progress:"
echo "  bash scripts/monitor_flux64_progress.sh"
