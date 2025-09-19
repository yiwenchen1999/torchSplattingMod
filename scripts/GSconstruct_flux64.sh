#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=gsconstruct_flux64
#SBATCH --mem=64
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h200:1
#SBATCH --output=myjob.gsconstruct_flux64.out
#SBATCH --error=myjob.gsconstruct_flux64.err

# Script to iterate through all Objaverse folders and:
# 1. Encode flux latents for GSTrain and GSTest if they exist
# 2. Train Gaussian Splatting with flux latents
# 3. Keep track of completed scenes

cd /projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/torchSplattingMod

# Configuration
DATA_ROOT="/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense"
PROGRESS_FILE="scripts/completed_flux64_scenes.txt"
FLUX_MODEL="black-forest-labs/FLUX.1-dev"
IMAGE_SIZE="64"
FEATURE_DIM="16"  # FLUX latent dimension

# Create progress file if it doesn't exist
touch "$PROGRESS_FILE"

echo "Starting GS Construction with FLUX64 encoding..."
echo "Data root: $DATA_ROOT"
echo "Progress file: $PROGRESS_FILE"
echo "FLUX model: $FLUX_MODEL"

# Function to check if a scene is already completed
is_scene_completed() {
    local scene_name="$1"
    grep -q "^$scene_name$" "$PROGRESS_FILE"
}

# Function to mark a scene as completed
mark_scene_completed() {
    local scene_name="$1"
    echo "$scene_name" >> "$PROGRESS_FILE"
    echo "  ✓ Marked $scene_name as completed"
}

# Function to encode flux latents
encode_flux_latents() {
    local input_folder="$1"
    local output_folder="$2"
    local folder_name="$3"
    
    echo "    Encoding FLUX latents for $folder_name..."
    
    cd data-preprocess
    python encode_with_flux.py \
        --input "$input_folder" \
        --output "$output_folder" \
        --flux "$FLUX_MODEL" \
        --size 512 \
        --device cuda \
        --fp16 \
        --sample \
        --save_meta
    
    local exit_code=$?
    cd ..
    
    if [ $exit_code -eq 0 ]; then
        echo "    ✓ FLUX encoding completed for $folder_name"
        return 0
    else
        echo "    ❌ FLUX encoding failed for $folder_name"
        return 1
    fi
}

# Function to train Gaussian Splatting
train_gaussian_splatting() {
    local input_folder="$1"
    local scene_name="$2"
    
    echo "    Training Gaussian Splatting for $scene_name..."
    
    python train_latents.py \
        --input_folder "$input_folder" \
        --scene_name "$scene_name" \
        --image_size "$IMAGE_SIZE" \
        --feature_dim "$FEATURE_DIM" \
        --latent_folder "flux_latents"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "    ✓ Training completed for $scene_name"
        return 0
    else
        echo "    ❌ Training failed for $scene_name"
        return 1
    fi
}

# Function to process a single folder
process_folder() {
    local folder_path="$1"
    local folder_name=$(basename "$folder_path")
    local scene_name="${folder_name}_flux"
    
    echo ""
    echo "Processing folder: $folder_name"
    echo "Scene name: $scene_name"
    
    # Check if already completed
    if is_scene_completed "$scene_name"; then
        echo "  ⏭️  Skipping $folder_name: already completed"
        return 0
    fi
    
    # Check if GSTrain and GSTest exist
    local gstrain_path="$folder_path/GSTrain"
    local gstest_path="$folder_path/GSTest"
    
    if [ ! -d "$gstrain_path" ] || [ ! -d "$gstest_path" ]; then
        echo "  ⏭️  Skipping $folder_name: GSTrain or GSTest not found"
        return 0
    fi
    
    # Check if GSTrain has enough images (at least 200)
    local train_image_count=$(find "$gstrain_path" -name "gt_*.png" | wc -l)
    if [ "$train_image_count" -lt 200 ]; then
        echo "  ⏭️  Skipping $folder_name: GSTrain has only $train_image_count images (need 200+)"
        return 0
    fi
    
    # Check if GSTest has enough images (at least 100)
    local test_image_count=$(find "$gstest_path" -name "gt_*.png" | wc -l)
    if [ "$test_image_count" -lt 100 ]; then
        echo "  ⏭️  Skipping $folder_name: GSTest has only $test_image_count images (need 100+)"
        return 0
    fi
    
    echo "  ✓ Found GSTrain ($train_image_count images) and GSTest ($test_image_count images)"
    
    # Check if flux latents already exist
    local train_flux_path="$gstrain_path/flux_latents_64"
    local test_flux_path="$gstest_path/flux_latents_64"
    
    local need_encoding=false
    if [ ! -d "$train_flux_path" ]; then
        echo "  📝 Need to encode GSTrain flux latents"
        need_encoding=true
    else
        echo "  ✓ GSTrain flux latents already exist"
    fi
    
    if [ ! -d "$test_flux_path" ]; then
        echo "  📝 Need to encode GSTest flux latents"
        need_encoding=true
    else
        echo "  ✓ GSTest flux latents already exist"
    fi
    
    # Encode flux latents if needed
    if [ "$need_encoding" = true ]; then
        echo "  🔄 Encoding FLUX latents..."
        
        if [ ! -d "$train_flux_path" ]; then
            if ! encode_flux_latents "$gstrain_path" "$train_flux_path" "GSTrain"; then
                echo "  ❌ Failed to encode GSTrain flux latents for $folder_name"
                return 1
            fi
        fi
        
        if [ ! -d "$test_flux_path" ]; then
            if ! encode_flux_latents "$gstest_path" "$test_flux_path" "GSTest"; then
                echo "  ❌ Failed to encode GSTest flux latents for $folder_name"
                return 1
            fi
        fi
    fi
    
    # Train Gaussian Splatting
    echo "  🎯 Training Gaussian Splatting..."
    if train_gaussian_splatting "$gstrain_path" "$scene_name"; then
        mark_scene_completed "$scene_name"
        echo "  🎉 Successfully completed $folder_name"
        return 0
    else
        echo "  ❌ Failed to train Gaussian Splatting for $folder_name"
        return 1
    fi
}

# Main processing loop
echo ""
echo "Starting main processing loop..."

processed_count=0
skipped_count=0
error_count=0

# Get all folders and sort them
folders=($(find "$DATA_ROOT" -maxdepth 1 -type d | grep -v "^$DATA_ROOT$" | sort))

echo "Found ${#folders[@]} folders to process"

for folder in "${folders[@]}"; do
    if process_folder "$folder"; then
        processed_count=$((processed_count + 1))
    else
        error_count=$((error_count + 1))
    fi
done

echo ""
echo "Processing complete!"
echo "✓ Successfully processed: $processed_count folders"
echo "❌ Errors: $error_count folders"
echo "📊 Total completed scenes: $(wc -l < "$PROGRESS_FILE")"

echo ""
echo "Completed scenes:"
cat "$PROGRESS_FILE"
