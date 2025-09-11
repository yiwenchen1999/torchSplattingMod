#!/bin/bash

# FLUX VAE encoding script
# Use FLUX encoder to encode images to latent representations

cd ../data-preprocess

# Target scene list
targets=("statue" "bus" "houseA" "houseB" "shop" "atlus" "chair" "lizard" "lamp" "apron" "sofa" "cake" "cheeseplate")

for target in "${targets[@]}"; do
    echo "Processing target: $target"
    
    # Use FLUX VAE to encode images to latent representations
    python encode_with_flux.py \
        --input ../../objaverse_synthetic/${target}_processed_train \
        --output ../../objaverse_synthetic/${target}_processed_train/flux_latents_64 \
        --flux black-forest-labs/FLUX.1-dev \
        --size 512 \
        --device cuda \
        --fp16 \
        --save_meta
    
    echo "Completed FLUX encoding for $target (64x64 latents)"
    echo "----------------------------------------"
done

# Example usage comments:
# Encode single scene:
# python encode_with_flux.py \
#     --input /path/to/images \
#     --output /path/to/latents \
#     --flux black-forest-labs/FLUX.1-dev \
#     --size 1024 \
#     --device cuda \
#     --fp16 \
#     --save_meta

# Encode different sizes:
# python encode_with_flux.py \
#     --input ../../objaverse_synthetic/statue_processed_train \
#     --output ../../objaverse_synthetic/statue_processed_train/flux_latents_128 \
#     --flux black-forest-labs/FLUX.1-dev \
#     --size 1024 \
#     --device cuda \
#     --fp16

# python encode_with_flux.py \
#     --input ../../objaverse_synthetic/statue_processed_train \
#     --output ../../objaverse_synthetic/statue_processed_train/flux_latents_256 \
#     --flux black-forest-labs/FLUX.1-dev \
#     --size 2048 \
#     --device cuda \
#     --fp16
