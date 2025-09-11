#!/bin/bash

# FLUX VAE decoding script
# Use FLUX decoder to decode latent representations to images

cd ../data-preprocess

# Decode FLUX latents to images
python decode_with_flux.py \
    --input ../../nerf_synthetic/ship_latents_processed_test/flux_latents_64 \
    --output ../../nerf_synthetic/ship_latents_processed_test/flux_latents_64_decoded \
    --flux black-forest-labs/FLUX.1-dev \
    --device cuda \
    --fp16

# Example usage comments:

# Decode single latent folder:
# python decode_with_flux.py \
#     --input /path/to/latents \
#     --output /path/to/decoded_images \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16

# Decode different sized latents:
# python decode_with_flux.py \
#     --input ../../objaverse_synthetic/statue_processed_train/flux_latents_64 \
#     --output ../../objaverse_synthetic/statue_processed_train/flux_latents_64_decoded \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16

# python decode_with_flux.py \
#     --input ../../objaverse_synthetic/statue_processed_train/flux_latents_128 \
#     --output ../../objaverse_synthetic/statue_processed_train/flux_latents_128_decoded \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16

# python decode_with_flux.py \
#     --input ../../objaverse_synthetic/statue_processed_train/flux_latents_256 \
#     --output ../../objaverse_synthetic/statue_processed_train/flux_latents_256_decoded \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16

# Decode training results:
# python decode_with_flux.py \
#     --input ../result/statue_flux_64/eval_step_150000/ \
#     --output ../decoded_result/statue_flux_64/eval_step_150000_decoded \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16

# Disable mask visualization:
# python decode_with_flux.py \
#     --input /path/to/latents \
#     --output /path/to/decoded_images \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16 \
#     --no_mask_viz

# Use different output format:
# python decode_with_flux.py \
#     --input /path/to/latents \
#     --output /path/to/decoded_images \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16 \
#     --ext .jpg
