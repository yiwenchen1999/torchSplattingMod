#!/bin/bash

# FLUX VAE encoding script
# Use FLUX encoder to encode images to latent representations

cd ../data-preprocess

python encode_with_flux.py \
    --input ../../nerf_synthetic/ship_latents_processed_test \
    --output ../../nerf_synthetic/ship_latents_processed_test/flux_latents_64 \
    --flux black-forest-labs/FLUX.1-dev \
    --size 512 \
    --device cuda \
    --fp16 \
    --sample \
    --save_meta

