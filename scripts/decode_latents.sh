cd ../data-preprocess

python decode_with_vae.py \
    --input ../../nerf_synthetic/ship_latents_processed_test/vae_latents_64 \
    --output ../../nerf_synthetic/ship_latents_processed_test/vae_latents_64_decoded_with_flux \
    --flux black-forest-labs/FLUX.1-dev \
    --device cuda \
    --fp16

