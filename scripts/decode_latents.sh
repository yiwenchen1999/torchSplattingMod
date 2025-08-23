cd ../data-preprocess

# python decode_with_vae.py \
# --input ../../nerf_synthetic/ship_latents_processed_test/vae_latents \
# --output ../../nerf_synthetic/ship_latents_processed_test/vae_latents_decoded

python decode_with_vae.py \
--input ../result/ship_latents \
--output ../result/ship_latents_decoded
