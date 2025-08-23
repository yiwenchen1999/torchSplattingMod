cd ../data-preprocess

# python decode_with_vae.py \
# --input ../../nerf_synthetic/ship_latents_processed_test/vae_latents \
# --output ../../nerf_synthetic/ship_latents_processed_test/vae_latents_decoded

python decode_with_vae.py \
--input ../result/ship_latents_64_adjustedInit \
--output ../result/ship_latents_64_decoded_adjustedInit

python decode_with_vae.py \
--input ../result/ship_latents_128_adjustedInit \
--output ../result/ship_latents_128_decoded_adjustedInit
