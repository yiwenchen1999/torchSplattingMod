cd ../data-preprocess

python decode_with_vae.py \
    --input ../../nerf_synthetic/ship_latents_processed_test/vae_latents_64 \
    --output ../../nerf_synthetic/ship_latents_processed_test/vae_latents_64_decoded_with_flux \
    --flux black-forest-labs/FLUX.1-dev \
    --device cuda \
    --fp16

# python decode_with_vae.py \
# --input ../result/ship_latents_fullEval_128/eval_step_100000/ \
# --output ../decoded_result/ship_latents_fullEval_128/eval_step_100000_decoded

# python decode_with_vae.py \
# --input ../result/ship_latents_64_fullEval/eval_step_100000/ \
# --output ../decoded_result/ship_latents_64_fullEval/eval_step_100000_decoded

# python decode_with_vae.py \
# --input ../../nerf_synthetic/ship_latents_processed_test/vae_latents_128 \
# --output ../decoded_result/ship_latents_processed_test/vae_latents_128_decoded

# python decode_with_vae.py \
# --input ../../nerf_synthetic/ship_latents_processed_test/vae_latents_64 \
# --output ../decoded_result/ship_latents_processed_test/vae_latents_64_decoded

#  python decode_with_vae.py \
# --input ../result/statue_64_fullEval/eval_step_100000/ \
# --output ../result/statue_64_fullEval/eval_step_100000_decoded

# python decode_with_vae.py \
# --input ../result/shop_64_fullEval/eval_step_100000/ \
# --output ../result/shop_64_fullEval/eval_step_100000_decoded


# python decode_with_vae.py \
# --input ../result/ship_latents_128_adjustedInit \
# --output ../result/ship_latents_128_decoded_adjustedInit
