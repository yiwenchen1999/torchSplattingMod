cd ../data-preprocess

# python decode_with_vae.py \
# --input ../../nerf_synthetic/ship_latents_processed_test/vae_latents \
# --output ../../nerf_synthetic/ship_latents_processed_test/vae_latents_decoded

python decode_with_vae.py \
--input ../result/atlus_64_fullEval/eval_step_100000/ \
--output ../result/atlus_64_fullEval/eval_step_100000_decoded

# python decode_with_vae.py \
# --input ../result/statue_64_fullEval/eval_step_100000/ \
# --output ../result/statue_64_fullEval/eval_step_100000_decoded

# python decode_with_vae.py \
# --input ../result/shop_64_fullEval/eval_step_100000/ \
# --output ../result/shop_64_fullEval/eval_step_100000_decoded


# python decode_with_vae.py \
# --input ../result/ship_latents_128_adjustedInit \
# --output ../result/ship_latents_128_decoded_adjustedInit
