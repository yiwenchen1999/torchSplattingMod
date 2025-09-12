cd ..
python train_latents.py \
--input_folder ../nerf_synthetic/ship_latents_processed_test \
--scene_name ship_latents_flux64_init256 \
--image_size 64 \
--feature_dim 16

# cd ../data-preprocess
# python decode_with_vae.py \
# --input ../result/ship_latents_flux64_64/eval_step_100000/ \
# --output ../decoded_result/ship_latents_flux64_64/eval_step_100000_decoded

python decode_with_flux.py \
    --input ../result/ship_latents_flux64_init128_64/eval_step_100000/ \
    --output ../decoded_result/ship_latents_flux64_init128_64/eval_step_100000_decoded \
    --flux black-forest-labs/FLUX.1-dev \
    --device cuda \
    --fp16
