cd ..
python train_latents.py \
--folder ../nerf_synthetic/ship_latents_processed_test \
--scene_name ship_latents_fullEval \
--image_size 128

cd ../data-preprocess
python decode_with_vae.py \
--input ../result/ship_latents_fullEval_128/eval_step_150000/ \
--output ../decoded_result/ship_latents_fullEval_128/eval_step_150000_decoded
