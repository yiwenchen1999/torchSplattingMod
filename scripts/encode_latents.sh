cd ../data-preprocess
python encode_with_vae.py \
--input ../../nerf_synthetic/ship_latents_processed_test \
--output ../../nerf_synthetic/ship_latents_processed_test/vae_latents_256 \
--size 2048

cd ..
python train_latents.py 

