cd ../data-preprocess
python encode_with_vae.py \
--input ../../nerf_synthetic/ship_latents_processed_test \
--output ../../nerf_synthetic/ship_latents_processed_test/vae_latents_64 \
--size 512

cd ..
python train_latents.py 

