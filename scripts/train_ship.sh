#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=train_ship
#SBATCH --mem=32
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h200:1
#SBATCH --output=myjob.train_ship.out
#SBATCH --error=myjob.train_ship.err

cd ..
# python train_latents.py \
# --input_folder ../nerf_synthetic/ship_latents_processed_test \
# --scene_name ship_latents_clip16 \
# --image_size 16 \
# --feature_dim 768 \
# --latent_folder clip_features_algined

python train_latents.py \
--input_folder ../nerf_synthetic/ship_latents_processed_test \
--scene_name ship_latents_clip32 \
--image_size 32 \
--feature_dim 768 \
--latent_folder clip_features_algined

python train_latents.py \
--input_folder ../nerf_synthetic/ship_latents_processed_test \
--scene_name ship_latents_clip64 \
--image_size 64 \
--feature_dim 768 \
--latent_folder clip_features_algined

# python train_latents.py \
# --input_folder ../nerf_synthetic/ship_latents_processed_test \
# --scene_name ship_latents_dino16 \
# --image_size 16 \
# --feature_dim 1024 \
# --latent_folder dino_features_v2

python train_latents.py \
--input_folder ../nerf_synthetic/ship_latents_processed_test \
--scene_name ship_latents_dino32 \
--image_size 32 \
--feature_dim 1024 \
--latent_folder dino_features_v2

python train_latents.py \
--input_folder ../nerf_synthetic/ship_latents_processed_test \
--scene_name ship_latents_dino64 \
--image_size 64 \
--feature_dim 1024 \
--latent_folder dino_features_v2

# cd ../data-preprocess
# python decode_with_vae.py \
# --input ../result/ship_latents_flux64_64/eval_step_100000/ \
# --output ../decoded_result/ship_latents_flux64_64/eval_step_100000_decoded

# python decode_with_flux.py \
#     --input ../result/ship_latents_flux64_init128_64/eval_step_100000/ \
#     --output ../decoded_result/ship_latents_flux64_init128_64/eval_step_100000_decoded \
#     --flux black-forest-labs/FLUX.1-dev \
#     --device cuda \
#     --fp16
