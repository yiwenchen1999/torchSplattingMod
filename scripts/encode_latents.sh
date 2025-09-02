cd ../data-preprocess
target=atlus
python encode_with_vae.py \
--input ../../objaverse_synthetic/${target}_processed_train \
--output ../../objaverse_synthetic/${target}_processed_train/vae_latents_64 \
--size 512

python encode_with_vae.py \
--input ../../objaverse_synthetic/${target}_processed_train \
--output ../../objaverse_synthetic/${target}_processed_train/vae_latents_128 \
--size 1024

target=shop
python encode_with_vae.py \
--input ../../objaverse_synthetic/${target}_processed_train \
--output ../../objaverse_synthetic/${target}_processed_train/vae_latents_64 \
--size 512

python encode_with_vae.py \
--input ../../objaverse_synthetic/${target}_processed_train \
--output ../../objaverse_synthetic/${target}_processed_train/vae_latents_128 \
--size 1024

target=statue
python encode_with_vae.py \
--input ../../objaverse_synthetic/${target}_processed_train \
--output ../../objaverse_synthetic/${target}_processed_train/vae_latents_64 \
--size 512

python encode_with_vae.py \
--input ../../objaverse_synthetic/${target}_processed_train \
--output ../../objaverse_synthetic/${target}_processed_train/vae_latents_128 \
--size 1024

