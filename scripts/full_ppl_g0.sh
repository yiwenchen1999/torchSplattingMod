cd ../data-preprocess
targets=("statue" "bus" "houseA" "houseB" "shop" "atlus" "chair" "lizard" "lamp" "apron" "sofa" "cake" "cheeseplate" "houseB")

for target in "${targets[@]}"; do
    echo "Processing target: $target"
    
    # python encode_with_vae.py \
    # --input ../../objaverse_synthetic/${target}_processed_train \
    # --output ../../objaverse_synthetic/${target}_processed_train/vae_latents_64 \
    # --size 512
    # cd ..
    # python train_latents.py \
    # --folder ../objaverse_synthetic \
    # --scene_name ${target} \
    # --image_size 64
    # cd data-preprocess
    python decode_with_vae.py \
    --input ../result/${target}_64/eval_step_150000/ \
    --output ../decoded_result/${target}_64/eval_step_150000_decoded

    # python encode_with_vae.py \
    # --input ../../objaverse_synthetic/${target}_processed_train \
    # --output ../../objaverse_synthetic/${target}_processed_train/vae_latents_128 \
    # --size 1024
    # cd ..
    # python train_latents.py \
    # --folder ../objaverse_synthetic \
    # --scene_name ${target} \
    # --image_size 128
    # cd data-preprocess
    # python decode_latents.py \
    # --input ../result/${target}_128/eval_step_150000/ \
    # --output ../decoded_result/${target}_128/eval_step_150000_decoded

    # python encode_with_vae.py \
    # --input ../../objaverse_synthetic/${target}_processed_train \
    # --output ../../objaverse_synthetic/${target}_processed_train/vae_latents_256 \
    # --size 2048
    # cd ..
    # python train_latents.py \
    # --folder ../objaverse_synthetic \
    # --scene_name ${target} \
    # --image_size 256
    # cd data-preprocess
    # python decode_latents.py \
    # --input ../result/${target}_256/eval_step_150000/ \
    # --output ../decoded_result/${target}_256/eval_step_150000_decoded

    
    echo "Completed processing for target: $target"
    echo "----------------------------------------"
done

