cd ../data-preprocess
targets=("houseA" "shop")

for target in "${targets[@]}"; do
    echo "Processing target: $target"
    
    if [ ! -d "../decoded_result/${target}_64/eval_step_100000_decoded" ] || [ -z "$(ls -A ../decoded_result/${target}_64/eval_step_100000_decoded 2>/dev/null)" ]; then
        python encode_with_vae.py \
        --input ../../objaverse_synthetic/${target}_processed_train \
        --output ../../objaverse_synthetic/${target}_processed_train/vae_latents_64 \
        --size 512
        cd ..
        python train_latents.py \
        --folder ../objaverse_synthetic \
        --scene_name ${target} \
        --image_size 64
        cd data-preprocess
        python decode_with_vae.py \
        --input ../result/${target}_64/eval_step_100000/ \
        --output ../decoded_result/${target}_64/eval_step_100000_decoded
    fi
    
    if [ ! -d "../decoded_result/${target}_128/eval_step_100000_decoded" ] || [ -z "$(ls -A ../decoded_result/${target}_128/eval_step_100000_decoded 2>/dev/null)"]; then
        if [ ! -d "../../objaverse_synthetic/${target}_processed_train/vae_latents_128" ]; then
            python encode_with_vae.py \
            --input ../../objaverse_synthetic/${target}_processed_train \
            --output ../../objaverse_synthetic/${target}_processed_train/vae_latents_128 \
            --size 1024
        fi
        cd ..
        python train_latents.py \
        --folder ../objaverse_synthetic \
        --scene_name ${target} \
        --image_size 128
        cd data-preprocess
        python decode_with_vae.py \
        --input ../result/${target}_128/eval_step_100000/ \
        --output ../decoded_result/${target}_128/eval_step_100000_decoded
    fi


    if [ ! -d "../decoded_result/${target}_256/eval_step_100000_decoded" ] || [ -z "$(ls -A ../decoded_result/${target}_256/eval_step_100000_decoded 2>/dev/null)" ]; then
        if [ ! -d "../../objaverse_synthetic/${target}_processed_train/vae_latents_256" ]; then
        python encode_with_vae.py \
        --input ../../objaverse_synthetic/${target}_processed_train \
        --output ../../objaverse_synthetic/${target}_processed_train/vae_latents_256 \
        --size 2048
        fi
        cd ..
        python train_latents.py \
        --folder ../objaverse_synthetic \
        --scene_name ${target} \
        --image_size 256
        cd data-preprocess
        python decode_with_vae.py \
        --input ../result/${target}_256/eval_step_100000/ \
        --output ../decoded_result/${target}_256/eval_step_100000_decoded
    fi
    
    echo "Completed processing for target: $target"
    echo "----------------------------------------"
done

