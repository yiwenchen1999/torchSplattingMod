#!/bin/bash

# CLIP特征编码脚本
# 使用CLIP ViT模型编码图像为特征图

cd ../data-preprocess

# python encode_with_clip.py \
#     --input ../../nerf_synthetic/ship_latents_processed_test \
#     --output ../../nerf_synthetic/ship_latents_processed_test/clip_features_raw \
#     --model ViT-L/14 \
#     --size 224 \
#     --device cuda \
#     --save_meta

# python encode_with_clip.py \
#     --input ../../nerf_synthetic/ship_latents_processed_test \
#     --output ../../nerf_synthetic/ship_latents_processed_test/clip_features_algined_16 \
#     --model ViT-L/14 \
#     --size 224 \
#     --device cuda \
#     --save_meta \
#     --spatial_mode processed

python encode_with_clip.py \
    --input ../../nerf_synthetic/ship_latents_processed_test \
    --output ../../nerf_synthetic/ship_latents_processed_test/clip_features_algined_32 \
    --model ViT-L/14 \
    --size 448 \
    --device cuda \
    --save_meta \
    --spatial_mode processed

python encode_with_clip.py \
    --input ../../nerf_synthetic/ship_latents_processed_test \
    --output ../../nerf_synthetic/ship_latents_processed_test/clip_features_algined_64 \
    --model ViT-L/14 \
    --size 896 \
    --device cuda \
    --save_meta \
    --spatial_mode processed