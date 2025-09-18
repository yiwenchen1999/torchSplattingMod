#!/bin/bash

# DINOv3特征编码脚本
# 使用DINOv3模型编码图像为特征图

cd ../data-preprocess

# python encode_with_dino.py \
#     --input ../../nerf_synthetic/ship_latents_processed_test \
#     --output ../../nerf_synthetic/ship_latents_processed_test/dino_features_v2_16 \
#     --model facebook/dinov2-large \
#     --size 224 \
#     --device cuda \
#     --save_meta

python encode_with_dino.py \
    --input ../../nerf_synthetic/ship_latents_processed_test \
    --output ../../nerf_synthetic/ship_latents_processed_test/dino_features_v2_32 \
    --model facebook/dinov2-large \
    --size 448 \
    --device cuda \
    --save_meta

python encode_with_dino.py \
    --input ../../nerf_synthetic/ship_latents_processed_test \
    --output ../../nerf_synthetic/ship_latents_processed_test/dino_features_v2_64 \
    --model facebook/dinov2-large \
    --size 896 \
    --device cuda \
    --save_meta


# python encode_with_dino.py \
#     --input ../../nerf_synthetic/ship_latents_processed_test \
#     --output ../../nerf_synthetic/ship_latents_processed_test/dino_features_v3 \
#     --model facebook/dinov3-vitl16-pretrain-lvd1689m \
#     --size 224 \
#     --device cuda \
#     --save_meta
