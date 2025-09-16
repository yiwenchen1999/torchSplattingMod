#!/bin/bash

# CLIP特征编码脚本
# 使用CLIP ViT模型编码图像为特征图

cd ../data-preprocess

python encode_with_clip.py \
    --input ../../nerf_synthetic/ship_latents_processed_test \
    --output ../../nerf_synthetic/ship_latents_processed_test/clip_features \
    --model ViT-L/14 \
    --size 224 \
    --device cuda \
    --save_meta
