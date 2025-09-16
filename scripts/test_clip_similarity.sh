#!/bin/bash

# CLIP相似性测试脚本
# 测试文本提示与CLIP特征的相似性并生成热力图

cd ../data-preprocess

python test_clip_similarity.py \
    --features_dir ../../nerf_synthetic/ship_latents_processed_test/clip_features \
    --output_dir ../../nerf_synthetic/ship_latents_processed_test/clip_similarity_visualization \
    --text_prompt "a ship at sea" \
    --model ViT-L/14 \
    --device cuda \
    --save_individual \
    --save_grid
