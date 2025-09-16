#!/bin/bash

# CLIP全局特征测试脚本
# 测试全局特征是否与标准CLIP编码的图像特征相同

cd ../data-preprocess

python test_clip_global_features.py \
    --model ViT-L/14 \
    --device cuda \
    --num_tests 10 \
    --save_results
