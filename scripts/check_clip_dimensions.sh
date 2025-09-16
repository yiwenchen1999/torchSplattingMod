#!/bin/bash

# CLIP模型维度检查脚本
# 检查不同CLIP模型版本的隐藏维度和最终维度

cd ../data-preprocess

python check_clip_dimensions.py \
    --models ViT-B/32 ViT-B/16 ViT-L/14 ViT-L/14@336px \
    --device cuda \
    --save_results
