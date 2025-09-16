#!/bin/bash

# 多提示CLIP相似性测试脚本
# 测试多个文本提示与CLIP特征的相似性

cd ../data-preprocess

# 定义要测试的提示列表
prompts=(
    "a ship at sea"
)

# 为每个提示运行测试
for prompt in "${prompts[@]}"; do
    echo "Testing prompt: '$prompt'"
    
    # 创建安全的输出目录名（替换空格和特殊字符）
    safe_prompt=$(echo "$prompt" | sed 's/[^a-zA-Z0-9]/_/g')
    output_dir="../../nerf_synthetic/ship_latents_processed_test/clip_similarity_${safe_prompt}"
    
    python test_clip_similarity.py \
        --features_dir ../../nerf_synthetic/ship_latents_processed_test/clip_features \
        --output_dir "$output_dir" \
        --text_prompt "$prompt" \
        --model ViT-L/14 \
        --device cuda \
        --save_individual \
        --save_grid
    
    echo "Results saved to: $output_dir"
    echo "---"
done

echo "All prompt tests completed!"
