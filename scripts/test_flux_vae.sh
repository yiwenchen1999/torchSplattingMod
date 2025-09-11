#!/bin/bash

# 测试FLUX VAE编码和解码脚本
# 这个脚本演示如何使用修改后的编码和解码脚本

echo "=== 测试FLUX VAE编码和解码脚本 ==="

# 设置路径
INPUT_DIR="datasamples/nerf_synthetic/ship_latents_processed_train"
OUTPUT_ENCODED="test_flux_encoded"
OUTPUT_DECODED="test_flux_decoded"

# FLUX模型路径 (需要根据实际情况调整)
FLUX_MODEL="black-forest-labs/FLUX.1-dev"

echo "1. 使用FLUX VAE编码图像..."
python data-preprocess/encode_with_vae.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_ENCODED" \
    --flux "$FLUX_MODEL" \
    --size 1024 \
    --device cuda \
    --fp16 \
    --overwrite

echo "2. 使用FLUX VAE解码潜在表示..."
python data-preprocess/decode_with_vae.py \
    --input "$OUTPUT_ENCODED" \
    --output "$OUTPUT_DECODED" \
    --flux "$FLUX_MODEL" \
    --device cuda \
    --fp16

echo "3. 对比原始SD VAE编码和解码..."
echo "使用SD VAE编码..."
python data-preprocess/encode_with_vae.py \
    --input "$INPUT_DIR" \
    --output "test_sd_encoded" \
    --sd "runwayml/stable-diffusion-v1-5" \
    --size 1024 \
    --device cuda \
    --fp16 \
    --overwrite

echo "使用SD VAE解码..."
python data-preprocess/decode_with_vae.py \
    --input "test_sd_encoded" \
    --output "test_sd_decoded" \
    --sd "runwayml/stable-diffusion-v1-5" \
    --device cuda \
    --fp16

echo "=== 测试完成 ==="
echo "FLUX VAE编码结果: $OUTPUT_ENCODED"
echo "FLUX VAE解码结果: $OUTPUT_DECODED"
echo "SD VAE编码结果: test_sd_encoded"
echo "SD VAE解码结果: test_sd_decoded"
echo ""
echo "请检查输出目录中的结果，比较FLUX和SD VAE的差异。"
