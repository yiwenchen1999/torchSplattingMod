# FLUX VAE 支持

本文档说明如何使用修改后的编码和解码脚本来支持FLUX扩散模型的VAE编码器。

## 概述

修改后的脚本现在支持三种VAE类型：
1. **标准VAE**: 直接指定VAE模型路径
2. **SD VAE**: Stable Diffusion模型的VAE (默认)
3. **FLUX VAE**: FLUX扩散模型的VAE (新增)

## 主要差异

### FLUX VAE vs SD VAE

| 特性 | SD VAE | FLUX VAE |
|------|--------|----------|
| scaling_factor | 0.18215 | 0.3611 |
| shift_factor | None | 0.1159 |
| 编码公式 | `latents * SCALE` | `(latents - shift_factor) * scaling_factor` |
| 解码公式 | `latents / SCALE` | `latents / scaling_factor + shift_factor` |

## 使用方法

### 编码脚本 (encode_with_vae.py)

#### 使用FLUX VAE编码
```bash
python data-preprocess/encode_with_vae.py \
    --input /path/to/images \
    --output /path/to/latents \
    --flux black-forest-labs/FLUX.1-dev \
    --size 1024 \
    --device cuda \
    --fp16
```

#### 使用SD VAE编码 (默认)
```bash
python data-preprocess/encode_with_vae.py \
    --input /path/to/images \
    --output /path/to/latents \
    --sd runwayml/stable-diffusion-v1-5 \
    --size 1024 \
    --device cuda \
    --fp16
```

#### 使用自定义VAE编码
```bash
python data-preprocess/encode_with_vae.py \
    --input /path/to/images \
    --output /path/to/latents \
    --vae /path/to/custom/vae \
    --size 1024 \
    --device cuda \
    --fp16
```

### 解码脚本 (decode_with_vae.py)

#### 使用FLUX VAE解码
```bash
python data-preprocess/decode_with_vae.py \
    --input /path/to/latents \
    --output /path/to/images \
    --flux black-forest-labs/FLUX.1-dev \
    --device cuda \
    --fp16
```

#### 使用SD VAE解码 (默认)
```bash
python data-preprocess/decode_with_vae.py \
    --input /path/to/latents \
    --output /path/to/images \
    --sd runwayml/stable-diffusion-v1-5 \
    --device cuda \
    --fp16
```

#### 使用自定义VAE解码
```bash
python data-preprocess/decode_with_vae.py \
    --input /path/to/latents \
    --output /path/to/images \
    --vae /path/to/custom/vae \
    --device cuda \
    --fp16
```

## 新增参数

### encode_with_vae.py
- `--flux`: 指定FLUX模型路径，将使用其VAE子文件夹

### decode_with_vae.py
- `--flux`: 指定FLUX模型路径，将使用其VAE子文件夹

## 元数据信息

当使用 `--save_meta` 参数时，编码脚本会保存额外的元数据信息：

```json
{
  "source": "path/to/source/image.png",
  "orig_size_hw": [1024, 1024],
  "size_input": 1024,
  "latent_shape": [4, 128, 128],
  "mask_shape": [128, 128],
  "dtype": "float16",
  "scale_factor": 0.3611,
  "shift_factor": 0.1159,
  "vae_type": "flux",
  "mode": "mean",
  "vae": "black-forest-labs/FLUX.1-dev/vae",
  "mask_note": "mask is 1 where input alpha>0, resized with nearest to latent resolution"
}
```

## 测试

运行测试脚本验证功能：

```bash
# 简单功能测试
python test_flux_vae_simple.py

# 完整编码解码测试
bash scripts/test_flux_vae.sh
```

## 注意事项

1. **兼容性**: 使用FLUX VAE编码的潜在表示必须使用FLUX VAE解码，反之亦然
2. **内存使用**: FLUX模型较大，建议使用 `--fp16` 参数减少内存使用
3. **设备**: 确保有足够的GPU内存，或使用 `--device cpu` 进行CPU推理
4. **模型下载**: 首次使用FLUX模型时会自动下载，请确保网络连接正常

## 错误排查

### 常见错误

1. **CUDA内存不足**
   - 使用 `--fp16` 参数
   - 使用 `--device cpu`
   - 减少 `--size` 参数

2. **模型加载失败**
   - 检查网络连接
   - 验证模型路径是否正确
   - 确保有足够的磁盘空间

3. **编码解码不匹配**
   - 确保编码和解码使用相同的VAE类型
   - 检查潜在表示文件是否完整

## 示例工作流

```bash
# 1. 使用FLUX VAE编码图像
python data-preprocess/encode_with_vae.py \
    --input datasamples/nerf_synthetic/ship_latents_processed_train \
    --output flux_encoded_latents \
    --flux black-forest-labs/FLUX.1-dev \
    --size 1024 \
    --device cuda \
    --fp16 \
    --save_meta

# 2. 使用FLUX VAE解码潜在表示
python data-preprocess/decode_with_vae.py \
    --input flux_encoded_latents \
    --output flux_decoded_images \
    --flux black-forest-labs/FLUX.1-dev \
    --device cuda \
    --fp16

# 3. 比较结果
ls -la flux_decoded_images/
```

这样就完成了FLUX VAE支持的集成！
