# CLIP相似性测试脚本说明

本文档介绍如何使用CLIP相似性测试脚本来分析文本提示与图像特征之间的相似性。

## 脚本概述

### 1. 主要脚本 (`test_clip_similarity.py`)

测试单个文本提示与CLIP特征之间的相似性，生成热力图可视化。

**功能：**
- 加载CLIP特征文件
- 编码文本提示
- 计算余弦相似性
- 生成归一化热力图
- 保存统计信息

### 2. 执行脚本

#### 单提示测试 (`test_clip_similarity.sh`)
```bash
./scripts/test_clip_similarity.sh
```

#### 多提示测试 (`test_multiple_prompts.sh`)
```bash
./scripts/test_multiple_prompts.sh
```

## 使用方法

### 基本用法

```bash
# 测试单个提示
python data-preprocess/test_clip_similarity.py \
    --features_dir /path/to/clip/features \
    --output_dir /path/to/output \
    --text_prompt "a ship at sea" \
    --model ViT-L/14 \
    --device cuda \
    --save_individual \
    --save_grid
```

### 参数说明

- `--features_dir`: CLIP特征文件目录（包含.npy文件）
- `--output_dir`: 输出可视化结果的目录
- `--text_prompt`: 要测试的文本提示（默认："a ship at sea"）
- `--model`: CLIP模型名称（默认：ViT-L/14）
- `--device`: 计算设备（默认：cuda）
- `--save_individual`: 保存每个图像的单独热力图
- `--save_grid`: 保存所有热力图的网格图

## 输出文件

### 1. 热力图文件
- **单独热力图**: `{feature_name}_heatmap.png`
- **网格热力图**: `similarity_grid.png`

### 2. 统计文件
- **统计信息**: `similarity_stats.json`

统计信息包含：
```json
{
  "text_prompt": "a ship at sea",
  "model": "ViT-L/14",
  "num_images": 100,
  "mean_similarity": 0.2345,
  "std_similarity": 0.1234,
  "min_similarity": 0.0123,
  "max_similarity": 0.4567,
  "feature_files": ["file1.npy", "file2.npy", ...]
}
```

## 热力图解释

### 颜色编码
- **红色/黄色**: 高相似性（与文本提示匹配的区域）
- **蓝色/黑色**: 低相似性（与文本提示不匹配的区域）

### 归一化
- 相似性分数被归一化到[0,1]范围
- 每个图像独立归一化，确保最佳可视化效果

## 示例用法

### 1. 测试船舶相关提示
```bash
python test_clip_similarity.py \
    --features_dir ../../nerf_synthetic/ship_latents_processed_test/clip_features \
    --output_dir ./ship_similarity \
    --text_prompt "a ship at sea" \
    --save_individual \
    --save_grid
```

### 2. 测试海洋相关提示
```bash
python test_clip_similarity.py \
    --features_dir ../../nerf_synthetic/ship_latents_processed_test/clip_features \
    --output_dir ./ocean_similarity \
    --text_prompt "ocean waves" \
    --save_individual \
    --save_grid
```

### 3. 批量测试多个提示
```bash
./scripts/test_multiple_prompts.sh
```

## 技术细节

### 相似性计算
1. **文本编码**: 使用CLIP文本编码器将提示转换为特征向量
2. **特征归一化**: 对图像特征进行L2归一化
3. **余弦相似性**: 计算文本特征与每个像素特征的余弦相似性
4. **空间映射**: 将相似性分数映射回空间维度

### 特征处理
- 输入特征形状: `(hidden_dim, height, width)`
- 输出热力图形状: `(height, width)`
- 对于ViT-L/14: `(1024, 16, 16)` → `(16, 16)`

## 依赖要求

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib
pip install numpy
pip install pillow
pip install tqdm
```

## 注意事项

1. **内存使用**: 大模型和大量图像可能需要较多GPU内存
2. **处理时间**: 相似性计算时间取决于图像数量和模型大小
3. **特征格式**: 确保输入特征文件是正确格式的.npy文件
4. **设备兼容**: 脚本会自动检测CUDA可用性

## 故障排除

### 常见问题

1. **CUDA内存不足**: 尝试使用CPU或减少批处理大小
2. **特征文件格式错误**: 检查.npy文件是否包含正确的CLIP特征
3. **模型加载失败**: 确保CLIP模型可以正常下载和加载

### 调试模式

可以修改脚本添加更多调试信息：
```python
# 在脚本中添加
print(f"Feature shape: {features.shape}")
print(f"Text feature shape: {text_features.shape}")
print(f"Similarity range: {similarity_map.min():.4f} - {similarity_map.max():.4f}")
```
