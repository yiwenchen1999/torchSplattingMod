# 特征编码脚本说明

本文档介绍如何使用新创建的特征编码脚本来提取CLIP和DINO v2特征。

## 脚本概述

### 1. CLIP特征编码 (`encode_with_clip.py`)

使用CLIP ViT模型的最后一层特征图（而不是统一向量）来编码图像。

**特点：**
- 使用CLIP ViT模型的最后一层特征图
- 支持多种CLIP模型（ViT-L/14, ViT-B/32等）
- 输出空间特征图而不是全局特征向量
- 包含详细的形状日志输出

**使用方法：**
```bash
# 使用bash脚本（推荐）
./scripts/encode_clip_features.sh --input /path/to/images --output /path/to/features

# 直接使用Python脚本
python data-preprocess/encode_with_clip.py --input /path/to/images --output /path/to/features --model ViT-L/14
```

### 2. DINO v2特征编码 (`encode_with_dino.py`)

使用DINO v2模型提取图像的特征图。

**特点：**
- 使用DINO v2的patch特征
- 支持多种DINO v2模型（small, base, large, giant）
- 输出空间特征图
- 包含详细的形状日志输出

**使用方法：**
```bash
# 使用bash脚本（推荐）
./scripts/encode_dino_features.sh --input /path/to/images --output /path/to/features

# 直接使用Python脚本
python data-preprocess/encode_with_dino.py --input /path/to/images --output /path/to/features --model facebook/dinov2-large
```

## 参数说明

### 通用参数

- `--input`: 输入图像目录（必需）
- `--output`: 输出特征目录（必需）
- `--size`: 输入图像尺寸（默认：224）
- `--device`: 计算设备（默认：cuda）
- `--fp16`: 使用float16精度
- `--overwrite`: 覆盖已存在的文件
- `--save_meta`: 保存元数据JSON文件

### CLIP特定参数

- `--model`: CLIP模型名称
  - `ViT-L/14` (默认)
  - `ViT-B/32`
  - `ViT-B/16`
  - 其他CLIP模型

### DINO v2特定参数

- `--model`: DINO v2模型名称
  - `facebook/dinov2-small` (小模型)
  - `facebook/dinov2-base` (基础模型)
  - `facebook/dinov2-large` (大模型，默认)
  - `facebook/dinov2-giant` (巨型模型)

## 输出格式

每个脚本会生成以下文件：

1. **特征文件** (`*.npy`): 包含特征图的NumPy数组
2. **掩码文件** (`*_mask.npy`): 包含alpha通道掩码
3. **元数据文件** (`*.json`): 包含处理信息（如果使用`--save_meta`）

### 特征形状

- **CLIP**: `(hidden_dim, grid_size, grid_size)`
  - ViT-L/14: `(1024, 16, 16)`
  - ViT-B/32: `(512, 7, 7)`

- **DINO v2**: `(hidden_dim, grid_size, grid_size)`
  - dinov2-large: `(1024, 16, 16)`
  - dinov2-base: `(768, 16, 16)`
  - dinov2-small: `(384, 16, 16)`

## 示例用法

### 基本用法

```bash
# CLIP特征编码
./scripts/encode_clip_features.sh \
    --input /path/to/input/images \
    --output /path/to/clip/features

# DINO v2特征编码
./scripts/encode_dino_features.sh \
    --input /path/to/input/images \
    --output /path/to/dino/features
```

### 高级用法

```bash
# 使用不同模型和精度
./scripts/encode_clip_features.sh \
    --input /path/to/input/images \
    --output /path/to/clip/features \
    --model ViT-B/32 \
    --fp16 \
    --save_meta

# 使用DINO v2基础模型
./scripts/encode_dino_features.sh \
    --input /path/to/input/images \
    --output /path/to/dino/features \
    --model facebook/dinov2-base \
    --overwrite
```

## 依赖要求

确保安装了以下Python包：

```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install pillow
pip install tqdm
pip install numpy
```

对于CLIP：
```bash
pip install git+https://github.com/openai/CLIP.git
```

## 注意事项

1. **内存使用**: 大模型（如ViT-L/14, dinov2-large）需要更多GPU内存
2. **处理时间**: 特征提取时间取决于模型大小和图像数量
3. **文件格式**: 支持常见图像格式（PNG, JPG, JPEG, WEBP, BMP, TIF, TIFF）
4. **Alpha通道**: 脚本会自动处理RGBA图像的alpha通道
5. **日志输出**: 每个处理步骤都会输出特征形状信息

## 故障排除

### 常见问题

1. **CUDA内存不足**: 尝试使用较小的模型或CPU
2. **模型下载失败**: 检查网络连接，可能需要手动下载模型
3. **图像加载错误**: 确保图像文件没有损坏

### 调试模式

使用`--save_meta`参数可以保存详细的处理信息，有助于调试问题。
