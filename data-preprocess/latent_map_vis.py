#!/usr/bin/env python3
"""
潜在映射可视化工具
读取.npy特征文件并将其可视化为RGB图像
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse

def visualize_feature_map_as_rgb(feature_path, output_path=None, method='activation', 
                                channel_indices=None, normalize=True, colormap='viridis'):
    """
    将特征映射可视化为RGB图像
    
    Args:
        feature_path: .npy特征文件路径
        output_path: 输出图像路径，如果为None则显示图像
        method: 可视化方法
            - 'activation': 激活映射（所有通道的平均值）
            - 'l2_norm': L2范数映射
            - 'first_3_channels': 使用前3个通道作为RGB
            - 'custom_channels': 使用指定的通道索引
        channel_indices: 当method='custom_channels'时，指定要使用的通道索引列表
        normalize: 是否标准化到[0,1]范围
        colormap: matplotlib颜色映射名称
    
    Returns:
        numpy.ndarray: 生成的RGB图像
    """
    # 加载特征数据
    features = np.load(feature_path)
    print(f"加载特征文件: {feature_path}")
    print(f"特征形状: {features.shape}")
    print(f"数据类型: {features.dtype}")
    print(f"数值范围: [{features.min():.4f}, {features.max():.4f}]")
    
    # 确保特征数据是3D的 (channels, height, width)
    if features.ndim == 1:
        # 如果是1D，尝试重塑为合理的3D形状
        # 假设是1024通道，16x16的空间维度
        if features.shape[0] == 262144:  # 1024 * 16 * 16
            features = features.reshape(1024, 16, 16)
        else:
            raise ValueError(f"无法自动重塑1D特征，形状: {features.shape}")
    elif features.ndim == 2:
        # 如果是2D，假设是 (channels, spatial)
        features = features.reshape(features.shape[0], int(np.sqrt(features.shape[1])), int(np.sqrt(features.shape[1])))
    elif features.ndim == 3:
        # 已经是3D，直接使用
        pass
    else:
        raise ValueError(f"不支持的特征维度: {features.ndim}")
    
    print(f"重塑后特征形状: {features.shape}")
    
    # 根据方法生成可视化
    if method == 'activation':
        # 激活映射：所有通道的平均值
        activation_map = np.mean(features, axis=0)
        print(f"激活映射形状: {activation_map.shape}")
        
        # 标准化
        if normalize:
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        
        # 应用颜色映射
        cmap = plt.cm.get_cmap(colormap)
        rgb_image = cmap(activation_map)[:, :, :3]  # 只取RGB，不要alpha通道
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
    elif method == 'l2_norm':
        # L2范数映射：每个空间位置的L2范数
        l2_norm_map = np.sqrt(np.sum(features**2, axis=0))
        print(f"L2范数映射形状: {l2_norm_map.shape}")
        
        # 标准化
        if normalize:
            l2_norm_map = (l2_norm_map - l2_norm_map.min()) / (l2_norm_map.max() - l2_norm_map.min())
        
        # 应用颜色映射
        cmap = plt.cm.get_cmap(colormap)
        rgb_image = cmap(l2_norm_map)[:, :, :3]
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
    elif method == 'first_3_channels':
        # 使用前3个通道作为RGB
        if features.shape[0] < 3:
            raise ValueError(f"特征通道数不足，需要至少3个通道，实际: {features.shape[0]}")
        
        rgb_image = features[:3, :, :].transpose(1, 2, 0)  # 转换为 (H, W, C)
        print(f"前3通道RGB形状: {rgb_image.shape}")
        
        # 标准化每个通道
        if normalize:
            for i in range(3):
                channel = rgb_image[:, :, i]
                rgb_image[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
        
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
    elif method == 'custom_channels':
        # 使用指定的通道
        if channel_indices is None:
            raise ValueError("使用custom_channels方法时必须指定channel_indices")
        
        if len(channel_indices) != 3:
            raise ValueError("channel_indices必须包含3个通道索引")
        
        for idx in channel_indices:
            if idx >= features.shape[0]:
                raise ValueError(f"通道索引 {idx} 超出范围 [0, {features.shape[0]-1}]")
        
        rgb_image = features[channel_indices, :, :].transpose(1, 2, 0)
        print(f"自定义通道RGB形状: {rgb_image.shape}")
        
        # 标准化每个通道
        if normalize:
            for i in range(3):
                channel = rgb_image[:, :, i]
                rgb_image[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
        
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
    else:
        raise ValueError(f"不支持的可视化方法: {method}")
    
    print(f"最终RGB图像形状: {rgb_image.shape}")
    print(f"RGB图像数值范围: [{rgb_image.min()}, {rgb_image.max()}]")
    
    # 保存像素级图像
    if output_path:
        # 使用OpenCV保存，确保是像素级图像
        cv2.imwrite(str(output_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        print(f"像素级图像已保存到: {output_path}")
        print(f"图像尺寸: {rgb_image.shape[1]}x{rgb_image.shape[0]} 像素")
    else:
        # 显示图像
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title(f'Feature Map Visualization - {method}')
        plt.axis('off')
        plt.show()
    
    return rgb_image

def create_high_res_pixel_visualization(feature_path, output_path, method='activation', 
                                      upscale_factor=8, colormap='viridis'):
    """
    创建高分辨率的像素级可视化
    
    Args:
        feature_path: .npy特征文件路径
        output_path: 输出图像路径
        method: 可视化方法
        upscale_factor: 上采样倍数
        colormap: 颜色映射
    """
    # 加载特征数据
    features = np.load(feature_path)
    print(f"创建高分辨率像素可视化: {feature_path}")
    print(f"原始特征形状: {features.shape}")
    
    # 重塑为3D
    if features.ndim == 1:
        if features.shape[0] == 262144:  # 1024 * 16 * 16
            features = features.reshape(1024, 16, 16)
        else:
            raise ValueError(f"无法自动重塑1D特征，形状: {features.shape}")
    elif features.ndim == 2:
        features = features.reshape(features.shape[0], int(np.sqrt(features.shape[1])), int(np.sqrt(features.shape[1])))
    
    # 生成基础可视化
    if method == 'activation':
        base_map = np.mean(features, axis=0)
    elif method == 'l2_norm':
        base_map = np.sqrt(np.sum(features**2, axis=0))
    elif method == 'first_3_channels':
        # 对于RGB方法，分别处理每个通道
        base_map = features[:3, :, :].transpose(1, 2, 0)
        # 标准化每个通道
        for i in range(3):
            channel = base_map[:, :, i]
            base_map[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
        
        # 上采样RGB图像
        upscaled = cv2.resize(base_map, 
                             (base_map.shape[1] * upscale_factor, base_map.shape[0] * upscale_factor),
                             interpolation=cv2.INTER_NEAREST)
        rgb_image = (upscaled * 255).astype(np.uint8)
        
        # 保存
        cv2.imwrite(str(output_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        print(f"高分辨率RGB像素图像已保存: {output_path}")
        print(f"图像尺寸: {rgb_image.shape[1]}x{rgb_image.shape[0]} 像素")
        return rgb_image
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    # 标准化
    base_map = (base_map - base_map.min()) / (base_map.max() - base_map.min())
    
    # 应用颜色映射
    cmap = plt.cm.get_cmap(colormap)
    rgb_image = cmap(base_map)[:, :, :3]
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # 上采样
    upscaled = cv2.resize(rgb_image, 
                         (rgb_image.shape[1] * upscale_factor, rgb_image.shape[0] * upscale_factor),
                         interpolation=cv2.INTER_NEAREST)
    
    # 保存高分辨率图像
    cv2.imwrite(str(output_path), cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))
    print(f"高分辨率像素图像已保存: {output_path}")
    print(f"图像尺寸: {upscaled.shape[1]}x{upscaled.shape[0]} 像素")
    
    return upscaled

def test_dino_feature_visualization():
    """
    测试DINO特征可视化
    """
    # DINO特征文件路径
    dino_feature_path = "../nerf_synthetic/ship_latents_processed_test/dino_features_v2_224/r_0.npy"
    
    if not Path(dino_feature_path).exists():
        print(f"DINO特征文件不存在: {dino_feature_path}")
        return
    
    # 创建输出目录
    output_dir = Path("feature_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("=== 测试DINO特征可视化 ===\n")
    
    # 1. 激活映射可视化
    print("1. 激活映射可视化")
    try:
        rgb_image = visualize_feature_map_as_rgb(
            dino_feature_path, 
            output_path=output_dir / "dino_activation_map.png",
            method='activation',
            colormap='viridis'
        )
        print("✓ 激活映射可视化完成\n")
    except Exception as e:
        print(f"✗ 激活映射可视化失败: {e}\n")
    
    # 2. L2范数映射可视化
    print("2. L2范数映射可视化")
    try:
        rgb_image = visualize_feature_map_as_rgb(
            dino_feature_path, 
            output_path=output_dir / "dino_l2_norm_map.png",
            method='l2_norm',
            colormap='plasma'
        )
        print("✓ L2范数映射可视化完成\n")
    except Exception as e:
        print(f"✗ L2范数映射可视化失败: {e}\n")
    
    # 3. 前3个通道作为RGB
    print("3. 前3个通道RGB可视化")
    try:
        rgb_image = visualize_feature_map_as_rgb(
            dino_feature_path, 
            output_path=output_dir / "dino_first_3_channels.png",
            method='first_3_channels'
        )
        print("✓ 前3通道RGB可视化完成\n")
    except Exception as e:
        print(f"✗ 前3通道RGB可视化失败: {e}\n")
    
    # 4. 自定义通道可视化
    print("4. 自定义通道可视化 (通道 0, 100, 500)")
    try:
        rgb_image = visualize_feature_map_as_rgb(
            dino_feature_path, 
            output_path=output_dir / "dino_custom_channels.png",
            method='custom_channels',
            channel_indices=[0, 100, 500]
        )
        print("✓ 自定义通道可视化完成\n")
    except Exception as e:
        print(f"✗ 自定义通道可视化失败: {e}\n")
    
    # 5. 不同颜色映射的激活映射
    print("5. 不同颜色映射的激活映射")
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool']
    for cmap in colormaps:
        try:
            rgb_image = visualize_feature_map_as_rgb(
                dino_feature_path, 
                output_path=output_dir / f"dino_activation_{cmap}.png",
                method='activation',
                colormap=cmap
            )
            print(f"✓ {cmap} 颜色映射完成")
        except Exception as e:
            print(f"✗ {cmap} 颜色映射失败: {e}")
    
    # 6. 高分辨率像素级可视化
    print("\n6. 高分辨率像素级可视化")
    try:
        # 高分辨率激活映射
        create_high_res_pixel_visualization(
            dino_feature_path,
            output_dir / "dino_activation_high_res.png",
            method='activation',
            upscale_factor=16,
            colormap='viridis'
        )
        print("✓ 高分辨率激活映射完成")
        
        # 高分辨率L2范数映射
        create_high_res_pixel_visualization(
            dino_feature_path,
            output_dir / "dino_l2_norm_high_res.png",
            method='l2_norm',
            upscale_factor=16,
            colormap='plasma'
        )
        print("✓ 高分辨率L2范数映射完成")
        
        # 高分辨率RGB映射
        create_high_res_pixel_visualization(
            dino_feature_path,
            output_dir / "dino_rgb_high_res.png",
            method='first_3_channels',
            upscale_factor=16
        )
        print("✓ 高分辨率RGB映射完成")
        
    except Exception as e:
        print(f"✗ 高分辨率可视化失败: {e}")
    
    print(f"\n所有像素级可视化结果已保存到: {output_dir}")

def main():
    """
    主函数，支持命令行参数
    """
    parser = argparse.ArgumentParser(description='可视化特征映射为RGB图像')
    parser.add_argument('--feature_path', type=str, 
                       default='../../nerf_synthetic/ship_latents_processed_test/dino_features_v2_224/r_0.npy',
                       help='特征文件路径')
    parser.add_argument('--output_path', type=str, default=None,
                       help='输出图像路径')
    parser.add_argument('--method', type=str, default='activation',
                       choices=['activation', 'l2_norm', 'first_3_channels', 'custom_channels'],
                       help='可视化方法')
    parser.add_argument('--channel_indices', type=int, nargs=3, default=[0, 100, 500],
                       help='自定义通道索引（当method=custom_channels时使用）')
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='颜色映射名称')
    parser.add_argument('--test', action='store_true',
                       help='运行测试函数')
    
    args = parser.parse_args()
    
    if args.test:
        test_dino_feature_visualization()
    else:
        visualize_feature_map_as_rgb(
            args.feature_path,
            args.output_path,
            args.method,
            args.channel_indices,
            colormap=args.colormap
        )

if __name__ == "__main__":
    # 如果直接运行脚本，执行测试
    test_dino_feature_visualization()
