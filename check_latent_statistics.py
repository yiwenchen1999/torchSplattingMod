#!/usr/bin/env python3
"""
检查潜在文件的统计信息
分析数值范围、L2范数范围和归一化情况
"""

import numpy as np
import os
from pathlib import Path
import glob

def check_latent_statistics(folder_path, folder_name):
    """
    检查指定文件夹中潜在文件的统计信息
    
    Args:
        folder_path: 潜在文件文件夹路径
        folder_name: 文件夹名称（用于显示）
    """
    print(f"\n=== 检查 {folder_name} 文件夹 ===")
    print(f"路径: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    # 获取所有.npy文件，排除alpha和depth相关文件
    all_files = glob.glob(os.path.join(folder_path, "*.npy"))
    main_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path).lower()
        # 排除包含alpha、depth、mask的文件
        if not any(suffix in filename for suffix in ['alpha', 'depth', 'mask']):
            main_files.append(file_path)
    
    main_files = sorted(main_files)
    print(f"找到 {len(main_files)} 个主要潜在文件")
    
    if len(main_files) == 0:
        print("❌ 没有找到有效的潜在文件")
        return
    
    # 分析前几个文件的详细信息
    print(f"\n--- 详细分析前 {min(5, len(main_files))} 个文件 ---")
    
    all_ranges = []
    all_norms = []
    all_means = []
    all_stds = []
    
    for i, file_path in enumerate(main_files[:5]):
        filename = os.path.basename(file_path)
        try:
            data = np.load(file_path)
            print(f"\n文件 {i+1}: {filename}")
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            
            # 如果是3D数据，展平为1D进行分析
            if data.ndim > 1:
                data_flat = data.flatten()
            else:
                data_flat = data
            
            # 数值范围
            min_val, max_val = data_flat.min(), data_flat.max()
            print(f"  数值范围: [{min_val:.6f}, {max_val:.6f}]")
            all_ranges.append((min_val, max_val))
            
            # L2范数
            if data.ndim > 1:
                l2_norms = np.linalg.norm(data, axis=tuple(range(1, data.ndim)))
                l2_norm_min, l2_norm_max = l2_norms.min(), l2_norms.max()
                l2_norm_mean = l2_norms.mean()
                print(f"  L2范数范围: [{l2_norm_min:.6f}, {l2_norm_max:.6f}]")
                print(f"  L2范数均值: {l2_norm_mean:.6f}")
                all_norms.append((l2_norm_min, l2_norm_max, l2_norm_mean))
            else:
                l2_norm = np.linalg.norm(data_flat)
                print(f"  L2范数: {l2_norm:.6f}")
                all_norms.append((l2_norm, l2_norm, l2_norm))
            
            # 统计信息
            mean_val = data_flat.mean()
            std_val = data_flat.std()
            print(f"  均值: {mean_val:.6f}")
            print(f"  标准差: {std_val:.6f}")
            all_means.append(mean_val)
            all_stds.append(std_val)
            
            # 检查是否归一化
            is_normalized = check_normalization(data_flat)
            print(f"  归一化状态: {is_normalized}")
            
        except Exception as e:
            print(f"  ❌ 读取文件失败: {e}")
    
    # 批量分析所有文件
    print(f"\n--- 批量分析所有 {len(main_files)} 个文件 ---")
    
    batch_ranges = []
    batch_norms = []
    batch_means = []
    
    print("正在批量处理文件...")
    for i, file_path in enumerate(main_files):
        if i % 20 == 0:
            print(f"  处理进度: {i}/{len(main_files)}")
        
        try:
            data = np.load(file_path)
            
            # 展平数据
            if data.ndim > 1:
                data_flat = data.flatten()
                l2_norms = np.linalg.norm(data, axis=tuple(range(1, data.ndim)))
                batch_norms.extend(l2_norms)
            else:
                data_flat = data
                batch_norms.append(np.linalg.norm(data_flat))
            
            # 统计信息
            batch_ranges.append((data_flat.min(), data_flat.max()))
            batch_means.append(data_flat.mean())
            
        except Exception as e:
            print(f"  ❌ 处理文件失败 {os.path.basename(file_path)}: {e}")
    
    # 汇总统计
    if batch_ranges:
        all_mins = [r[0] for r in batch_ranges]
        all_maxs = [r[1] for r in batch_ranges]
        
        print(f"\n--- {folder_name} 汇总统计 ---")
        print(f"全局数值范围: [{min(all_mins):.6f}, {max(all_maxs):.6f}]")
        print(f"全局均值范围: [{min(batch_means):.6f}, {max(batch_means):.6f}]")
        print(f"全局均值: {np.mean(batch_means):.6f}")
        
        if batch_norms:
            print(f"全局L2范数范围: [{min(batch_norms):.6f}, {max(batch_norms):.6f}]")
            print(f"全局L2范数均值: {np.mean(batch_norms):.6f}")
        
        # 归一化分析
        print(f"\n--- 归一化分析 ---")
        analyze_normalization_status(batch_ranges, batch_norms, batch_means)

def check_normalization(data):
    """
    检查单个数据是否归一化
    """
    # 检查L2范数是否接近1
    l2_norm = np.linalg.norm(data)
    is_unit_norm = abs(l2_norm - 1.0) < 0.01
    
    # 检查是否在[-1, 1]范围内
    is_bounded = data.min() >= -1.0 and data.max() <= 1.0
    
    # 检查是否在[0, 1]范围内
    is_positive_bounded = data.min() >= 0.0 and data.max() <= 1.0
    
    # 检查是否零均值
    is_zero_mean = abs(data.mean()) < 0.01
    
    # 检查是否单位方差
    is_unit_var = abs(data.std() - 1.0) < 0.01
    
    status = []
    if is_unit_norm:
        status.append("单位L2范数")
    if is_bounded:
        status.append("[-1,1]范围")
    if is_positive_bounded:
        status.append("[0,1]范围")
    if is_zero_mean:
        status.append("零均值")
    if is_unit_var:
        status.append("单位方差")
    
    return ", ".join(status) if status else "未归一化"

def analyze_normalization_status(ranges, norms, means):
    """
    分析整体的归一化状态
    """
    # 分析L2范数分布
    if norms:
        norm_mean = np.mean(norms)
        norm_std = np.std(norms)
        print(f"L2范数统计: 均值={norm_mean:.6f}, 标准差={norm_std:.6f}")
        
        if abs(norm_mean - 1.0) < 0.1 and norm_std < 0.1:
            print("✅ 可能进行了L2归一化")
        else:
            print("❌ 未进行L2归一化")
    
    # 分析数值范围
    all_mins = [r[0] for r in ranges]
    all_maxs = [r[1] for r in ranges]
    global_min, global_max = min(all_mins), max(all_maxs)
    
    print(f"全局数值范围: [{global_min:.6f}, {global_max:.6f}]")
    
    if global_min >= -1.0 and global_max <= 1.0:
        print("✅ 数值在[-1, 1]范围内")
    elif global_min >= 0.0 and global_max <= 1.0:
        print("✅ 数值在[0, 1]范围内")
    else:
        print("❌ 数值超出[-1, 1]范围")
    
    # 分析均值
    mean_mean = np.mean(means)
    print(f"全局均值: {mean_mean:.6f}")
    
    if abs(mean_mean) < 0.1:
        print("✅ 接近零均值")
    else:
        print("❌ 非零均值")

def main():
    """
    主函数
    """
    print("=== 潜在文件统计信息检查 ===\n")
    
    # 检查CLIP特征文件夹
    clip_folder = "../nerf_synthetic/ship_latents_processed_test/clip_features_algined_16"
    check_latent_statistics(clip_folder, "CLIP特征 (aligned_16)")
    
    # 检查DINO特征文件夹
    dino_folder = "../nerf_synthetic/ship_latents_processed_test/dino_features_v2_16"
    check_latent_statistics(dino_folder, "DINO特征 (v2_16)")
    
    print(f"\n=== 检查完成 ===")

if __name__ == "__main__":
    main()
