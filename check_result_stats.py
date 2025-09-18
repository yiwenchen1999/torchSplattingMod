#!/usr/bin/env python3
"""
检查结果文件夹中的.npy文件统计信息
"""

import numpy as np
import os
from pathlib import Path
import glob

def check_result_folder(folder_path, folder_name):
    """
    检查结果文件夹中的统计信息
    """
    print(f"\n=== 检查 {folder_name} ===")
    print(f"路径: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    # 查找所有.npy文件
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    print(f"找到 {len(npy_files)} 个.npy文件")
    
    if len(npy_files) == 0:
        print("❌ 没有找到.npy文件")
        return
    
    # 分析所有文件
    print(f"\n--- 分析所有 {len(npy_files)} 个文件 ---")
    
    all_ranges = []
    all_norms = []
    all_means = []
    all_stds = []
    file_stats = []
    
    for i, file_path in enumerate(npy_files):
        filename = os.path.basename(file_path)
        try:
            data = np.load(file_path)
            
            # 数值范围
            min_val, max_val = data.min(), data.max()
            all_ranges.append((min_val, max_val))
            
            # L2范数
            if data.ndim > 1:
                l2_norms = np.linalg.norm(data, axis=tuple(range(1, data.ndim)))
                l2_norm_min, l2_norm_max = l2_norms.min(), l2_norms.max()
                l2_norm_mean = l2_norms.mean()
                all_norms.extend(l2_norms)
            else:
                l2_norm = np.linalg.norm(data)
                l2_norm_min = l2_norm_max = l2_norm_mean = l2_norm
                all_norms.append(l2_norm)
            
            # 统计信息
            mean_val = data.mean()
            std_val = data.std()
            all_means.append(mean_val)
            all_stds.append(std_val)
            
            # 保存文件统计信息
            file_stats.append({
                'filename': filename,
                'shape': data.shape,
                'dtype': data.dtype,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'l2_norm_range': (l2_norm_min, l2_norm_max),
                'l2_norm_mean': l2_norm_mean
            })
            
            if i < 5:  # 显示前5个文件的详细信息
                print(f"\n文件 {i+1}: {filename}")
                print(f"  形状: {data.shape}")
                print(f"  数据类型: {data.dtype}")
                print(f"  数值范围: [{min_val:.6f}, {max_val:.6f}]")
                print(f"  均值: {mean_val:.6f}")
                print(f"  标准差: {std_val:.6f}")
                print(f"  L2范数范围: [{l2_norm_min:.6f}, {l2_norm_max:.6f}]")
                print(f"  L2范数均值: {l2_norm_mean:.6f}")
                
        except Exception as e:
            print(f"❌ 读取文件失败 {filename}: {e}")
    
    # 汇总统计
    if all_ranges:
        print(f"\n--- {folder_name} 汇总统计 ---")
        
        # 全局数值范围
        all_mins = [r[0] for r in all_ranges]
        all_maxs = [r[1] for r in all_ranges]
        global_min, global_max = min(all_mins), max(all_maxs)
        print(f"全局数值范围: [{global_min:.6f}, {global_max:.6f}]")
        
        # 全局均值
        global_mean = np.mean(all_means)
        global_std = np.mean(all_stds)
        print(f"全局均值: {global_mean:.6f}")
        print(f"全局标准差: {global_std:.6f}")
        
        # L2范数统计
        if all_norms:
            l2_norm_min = min(all_norms)
            l2_norm_max = max(all_norms)
            l2_norm_mean = np.mean(all_norms)
            l2_norm_std = np.std(all_norms)
            
            print(f"全局L2范数范围: [{l2_norm_min:.6f}, {l2_norm_max:.6f}]")
            print(f"全局L2范数均值: {l2_norm_mean:.6f}")
            print(f"全局L2范数标准差: {l2_norm_std:.6f}")
            
            # 归一化分析
            print(f"\n--- 归一化分析 ---")
            
            # 检查L2范数是否接近1
            if abs(l2_norm_mean - 1.0) < 0.1 and l2_norm_std < 0.1:
                print("✅ 可能进行了L2归一化")
            else:
                print("❌ 未进行L2归一化")
            
            # 检查数值范围
            if global_min >= -1.0 and global_max <= 1.0:
                print("✅ 数值在[-1, 1]范围内")
            elif global_min >= 0.0 and global_max <= 1.0:
                print("✅ 数值在[0, 1]范围内")
            else:
                print("❌ 数值超出[-1, 1]范围")
            
            # 检查均值
            if abs(global_mean) < 0.1:
                print("✅ 接近零均值")
            else:
                print("❌ 非零均值")
            
            # 检查标准差
            if abs(global_std - 1.0) < 0.1:
                print("✅ 接近单位标准差")
            else:
                print("❌ 非单位标准差")

def main():
    """
    主函数
    """
    print("=== 检查结果文件夹中的.npy文件统计信息 ===\n")
    
    # 检查两个文件夹
    dino_folder = "result/ship_latents_dino16_16/eval_step_50000"
    clip_folder = "result/ship_latents_clip16_16/eval_step_50000"
    
    check_result_folder(dino_folder, "DINO结果 (eval_step_50000)")
    check_result_folder(clip_folder, "CLIP结果 (eval_step_50000)")
    
    print(f"\n=== 检查完成 ===")

if __name__ == "__main__":
    main()
