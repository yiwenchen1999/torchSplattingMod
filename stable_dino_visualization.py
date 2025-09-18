#!/usr/bin/env python3
"""
稳定的DINO特征可视化脚本
专门处理高维DINO特征（1024维）的可视化，避免数值溢出问题
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def stable_dino_visualization(data_folder, max_files=100, output_dir="./dino_visualization"):
    """
    稳定的DINO特征可视化
    
    Args:
        data_folder: 包含.npy文件的文件夹路径
        max_files: 最大加载文件数量
        output_dir: 输出目录
    """
    data_path = Path(data_folder)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 加载DINO特征
    dino_features = load_dino_features_stable(data_path, max_files)
    
    if len(dino_features) == 0:
        print("未找到DINO特征文件")
        return
    
    print(f"加载了 {len(dino_features)} 个DINO特征，每个维度: {dino_features.shape[1]}")
    
    # 1. 基本统计信息
    print_basic_statistics(dino_features)
    
    # 2. 特征分布可视化（使用对数缩放）
    visualize_feature_distributions_stable(dino_features, output_path)
    
    # 3. 降维可视化
    visualize_dimensionality_reduction_stable(dino_features, output_path)
    
    # 4. 特征相似性分析
    visualize_feature_similarity_stable(dino_features, output_path)
    
    # 5. 创建热图可视化
    create_dino_heatmaps(dino_features, output_path)
    
    print(f"所有DINO特征可视化已保存到: {output_path}")

def load_dino_features_stable(data_path, max_files):
    """
    稳定地加载DINO特征文件
    """
    # 查找所有.npy文件，排除alpha、depth、mask文件
    all_npy_files = sorted(glob.glob(str(data_path / "r_*.npy")))
    dino_files = []
    
    for file_path in all_npy_files:
        filename = Path(file_path).name.lower()
        # 跳过包含alpha、depth、mask的文件
        if any(suffix in filename for suffix in ['alpha', 'depth', 'mask']):
            continue
        dino_files.append(file_path)
    
    dino_files = dino_files[:max_files]
    print(f"找到 {len(dino_files)} 个主要DINO特征文件")
    
    # 加载特征
    features = []
    for file_path in dino_files:
        try:
            feature = np.load(file_path)
            # 转换为float64以提高数值稳定性
            if feature.ndim > 1:
                feature = feature.flatten()
            feature = feature.astype(np.float64)
            features.append(feature)
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    
    return np.array(features) if features else np.array([])

def print_basic_statistics(features):
    """
    打印基本统计信息
    """
    print("\n=== DINO特征基本统计信息 ===")
    print(f"特征数量: {features.shape[0]}")
    print(f"特征维度: {features.shape[1]}")
    print(f"数据范围: [{features.min():.2e}, {features.max():.2e}]")
    print(f"均值: {features.mean():.2e}")
    print(f"标准差: {features.std():.2e}")
    
    # 计算L2范数（使用更稳定的方法）
    norms = np.sqrt(np.sum(features**2, axis=1))
    print(f"L2范数范围: [{norms.min():.2e}, {norms.max():.2e}]")
    print(f"L2范数均值: {norms.mean():.2e}")

def visualize_feature_distributions_stable(features, output_path):
    """
    稳定地可视化特征分布
    """
    print("创建特征分布可视化...")
    
    # 使用对数缩放处理大数值
    log_features = np.log10(np.abs(features) + 1e-10)
    
    plt.figure(figsize=(15, 10))
    
    # 1. 所有特征值的分布（对数缩放）
    plt.subplot(2, 3, 1)
    all_values = log_features.flatten()
    plt.hist(all_values, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('log10(|Feature Value|)')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Feature Values (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # 2. 每个特征维度的均值分布
    plt.subplot(2, 3, 2)
    mean_values = np.mean(features, axis=0)
    log_means = np.log10(np.abs(mean_values) + 1e-10)
    plt.hist(log_means, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('log10(|Mean Feature Value|)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Means (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # 3. 每个特征维度的标准差分布
    plt.subplot(2, 3, 3)
    std_values = np.std(features, axis=0)
    log_stds = np.log10(std_values + 1e-10)
    plt.hist(log_stds, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('log10(Feature Std)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Standard Deviations (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # 4. 特征范数分布（对数缩放）
    plt.subplot(2, 3, 4)
    norms = np.sqrt(np.sum(features**2, axis=1))
    log_norms = np.log10(norms + 1e-10)
    plt.hist(log_norms, bins=50, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('log10(L2 Norm)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature L2 Norms (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # 5. 特征相关性热图（前50个维度）
    plt.subplot(2, 3, 5)
    n_dims = min(50, features.shape[1])
    # 使用更稳定的相关性计算
    corr_matrix = np.corrcoef(features[:, :n_dims].T)
    # 处理NaN值
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Correlation'})
    plt.title(f'Feature Correlation Matrix (First {n_dims} dims)')
    
    # 6. 特征激活热图（前20个样本，前50个特征）
    plt.subplot(2, 3, 6)
    n_samples = min(20, features.shape[0])
    n_features = min(50, features.shape[1])
    # 使用对数缩放
    log_heatmap = np.log10(np.abs(features[:n_samples, :n_features]) + 1e-10)
    sns.heatmap(log_heatmap, 
                cmap='viridis', xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'log10(|Feature Value|)'})
    plt.title(f'Feature Activation Heatmap ({n_samples}x{n_features})')
    
    plt.tight_layout()
    plt.savefig(output_path / 'dino_feature_distributions_stable.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_dimensionality_reduction_stable(features, output_path):
    """
    使用降维方法稳定地可视化DINO特征
    """
    print("创建降维可视化...")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 1. PCA可视化
    print("执行PCA降维...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # PCA散点图
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=range(len(features_pca)), cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Sample Index')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of DINO Features')
    plt.grid(True, alpha=0.3)
    
    # PCA解释方差比
    plt.subplot(1, 2, 2)
    pca_full = PCA()
    pca_full.fit(features_scaled)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, min(101, len(cumsum)+1)), cumsum[:100], 'b-', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'dino_pca_visualization_stable.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE可视化（如果样本数不太多）
    if len(features) <= 100:
        print("执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        features_tsne = tsne.fit_transform(features_scaled)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                             c=range(len(features_tsne)), cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Sample Index')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Visualization of DINO Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'dino_tsne_visualization_stable.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_feature_similarity_stable(features, output_path):
    """
    稳定地可视化特征相似性
    """
    print("创建特征相似性可视化...")
    
    # 计算相似性矩阵（使用余弦相似性）
    n_samples = min(50, len(features))  # 限制样本数以节省计算时间
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # 余弦相似性
                norm_i = np.linalg.norm(features[i])
                norm_j = np.linalg.norm(features[j])
                if norm_i == 0 or norm_j == 0:
                    similarity_matrix[i, j] = 0.0
                else:
                    similarity = np.dot(features[i], features[j]) / (norm_i * norm_j)
                    similarity_matrix[i, j] = np.clip(similarity, -1.0, 1.0)
    
    # 创建相似性热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='RdYlBu_r', center=0,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f'DINO Feature Similarity Matrix ({n_samples}x{n_samples})')
    plt.tight_layout()
    plt.savefig(output_path / 'dino_similarity_matrix_stable.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 相似性分布
    plt.figure(figsize=(10, 6))
    # 提取上三角矩阵（不包括对角线）
    upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    plt.hist(upper_tri, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Similarities')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'dino_similarity_distribution_stable.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dino_heatmaps(features, output_path):
    """
    创建DINO特征的热图可视化
    """
    print("创建DINO特征热图...")
    
    # 1. 特征重要性热图
    feature_vars = np.var(features, axis=0)
    feature_means = np.mean(features, axis=0)
    
    # 选择前100个最重要的特征
    n_features = min(100, features.shape[1])
    top_features_idx = np.argsort(feature_vars)[-n_features:]
    
    plt.figure(figsize=(15, 8))
    
    # 特征方差热图
    plt.subplot(2, 2, 1)
    var_heatmap = feature_vars[top_features_idx].reshape(10, 10)
    sns.heatmap(var_heatmap, cmap='viridis', 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Variance'})
    plt.title('Top 100 Features Variance Heatmap')
    
    # 特征均值热图
    plt.subplot(2, 2, 2)
    mean_heatmap = feature_means[top_features_idx].reshape(10, 10)
    sns.heatmap(mean_heatmap, cmap='coolwarm', center=0,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Mean Value'})
    plt.title('Top 100 Features Mean Heatmap')
    
    # 特征激活模式热图
    plt.subplot(2, 2, 3)
    n_samples = min(20, features.shape[0])
    activation_heatmap = features[:n_samples, top_features_idx]
    sns.heatmap(activation_heatmap, cmap='viridis',
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Feature Value'})
    plt.title(f'Feature Activation Pattern ({n_samples} samples)')
    
    # 特征相关性热图
    plt.subplot(2, 2, 4)
    corr_matrix = np.corrcoef(features[:, top_features_idx].T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Correlation'})
    plt.title('Top 100 Features Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig(output_path / 'dino_heatmaps_stable.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 使用DINO特征路径
    data_folder = "../nerf_synthetic/ship_latents_processed_test/dino_features_v2_224"
    
    print("开始稳定的DINO特征可视化...")
    stable_dino_visualization(data_folder, max_files=100)
    print("DINO特征可视化完成！")
