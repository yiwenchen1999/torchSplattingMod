#!/usr/bin/env python3
"""
Create heatmap visualizations for L2 norms and similarity scores
Save results as pixel-based heatmaps instead of plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import cv2

def create_heatmap_visualization(data_folder, max_files=200, output_dir="./heatmap_output"):
    """
    Create heatmap visualizations for embedding data
    
    Args:
        data_folder: Path to folder containing .npy files
        max_files: Maximum number of files to load
        output_dir: Directory to save heatmap images
    """
    data_path = Path(data_folder)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get only main encoded map files (like r_0.npy, r_1.npy, etc.)
    all_npy_files = sorted(glob.glob(str(data_path / "r_*.npy")))
    npy_files = []
    for f in all_npy_files:
        filename = Path(f).name
        # Only select files with format r_number.npy
        if filename.startswith('r_') and filename.endswith('.npy') and '_' not in filename[2:-4]:
            npy_files.append(f)
    
    npy_files = npy_files[:max_files]
    
    print(f"Found {len(npy_files)} main encoded files, loading first {min(max_files, len(npy_files))}")
    
    # Load embeddings
    embeddings = []
    for file_path in npy_files:
        embedding = np.load(file_path)
        # Convert to float64 for numerical stability
        embedding_flat = embedding.flatten().astype(np.float64)
        embeddings.append(embedding_flat)
    
    embeddings = np.array(embeddings)
    print(f"Loading complete, shape: {embeddings.shape}")
    
    # Calculate similarity to first embedding
    first_embedding = embeddings[0]
    similarities = []
    
    print("Calculating similarities...")
    for i, embedding in enumerate(embeddings):
        if i == 0:
            similarities.append(1.0)
            continue
            
        first_norm = np.linalg.norm(first_embedding)
        embed_norm = np.linalg.norm(embedding)
        
        if first_norm == 0 or embed_norm == 0:
            similarity = 0.0
        else:
            first_normalized = first_embedding / first_norm
            embed_normalized = embedding / embed_norm
            similarity = np.dot(first_normalized, embed_normalized)
            similarity = np.clip(similarity, -1.0, 1.0)
        
        similarities.append(similarity)
    
    similarities = np.array(similarities)
    
    # Calculate L2 norms
    print("Calculating L2 norms...")
    l2_norms = np.linalg.norm(embeddings, axis=1)
    
    # Create similarity heatmap
    print("Creating similarity heatmap...")
    create_similarity_heatmap(similarities, output_path)
    
    # Create L2 norm heatmap
    print("Creating L2 norm heatmap...")
    create_l2_norm_heatmap(l2_norms, output_path)
    
    # Create embedding similarity matrix heatmap
    print("Creating embedding similarity matrix heatmap...")
    create_similarity_matrix_heatmap(embeddings, output_path, max_embeddings=100)
    
    # Create L2 distance matrix heatmap
    print("Creating L2 distance matrix heatmap...")
    create_distance_matrix_heatmap(embeddings, output_path, max_embeddings=100)
    
    # Create individual heatmaps for each .npy file
    print("Creating individual heatmaps for each .npy file...")
    create_individual_embedding_heatmaps(embeddings, npy_files, output_path)
    
    # Create channel-wise heatmaps for selected files
    print("Creating channel-wise heatmaps...")
    create_channel_wise_heatmaps(embeddings, npy_files, output_path, max_files=20, channels_per_file=16)
    
    print(f"All heatmaps saved to: {output_path}")
    
    return similarities, l2_norms

def create_similarity_heatmap(similarities, output_path, width=800, height=400):
    """
    Create a heatmap showing similarity scores as pixels
    """
    # Reshape similarities into a 2D grid
    n = len(similarities)
    # Find optimal grid dimensions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Pad with zeros if necessary
    padded_similarities = np.zeros(rows * cols)
    padded_similarities[:n] = similarities
    
    # Reshape to 2D
    similarity_grid = padded_similarities.reshape(rows, cols)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_grid, 
                cmap='RdYlBu_r', 
                center=0.5,
                cbar_kws={'label': 'Cosine Similarity'},
                xticklabels=False, 
                yticklabels=False)
    plt.title('Similarity to First Embedding (Pixel Heatmap)')
    plt.tight_layout()
    plt.savefig(output_path / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as raw image
    # Normalize to 0-255 range
    normalized = ((similarity_grid + 1) / 2 * 255).astype(np.uint8)
    cv2.imwrite(str(output_path / 'similarity_heatmap_raw.png'), normalized)

def create_l2_norm_heatmap(l2_norms, output_path, width=800, height=400):
    """
    Create a heatmap showing L2 norms as pixels
    """
    # Reshape L2 norms into a 2D grid
    n = len(l2_norms)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Pad with zeros if necessary
    padded_norms = np.zeros(rows * cols)
    padded_norms[:n] = l2_norms
    
    # Reshape to 2D
    norm_grid = padded_norms.reshape(rows, cols)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(norm_grid, 
                cmap='viridis',
                cbar_kws={'label': 'L2 Norm'},
                xticklabels=False, 
                yticklabels=False)
    plt.title('L2 Norms (Pixel Heatmap)')
    plt.tight_layout()
    plt.savefig(output_path / 'l2_norm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as raw image
    # Normalize to 0-255 range
    min_val, max_val = norm_grid.min(), norm_grid.max()
    normalized = ((norm_grid - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    cv2.imwrite(str(output_path / 'l2_norm_heatmap_raw.png'), normalized)

def create_similarity_matrix_heatmap(embeddings, output_path, max_embeddings=100):
    """
    Create a heatmap showing similarity matrix between embeddings
    """
    n_embeddings = min(max_embeddings, len(embeddings))
    similarity_matrix = np.zeros((n_embeddings, n_embeddings))
    
    print(f"Computing similarity matrix for {n_embeddings} embeddings...")
    for i in range(n_embeddings):
        for j in range(n_embeddings):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Calculate cosine similarity
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[j])
                if norm_i == 0 or norm_j == 0:
                    similarity_matrix[i, j] = 0.0
                else:
                    similarity = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
                    similarity_matrix[i, j] = np.clip(similarity, -1.0, 1.0)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, 
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Cosine Similarity'},
                xticklabels=False, 
                yticklabels=False)
    plt.title(f'Embedding Similarity Matrix ({n_embeddings}x{n_embeddings})')
    plt.tight_layout()
    plt.savefig(output_path / 'similarity_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save as raw image
    normalized = ((similarity_matrix + 1) / 2 * 255).astype(np.uint8)
    cv2.imwrite(str(output_path / 'similarity_matrix_heatmap_raw.png'), normalized)

def create_distance_matrix_heatmap(embeddings, output_path, max_embeddings=100):
    """
    Create a heatmap showing L2 distance matrix between embeddings
    """
    n_embeddings = min(max_embeddings, len(embeddings))
    distance_matrix = np.zeros((n_embeddings, n_embeddings))
    
    print(f"Computing distance matrix for {n_embeddings} embeddings...")
    for i in range(n_embeddings):
        for j in range(n_embeddings):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                distance_matrix[i, j] = distance
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix, 
                cmap='viridis',
                cbar_kws={'label': 'L2 Distance'},
                xticklabels=False, 
                yticklabels=False)
    plt.title(f'Embedding Distance Matrix ({n_embeddings}x{n_embeddings})')
    plt.tight_layout()
    plt.savefig(output_path / 'distance_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save as raw image
    min_val, max_val = distance_matrix.min(), distance_matrix.max()
    normalized = ((distance_matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    cv2.imwrite(str(output_path / 'distance_matrix_heatmap_raw.png'), normalized)

def create_embedding_space_heatmap(embeddings, output_path, method='pca'):
    """
    Create a heatmap of the embedding space using dimensionality reduction
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print(f"Creating embedding space heatmap using {method.upper()}...")
    
    # Reduce dimensionality
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create a grid and interpolate values
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    
    # Create grid
    grid_size = 100
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate values (using simple nearest neighbor for now)
    from scipy.spatial import cKDTree
    tree = cKDTree(embeddings_2d)
    
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    distances, indices = tree.query(grid_points)
    
    # Create heatmap based on distances
    distance_grid = distances.reshape(grid_size, grid_size)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_grid, 
                cmap='viridis',
                cbar_kws={'label': 'Distance to Nearest Embedding'},
                xticklabels=False, 
                yticklabels=False)
    plt.title(f'Embedding Space Density ({method.upper()})')
    plt.tight_layout()
    plt.savefig(output_path / f'embedding_space_{method}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_embedding_heatmaps(embeddings, npy_files, output_path, max_files=50):
    """
    Create individual heatmaps for each .npy file
    Each heatmap shows the 2D structure of the embedding (1024, 16, 16)
    """
    # Create subdirectory for individual heatmaps
    individual_dir = output_path / "individual_heatmaps"
    individual_dir.mkdir(exist_ok=True)
    
    # Limit number of files to process to avoid too many files
    n_files = min(max_files, len(embeddings))
    
    print(f"Creating individual heatmaps for {n_files} files...")
    
    for i in range(n_files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{n_files}")
        
        # Get the original filename
        filename = Path(npy_files[i]).stem  # e.g., "r_0"
        
        # Reshape embedding back to original 3D structure (1024, 16, 16)
        embedding_3d = embeddings[i].reshape(1024, 16, 16)
        
        # Create multiple views of the embedding
        create_embedding_views(embedding_3d, individual_dir, filename)
    
    print(f"Individual heatmaps saved to: {individual_dir}")

def create_embedding_views(embedding_3d, output_dir, filename):
    """
    Create different views of a single embedding
    """
    # View 1: Average across the first dimension (1024 -> 16x16)
    avg_view = np.mean(embedding_3d, axis=0)
    
    # View 2: Standard deviation across the first dimension
    std_view = np.std(embedding_3d, axis=0)
    
    # View 3: Max values across the first dimension
    max_view = np.max(embedding_3d, axis=0)
    
    # View 4: Min values across the first dimension
    min_view = np.min(embedding_3d, axis=0)
    
    # View 5: First few channels (e.g., first 64 channels)
    first_channels = embedding_3d[:64, :, :]
    first_channels_avg = np.mean(first_channels, axis=0)
    
    # Create heatmaps for each view
    views = {
        'avg': avg_view,
        'std': std_view,
        'max': max_view,
        'min': min_view,
        'first_64_channels': first_channels_avg
    }
    
    for view_name, view_data in views.items():
        # Create heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(view_data, 
                    cmap='viridis',
                    cbar_kws={'label': f'{view_name.title()} Value'},
                    xticklabels=False, 
                    yticklabels=False)
        plt.title(f'{filename} - {view_name.title()} View')
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename}_{view_name}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as raw image
        min_val, max_val = view_data.min(), view_data.max()
        if max_val > min_val:
            normalized = ((view_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(view_data, dtype=np.uint8)
        cv2.imwrite(str(output_dir / f'{filename}_{view_name}_raw.png'), normalized)
    
    # Create a combined view showing multiple statistics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    view_names = ['avg', 'std', 'max', 'min', 'first_64_channels']
    for i, view_name in enumerate(view_names):
        if i < len(axes):
            sns.heatmap(views[view_name], 
                       cmap='viridis',
                       ax=axes[i],
                       cbar_kws={'label': f'{view_name.title()}'},
                       xticklabels=False, 
                       yticklabels=False)
            axes[i].set_title(f'{view_name.title()}')
    
    # Hide the last subplot if not needed
    if len(view_names) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle(f'{filename} - All Views', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename}_combined_views.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_channel_wise_heatmaps(embeddings, npy_files, output_path, max_files=20, channels_per_file=16):
    """
    Create heatmaps showing individual channels of embeddings
    """
    # Create subdirectory for channel-wise heatmaps
    channel_dir = output_path / "channel_wise_heatmaps"
    channel_dir.mkdir(exist_ok=True)
    
    n_files = min(max_files, len(embeddings))
    
    print(f"Creating channel-wise heatmaps for {n_files} files...")
    
    for i in range(n_files):
        if i % 5 == 0:
            print(f"Processing channels for file {i+1}/{n_files}")
        
        filename = Path(npy_files[i]).stem
        
        # Reshape embedding back to original 3D structure (1024, 16, 16)
        embedding_3d = embeddings[i].reshape(1024, 16, 16)
        
        # Create heatmaps for first few channels
        n_channels = min(channels_per_file, embedding_3d.shape[0])
        
        # Create a grid of channel heatmaps
        cols = 4
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for ch in range(n_channels):
            row = ch // cols
            col = ch % cols
            
            sns.heatmap(embedding_3d[ch, :, :], 
                       cmap='viridis',
                       ax=axes[row, col],
                       cbar_kws={'label': f'Channel {ch}'},
                       xticklabels=False, 
                       yticklabels=False)
            axes[row, col].set_title(f'Channel {ch}')
        
        # Hide unused subplots
        for ch in range(n_channels, rows * cols):
            row = ch // cols
            col = ch % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'{filename} - First {n_channels} Channels', fontsize=16)
        plt.tight_layout()
        plt.savefig(channel_dir / f'{filename}_channels.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Use default path
    data_folder = "../nerf_synthetic/ship_latents_processed_test/clip_features_raw"
    
    print("Creating heatmap visualizations...")
    similarities, l2_norms = create_heatmap_visualization(data_folder, max_files=200)
    print("Heatmap visualization complete!")
