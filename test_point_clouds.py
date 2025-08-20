import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from pathlib import Path

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.point_utils import get_point_clouds, PointCloud
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera

def test_point_cloud_generation():
    """
    Test the get_point_clouds function with sample data
    """
    print("=== Testing Point Cloud Generation ===")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folder = '../ship_latents_processed'
    # folder = 'B075X65R3X'
    
    print(f"Using device: {device}")
    print(f"Loading data from: {folder}")
    
    # Load data (same as in train.py)
    try:
        data = read_all(folder, resize_factor=256.0/800.0)
        data = {k: v.to(device) for k, v in data.items()}
        data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)
        print(f"Successfully loaded data with {len(data['rgb'])} images")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Print data shapes and statistics
    print("\n=== Data Statistics ===")
    print(f"RGB shape: {data['rgb'].shape}")
    print(f"Depth shape: {data['depth'].shape}")
    print(f"Alpha shape: {data['alpha'].shape}")
    print(f"Camera shape: {data['camera'].shape}")
    # print(f"Latent shape: {data['latent'].shape}")
    #
    print(f"RGB range: [{data['rgb'].min():.3f}, {data['rgb'].max():.3f}]")
    print(f"Depth range: [{data['depth'].min():.3f}, {data['depth'].max():.3f}]")
    print(f"Alpha range: [{data['alpha'].min():.3f}, {data['alpha'].max():.3f}]")
    
    # Generate point clouds
    print("\n=== Generating Point Clouds ===")
    try:
        points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
        print(f"Generated point cloud with {len(points.coords)} points")
        print(f"Available channels: {list(points.channels.keys())}")
    except Exception as e:
        print(f"Error generating point clouds: {e}")
        return None
    
    return data, points

def analyze_point_cloud(points):
    """
    Analyze the generated point cloud
    """
    print("\n=== Point Cloud Analysis ===")
    
    coords = points.coords
    channels = points.channels
    
    print(f"Total points: {len(coords)}")
    print(f"Coordinate range:")
    print(f"  X: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    print(f"  Y: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
    print(f"  Z: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
    
    if 'R' in channels and 'G' in channels and 'B' in channels:
        rgb = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"RGB mean: [{rgb.mean(axis=0)}]")
    
    if 'A' in channels:
        alpha = channels['A']
        print(f"Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
        print(f"Alpha mean: {alpha.mean():.3f}")
        print(f"Points with alpha > 0.5: {np.sum(alpha > 0.5)}")
    
    # Calculate point density statistics
    from scipy.spatial.distance import pdist, squareform
    if len(coords) > 1000:
        # Sample a subset for distance calculation
        sample_indices = np.random.choice(len(coords), 1000, replace=False)
        sample_coords = coords[sample_indices]
        distances = pdist(sample_coords)
        print(f"Average distance between points (sampled): {distances.mean():.4f}")
        print(f"Min distance between points (sampled): {distances.min():.4f}")
        print(f"Max distance between points (sampled): {distances.max():.4f}")

def visualize_point_cloud_2d(points, save_path="point_cloud_2d.png"):
    """
    Create 2D scatter plots of the point cloud from different viewpoints
    """
    print(f"\n=== Creating 2D Visualizations ===")
    
    coords = points.coords
    channels = points.channels
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Point Cloud 2D Projections', fontsize=16)
    
    # XY projection
    if 'R' in channels and 'G' in channels and 'B' in channels:
        colors = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        colors = np.clip(colors, 0, 1)  # Ensure colors are in [0, 1]
    else:
        colors = 'blue'
    
    # XY projection
    axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=colors, s=0.1, alpha=0.6)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('XY Projection')
    axes[0, 0].set_aspect('equal')
    
    # XZ projection
    axes[0, 1].scatter(coords[:, 0], coords[:, 2], c=colors, s=0.1, alpha=0.6)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('XZ Projection')
    axes[0, 1].set_aspect('equal')
    
    # YZ projection
    axes[1, 0].scatter(coords[:, 1], coords[:, 2], c=colors, s=0.1, alpha=0.6)
    axes[1, 0].set_xlabel('Y')
    axes[1, 0].set_ylabel('Z')
    axes[1, 0].set_title('YZ Projection')
    axes[1, 0].set_aspect('equal')
    
    # Color distribution
    if 'R' in channels and 'G' in channels and 'B' in channels:
        rgb = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        axes[1, 1].hist(rgb.flatten(), bins=50, alpha=0.7, label=['R', 'G', 'B'])
        axes[1, 1].set_xlabel('Color Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Color Distribution')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D visualization saved to {save_path}")
    plt.close()

def visualize_point_cloud_3d(points, save_path="point_cloud_3d.png"):
    """
    Create 3D visualization of the point cloud
    """
    print(f"\n=== Creating 3D Visualization ===")
    
    coords = points.coords
    channels = points.channels
    
    # Sample points if there are too many for visualization
    if len(coords) > 10000:
        sample_indices = np.random.choice(len(coords), 10000, replace=False)
        coords = coords[sample_indices]
        if 'R' in channels and 'G' in channels and 'B' in channels:
            colors = np.stack([channels['R'][sample_indices], 
                             channels['G'][sample_indices], 
                             channels['B'][sample_indices]], axis=1)
        else:
            colors = 'blue'
    else:
        if 'R' in channels and 'G' in channels and 'B' in channels:
            colors = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        else:
            colors = 'blue'
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                        c=colors, s=0.5, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Visualization')
    
    # Set equal aspect ratio
    max_range = np.array([coords[:, 0].max()-coords[:, 0].min(),
                         coords[:, 1].max()-coords[:, 1].min(),
                         coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
    
    mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D visualization saved to {save_path}")
    plt.close()

def test_point_cloud_operations(points):
    """
    Test various point cloud operations
    """
    print("\n=== Testing Point Cloud Operations ===")
    
    # Test random sampling
    print("Testing random sampling...")
    sampled_points = points.random_sample(1000)
    print(f"Sampled {len(sampled_points.coords)} points from {len(points.coords)} original points")
    
    # Test farthest point sampling
    print("Testing farthest point sampling...")
    try:
        fps_points = points.farthest_point_sample(1000)
        print(f"Farthest point sampling: {len(fps_points.coords)} points")
    except Exception as e:
        print(f"Farthest point sampling failed: {e}")
    
    # Test saving and loading
    print("Testing save/load functionality...")
    try:
        points.save("test_point_cloud.npz")
        loaded_points = PointCloud.load("test_point_cloud.npz")
        print(f"Save/load successful: {len(loaded_points.coords)} points loaded")
        
        # Clean up
        os.remove("test_point_cloud.npz")
    except Exception as e:
        print(f"Save/load test failed: {e}")
    
    # Test PLY export
    print("Testing PLY export...")
    try:
        with open("test_point_cloud.ply", "wb") as f:
            points.write_ply(f)
        print("PLY export successful")
        
        # # Clean up
        # os.remove("test_point_cloud.ply")
    except Exception as e:
        print(f"PLY export failed: {e}")

def compare_with_original_data(data, points):
    """
    Compare the generated point cloud with the original data
    """
    print("\n=== Comparing with Original Data ===")
    
    # Check if point cloud coordinates match depth values
    coords = points.coords
    
    # Sample a few points and check their depth values
    if len(coords) > 0:
        print(f"Point cloud coordinate statistics:")
        print(f"  X: mean={coords[:, 0].mean():.3f}, std={coords[:, 0].std():.3f}")
        print(f"  Y: mean={coords[:, 1].mean():.3f}, std={coords[:, 1].std():.3f}")
        print(f"  Z: mean={coords[:, 2].mean():.3f}, std={coords[:, 2].std():.3f}")
        
        # Check if Z coordinates match depth range
        depth_range = data['depth_range']
        print(f"Original depth range: {depth_range[0].cpu().numpy()}")
        print(f"Point cloud Z range: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
        
        # Check alpha channel consistency
        if 'A' in points.channels:
            alpha_values = points.channels['A']
            original_alpha = data['alpha'].cpu().numpy()
            print(f"Original alpha range: [{original_alpha.min():.3f}, {original_alpha.max():.3f}]")
            print(f"Point cloud alpha range: [{alpha_values.min():.3f}, {alpha_values.max():.3f}]")

def main():
    """
    Main function to run all tests
    """
    print("Starting Point Cloud Testing Script")
    print("=" * 50)
    
    # Test point cloud generation
    result = test_point_cloud_generation()
    if result is None:
        print("Failed to generate point clouds. Exiting.")
        return
    
    data, points = result
    
    # Analyze the point cloud
    analyze_point_cloud(points)
    
    # Create visualizations
    visualize_point_cloud_2d(points)
    visualize_point_cloud_3d(points)
    
    # Test point cloud operations
    test_point_cloud_operations(points)
    
    # Compare with original data
    compare_with_original_data(data, points)
    
    print("\n" + "=" * 50)
    print("Point Cloud Testing Complete!")
    print("Generated files:")
    print("- point_cloud_2d.png: 2D projections and color distribution")
    print("- point_cloud_3d.png: 3D visualization")

if __name__ == "__main__":
    main()
