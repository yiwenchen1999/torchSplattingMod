import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.utils.data_utils_og import read_all
from gaussian_splatting.utils.point_utils import get_point_clouds, PointCloud

def find_data_folder():
    """
    Try to find a suitable data folder for testing
    """
    possible_folders = [
        '../nerf_synthetic/ship_latents_processed',
        '../ship_latents',
        'ship_latents',
        '../datasamples/000074a334c541878360457c672b6c2e',
        '../datasamples/00e9e6b9dbc34dc1ae691529688ab000',
    ]
    
    for folder in possible_folders:
        if os.path.exists(folder):
            print(f"Found data folder: {folder}")
            return folder
    
    print("No suitable data folder found. Please check the data paths.")
    return None

def test_point_clouds_basic():
    """
    Basic test of the get_point_clouds function
    """
    print("=== Basic Point Cloud Test ===")
    
    # Find data folder
    folder = find_data_folder()
    if folder is None:
        print("Cannot proceed without data folder")
        return None
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load data
        print(f"Loading data from: {folder}")
        data = read_all(folder, resize_factor=0.5)
        data = {k: v.to(device) for k, v in data.items()}
        data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)
        
        print(f"Data loaded successfully:")
        print(f"  - Number of images: {len(data['rgb'])}")
        print(f"  - RGB shape: {data['rgb'].shape}")
        print(f"  - Depth shape: {data['depth'].shape}")
        print(f"  - Alpha shape: {data['alpha'].shape}")
        
        # Generate point clouds
        print("\nGenerating point clouds...")
        points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
        
        print(f"Point cloud generated successfully:")
        print(f"  - Number of points: {len(points.coords)}")
        print(f"  - Available channels: {list(points.channels.keys())}")
        
        # Basic statistics
        coords = points.coords
        print(f"  - Coordinate range:")
        print(f"    X: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
        print(f"    Y: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
        print(f"    Z: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
        
        if 'R' in points.channels and 'G' in points.channels and 'B' in points.channels:
            rgb = np.stack([points.channels['R'], points.channels['G'], points.channels['B']], axis=1)
            print(f"  - RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        
        return data, points
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_points_simple(points, save_path="point_cloud_simple.png"):
    """
    Simple visualization of the point cloud
    """
    if points is None:
        print("No points to visualize")
        return
    
    print(f"\nCreating simple visualization...")
    
    coords = points.coords
    channels = points.channels
    
    # Sample points if there are too many
    if len(coords) > 5000:
        indices = np.random.choice(len(coords), 5000, replace=False)
        coords = coords[indices]
        if 'R' in channels and 'G' in channels and 'B' in channels:
            colors = np.stack([channels['R'][indices], channels['G'][indices], channels['B'][indices]], axis=1)
        else:
            colors = 'blue'
    else:
        if 'R' in channels and 'G' in channels and 'B' in channels:
            colors = np.stack([channels['R'], channels['G'], channels['B']], axis=1)
        else:
            colors = 'blue'
    
    # Create 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY projection
    axes[0].scatter(coords[:, 0], coords[:, 1], c=colors, s=0.5, alpha=0.6)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Projection')
    axes[0].set_aspect('equal')
    
    # XZ projection
    axes[1].scatter(coords[:, 0], coords[:, 2], c=colors, s=0.5, alpha=0.6)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Projection')
    axes[1].set_aspect('equal')
    
    # YZ projection
    axes[2].scatter(coords[:, 1], coords[:, 2], c=colors, s=0.5, alpha=0.6)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Projection')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

def test_point_operations(points):
    """
    Test basic point cloud operations
    """
    if points is None:
        print("No points to test operations on")
        return
    
    print("\n=== Testing Point Cloud Operations ===")
    
    # Test random sampling
    print("Testing random sampling...")
    sampled = points.random_sample(1000)
    print(f"  - Sampled {len(sampled.coords)} points from {len(points.coords)}")
    
    # Test save/load
    print("Testing save/load...")
    try:
        points.save("test_points.npz")
        loaded = PointCloud.load("test_points.npz")
        print(f"  - Save/load successful: {len(loaded.coords)} points")
        os.remove("test_points.npz")
    except Exception as e:
        print(f"  - Save/load failed: {e}")
    
    # Test PLY export
    print("Testing PLY export...")
    try:
        with open("test_points.ply", "wb") as f:
            points.write_ply(f)
        print("  - PLY export successful")
        os.remove("test_points.ply")
    except Exception as e:
        print(f"  - PLY export failed: {e}")

def main():
    """
    Main function for simple testing
    """
    print("Simple Point Cloud Testing Script")
    print("=" * 40)
    
    # Test point cloud generation
    result = test_point_clouds_basic()
    if result is None:
        print("Test failed. Exiting.")
        return
    
    data, points = result
    
    # Create visualization
    visualize_points_simple(points)
    
    # Test operations
    test_point_operations(points)
    
    print("\n" + "=" * 40)
    print("Simple testing complete!")
    print("Generated files:")
    print("- point_cloud_simple.png: 2D projections")

if __name__ == "__main__":
    main()
