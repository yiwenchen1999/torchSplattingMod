#!/usr/bin/env python3
"""
Test script to visualize alpha masks before point cloud creation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.point_utils import get_point_clouds

def visualize_alpha_masks(folder_path, output_dir="alpha_visualizations"):
    """
    Visualize alpha masks from a dataset folder before point cloud creation
    
    Args:
        folder_path: Path to the dataset folder
        output_dir: Directory to save visualization outputs
    """
    
    print(f"=== Alpha Mask Visualization for {folder_path} ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the dataset
        print("Loading dataset...")
        data = read_all(folder_path, resize_factor=1.0, latent_model=False)
        
        print(f"Dataset loaded successfully!")
        print(f"RGB shape: {data['rgb'].shape}")
        print(f"Alpha shape: {data['alpha'].shape}")
        print(f"Depth shape: {data['depth'].shape}")
        print(f"Camera shape: {data['camera'].shape}")
        
        # Get individual components
        rgbs = data['rgb']  # [N, H, W, 3]
        alphas = data['alpha']  # [N, H, W]
        depths = data['depth']  # [N, H, W]
        cameras = data['camera']  # [N, 15]
        
        num_images = rgbs.shape[0]
        print(f"Number of images: {num_images}")
        
        # Visualize first few images
        num_to_visualize = min(5, num_images)
        
        for i in range(num_to_visualize):
            print(f"\n--- Processing Image {i+1}/{num_to_visualize} ---")
            
            # Get current image data
            rgb = rgbs[i]  # [H, W, 3]
            alpha = alphas[i]  # [H, W]
            depth = depths[i]  # [H, W]
            camera = cameras[i]  # [15]
            
            print(f"Image {i}: RGB {rgb.shape}, Alpha {alpha.shape}, Depth {depth.shape}")
            print(f"Alpha range: {alpha.min():.4f} to {alpha.max():.4f}")
            print(f"Depth range: {depth.min():.4f} to {depth.max():.4f}")
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Image {i+1} Analysis', fontsize=16)
            
            # RGB image
            axes[0, 0].imshow(rgb.cpu().numpy())
            axes[0, 0].set_title('RGB Image')
            axes[0, 0].axis('off')
            
            # Alpha mask
            alpha_vis = alpha.cpu().numpy()
            im1 = axes[0, 1].imshow(alpha_vis, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title('Alpha Mask')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Depth map
            depth_vis = depth.cpu().numpy()
            im2 = axes[0, 2].imshow(depth_vis, cmap='viridis')
            axes[0, 2].set_title('Depth Map')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2])
            
            # Alpha histogram
            axes[1, 0].hist(alpha_vis.flatten(), bins=50, alpha=0.7, color='blue')
            axes[1, 0].set_title('Alpha Distribution')
            axes[1, 0].set_xlabel('Alpha Value')
            axes[1, 0].set_ylabel('Frequency')
            
            # Depth histogram
            axes[1, 1].hist(depth_vis.flatten(), bins=50, alpha=0.7, color='green')
            axes[1, 1].set_title('Depth Distribution')
            axes[1, 1].set_xlabel('Depth Value')
            axes[1, 1].set_ylabel('Frequency')
            
            # Alpha threshold visualization
            alpha_threshold = 0.5
            alpha_binary = (alpha_vis > alpha_threshold).astype(np.float32)
            axes[1, 2].imshow(alpha_binary, cmap='gray', vmin=0, vmax=1)
            axes[1, 2].set_title(f'Alpha > {alpha_threshold}')
            axes[1, 2].axis('off')
            
            # Add statistics text
            stats_text = f"""Alpha Stats:
Mean: {alpha_vis.mean():.4f}
Std: {alpha_vis.std():.4f}
Min: {alpha_vis.min():.4f}
Max: {alpha_vis.max():.4f}
Pixels > 0.5: {(alpha_vis > 0.5).sum()}
Total pixels: {alpha_vis.size}
Coverage: {(alpha_vis > 0.5).sum() / alpha_vis.size * 100:.2f}%"""
            
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            
            # Save visualization
            output_file = os.path.join(output_dir, f'alpha_analysis_image_{i+1}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_file}")
            
            plt.close()
        
        # Create summary visualization
        print("\n--- Creating Summary Visualization ---")
        create_summary_visualization(alphas, depths, output_dir)
        
        # Test point cloud creation with first image
        print("\n--- Testing Point Cloud Creation ---")
        test_point_cloud_creation(rgbs[4:5], alphas[4:5], depths[4:5], cameras[4:5], output_dir)
        
        print(f"\n=== Visualization Complete! ===")
        print(f"Check the '{output_dir}' folder for outputs.")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

def create_summary_visualization(alphas, depths, output_dir):
    """Create a summary visualization across all images"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Summary Analysis', fontsize=16)
    
    # Alpha statistics across all images
    alpha_means = [alpha.mean().item() for alpha in alphas]
    alpha_stds = [alpha.std().item() for alpha in alphas]
    
    axes[0, 0].plot(alpha_means, 'b-', label='Mean Alpha')
    axes[0, 0].fill_between(range(len(alpha_means)), 
                           [m - s for m, s in zip(alpha_means, alpha_stds)],
                           [m + s for m, s in zip(alpha_means, alpha_stds)],
                           alpha=0.3, color='blue')
    axes[0, 0].set_title('Alpha Statistics Across Images')
    axes[0, 0].set_xlabel('Image Index')
    axes[0, 0].set_ylabel('Alpha Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Depth statistics across all images
    depth_means = [depth.mean().item() for depth in depths]
    depth_stds = [depth.std().item() for depth in depths]
    
    axes[0, 1].plot(depth_means, 'g-', label='Mean Depth')
    axes[0, 1].fill_between(range(len(depth_means)), 
                           [m - s for m, s in zip(depth_means, depth_stds)],
                           [m + s for m, s in zip(depth_means, depth_stds)],
                           alpha=0.3, color='green')
    axes[0, 1].set_title('Depth Statistics Across Images')
    axes[0, 1].set_xlabel('Image Index')
    axes[0, 1].set_ylabel('Depth Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coverage percentage
    coverage_percentages = [(alpha > 0.5).sum().item() / alpha.numel() * 100 
                           for alpha in alphas]
    axes[1, 0].plot(coverage_percentages, 'r-', marker='o')
    axes[1, 0].set_title('Alpha Coverage Percentage')
    axes[1, 0].set_xlabel('Image Index')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined histogram
    all_alphas = torch.cat([alpha.flatten() for alpha in alphas])
    all_depths = torch.cat([depth.flatten() for depth in depths])
    
    axes[1, 1].hist(all_alphas.numpy(), bins=50, alpha=0.7, color='blue', 
                     label=f'Alpha (N={len(all_alphas)})', density=True)
    axes[1, 1].hist(all_depths.numpy(), bins=50, alpha=0.7, color='green', 
                     label=f'Depth (N={len(all_depths)})', density=True)
    axes[1, 1].set_title('Combined Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary
    output_file = os.path.join(output_dir, 'dataset_summary.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved summary visualization to: {output_file}")
    plt.close()

def test_point_cloud_creation(rgbs, alphas, depths, cameras, output_dir):
    """Test point cloud creation with a small subset"""
    
    try:
        print("Creating point cloud...")
        point_cloud = get_point_clouds(cameras, depths, alphas, rgbs)
        
        print(f"Point cloud created successfully!")
        print(f"Number of points: {point_cloud.coords.shape[0]}")
        print(f"Available channels: {list(point_cloud.channels.keys())}")
        
        # Save point cloud as PLY file
        ply_file = os.path.join(output_dir, 'test_point_cloud.ply')
        with open(ply_file, 'wb') as f:
            point_cloud.write_ply(f)
        print(f"Saved point cloud to: {ply_file}")
        
        # Create point cloud visualization
        if point_cloud.coords.shape[0] > 0:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Sample points for visualization (limit to 10000 for performance)
            if point_cloud.coords.shape[0] > 10000:
                indices = np.random.choice(point_cloud.coords.shape[0], 10000, replace=False)
                coords = point_cloud.coords[indices]
                colors = point_cloud.channels['R'][indices], point_cloud.channels['G'][indices], point_cloud.channels['B'][indices]
            else:
                coords = point_cloud.coords
                colors = point_cloud.channels['R'], point_cloud.channels['G'], point_cloud.channels['B']
            
            # Normalize colors to [0, 1]
            colors = np.stack(colors, axis=1)
            colors = np.clip(colors, 0, 1)
            
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                               c=colors, s=0.1, alpha=0.8)
            
            ax.set_title(f'Point Cloud Visualization ({coords.shape[0]} points)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            plt.tight_layout()
            
            # Save 3D visualization
            output_file = os.path.join(output_dir, 'point_cloud_3d.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved 3D visualization to: {output_file}")
            plt.close()
        
    except Exception as e:
        print(f"Error during point cloud creation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the visualization"""
    
    # Check if dataset path is provided
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # Default to looking for common dataset folders
        possible_paths = [
            "../datasamples/objaverse_synthetic/houseA_processed_train", 
        ]
        
        folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                folder_path = path
                break
        
        if folder_path is None:
            print("No dataset folder found. Please provide a path as argument.")
            print("Usage: python test_point_clouds.py <dataset_folder_path>")
            print("\nPossible paths:")
            for path in possible_paths:
                print(f"  {path}")
            return
    
    if not os.path.exists(folder_path):
        print(f"Dataset folder not found: {folder_path}")
        return
    
    print(f"Using dataset folder: {folder_path}")
    visualize_alpha_masks(folder_path)

if __name__ == "__main__":
    main()
