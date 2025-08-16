#!/usr/bin/env python3
"""
Create depth visualizations from ship_latents/test and store them in vis_depths_gt folder.
"""

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def create_depth_visualization(depth_data, output_path, filename, colormap='viridis'):
    """Create a single depth visualization with colormap"""
    plt.figure(figsize=(12, 10))
    
    # Create the visualization
    im = plt.imshow(depth_data, cmap=colormap)
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Depth', rotation=270, labelpad=15)
    
    # Set title and labels
    plt.title(f'Depth Visualization: {filename}\nRange: [{depth_data.min():.3f}, {depth_data.max():.3f}]', 
              fontsize=14, pad=20)
    plt.axis('off')
    
    # Save the visualization
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def process_depth_file(depth_file_path, output_dir, filename):
    """Process a single depth file and create visualization"""
    try:
        # Load the depth image
        depth_img = imageio.imread(depth_file_path)
        
        # Extract depth from any channel (they're all identical)
        depth_data = depth_img[..., 0].astype(np.float32) / 255.0
        
        # Create output filename
        output_filename = f"{filename}_depth_gt.png"
        output_path = output_dir / output_filename
        
        # Create visualization
        create_depth_visualization(depth_data, output_path, filename)
        
        print(f"Created: {output_filename}")
        return True
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

def create_multiple_colormap_visualization(depth_data, output_dir, filename):
    """Create visualization with multiple colormaps for comparison"""
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'turbo']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Depth Visualization Comparison: {filename}', fontsize=16, y=0.95)
    
    # Plot each colormap
    for i, cmap in enumerate(colormaps):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        im = ax.imshow(depth_data, cmap=cmap)
        ax.set_title(f'{cmap}\nRange: [{depth_data.min():.3f}, {depth_data.max():.3f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide the last subplot if we have odd number of colormaps
    if len(colormaps) < 6:
        axes[1, 2].axis('off')
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    
    # Save the comparison
    output_filename = f"{filename}_depth_comparison.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created: {output_filename}")

def main():
    # Setup paths
    ship_latents_test_dir = Path("../ship_latents/test")
    vis_depths_gt_dir = ship_latents_test_dir / "vis_depths_gt"
    
    # Create output directory
    vis_depths_gt_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {vis_depths_gt_dir}")
    
    # Find all depth files
    depth_files = []
    for file in ship_latents_test_dir.glob("*depth*.png"):
        depth_files.append(file)
    
    print(f"Found {len(depth_files)} depth files")
    
    if len(depth_files) == 0:
        print("No depth files found!")
        return
    
    # Process each depth file
    processed_count = 0
    
    for i, depth_file in enumerate(depth_files):
        # Extract filename without extension
        filename = depth_file.stem
        
        print(f"\nProcessing {i+1}/{len(depth_files)}: {filename}")
        
        # Create standard visualization
        success = process_depth_file(depth_file, vis_depths_gt_dir, filename)
        
        if success:
            processed_count += 1
            
            # Create comparison visualization for first few files
            if i < 5:  # Only for first 5 files to avoid too many files
                try:
                    depth_img = imageio.imread(depth_file)
                    depth_data = depth_img[..., 0].astype(np.float32) / 255.0
                    create_multiple_colormap_visualization(depth_data, vis_depths_gt_dir, filename)
                except Exception as e:
                    print(f"Error creating comparison for {filename}: {e}")
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Processed: {processed_count}/{len(depth_files)} files")
    print(f"Output directory: {vis_depths_gt_dir}")
    print(f"Files created:")
    print(f"  - {processed_count} standard depth visualizations (*_depth_gt.png)")
    print(f"  - 5 comparison visualizations (*_depth_comparison.png)")
    
    # Create a summary visualization
    create_summary_visualization(vis_depths_gt_dir, ship_latents_test_dir)

def create_summary_visualization(vis_dir, test_dir):
    """Create a summary visualization showing a few examples"""
    # Find some sample depth files
    depth_files = list(test_dir.glob("*depth*.png"))[:9]  # First 9 files
    
    if len(depth_files) < 9:
        print("Not enough depth files for summary visualization")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Depth Visualization Examples (ship_latents/test)', fontsize=16, y=0.95)
    
    for i, depth_file in enumerate(depth_files):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        try:
            # Load and process depth
            depth_img = imageio.imread(depth_file)
            depth_data = depth_img[..., 0].astype(np.float32) / 255.0
            
            # Create visualization
            im = ax.imshow(depth_data, cmap='viridis')
            ax.set_title(f'{depth_file.stem}\nRange: [{depth_data.min():.3f}, {depth_data.max():.3f}]')
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{depth_file.name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = vis_dir / "depth_summary_examples.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created: depth_summary_examples.png")

if __name__ == "__main__":
    main()
