#!/usr/bin/env python3
"""
Analyze depth data in ship_latents/test to determine mapping to real depth values.
"""

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def analyze_depth_file(depth_file_path):
    """Analyze a single depth file to understand its format and range"""
    print(f"\nAnalyzing: {depth_file_path}")
    
    # Load the depth image
    depth_img = imageio.imread(depth_file_path)
    
    print(f"Shape: {depth_img.shape}")
    print(f"Data type: {depth_img.dtype}")
    print(f"Min value: {depth_img.min()}")
    print(f"Max value: {depth_img.max()}")
    print(f"Mean value: {depth_img.mean():.3f}")
    print(f"Std value: {depth_img.std():.3f}")
    
    # Check if it's multi-channel
    if len(depth_img.shape) == 3:
        print(f"Channels: {depth_img.shape[2]}")
        for i in range(depth_img.shape[2]):
            channel = depth_img[..., i]
            print(f"  Channel {i}: min={channel.min()}, max={channel.max()}, mean={channel.mean():.3f}")
    
    # Count zero pixels (background)
    zero_pixels = np.sum(depth_img == 0)
    total_pixels = depth_img.size
    print(f"Zero pixels: {zero_pixels} ({100*zero_pixels/total_pixels:.1f}%)")
    
    return depth_img

def test_different_normalizations(depth_data):
    """Test different normalization approaches to find the best mapping"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT NORMALIZATION APPROACHES")
    print("="*60)
    
    # Extract single channel if multi-channel
    if len(depth_data.shape) == 3:
        depth_single = depth_data[..., 0]
    else:
        depth_single = depth_data
    
    # Remove zero pixels (background) for analysis
    non_zero_mask = depth_single > 0
    non_zero_depth = depth_single[non_zero_mask]
    
    if len(non_zero_depth) == 0:
        print("No non-zero depth values found!")
        return
    
    print(f"Non-zero depth range: [{non_zero_depth.min()}, {non_zero_depth.max()}]")
    print(f"Non-zero depth mean: {non_zero_depth.mean():.3f}")
    
    # Test different normalization factors
    normalization_factors = [255, 512, 1024, 2048, 4096, 65535]
    
    print("\nNormalization Results:")
    print("-" * 50)
    for factor in normalization_factors:
        normalized = non_zero_depth.astype(np.float32) / factor
        print(f"Factor {factor:5d}: range [{normalized.min():.3f}, {normalized.max():.3f}], mean {normalized.mean():.3f}")
    
    # Test NeRF synthetic normalization specifically
    print("\nNeRF Synthetic Normalization (/512):")
    nerf_normalized = non_zero_depth.astype(np.float32) / 512
    print(f"Range: [{nerf_normalized.min():.3f}, {nerf_normalized.max():.3f}]")
    print(f"Mean: {nerf_normalized.mean():.3f}")
    print(f"Typical ship depth should be ~2-4 meters")
    
    # Test standard 8-bit normalization
    print("\nStandard 8-bit Normalization (/255):")
    std_normalized = non_zero_depth.astype(np.float32) / 255
    print(f"Range: [{std_normalized.min():.3f}, {std_normalized.max():.3f}]")
    print(f"Mean: {std_normalized.mean():.3f}")

def create_depth_visualization(depth_data, output_path, title):
    """Create visualization of depth data with different normalizations"""
    plt.figure(figsize=(20, 12))
    
    # Extract single channel if multi-channel
    if len(depth_data.shape) == 3:
        depth_single = depth_data[..., 0]
    else:
        depth_single = depth_data
    
    # Create subplots for different normalizations
    normalizations = [
        ("Raw", depth_single, "viridis"),
        ("/255", depth_single.astype(np.float32) / 255, "plasma"),
        ("/512 (NeRF)", depth_single.astype(np.float32) / 512, "inferno"),
        ("/65535", depth_single.astype(np.float32) / 65535, "magma")
    ]
    
    for i, (name, data, cmap) in enumerate(normalizations):
        plt.subplot(2, 2, i+1)
        
        # Create masked array to handle zeros
        masked_data = np.ma.masked_equal(data, 0)
        
        im = plt.imshow(masked_data, cmap=cmap)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"{name}\nRange: [{data.min():.1f}, {data.max():.1f}]")
        plt.axis('off')
    
    plt.suptitle(f"Depth Visualizations: {title}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created visualization: {output_path}")

def analyze_multiple_files():
    """Analyze multiple depth files to get a comprehensive understanding"""
    test_dir = Path("../ship_latents/test")
    depth_files = list(test_dir.glob("*depth*.png"))
    
    print(f"Found {len(depth_files)} depth files")
    
    if len(depth_files) == 0:
        print("No depth files found!")
        return
    
    # Analyze first few files
    sample_files = depth_files[:3]
    
    all_depths = []
    for depth_file in sample_files:
        depth_data = analyze_depth_file(depth_file)
        all_depths.append(depth_data)
        
        # Test normalizations for this file
        test_different_normalizations(depth_data)
        
        # Create visualization
        output_path = f"depth_analysis_{depth_file.stem}.png"
        create_depth_visualization(depth_data, output_path, depth_file.stem)
    
    # Aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    
    all_min = min(d.min() for d in all_depths)
    all_max = max(d.max() for d in all_depths)
    all_mean = np.mean([d.mean() for d in all_depths])
    
    print(f"Overall range: [{all_min}, {all_max}]")
    print(f"Overall mean: {all_mean:.3f}")
    
    # Determine best normalization
    print("\nRECOMMENDED NORMALIZATION:")
    if all_max <= 255:
        print("Data appears to be 8-bit. Use /255 normalization.")
        print("Real depth = normalized_depth * max_depth")
    elif all_max <= 65535:
        print("Data appears to be 16-bit. Testing best factor...")
        
        # Test common factors
        factors = [255, 512, 1024, 2048, 4096, 65535]
        best_factor = None
        best_range = None
        
        for factor in factors:
            normalized_max = all_max / factor
            if 1.0 <= normalized_max <= 10.0:  # Reasonable depth range
                if best_factor is None or abs(normalized_max - 4.0) < abs(best_range - 4.0):
                    best_factor = factor
                    best_range = normalized_max
        
        if best_factor:
            print(f"Recommended factor: /{best_factor}")
            print(f"This gives depth range: [0, {all_max/best_factor:.2f}] meters")
            print(f"Real depth = raw_depth / {best_factor}")
        else:
            print("No clear normalization factor found. Manual inspection needed.")
    
    return all_depths

def main():
    print("DEPTH DATA ANALYSIS FOR SHIP_LATENTS/TEST")
    print("="*60)
    
    # Analyze the depth files
    depth_data_list = analyze_multiple_files()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("1. Check the generated visualization files for visual inspection")
    print("2. Use the recommended normalization factor above")
    print("3. For torchSplattingMod, update the depth loading accordingly")
    print("4. Consider updating max_depth in info.json if needed")

if __name__ == "__main__":
    main()
