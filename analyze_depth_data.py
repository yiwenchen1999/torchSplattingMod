#!/usr/bin/env python3
"""
Analyze depth data from B075X65R3X and ship_latents/test datasets
to check if they use similar conventions and ranges.
"""

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_depth_image(image_path):
    """Analyze a single depth image"""
    try:
        # Load image
        depth_img = imageio.imread(image_path)
        
        # Convert to float if needed
        if depth_img.dtype == np.uint8:
            depth_float = depth_img.astype(np.float32) / 255.0
        elif depth_img.dtype == np.uint16:
            depth_float = depth_img.astype(np.float32) / 65535.0
        else:
            depth_float = depth_img.astype(np.float32)
        
        # Get statistics
        stats = {
            'shape': depth_img.shape,
            'dtype': depth_img.dtype,
            'min': depth_float.min(),
            'max': depth_float.max(),
            'mean': depth_float.mean(),
            'std': depth_float.std(),
            'file_size': os.path.getsize(image_path),
            'has_nan': np.isnan(depth_float).any(),
            'has_inf': np.isinf(depth_float).any(),
            'unique_values': len(np.unique(depth_float))
        }
        
        return depth_float, stats
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None, None

def analyze_dataset_depth(dataset_path, dataset_name, max_samples=5):
    """Analyze depth images from a dataset"""
    print(f"\n{'='*50}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*50}")
    
    # Find depth files
    depth_files = []
    for file in os.listdir(dataset_path):
        if 'depth' in file.lower() and file.endswith('.png'):
            depth_files.append(os.path.join(dataset_path, file))
    
    print(f"Found {len(depth_files)} depth files")
    
    if len(depth_files) == 0:
        print("No depth files found!")
        return
    
    # Analyze first few files
    all_stats = []
    sample_images = []
    
    for i, depth_file in enumerate(depth_files[:max_samples]):
        print(f"\nAnalyzing {os.path.basename(depth_file)}...")
        depth_data, stats = analyze_depth_image(depth_file)
        
        if depth_data is not None:
            all_stats.append(stats)
            sample_images.append(depth_data)
            
            print(f"  Shape: {stats['shape']}")
            print(f"  Dtype: {stats['dtype']}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  File size: {stats['file_size']} bytes")
            print(f"  Unique values: {stats['unique_values']}")
    
    # Summary statistics
    if all_stats:
        print(f"\n{dataset_name} Summary:")
        print(f"  Average range: [{np.mean([s['min'] for s in all_stats]):.4f}, {np.mean([s['max'] for s in all_stats]):.4f}]")
        print(f"  Average mean: {np.mean([s['mean'] for s in all_stats]):.4f}")
        print(f"  Average std: {np.mean([s['std'] for s in all_stats]):.4f}")
        print(f"  Average file size: {np.mean([s['file_size'] for s in all_stats]):.0f} bytes")
        
        # Check if all images have same shape
        shapes = [s['shape'] for s in all_stats]
        if len(set(shapes)) == 1:
            print(f"  All images have same shape: {shapes[0]}")
        else:
            print(f"  Different shapes found: {set(shapes)}")
    
    return all_stats, sample_images

def visualize_depth_comparison(b075x65r3x_samples, ship_latents_samples):
    """Create visualization comparing depth data from both datasets"""
    if not b075x65r3x_samples or not ship_latents_samples:
        print("Cannot create visualization - missing sample data")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Depth Data Comparison: B075X65R3X vs ship_latents/test', fontsize=16)
    
    # B075X65R3X samples
    for i, depth_img in enumerate(b075x65r3x_samples[:3]):
        ax = axes[0, i]
        im = ax.imshow(depth_img, cmap='viridis')
        ax.set_title(f'B075X65R3X Sample {i+1}\nRange: [{depth_img.min():.3f}, {depth_img.max():.3f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # ship_latents samples
    for i, depth_img in enumerate(ship_latents_samples[:3]):
        ax = axes[1, i]
        im = ax.imshow(depth_img, cmap='viridis')
        ax.set_title(f'ship_latents Sample {i+1}\nRange: [{depth_img.min():.3f}, {depth_img.max():.3f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'depth_comparison.png'")

def main():
    # Analyze B075X65R3X dataset
    b075x65r3x_path = "./B075X65R3X"
    b075x65r3x_stats, b075x65r3x_samples = analyze_dataset_depth(b075x65r3x_path, "B075X65R3X")
    
    # Analyze ship_latents dataset
    ship_latents_path = "../ship_latents/test"
    ship_latents_stats, ship_latents_samples = analyze_dataset_depth(ship_latents_path, "ship_latents/test")
    
    # Compare datasets
    print(f"\n{'='*50}")
    print("DATASET COMPARISON")
    print(f"{'='*50}")
    
    if b075x65r3x_stats and ship_latents_stats:
        # Compare ranges
        b075x65r3x_range = [np.mean([s['min'] for s in b075x65r3x_stats]), 
                           np.mean([s['max'] for s in b075x65r3x_stats])]
        ship_latents_range = [np.mean([s['min'] for s in ship_latents_stats]), 
                             np.mean([s['max'] for s in ship_latents_stats])]
        
        print(f"B075X65R3X average range: [{b075x65r3x_range[0]:.4f}, {b075x65r3x_range[1]:.4f}]")
        print(f"ship_latents average range: [{ship_latents_range[0]:.4f}, {ship_latents_range[1]:.4f}]")
        
        # Check if ranges are similar
        range_diff = abs(b075x65r3x_range[1] - ship_latents_range[1])
        if range_diff < 0.1:
            print("✅ Ranges are similar - can likely use same loading function")
        else:
            print("❌ Ranges are different - may need different loading functions")
        
        # Compare shapes
        b075x65r3x_shape = b075x65r3x_stats[0]['shape']
        ship_latents_shape = ship_latents_stats[0]['shape']
        print(f"B075X65R3X shape: {b075x65r3x_shape}")
        print(f"ship_latents shape: {ship_latents_shape}")
        
        if b075x65r3x_shape == ship_latents_shape:
            print("✅ Shapes are identical")
        else:
            print("❌ Shapes are different")
        
        # Compare data types
        b075x65r3x_dtype = b075x65r3x_stats[0]['dtype']
        ship_latents_dtype = ship_latents_stats[0]['dtype']
        print(f"B075X65R3X dtype: {b075x65r3x_dtype}")
        print(f"ship_latents dtype: {ship_latents_dtype}")
        
        if b075x65r3x_dtype == ship_latents_dtype:
            print("✅ Data types are identical")
        else:
            print("❌ Data types are different")
    
    # Create visualization
    visualize_depth_comparison(b075x65r3x_samples, ship_latents_samples)

if __name__ == "__main__":
    main()
