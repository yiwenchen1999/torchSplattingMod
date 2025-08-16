#!/usr/bin/env python3
"""
Analyze depth channels in ship_latents/test data and visualize with colormaps.
"""

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_depth_channels(image_path):
    """Analyze all channels in a depth image"""
    try:
        # Load image
        depth_img = imageio.imread(image_path)
        
        if len(depth_img.shape) != 3 or depth_img.shape[-1] != 4:
            print(f"Image {image_path} is not 4-channel RGBA")
            return None
        
        # Convert to float
        depth_float = depth_img.astype(np.float32) / 255.0
        
        # Analyze each channel
        channels = ['R', 'G', 'B', 'A']
        channel_stats = {}
        
        for i, channel in enumerate(channels):
            channel_data = depth_float[..., i]
            stats = {
                'min': channel_data.min(),
                'max': channel_data.max(),
                'mean': channel_data.mean(),
                'std': channel_data.std(),
                'unique_values': len(np.unique(channel_data)),
                'zero_pixels': np.sum(channel_data == 0),
                'total_pixels': channel_data.size,
                'zero_percentage': np.sum(channel_data == 0) / channel_data.size * 100
            }
            channel_stats[channel] = stats
        
        return depth_float, channel_stats
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None, None

def visualize_depth_channels(depth_data, channel_stats, filename):
    """Create visualization of all depth channels"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Depth Channel Analysis: {filename}', fontsize=16)
    
    channels = ['R', 'G', 'B', 'A']
    colormaps = ['viridis', 'plasma', 'inferno', 'magma']
    
    # Plot each channel with different colormaps
    for i, (channel, cmap) in enumerate(zip(channels, colormaps)):
        channel_data = depth_data[..., i]
        
        # Top row: channel visualization
        ax1 = axes[0, i]
        im1 = ax1.imshow(channel_data, cmap=cmap)
        ax1.set_title(f'{channel} Channel\nRange: [{channel_data.min():.3f}, {channel_data.max():.3f}]')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Bottom row: histogram
        ax2 = axes[1, i]
        ax2.hist(channel_data.flatten(), bins=50, alpha=0.7, color='blue')
        ax2.set_title(f'{channel} Histogram\nMean: {channel_data.mean():.3f}, Std: {channel_data.std():.3f}')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_channels_across_images(dataset_path, num_samples=5):
    """Compare channels across multiple images"""
    print(f"\n{'='*60}")
    print("COMPARING CHANNELS ACROSS MULTIPLE IMAGES")
    print(f"{'='*60}")
    
    # Find depth files
    depth_files = []
    for file in os.listdir(dataset_path):
        if 'depth' in file.lower() and file.endswith('.png'):
            depth_files.append(os.path.join(dataset_path, file))
    
    if len(depth_files) == 0:
        print("No depth files found!")
        return
    
    # Analyze first few files
    all_channel_stats = []
    
    for i, depth_file in enumerate(depth_files[:num_samples]):
        print(f"\nAnalyzing {os.path.basename(depth_file)}...")
        depth_data, channel_stats = analyze_depth_channels(depth_file)
        
        if channel_stats:
            all_channel_stats.append(channel_stats)
            
            # Print channel comparison
            channels = ['R', 'G', 'B', 'A']
            for channel in channels:
                stats = channel_stats[channel]
                print(f"  {channel}: range=[{stats['min']:.3f}, {stats['max']:.3f}], "
                      f"mean={stats['mean']:.3f}, zero_pixels={stats['zero_pixels']} "
                      f"({stats['zero_percentage']:.1f}%)")
    
    # Compare channels across images
    if all_channel_stats:
        print(f"\n{'='*60}")
        print("CHANNEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        channels = ['R', 'G', 'B', 'A']
        for channel in channels:
            # Calculate average stats for this channel across all images
            avg_min = np.mean([stats[channel]['min'] for stats in all_channel_stats])
            avg_max = np.mean([stats[channel]['max'] for stats in all_channel_stats])
            avg_mean = np.mean([stats[channel]['mean'] for stats in all_channel_stats])
            avg_zero_pct = np.mean([stats[channel]['zero_percentage'] for stats in all_channel_stats])
            
            print(f"{channel} Channel Average:")
            print(f"  Range: [{avg_min:.3f}, {avg_max:.3f}]")
            print(f"  Mean: {avg_mean:.3f}")
            print(f"  Zero pixels: {avg_zero_pct:.1f}%")
            print()

def create_depth_visualization(depth_data, filename, output_dir):
    """Create various depth visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Try different channels as depth
    channels = ['R', 'G', 'B', 'A']
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'turbo']
    
    for i, channel in enumerate(channels):
        channel_data = depth_data[..., i]
        
        # Create figure with multiple colormaps
        fig, axes = plt.subplots(1, len(colormaps), figsize=(20, 4))
        fig.suptitle(f'{channel} Channel Depth Visualization: {filename}', fontsize=16)
        
        for j, cmap in enumerate(colormaps):
            ax = axes[j]
            im = ax.imshow(channel_data, cmap=cmap)
            ax.set_title(f'{cmap}\nRange: [{channel_data.min():.3f}, {channel_data.max():.3f}]')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename}_{channel}_depth_visualization.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'All Channels Depth Visualization: {filename}', fontsize=16)
    
    for i, channel in enumerate(channels):
        row, col = i // 2, i % 2
        channel_data = depth_data[..., i]
        
        im = axes[row, col].imshow(channel_data, cmap='viridis')
        axes[row, col].set_title(f'{channel} Channel\nRange: [{channel_data.min():.3f}, {channel_data.max():.3f}]')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename}_all_channels.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    dataset_path = "../ship_latents/test"
    
    # Find a sample depth file
    depth_files = []
    for file in os.listdir(dataset_path):
        if 'depth' in file.lower() and file.endswith('.png'):
            depth_files.append(os.path.join(dataset_path, file))
    
    if len(depth_files) == 0:
        print("No depth files found!")
        return
    
    # Analyze first depth file in detail
    sample_file = depth_files[0]
    print(f"Analyzing sample file: {os.path.basename(sample_file)}")
    
    depth_data, channel_stats = analyze_depth_channels(sample_file)
    
    if depth_data is not None:
        # Create detailed visualization
        filename = os.path.basename(sample_file).replace('.png', '')
        fig = visualize_depth_channels(depth_data, channel_stats, filename)
        plt.savefig(f'depth_channel_analysis_{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create depth visualizations
        create_depth_visualization(depth_data, filename, 'depth_visualizations')
        
        print(f"\nVisualizations saved:")
        print(f"- depth_channel_analysis_{filename}.png")
        print(f"- depth_visualizations/{filename}_*_depth_visualization.png")
        print(f"- depth_visualizations/{filename}_all_channels.png")
    
    # Compare channels across multiple images
    compare_channels_across_images(dataset_path, num_samples=3)

if __name__ == "__main__":
    main()
