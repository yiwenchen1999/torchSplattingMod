#!/usr/bin/env python3
"""
Determine the best mapping from ship_latents/test depth data to real depth values.
"""

import numpy as np
import imageio
from pathlib import Path

def analyze_depth_mapping():
    """Analyze depth data and determine the best mapping to real depth"""
    
    print("DEPTH MAPPING ANALYSIS FOR SHIP_LATENTS/TEST")
    print("="*60)
    
    # Based on the analysis results
    print("ANALYSIS RESULTS:")
    print("-" * 40)
    print("Data format: 4-channel uint8 images")
    print("Raw range: [0, 167] (8-bit, not 16-bit)")
    print("All channels are identical")
    print("Background: 0 (68.6% of pixels)")
    print("Non-zero range: [83, 167]")
    print("Non-zero mean: ~137")
    
    print("\nNORMALIZATION OPTIONS:")
    print("-" * 40)
    
    # Test different normalization approaches
    non_zero_min, non_zero_max = 83, 167
    non_zero_mean = 137
    
    print("1. Standard 8-bit normalization (/255):")
    std_min = non_zero_min / 255
    std_max = non_zero_max / 255
    std_mean = non_zero_mean / 255
    print(f"   Range: [{std_min:.3f}, {std_max:.3f}]")
    print(f"   Mean: {std_mean:.3f}")
    print(f"   If max_depth=5.0: Real depth range [{std_min*5:.2f}, {std_max*5:.2f}] meters")
    
    print("\n2. NeRF synthetic normalization (/512):")
    nerf_min = non_zero_min / 512
    nerf_max = non_zero_max / 512
    nerf_mean = non_zero_mean / 512
    print(f"   Range: [{nerf_min:.3f}, {nerf_max:.3f}]")
    print(f"   Mean: {nerf_mean:.3f}")
    print(f"   Real depth range: [{nerf_min:.2f}, {nerf_max:.2f}] meters")
    
    print("\n3. Custom normalization for ship scene:")
    # Assuming ship is typically 2-4 meters from camera
    target_min, target_max = 2.0, 4.0
    scale_factor = (target_max - target_min) / (non_zero_max - non_zero_min)
    custom_min = non_zero_min * scale_factor + target_min
    custom_max = non_zero_max * scale_factor + target_min
    print(f"   Scale factor: {scale_factor:.4f}")
    print(f"   Real depth range: [{custom_min:.2f}, {custom_max:.2f}] meters")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    
    print("Option A: Use NeRF synthetic normalization (/512)")
    print(f"   - Real depth = raw_depth / 512")
    print(f"   - Gives reasonable range: [{nerf_min:.2f}, {nerf_max:.2f}] meters")
    print(f"   - Consistent with NeRF paper conventions")
    
    print("\nOption B: Use standard normalization with custom max_depth")
    print(f"   - Real depth = (raw_depth / 255) * max_depth")
    print(f"   - Set max_depth = {std_max/0.8:.1f} for 80% of max value")
    print(f"   - This gives range: [0, {std_max/0.8:.1f}] meters")
    
    print("\nOption C: Custom scaling for ship scene")
    print(f"   - Real depth = (raw_depth - {non_zero_min}) * {scale_factor:.4f} + {target_min}")
    print(f"   - Maps to typical ship depth range: [{target_min}, {target_max}] meters")
    
    return {
        'raw_range': (0, 167),
        'non_zero_range': (83, 167),
        'std_normalization': (std_min, std_max),
        'nerf_normalization': (nerf_min, nerf_max),
        'custom_scaling': (custom_min, custom_max),
        'scale_factor': scale_factor
    }

def create_mapping_functions():
    """Create functions for different depth mapping approaches"""
    
    print("\n" + "="*60)
    print("MAPPING FUNCTIONS")
    print("="*60)
    
    # Based on analysis
    non_zero_min = 83
    target_min, target_max = 2.0, 4.0
    scale_factor = (target_max - target_min) / (167 - non_zero_min)
    
    print("Python functions for depth mapping:")
    print("-" * 40)
    
    print("1. NeRF synthetic normalization:")
    print("def map_depth_nerf(raw_depth):")
    print("    return raw_depth.astype(np.float32) / 512")
    print()
    
    print("2. Standard normalization with max_depth:")
    print("def map_depth_standard(raw_depth, max_depth=6.25):")
    print("    return (raw_depth.astype(np.float32) / 255) * max_depth")
    print()
    
    print("3. Custom ship scene mapping:")
    print("def map_depth_ship(raw_depth):")
    print(f"    return (raw_depth - {non_zero_min}) * {scale_factor:.4f} + {target_min}")
    print()
    
    print("4. For torchSplattingMod data_utils.py:")
    print("-" * 40)
    print("Current line:")
    print("depth = torch.from_numpy(imageio.imread(rgb_file[:-7]+'depth.png').astype(np.float32) / 255.0 * max_depth)")
    print()
    print("For NeRF normalization:")
    print("depth = torch.from_numpy(imageio.imread(rgb_file[:-7]+'depth.png').astype(np.float32) / 512)")
    print()
    print("For custom ship mapping:")
    print(f"raw_depth = imageio.imread(rgb_file[:-7]+'depth.png').astype(np.float32)")
    print(f"depth = torch.from_numpy((raw_depth - {non_zero_min}) * {scale_factor:.4f} + {target_min})")

def test_mapping_on_sample():
    """Test the mapping functions on a sample depth file"""
    
    print("\n" + "="*60)
    print("TESTING MAPPING ON SAMPLE FILE")
    print("="*60)
    
    # Load a sample depth file
    sample_file = Path("../ship_latents/test/r_0_depth_0002.png")
    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        return
    
    depth_img = imageio.imread(sample_file)
    depth_single = depth_img[..., 0]  # Extract first channel
    
    # Remove background
    non_zero_mask = depth_single > 0
    non_zero_depth = depth_single[non_zero_mask]
    
    print(f"Sample file: {sample_file.name}")
    print(f"Raw depth range: [{depth_single.min()}, {depth_single.max()}]")
    print(f"Non-zero depth range: [{non_zero_depth.min()}, {non_zero_depth.max()}]")
    print(f"Non-zero depth mean: {non_zero_depth.mean():.2f}")
    
    # Test different mappings
    print("\nMapping Results:")
    print("-" * 30)
    
    # NeRF normalization
    nerf_depth = non_zero_depth.astype(np.float32) / 512
    print(f"NeRF (/512): range [{nerf_depth.min():.3f}, {nerf_depth.max():.3f}], mean {nerf_depth.mean():.3f}")
    
    # Standard normalization
    std_depth = non_zero_depth.astype(np.float32) / 255
    print(f"Standard (/255): range [{std_depth.min():.3f}, {std_depth.max():.3f}], mean {std_depth.mean():.3f}")
    
    # Custom ship mapping
    non_zero_min = 83
    target_min, target_max = 2.0, 4.0
    scale_factor = (target_max - target_min) / (167 - non_zero_min)
    custom_depth = (non_zero_depth - non_zero_min) * scale_factor + target_min
    print(f"Custom ship: range [{custom_depth.min():.3f}, {custom_depth.max():.3f}], mean {custom_depth.mean():.3f}")

def main():
    # Analyze the mapping options
    mapping_info = analyze_depth_mapping()
    
    # Create mapping functions
    create_mapping_functions()
    
    # Test on sample file
    test_mapping_on_sample()
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    print("For ship_latents/test depth data, I recommend:")
    print()
    print("1. PRIMARY: Use NeRF synthetic normalization (/512)")
    print("   - Real depth = raw_depth / 512")
    print("   - Gives range: [0.16, 0.33] meters")
    print("   - Consistent with NeRF conventions")
    print()
    print("2. ALTERNATIVE: Use custom ship mapping")
    print("   - Maps to typical ship depth range: [2.0, 4.0] meters")
    print("   - More realistic for ship scenes")
    print()
    print("3. UPDATE torchSplattingMod:")
    print("   - Modify data_utils.py line 79")
    print("   - Change from '/ 255.0 * max_depth' to '/ 512'")
    print("   - Or implement custom ship mapping")

if __name__ == "__main__":
    main()
