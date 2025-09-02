#!/usr/bin/env python3
"""
Test script to verify EXR file reading functionality
"""

import os
import sys
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_splatting.utils.data_utils import read_exr_depth, HAS_OPENEXR, HAS_CV2

def test_exr_reading():
    """Test EXR file reading with different methods"""
    
    print("=== EXR Reading Test ===")
    print(f"OpenEXR available: {HAS_OPENEXR}")
    print(f"OpenCV available: {HAS_CV2}")
    
    # Look for EXR files in the datasamples directory
    datasamples_dir = "../datasamples"
    exr_files = []
    
    for root, dirs, files in os.walk(datasamples_dir):
        for file in files:
            if file.endswith('.exr'):
                exr_files.append(os.path.join(root, file))
    
    if not exr_files:
        print("No EXR files found in datasamples directory")
        return
    
    print(f"\nFound {len(exr_files)} EXR files:")
    for exr_file in exr_files[:5]:  # Show first 5
        print(f"  {exr_file}")
    
    # Test reading the first EXR file
    test_file = exr_files[0]
    print(f"\nTesting with: {test_file}")
    
    try:
        if HAS_OPENEXR:
            print("\n--- Testing OpenEXR ---")
            depth_array = read_exr_depth(test_file)
            print(f"Success! Depth shape: {depth_array.shape}")
            print(f"Depth dtype: {depth_array.dtype}")
            print(f"Depth range: {depth_array.min():.6f} to {depth_array.max():.6f}")
            print(f"Sample values [0:5, 0:5]:\n{depth_array[0:5, 0:5]}")
        else:
            print("OpenEXR not available, skipping test")
            
    except Exception as e:
        print(f"OpenEXR test failed: {e}")
    
    # Test with OpenCV if available
    if HAS_CV2:
        try:
            print("\n--- Testing OpenCV ---")
            import cv2
            depth_array = cv2.imread(test_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if depth_array is not None:
                print(f"Success! Depth shape: {depth_array.shape}")
                print(f"Depth dtype: {depth_array.dtype}")
                print(f"Depth range: {depth_array.min():.6f} to {depth_array.max():.6f}")
                print(f"Sample values [0:5, 0:5]:\n{depth_array[0:5, 0:5]}")
            else:
                print("OpenCV failed to read EXR file")
        except Exception as e:
            print(f"OpenCV test failed: {e}")
    
    # Test with imageio
    try:
        print("\n--- Testing ImageIO ---")
        import imageio
        depth_array = imageio.imread(test_file)
        print(f"Success! Depth shape: {depth_array.shape}")
        print(f"Depth dtype: {depth_array.dtype}")
        if hasattr(depth_array, 'min') and hasattr(depth_array, 'max'):
            print(f"Depth range: {depth_array.min():.6f} to {depth_array.max():.6f}")
        print(f"Sample values [0:5, 0:5]:\n{depth_array[0:5, 0:5]}")
    except Exception as e:
        print(f"ImageIO test failed: {e}")

if __name__ == "__main__":
    test_exr_reading()
