#!/usr/bin/env python3
"""
Preprocessing script to generate info.json for ship_latents folder
This script creates the necessary info.json file that torchSplattingMod can read.
It uses depth data from the test folder and generates alpha masks from RGB images.
"""

import os
import json
import numpy as np
import imageio
from PIL import Image
import argparse
from pathlib import Path
import cv2

def create_alpha_mask(rgb_path):
    """
    Create alpha mask from RGB image by thresholding
    """
    try:
        # Load image
        img = imageio.imread(rgb_path)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Create alpha mask by thresholding
        # Assuming background is dark/black
        _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Normalize to 0-1
        alpha = alpha.astype(np.float32) / 255.0
        
        return alpha
    except Exception as e:
        print(f"Error creating alpha mask for {rgb_path}: {e}")
        # Return all ones as fallback
        return np.ones((512, 512), dtype=np.float32)

def get_depth_from_test(rgb_filename, test_dir):
    """
    Try to find corresponding depth file in test directory
    """
    # Extract the base name (e.g., 'r_0' from 'r_0.png')
    base_name = rgb_filename.replace('.png', '')
    
    # Look for depth file with pattern like r_0_depth_0002.png
    depth_patterns = [
        f"{base_name}_depth_0002.png",
        f"{base_name}_depth.png",
        f"{base_name}_depth_0001.png"
    ]
    
    for pattern in depth_patterns:
        depth_path = os.path.join(test_dir, pattern)
        if os.path.exists(depth_path):
            return pattern
    
    print(f"Warning: No depth file found for {rgb_filename}")
    return None

def process_transforms_file(transforms_path, test_dir, output_dir):
    """
    Process transforms file and create info.json using only test set data
    """
    # Read transforms file
    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)
    
    # Get camera parameters
    camera_angle_x = transforms_data.get('camera_angle_x', 0.6911112070083618)
    
    # Calculate focal length from camera angle
    # Assuming image width of 512 (standard for this dataset)
    image_width = 512
    focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
    
    # Create intrinsic matrix
    intrinsic = [
        [focal_length, 0.0, image_width / 2],
        [0.0, focal_length, image_width / 2],
        [0.0, 0.0, 1.0]
    ]
    
    images = []
    
    for i, frame in enumerate(transforms_data['frames']):
        # Get file path and extract filename
        file_path = frame['file_path']
        rgb_filename = os.path.basename(file_path) + '.png'
        rgb_path = os.path.join(test_dir, rgb_filename)
        
        # Check if RGB file exists
        if not os.path.exists(rgb_path):
            print(f"Warning: RGB file not found: {rgb_path}")
            continue
        
        # Get transform matrix
        transform_matrix = frame['transform_matrix']
        
        # Convert from NeRF format to OpenCV format
        # NeRF uses different coordinate system
        pose = np.array(transform_matrix)
        
        # Create alpha mask
        alpha_filename = rgb_filename.replace('.png', '_alpha.png')
        alpha_path = os.path.join(output_dir, alpha_filename)
        alpha_mask = create_alpha_mask(rgb_path)
        imageio.imwrite(alpha_path, (alpha_mask * 255).astype(np.uint8))
        
        # Find depth file in the same test directory
        depth_filename = get_depth_from_test(rgb_filename, test_dir)
        
        # Create image entry
        image_entry = {
            "intrinsic": intrinsic,
            "pose": pose.tolist(),
            "rgb": rgb_filename,
            "alpha": alpha_filename,
            "max_depth": 5.0,
            "HW": [image_width, image_width]
        }
        
        if depth_filename:
            image_entry["depth"] = depth_filename
            # Copy depth file to output directory
            src_depth_path = os.path.join(test_dir, depth_filename)
            dst_depth_path = os.path.join(output_dir, depth_filename)
            if os.path.exists(src_depth_path):
                import shutil
                shutil.copy2(src_depth_path, dst_depth_path)
        
        images.append(image_entry)
        
        # Copy RGB file to output directory
        dst_rgb_path = os.path.join(output_dir, rgb_filename)
        import shutil
        shutil.copy2(rgb_path, dst_rgb_path)
    
    return images

def main():
    parser = argparse.ArgumentParser(description='Preprocess ship_latents for torchSplattingMod')
    parser.add_argument('--ship_latents_dir', type=str, default='../ship_latents',
                       help='Path to ship_latents directory')
    parser.add_argument('--output_dir', type=str, default='../ship_latents_processed',
                       help='Output directory for processed data')
    parser.add_argument('--transforms_file', type=str, default='transforms_test.json',
                       help='Transforms file to process (default: transforms_test.json)')
    
    args = parser.parse_args()
    
    # Setup paths
    ship_latents_dir = Path(args.ship_latents_dir)
    output_dir = Path(args.output_dir)
    test_dir = ship_latents_dir / 'test'
    transforms_path = ship_latents_dir / args.transforms_file
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check if directories exist
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not transforms_path.exists():
        raise FileNotFoundError(f"Transforms file not found: {transforms_path}")
    
    print(f"Processing {transforms_path}...")
    print(f"Test directory: {test_dir}")
    print(f"Output directory: {output_dir}")
    print("Using only test set data (ignoring train set)")
    
    # Process transforms file
    images = process_transforms_file(transforms_path, test_dir, output_dir)
    
    # Create info.json
    info_data = {
        "backend": "CYCLES",
        "light_mode": "uniform",
        "fast_mode": False,
        "format_version": 6,
        "channels": ["R", "G", "B", "A", "D"],
        "scale": 0.5,
        "images": images,
        "bbox": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]  # Default bounding box
    }
    
    # Save info.json
    info_path = output_dir / 'info.json'
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2)
    
    print(f"Successfully created {info_path}")
    print(f"Processed {len(images)} images")
    print(f"Output directory: {output_dir}")
    print("\nTo use with torchSplattingMod:")
    print(f"1. Copy the contents of {output_dir} to your torchSplattingMod data directory")
    print("2. Update the folder path in train.py to point to this directory")

if __name__ == "__main__":
    main()
