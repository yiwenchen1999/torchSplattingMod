#!/usr/bin/env python3
"""
Script to generate info.json files for bus dataset train and test splits.
Converts from bus dataset format to torchSplattingMod format.
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import cv2

def calculate_intrinsic_matrix(fov_degrees, image_width, image_height):
    """
    Calculate intrinsic matrix from FOV and image dimensions
    """
    fov_radians = np.radians(fov_degrees)
    focal_length = 0.5 * image_width / np.tan(0.5 * fov_radians)
    
    intrinsic = [
        [focal_length, 0.0, image_width / 2],
        [0.0, focal_length, image_height / 2],
        [0.0, 0.0, 1.0]
    ]
    return intrinsic

def create_alpha_mask_from_rgb(rgb_path):
    """
    Create alpha mask from RGB image by thresholding
    """
    try:
        # Load image
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            # Try with PIL as fallback
            pil_img = Image.open(rgb_path)
            img = np.array(pil_img)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        if os.path.exists(rgb_path):
            img = Image.open(rgb_path)
            return np.ones(img.size[::-1], dtype=np.float32)  # PIL size is (w,h), we need (h,w)
        else:
            return np.ones((512, 512), dtype=np.float32)

def process_bus_dataset(data_dir, split, output_dir):
    """
    Process bus dataset and generate info.json
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Read camera data
    cameras_file = data_path / split / "cameras.json"
    if not cameras_file.exists():
        raise FileNotFoundError(f"Camera file not found: {cameras_file}")
    
    with open(cameras_file, 'r') as f:
        cameras_data = json.load(f)
    
    # Get image dimensions from first image
    if split == "train":
        # Look for gt files in white_env_0
        rgb_files = list(data_path.glob(f"{split}/white_env_0/gt_*.png"))
        if not rgb_files:
            raise FileNotFoundError(f"No RGB files found in {data_path}/{split}/white_env_0")
        sample_rgb = rgb_files[0]
    else:  # test
        # Look for gt files in white_env_0
        rgb_files = list(data_path.glob(f"{split}/white_env_0/gt_*.png"))
        if not rgb_files:
            raise FileNotFoundError(f"No RGB files found in {data_path}/{split}/white_env_0")
        sample_rgb = rgb_files[0]
    
    # Get image dimensions
    with Image.open(sample_rgb) as img:
        image_width, image_height = img.size
    
    print(f"Image dimensions: {image_width}x{image_height}")
    
    # Calculate intrinsic matrix (assuming FOV from camera data)
    fov = cameras_data[0].get('fov', 30)  # Default to 30 degrees
    intrinsic = calculate_intrinsic_matrix(fov, image_width, image_height)
    
    images = []
    
    # Track statistics
    depth_files_found = 0
    placeholders_created = 0
    
    # Process each camera
    for i, camera in enumerate(cameras_data):
        eye_idx = camera['eye_idx']
        c2w_matrix = np.array(camera['c2w'])
        
        print(f"Processing camera {i+1}/{len(cameras_data)} (eye_idx: {eye_idx})")
        
        # Determine file paths based on split
        if split == "train":
            # Train uses gt_X.png naming in white_env_0 subdirectory
            rgb_filename = f"gt_{eye_idx}.png"
            rgb_path = data_path / split / "white_env_0" / rgb_filename
            
            # Look for corresponding depth file in depth folder
            depth_filename = None
            depth_patterns = [
                f"depth_{eye_idx}.exr",
                f"depth_{eye_idx}0001.exr",
                f"depth_{eye_idx:06d}.exr"
            ]
            
            for pattern in depth_patterns:
                depth_path = data_path / split / "depth" / pattern
                if depth_path.exists():
                    depth_filename = pattern
                    break
        else:  # test
            # Test uses gt_X.png naming in white_env_0 subdirectory
            rgb_filename = f"gt_{eye_idx}.png"
            rgb_path = data_path / split / "white_env_0" / rgb_filename
            
            # Look for corresponding depth file in depth folder (same as train)
            depth_filename = None
            depth_patterns = [
                f"depth_{eye_idx}.exr",
                f"depth_{eye_idx}0001.exr",
                f"depth_{eye_idx:06d}.exr"
            ]
            
            for pattern in depth_patterns:
                depth_path = data_path / split / "depth" / pattern
                if depth_path.exists():
                    depth_filename = pattern
                    break
        
        # Check if RGB file exists
        if not rgb_path.exists():
            print(f"Warning: RGB file not found: {rgb_path}")
            continue
        
        # Create alpha mask
        alpha_filename = rgb_filename.replace('.png', '_alpha.png')
        alpha_path = output_path / alpha_filename
        
        # Generate alpha mask
        alpha_mask = create_alpha_mask_from_rgb(str(rgb_path))
        alpha_img = Image.fromarray((alpha_mask * 255).astype(np.uint8))
        alpha_img.save(alpha_path)
        
        # Create image entry
        image_entry = {
            "intrinsic": intrinsic,
            "pose": c2w_matrix.tolist(),
            "rgb": rgb_filename,
            "alpha": alpha_filename,
            "max_depth": 5.0,
            "HW": [image_height, image_width]  # Note: HW is [height, width]
        }
        
        if depth_filename:
            image_entry["depth"] = depth_filename
            # Copy depth file to output directory
            src_depth_path = data_path / split / "depth" / depth_filename
            dst_depth_path = output_path / depth_filename
            if src_depth_path.exists():
                # Copy the .exr file as is
                import shutil
                shutil.copy2(src_depth_path, dst_depth_path)
                print(f"  ✓ Copied depth file: {depth_filename}")
                depth_files_found += 1
            else:
                print(f"  ⚠ Warning: Depth file not found: {src_depth_path}")
                # Create black depth placeholder as fallback
                depth_filename = rgb_filename.replace('.png', '_depth_placeholder.png')
                image_entry["depth"] = depth_filename
                depth_placeholder = np.zeros((image_height, image_width), dtype=np.uint8)
                depth_img = Image.fromarray(depth_placeholder)
                dst_depth_path = output_path / depth_filename
                depth_img.save(dst_depth_path)
                print(f"  ⚠ Created black depth placeholder as fallback: {depth_filename}")
                placeholders_created += 1
        else:
            # No depth file found, create black depth placeholder
            depth_filename = rgb_filename.replace('.png', '_depth_placeholder.png')
            image_entry["depth"] = depth_filename
            depth_placeholder = np.zeros((image_height, image_width), dtype=np.uint8)
            depth_img = Image.fromarray(depth_placeholder)
            dst_depth_path = output_path / depth_filename
            depth_img.save(dst_depth_path)
            print(f"  ⚠ Created black depth placeholder: {depth_filename}")
            placeholders_created += 1
        
        # Copy RGB file to output directory
        dst_rgb_path = output_path / rgb_filename
        import shutil
        shutil.copy2(rgb_path, dst_rgb_path)
        
        images.append(image_entry)
    
    return images

def cleanup_input_directory(data_dir, split, dry_run=False):
    """
    Clean up the input directory by removing .png and .exr files 
    that are not in the depth/ or white_env_0/ (rgb) folders
    """
    data_path = Path(data_dir) / split
    
    if not data_path.exists():
        print(f"Warning: Input directory {data_path} not found, skipping cleanup")
        return
    
    print(f"\nCleaning up input directory: {data_path}")
    
    # Define folders that should keep their files
    keep_folders = {"depth", "white_env_0"}
    
    # Show current directory structure
    print(f"Current directory structure:")
    
    # Show root-level files first
    root_files = list(data_path.glob("*.png")) + list(data_path.glob("*.exr"))
    if root_files:
        print(f"  📄 Root level files ({len(root_files)}):")
        for file_path in sorted(root_files):
            print(f"    - {file_path.name}")
    
    # Show subdirectories
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            file_count = len(list(item.glob("*.png"))) + len(list(item.glob("*.exr")))
            print(f"  📁 {item.name}/ ({file_count} image files)")
    
    # Get all .png and .exr files in the split directory (root level)
    root_files = list(data_path.glob("*.png")) + list(data_path.glob("*.exr"))
    
    # Get all .png and .exr files in subdirectories (excluding keep_folders)
    subdir_files = []
    for subdir in data_path.iterdir():
        if subdir.is_dir() and subdir.name not in keep_folders:
            subdir_files.extend(list(subdir.glob("*.png")) + list(subdir.glob("*.exr")))
    
    # Combine all files
    all_files = root_files + subdir_files
    
    if not all_files:
        print("No .png or .exr files found to clean up")
        return
    
    # Files to remove (those not in keep folders)
    # This includes:
    # - Root-level files (e.g., rgb_for_depth_86.png, depth_310001.exr)
    # - Files in other subdirectories (e.g., black_env_0/, colored_env_0/)
    files_to_remove = []
    for file_path in all_files:
        # Check if file is in a keep folder
        in_keep_folder = False
        for keep_folder in keep_folders:
            if keep_folder in file_path.parts:
                in_keep_folder = True
                break
        
        if not in_keep_folder:
            files_to_remove.append(file_path)
    
    if not files_to_remove:
        print("No unnecessary files found to remove")
        return
    
    print(f"Found {len(files_to_remove)} unnecessary files to remove:")
    for file_path in sorted(files_to_remove):
        print(f"  🗑️  {file_path.relative_to(data_path)}")
    
    # Show summary
    total_size = sum(f.stat().st_size for f in files_to_remove)
    print(f"\nTotal size to be freed: {total_size / (1024*1024):.2f} MB")
    
    if dry_run:
        print(f"\nDRY RUN MODE: No files were actually deleted")
        print(f"This was just a preview of what would be cleaned up")
        return
    
    # Ask for confirmation before deletion
    response = input(f"\nProceed to delete {len(files_to_remove)} files? (y/N): ")
    if response.lower() != 'y':
        print("Cleanup cancelled by user")
        return
    
    # Remove the files
    removed_count = 0
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"  ✓ Removed: {file_path.relative_to(data_path)}")
            removed_count += 1
        except Exception as e:
            print(f"  ⚠ Warning: Could not remove {file_path.relative_to(data_path)}: {e}")
    
    print(f"\nInput directory cleanup complete: removed {removed_count} files")

def cleanup_unnecessary_files(output_dir):
    """
    Remove all .png and .exr files that are not in the rgb and depth folders
    """
    output_path = Path(output_dir)
    
    # Get list of all files in output directory
    all_files = list(output_path.glob("*.png")) + list(output_path.glob("*.exr"))
    
    # Get list of files that should be kept (from info.json)
    info_file = output_path / "info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            info_data = json.load(f)
        
        # Collect all file names that should be kept
        keep_files = set()
        for image in info_data['images']:
            if 'rgb' in image:
                keep_files.add(image['rgb'])
            if 'alpha' in image:
                keep_files.add(image['alpha'])
            if 'depth' in image:
                keep_files.add(image['depth'])
        
        print(f"Files to keep ({len(keep_files)}):")
        for file_name in sorted(keep_files):
            print(f"  ✓ {file_name}")
        
        print(f"\nFiles to remove ({len(all_files) - len(keep_files)}):")
        removed_count = 0
        for file_path in all_files:
            if file_path.name not in keep_files:
                try:
                    file_path.unlink()
                    print(f"  🗑️  Removed: {file_path.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Could not remove {file_path.name}: {e}")
        
        print(f"\nCleanup complete: removed {removed_count} unnecessary files")
    else:
        print("Warning: info.json not found, skipping cleanup")

def calculate_bbox(images):
    """
    Calculate bounding box from camera poses
    """
    positions = []
    for image in images:
        pose = np.array(image['pose'])
        # Extract camera position (translation part)
        position = pose[:3, 3]
        positions.append(position)
    
    if not positions:
        return [[-1, -1, -1], [1, 1, 1]]  # Default bbox
    
    positions = np.array(positions)
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    
    # Add some padding
    padding = 0.1
    min_coords -= padding
    max_coords += padding
    
    return [min_coords.tolist(), max_coords.tolist()]

def main():
    parser = argparse.ArgumentParser(description='Generate info.json for bus dataset')
    parser.add_argument('--data_dir', type=str, default='datasamples/bus',
                       help='Path to bus dataset directory')
    parser.add_argument('--output_dir', type=str, default='datasamples/bus_processed',
                       help='Output directory for processed data')
    parser.add_argument('--split', type=str, choices=['train', 'test'], required=True,
                       help='Which split to process (train or test)')
    parser.add_argument('--cleanup_input', action='store_true',
                       help='Clean up unnecessary .png and .exr files in the input directory (removes files not in depth/ or white_env_0/ folders, including root-level files)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be cleaned up without actually deleting files (use with --cleanup_input)')
    
    args = parser.parse_args()
    
    print(f"Processing {args.split} split...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Process the dataset
    images = process_bus_dataset(args.data_dir, args.split, args.output_dir)
    
    # Calculate bounding box
    bbox = calculate_bbox(images)
    
    # Create info.json
    info_data = {
        "backend": "CYCLES",
        "light_mode": "uniform",
        "fast_mode": False,
        "format_version": 6,
        "channels": ["R", "G", "B", "A", "D"],
        "scale": 0.5,
        "images": images,
        "bbox": bbox
    }
    
    # Save info.json
    info_path = Path(args.output_dir) / 'info.json'
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2)
    
    print(f"Successfully created {info_path}")
    print(f"Processed {len(images)} images")
    print(f"Bounding box: {bbox}")
    print(f"Output directory: {args.output_dir}")
        
    # Clean up unnecessary files in output directory
    print("\nCleaning up unnecessary files in output directory...")
    cleanup_unnecessary_files(args.output_dir)
    
    # Clean up input directory if requested
    if args.cleanup_input:
        cleanup_input_directory(args.data_dir, args.split, args.dry_run)

if __name__ == "__main__":
    main()
