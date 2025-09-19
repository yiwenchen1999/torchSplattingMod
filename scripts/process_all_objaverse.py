#!/usr/bin/env python3
"""
Automated script to process all Objaverse data in /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import cv2
import shutil
from tqdm import tqdm
import glob

# Add the data-preprocess directory to the path so we can import the preprocessing functions
sys.path.append(str(Path(__file__).parent.parent / "data-preprocess"))
from preprocess_objaverse_data import blend_rgba_image, calculate_intrinsic_matrix

def check_alpha_quality(alpha_mask, threshold=0.1):
    """
    Check if alpha mask is mostly dark (low quality)
    Returns True if alpha mask is mostly dark
    """
    mean_alpha = np.mean(alpha_mask)
    return mean_alpha < threshold

def process_env_folder(env_path, output_path):
    """
    Process all images in an env folder and apply alpha blending
    """
    print(f"    Processing {env_path.name}...")
    
    # Get all image files
    image_files = list(env_path.glob("*.png")) + list(env_path.glob("*.jpg")) + list(env_path.glob("*.jpeg"))
    if not image_files:
        print(f"    No images found in {env_path}")
        return
    
    processed_count = 0
    for img_path in image_files:
        try:
            # Skip if it's already an alpha file
            if "alpha" in img_path.name.lower():
                continue
                
            # Process with alpha blending
            success, message, alpha_mask, blended_rgb = blend_rgba_image(str(img_path))
            
            if success:
                # Overwrite the original image with blended version
                blended_rgb_pil = Image.fromarray(blended_rgb)
                blended_rgb_pil.save(img_path)
                processed_count += 1
            else:
                print(f"    Warning: Failed to process {img_path.name}: {message}")
                
        except Exception as e:
            print(f"    Error processing {img_path.name}: {e}")
    
    print(f"    Processed {processed_count} images in {env_path.name}")
    
    # Create blended.txt flag
    blended_flag = env_path / "blended.txt"
    blended_flag.touch()

def check_quality_and_mark_broken(object_path):
    """
    Check the first 3 images for dark alpha masks and mark as broken if needed
    Returns True if marked as broken
    """
    # Look for images in white_env_0
    white_env_path = object_path / "train" / "white_env_0"
    if not white_env_path.exists():
        return False
    
    # Get first 3 images
    image_files = sorted(list(white_env_path.glob("gt_*.png")))[:3]
    if len(image_files) < 3:
        return False
    
    dark_count = 0
    for img_path in image_files:
        try:
            success, message, alpha_mask, _ = blend_rgba_image(str(img_path))
            if success and check_alpha_quality(alpha_mask):
                dark_count += 1
        except Exception as e:
            print(f"    Error checking quality for {img_path.name}: {e}")
    
    # If 2 or more of the first 3 images have dark alpha masks, mark as broken
    if dark_count >= 2:
        broken_flag = object_path / "broken.txt"
        broken_flag.touch()
        print(f"    Marked as broken: {dark_count}/3 images have dark alpha masks")
        return True
    
    return False

def process_object_folder(object_path):
    """
    Process a single object folder
    """
    print(f"\nProcessing object: {object_path.name}")
    
    # Check if already processed
    if (object_path / "broken.txt").exists():
        print("  Skipping: marked as broken")
        return False
    
    # Check if done.txt exists
    if not (object_path / "done.txt").exists():
        print("  Skipping: no done.txt flag")
        return False
    
    # Check if already has GSTrain and GSTest with correct counts
    gstrain_path = object_path / "GSTrain"
    gstest_path = object_path / "GSTest"
    
    if gstrain_path.exists() and gstest_path.exists():
        # Check if they have the expected number of images
        train_images = len(list(gstrain_path.glob("gt_*.png")))
        test_images = len(list(gstest_path.glob("gt_*.png")))
        
        if train_images >= 200 and test_images >= 100:
            print(f"  Skipping: already processed (train: {train_images}, test: {test_images})")
            return False
    
    # Check quality and mark as broken if needed
    if check_quality_and_mark_broken(object_path):
        return False
    
    # Process all env folders first
    train_path = object_path / "train"
    test_path = object_path / "test"
    
    if train_path.exists():
        print("  Processing env folders in train...")
        env_folders = sorted([f for f in train_path.iterdir() if f.is_dir() and f.name.startswith("env_")])
        for env_folder in env_folders:
            if not (env_folder / "blended.txt").exists():
                process_env_folder(env_folder, train_path)
            else:
                print(f"    Skipping {env_folder.name}: already blended")
    
    if test_path.exists():
        print("  Processing env folders in test...")
        env_folders = sorted([f for f in test_path.iterdir() if f.is_dir() and f.name.startswith("env_")])
        for env_folder in env_folders:
            if not (env_folder / "blended.txt").exists():
                process_env_folder(env_folder, test_path)
            else:
                print(f"    Skipping {env_folder.name}: already blended")
    
    # Process train split
    if train_path.exists():
        print("  Processing train split...")
        gstrain_output = object_path / "GSTrain"
        gstrain_output.mkdir(exist_ok=True)
        
        # Run preprocessing for train
        try:
            from preprocess_objaverse_data import process_bus_dataset, calculate_bbox, cleanup_unnecessary_files
            
            images = process_bus_dataset(str(object_path), "train", str(gstrain_output))
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
            
            info_path = gstrain_output / 'info.json'
            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            print(f"  ✓ Created GSTrain with {len(images)} images")
            cleanup_unnecessary_files(str(gstrain_output))
            
        except Exception as e:
            print(f"  Error processing train split: {e}")
    
    # Process test split
    if test_path.exists():
        print("  Processing test split...")
        gstest_output = object_path / "GSTest"
        gstest_output.mkdir(exist_ok=True)
        
        # Run preprocessing for test
        try:
            from preprocess_objaverse_data import process_bus_dataset, calculate_bbox, cleanup_unnecessary_files
            
            images = process_bus_dataset(str(object_path), "test", str(gstest_output))
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
            
            info_path = gstest_output / 'info.json'
            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            print(f"  ✓ Created GSTest with {len(images)} images")
            cleanup_unnecessary_files(str(gstest_output))
            
        except Exception as e:
            print(f"  Error processing test split: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process all Objaverse data')
    parser.add_argument('--data_root', type=str, 
                       default='/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense',
                       help='Root directory containing all object folders')
    parser.add_argument('--max_objects', type=int, default=None,
                       help='Maximum number of objects to process (for testing)')
    parser.add_argument('--start_from', type=str, default=None,
                       help='Start processing from this object folder name')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be processed without actually processing')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        return
    
    # Get all object folders
    object_folders = [f for f in data_root.iterdir() if f.is_dir()]
    object_folders.sort()
    
    print(f"Found {len(object_folders)} object folders")
    
    # Filter by start_from if specified
    if args.start_from:
        start_idx = None
        for i, folder in enumerate(object_folders):
            if folder.name == args.start_from:
                start_idx = i
                break
        if start_idx is not None:
            object_folders = object_folders[start_idx:]
            print(f"Starting from: {args.start_from}")
        else:
            print(f"Warning: Start folder {args.start_from} not found")
    
    # Limit by max_objects if specified
    if args.max_objects:
        object_folders = object_folders[:args.max_objects]
        print(f"Processing first {args.max_objects} objects")
    
    if args.dry_run:
        print("\nDRY RUN - Objects that would be processed:")
        for folder in object_folders:
            if (folder / "done.txt").exists() and not (folder / "broken.txt").exists():
                gstrain_path = folder / "GSTrain"
                gstest_path = folder / "GSTest"
                if not (gstrain_path.exists() and gstest_path.exists()):
                    print(f"  {folder.name}")
        return
    
    # Process each object folder
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for object_folder in tqdm(object_folders, desc="Processing objects"):
        try:
            success = process_object_folder(object_folder)
            if success:
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"\nError processing {object_folder.name}: {e}")
            error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"✓ Successfully processed: {processed_count} objects")
    print(f"⏭️  Skipped: {skipped_count} objects") 
    print(f"❌ Errors: {error_count} objects")

if __name__ == "__main__":
    main()
