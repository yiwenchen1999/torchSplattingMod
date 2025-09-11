#!/usr/bin/env python3
"""
Script to process RGBA images and save blended RGB images with alpha multiplication
Replaces RGBA images with RGB images where RGB values are multiplied by alpha values
Automatically skips images with "depth" or "alpha" in their filenames
"""

import argparse
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
    """List all image files in folder, excluding depth and alpha images"""
    p = Path(folder)
    all_images = [f for f in p.rglob("*") if f.suffix.lower() in exts]
    
    # Filter out images with "depth" or "alpha" in their names
    filtered_images = []
    for img in all_images:
        img_name_lower = img.name.lower()
        if "depth" not in img_name_lower and "alpha" not in img_name_lower:
            filtered_images.append(img)
    
    return sorted(filtered_images)

def blend_rgba_image(image_path: str, output_path: str = None):
    """
    Process RGBA image and create blended RGB image with alpha multiplication
    Args:
        image_path: Path to input RGBA image
        output_path: Path to save blended RGB image (if None, overwrites original)
    """
    # Load with OpenCV
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Check if image has alpha channel
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        # Image has alpha channel (BGRA)
        b, g, r, a = cv2.split(img_bgr)
        
        # Convert to float32 for processing
        r_float = r.astype(np.float32) / 255.0
        g_float = g.astype(np.float32) / 255.0
        b_float = b.astype(np.float32) / 255.0
        a_float = a.astype(np.float32) / 255.0
        
        # Multiply RGB with alpha values
        r_blended = (r_float * a_float * 255).astype(np.uint8)
        g_blended = (g_float * a_float * 255).astype(np.uint8)
        b_blended = (b_float * a_float * 255).astype(np.uint8)
        
        # Create blended RGB image (cv2.split gives B,G,R,A, so we need to reorder to RGB)
        rgb_blended = np.stack([r_blended, g_blended, b_blended], axis=2)
        
        # Save blended RGB image
        if output_path is None:
            output_path = image_path
        
        # Save with PIL to ensure proper RGB format
        rgb_pil = Image.fromarray(rgb_blended)
        rgb_pil.save(output_path)
        
        return True, "RGBA -> RGB blended"
        
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        # Image has no alpha channel (BGR)
        # Convert BGR to RGB and save as is
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if output_path is None:
            output_path = image_path
        
        rgb_pil = Image.fromarray(rgb_image)
        rgb_pil.save(output_path)
        
        return True, "BGR -> RGB (no alpha)"
        
    else:
        # Grayscale image
        if output_path is None:
            output_path = image_path
        
        # Convert grayscale to RGB
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        rgb_pil = Image.fromarray(rgb_image)
        rgb_pil.save(output_path)
        
        return True, "Grayscale -> RGB"

def process_folder(input_folder: str, output_folder: str = None, overwrite: bool = False):
    """
    Process all images in a folder
    Args:
        input_folder: Input folder containing images
        output_folder: Output folder (if None, overwrites original files)
        overwrite: Whether to overwrite existing files
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        return
    
    # Set output folder
    if output_folder is None:
        output_path = input_path
        print(f"Processing images in place: {input_folder}")
    else:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Processing images from {input_folder} to {output_folder}")
    
    # Get all image files
    images = list_images(input_folder)
    if not images:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(images)} images to process (excluding depth and alpha images)")
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(images, desc="Processing images"):
        try:
            # Determine output path
            if output_folder is None:
                # Overwrite original
                out_path = img_path
            else:
                # Save to output folder maintaining structure
                rel_path = img_path.relative_to(input_path)
                out_path = output_path / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if output file exists
            if out_path.exists() and not overwrite:
                print(f"Skipping {img_path.name} (already exists)")
                continue
            
            # Process the image
            success, message = blend_rgba_image(str(img_path), str(out_path))
            
            if success:
                success_count += 1
                if success_count <= 5:  # Show first 5 for verification
                    print(f"✅ {img_path.name}: {message}")
            else:
                error_count += 1
                print(f"❌ Failed to process {img_path.name}")
                
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {img_path.name}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"✅ Successfully processed: {success_count} images")
    print(f"❌ Errors: {error_count} images")
    
    if output_folder is None:
        print(f"All images have been processed in place in: {input_folder}")
    else:
        print(f"Processed images saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Process RGBA images and create blended RGB images")
    parser.add_argument("--input", required=True, help="Input folder containing images")
    parser.add_argument("--output", help="Output folder (if not specified, overwrites original files)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--test", action="store_true", help="Test mode - process only first 5 images")
    args = parser.parse_args()
    
    if args.test:
        # Test mode - process only first 5 images
        input_path = Path(args.input)
        images = list_images(args.input)
        if images:
            test_images = images[:5]
            print(f"Test mode: Processing first {len(test_images)} images")
            
            for img_path in test_images:
                try:
                    success, message = blend_rgba_image(str(img_path))
                    print(f"✅ {img_path.name}: {message}")
                except Exception as e:
                    print(f"❌ Error processing {img_path.name}: {e}")
        else:
            print("No images found for testing")
    else:
        # Full processing
        process_folder(args.input, args.output, args.overwrite)

if __name__ == "__main__":
    main()
