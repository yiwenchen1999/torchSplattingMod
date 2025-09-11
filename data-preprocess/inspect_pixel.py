#!/usr/bin/env python3
"""
Inspect specific pixel in an image to understand transparency issues
"""

import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path

def inspect_pixel(image_path: str, x: int, y: int):
    """
    Inspect a specific pixel in an image
    """
    print(f"Inspecting pixel ({x}, {y}) in image: {image_path}")
    
    # Load with OpenCV
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print(f"Error: Could not load image with OpenCV")
        return
    
    print(f"OpenCV image shape: {img_bgr.shape}")
    print(f"OpenCV image dtype: {img_bgr.dtype}")
    
    # Check bounds
    h, w = img_bgr.shape[:2]
    if x >= w or y >= h or x < 0 or y < 0:
        print(f"Error: Pixel ({x}, {y}) is out of bounds. Image size: {w}x{h}")
        return
    
    # Extract pixel values
    if len(img_bgr.shape) == 3:
        if img_bgr.shape[2] == 4:  # BGRA
            b, g, r, a = img_bgr[y, x]
            print(f"OpenCV BGRA pixel ({x}, {y}): B={b}, G={g}, R={r}, A={a}")
        elif img_bgr.shape[2] == 3:  # BGR
            b, g, r = img_bgr[y, x]
            a = 255  # No alpha channel
            print(f"OpenCV BGR pixel ({x}, {y}): B={b}, G={g}, R={r} (no alpha)")
    else:  # Grayscale
        gray = img_bgr[y, x]
        b = g = r = gray
        a = 255
        print(f"OpenCV Grayscale pixel ({x}, {y}): Gray={gray} (no alpha)")
    
    # Load with PIL for comparison
    try:
        pil_img = Image.open(image_path)
        print(f"PIL image mode: {pil_img.mode}")
        print(f"PIL image size: {pil_img.size}")
        
        if pil_img.mode == "RGBA":
            r_pil, g_pil, b_pil, a_pil = pil_img.getpixel((x, y))
            print(f"PIL RGBA pixel ({x}, {y}): R={r_pil}, G={g_pil}, B={b_pil}, A={a_pil}")
        elif pil_img.mode == "RGB":
            r_pil, g_pil, b_pil = pil_img.getpixel((x, y))
            print(f"PIL RGB pixel ({x}, {y}): R={r_pil}, G={g_pil}, B={b_pil} (no alpha)")
        elif pil_img.mode == "L":
            gray_pil = pil_img.getpixel((x, y))
            print(f"PIL Grayscale pixel ({x}, {y}): Gray={gray_pil} (no alpha)")
        else:
            print(f"PIL pixel ({x}, {y}): {pil_img.getpixel((x, y))} (mode: {pil_img.mode})")
            
    except Exception as e:
        print(f"Error loading with PIL: {e}")
    
    # Analyze the pixel
    print(f"\nPixel Analysis:")
    print(f"  Coordinates: ({x}, {y})")
    
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        # Has alpha channel
        alpha_value = a
        print(f"  Alpha value: {alpha_value}")
        
        if alpha_value == 0:
            print(f"  → Pixel is fully transparent (alpha=0)")
        elif alpha_value == 255:
            print(f"  → Pixel is fully opaque (alpha=255)")
        else:
            print(f"  → Pixel is partially transparent (alpha={alpha_value})")
        
        # Check RGB values
        print(f"  RGB values: R={r}, G={g}, B={b}")
        
        # Check if RGB values might make it appear transparent
        if r == 0 and g == 0 and b == 0:
            print(f"  → RGB is black (0,0,0) - might appear transparent against dark background")
        elif r == 255 and g == 255 and b == 255:
            print(f"  → RGB is white (255,255,255) - might appear transparent against light background")
        else:
            print(f"  → RGB has color values")
        
        # Check if it's a premultiplied alpha issue
        if alpha_value > 0 and alpha_value < 255:
            # Check if RGB values are premultiplied
            expected_r = int(r * 255 / alpha_value) if alpha_value > 0 else 0
            expected_g = int(g * 255 / alpha_value) if alpha_value > 0 else 0
            expected_b = int(b * 255 / alpha_value) if alpha_value > 0 else 0
            
            print(f"  → If premultiplied alpha, expected RGB would be: ({expected_r}, {expected_g}, {expected_b})")
    
    # Check surrounding pixels for context
    print(f"\nSurrounding pixels (3x3 around ({x}, {y})):")
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
                    nb, ng, nr, na = img_bgr[ny, nx]
                    print(f"    ({nx:3d},{ny:3d}): B={nb:3d} G={ng:3d} R={nr:3d} A={na:3d}")
                else:
                    val = img_bgr[ny, nx]
                    print(f"    ({nx:3d},{ny:3d}): {val}")
            else:
                print(f"    ({nx:3d},{ny:3d}): out of bounds")

def extract_alpha_channel(image_path: str, output_path: str = None):
    """
    Extract alpha channel from an image and save as grayscale
    """
    print(f"Extracting alpha channel from: {image_path}")
    
    # Load with OpenCV
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print(f"Error: Could not load image with OpenCV")
        return
    
    print(f"OpenCV image shape: {img_bgr.shape}")
    print(f"OpenCV image dtype: {img_bgr.dtype}")
    
    # Check if image has alpha channel
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        print("Image has alpha channel (BGRA)")
        b, g, r, a = cv2.split(img_bgr)
        
        # Analyze alpha channel
        print(f"Alpha channel analysis:")
        print(f"  Shape: {a.shape}")
        print(f"  Dtype: {a.dtype}")
        print(f"  Value range: [{a.min()}, {a.max()}]")
        print(f"  Mean: {a.mean():.2f}")
        print(f"  Std: {a.std():.2f}")
        
        # Detailed alpha histogram
        unique_values, counts = np.unique(a, return_counts=True)
        print(f"  Unique values: {len(unique_values)}")
        if len(unique_values) <= 20:
            print(f"  Alpha histogram:")
            for val, count in zip(unique_values, counts):
                percentage = count / a.size * 100
                print(f"    Alpha {val:3d}: {count:6d} pixels ({percentage:5.1f}%)")
        else:
            print(f"  Too many unique values ({len(unique_values)}), showing first 10:")
            for val, count in zip(unique_values[:10], counts[:10]):
                percentage = count / a.size * 100
                print(f"    Alpha {val:3d}: {count:6d} pixels ({percentage:5.1f}%)")
            print(f"  ... and {len(unique_values)-10} more values")
        
        # Save alpha channel as grayscale image
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_alpha_extracted.png"
        
        # Save with OpenCV
        cv2.imwrite(str(output_path), a)
        print(f"Alpha channel saved to: {output_path}")
        
        # Also save with PIL for comparison
        pil_output = Path(output_path).parent / f"{Path(output_path).stem}_pil.png"
        a_pil = Image.fromarray(a, mode="L")
        a_pil.save(pil_output)
        print(f"Alpha channel (PIL) saved to: {pil_output}")
        
        return a
        
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        print("Image has no alpha channel (BGR)")
        # Create a fully opaque alpha channel
        h, w = img_bgr.shape[:2]
        a = np.ones((h, w), dtype=np.uint8) * 255
        print("Created fully opaque alpha channel (all values = 255)")
        
        # Save the created alpha channel
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_alpha_created.png"
        
        cv2.imwrite(str(output_path), a)
        print(f"Created alpha channel saved to: {output_path}")
        
        return a
        
    else:
        print("Image is grayscale")
        # Create a fully opaque alpha channel
        h, w = img_bgr.shape[:2]
        a = np.ones((h, w), dtype=np.uint8) * 255
        print("Created fully opaque alpha channel (all values = 255)")
        
        # Save the created alpha channel
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_alpha_created.png"
        
        cv2.imwrite(str(output_path), a)
        print(f"Created alpha channel saved to: {output_path}")
        
        return a

def main():
    parser = argparse.ArgumentParser(description="Inspect pixel or extract alpha channel from an image")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--x", type=int, help="X coordinate (for pixel inspection)")
    parser.add_argument("--y", type=int, help="Y coordinate (for pixel inspection)")
    parser.add_argument("--extract-alpha", action="store_true", help="Extract alpha channel")
    parser.add_argument("--output", help="Output path for alpha channel")
    args = parser.parse_args()
    
    if args.extract_alpha:
        extract_alpha_channel(args.image, args.output)
    elif args.x is not None and args.y is not None:
        inspect_pixel(args.image, args.x, args.y)
    else:
        print("Please specify either --extract-alpha or both --x and --y coordinates")
        print("Use --help for more information")

if __name__ == "__main__":
    main()
