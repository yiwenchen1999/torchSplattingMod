#!/usr/bin/env python3
"""
Manual debug script for preprocess_rgba function
Usage: python debug_preprocess_rgba.py --image path/to/image.png --size 512
"""

import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
import cv2

def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Keep aspect ratio, resize shortest side to size, then center crop to (size,size)"""
    w, h = img.size
    if w == h:
        return img.resize((size, size), Image.BICUBIC)
    if w < h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_w, new_h = int(round(w * size / h)), size
    res = img.resize((new_w, new_h), Image.BICUBIC)
    left = (res.width - size) // 2
    top = (res.height - size) // 2
    return res.crop((left, top, left + size, top + size))

def preprocess_rgba_opencv(image_path: str, size: int):
    """
    Preprocess RGBA image using OpenCV - inspect alpha channel range
    Returns:
      x: (1,3,H,W) in [-1,1] RGB with transparent areas set to white
      mask_img: PIL 'L' (H,W) binary mask (255 where alpha>0) - for reference only
    """
    print(f"Loading image with OpenCV: {image_path}")
    
    # Load image with OpenCV (BGR format)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(f"OpenCV image shape: {img_bgr.shape}")
    print(f"OpenCV image dtype: {img_bgr.dtype}")
    
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Check if image has alpha channel
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        print("Image has alpha channel (BGRA)")
        b, g, r, a = cv2.split(img_bgr)
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        print("Image has no alpha channel (BGR)")
        b, g, r = cv2.split(img_bgr)
        a = np.ones_like(b) * 255  # Create opaque alpha
    else:
        print("Image is grayscale")
        gray = img_bgr
        b = g = r = gray
        a = np.ones_like(gray) * 255  # Create opaque alpha
    
    print(f"Channel shapes: B={b.shape}, G={g.shape}, R={r.shape}, A={a.shape}")
    print(f"Channel dtypes: B={b.dtype}, G={g.dtype}, R={r.dtype}, A={a.dtype}")
    
    # Inspect alpha channel in detail
    print(f"\nAlpha channel analysis:")
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
    
    # Resize and crop using OpenCV
    h, w = a.shape[:2]
    if h == w:
        resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    else:
        if w < h:
            new_w, new_h = size, int(round(h * size / w))
        else:
            new_w, new_h = int(round(w * size / h)), size
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Center crop
        start_x = (new_w - size) // 2
        start_y = (new_h - size) // 2
        resized = resized[start_y:start_y+size, start_x:start_x+size]
    
    print(f"Resized image shape: {resized.shape}")
    
    # Split resized image
    if len(resized.shape) == 3 and resized.shape[2] == 4:
        b_resized, g_resized, r_resized, a_resized = cv2.split(resized)
    else:
        b_resized, g_resized, r_resized = cv2.split(resized)
        a_resized = np.ones_like(b_resized) * 255
    
    print(f"Resized channel shapes: B={b_resized.shape}, G={g_resized.shape}, R={r_resized.shape}, A={a_resized.shape}")
    print(f"Resized alpha range: [{a_resized.min()}, {a_resized.max()}]")
    
    # Make transparent areas white
    transparent_mask = a_resized == 0
    transparent_count = np.sum(transparent_mask)
    print(f"Transparent pixels: {transparent_count} ({transparent_count/a_resized.size*100:.1f}%)")
    
    # Create RGB with white background
    r_white = r_resized.copy()
    g_white = g_resized.copy()
    b_white = b_resized.copy()
    
    r_white[transparent_mask] = 255
    g_white[transparent_mask] = 255
    b_white[transparent_mask] = 255
    
    # Convert BGR to RGB for PyTorch
    rgb_white = np.stack([r_white, g_white, b_white], axis=2)
    
    # Convert to PIL for tensor conversion
    rgb_pil = Image.fromarray(rgb_white.astype(np.uint8))
    x = TF.to_tensor(rgb_pil) * 2.0 - 1.0  # (3,H,W) [-1,1]
    
    # Create binary mask
    mask_np = (a_resized > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode="L")
    
    print(f"Final tensor shape: {x.shape}, dtype: {x.dtype}")
    print(f"Final tensor range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Final mask white pixels: {np.sum(mask_np == 255)} ({np.sum(mask_np == 255)/mask_np.size*100:.1f}%)")
    
    return x.unsqueeze(0), mask_img  # (1,3,H,W), PIL(L)


def main():
    parser = argparse.ArgumentParser(description="Debug preprocess_rgba function on a specific image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--size", type=int, default=512, help="Output size (default: 512)")
    parser.add_argument("--output", help="Output directory (default: debug_output)")
    args = parser.parse_args()
    
    # Load and process image with OpenCV
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"\nProcessing image with OpenCV, size={args.size}...")
    try:
        x_tensor, mask_img = preprocess_rgba_opencv(str(image_path), args.size)
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save outputs
    output_dir = Path(args.output) if args.output else Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    # Convert tensor back to image for visualization
    x_vis = (x_tensor.squeeze(0).permute(1, 2, 0) + 1) / 2  # Convert from [-1,1] to [0,1]
    x_vis = torch.clamp(x_vis, 0, 1)
    x_vis_np = (x_vis.numpy() * 255).astype(np.uint8)
    x_vis_img = Image.fromarray(x_vis_np)
    
    # Save files
    base_name = image_path.stem
    x_vis_img.save(output_dir / f"{base_name}_processed_rgb.png")
    mask_img.save(output_dir / f"{base_name}_mask.png")
    
    # Save original for comparison (load with PIL for saving)
    try:
        original_pil = Image.open(image_path)
        original_pil.save(output_dir / f"{base_name}_original.png")
    except Exception as e:
        print(f"Warning: Could not save original image: {e}")
    
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  {base_name}_original.png - Original image")
    print(f"  {base_name}_processed_rgb.png - Processed RGB (what gets encoded)")
    print(f"  {base_name}_mask.png - Binary mask")
    
    print(f"\nProcessing completed successfully!")
    print(f"Final tensor shape: {x_tensor.shape}")
    print(f"Final tensor dtype: {x_tensor.dtype}")

if __name__ == "__main__":
    main()
