#!/usr/bin/env python3
"""
Test script for RGBA processing function with alpha premultiplication
Tests the preprocess_rgba function without requiring FLUX VAE
"""

import os
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
    Preprocess RGBA image using OpenCV, multiply RGB with alpha values
    Returns:
      x: (1,3,H,W) in [-1,1] with RGB multiplied by alpha
      alpha_img: PIL 'L' (H,W) alpha channel for reference
    """
    # Load with OpenCV
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
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
        
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        print("Image has no alpha channel (BGR)")
        # Create a fully opaque alpha channel
        h, w = img_bgr.shape[:2]
        a = np.ones((h, w), dtype=np.uint8) * 255
        b, g, r = cv2.split(img_bgr)
        print("Created fully opaque alpha channel (all values = 255)")
        
    else:
        print("Image is grayscale")
        # Create a fully opaque alpha channel
        h, w = img_bgr.shape[:2]
        a = np.ones((h, w), dtype=np.uint8) * 255
        b = g = r = img_bgr
        print("Created fully opaque alpha channel (all values = 255)")
    
    # Resize and center crop using OpenCV
    h, w = img_bgr.shape[:2]
    if w == h:
        # Already square, just resize
        r_resized = cv2.resize(r, (size, size), interpolation=cv2.INTER_CUBIC)
        g_resized = cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)
        b_resized = cv2.resize(b, (size, size), interpolation=cv2.INTER_CUBIC)
        a_resized = cv2.resize(a, (size, size), interpolation=cv2.INTER_CUBIC)
    else:
        # Resize maintaining aspect ratio
        if w < h:
            new_w, new_h = size, int(round(h * size / w))
        else:
            new_w, new_h = int(round(w * size / h)), size
        
        r_resized = cv2.resize(r, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        g_resized = cv2.resize(g, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        b_resized = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        a_resized = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Center crop
        start_x = (new_w - size) // 2
        start_y = (new_h - size) // 2
        r_resized = r_resized[start_y:start_y+size, start_x:start_x+size]
        g_resized = g_resized[start_y:start_y+size, start_x:start_x+size]
        b_resized = b_resized[start_y:start_y+size, start_x:start_x+size]
        a_resized = a_resized[start_y:start_y+size, start_x:start_x+size]
    
    # Convert to float32 for processing
    r_float = r_resized.astype(np.float32) / 255.0
    g_float = g_resized.astype(np.float32) / 255.0
    b_float = b_resized.astype(np.float32) / 255.0
    a_float = a_resized.astype(np.float32) / 255.0
    
    # Multiply RGB with alpha values
    r_multiplied = (r_float * a_float * 255).astype(np.uint8)
    g_multiplied = (g_float * a_float * 255).astype(np.uint8)
    b_multiplied = (b_float * a_float * 255).astype(np.uint8)
    
    # Create RGB image with alpha multiplication
    rgb_multiplied = np.stack([r_multiplied, g_multiplied, b_multiplied], axis=2)
    rgb_pil = Image.fromarray(rgb_multiplied.astype(np.uint8))
    
    # Convert to tensor
    x = TF.to_tensor(rgb_pil) * 2.0 - 1.0  # (3,H,W) [-1,1]
    
    # Save alpha channel for reference
    alpha_img = Image.fromarray(a_resized, mode="L")
    
    return x.unsqueeze(0), alpha_img  # (1,3,H,W), PIL(L)

def preprocess_rgba_rgb_only(image: Image.Image, size: int):
    """
    Preprocess RGBA image - use RGB only, discard alpha channel
    Returns:
      x: (1,3,H,W) in [-1,1] RGB only
      mask_img: PIL 'L' (H,W) binary mask (255 where alpha>0) - for reference only
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    rgba = resize_center_crop(image, size)
    r, g, b, a = rgba.split()
    
    # Just use RGB channels directly, no alpha processing
    rgb = Image.merge("RGB", (r, g, b))
    x = TF.to_tensor(rgb) * 2.0 - 1.0  # (3,H,W) [-1,1]
    
    # Create binary mask for reference (but won't be used for encoding)
    a_np = np.array(a, dtype=np.uint8)
    mask_np = (a_np > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode="L")
    return x.unsqueeze(0), mask_img  # (1,3,H,W), PIL(L)

def test_rgba_processing():
    """Test the RGBA processing function with OpenCV-based preprocessing"""
    
    # Test directory
    test_dir = Path("../../datasamples/nerf_synthetic/ship_latents_processed_test")
    output_dir = Path("../../datasamples/nerf_synthetic/ship_latents_processed_test/rgba_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Find RGBA images (single files with alpha channel)
    rgba_images = [f for f in test_dir.glob("r_*.png") if not f.name.endswith("_alpha.png") and not f.name.endswith("_depth_0002.png")]
    if not rgba_images:
        print("No RGBA images found!")
        return
    
    print(f"Found {len(rgba_images)} RGBA images")
    
    # Test with first few images
    test_images = rgba_images[:3]  # Test first 3 images
    
    for i, rgba_path in enumerate(test_images):
        print(f"\nTesting image {i+1}: {rgba_path.name}")
        
        try:
            # Test OpenCV-based preprocessing
            print("  Testing OpenCV-based alpha multiplication approach...")
            x_tensor_opencv, alpha_img = preprocess_rgba_opencv(str(rgba_path), size=512)
            
            print(f"  Processed tensor shape: {x_tensor_opencv.shape}")
            print(f"  Tensor dtype: {x_tensor_opencv.dtype}")
            print(f"  Tensor range: [{x_tensor_opencv.min():.3f}, {x_tensor_opencv.max():.3f}]")
            print(f"  Alpha channel size: {alpha_img.size}")
            
            # Convert tensor back to image for visualization
            def tensor_to_image(tensor):
                vis = (tensor.squeeze(0).permute(1, 2, 0) + 1) / 2  # Convert from [-1,1] to [0,1]
                vis = torch.clamp(vis, 0, 1)
                vis_np = (vis.numpy() * 255).astype(np.uint8)
                return Image.fromarray(vis_np)
            
            x_vis_opencv = tensor_to_image(x_tensor_opencv)
            
            # Save test outputs
            base_name = rgba_path.stem
            x_vis_opencv.save(output_dir / f"{base_name}_opencv_alpha_mult.png")
            alpha_img.save(output_dir / f"{base_name}_alpha_channel.png")
            
            # Load original image for comparison
            original_img = Image.open(rgba_path)
            if original_img.mode == "RGBA":
                # Save original RGB and alpha separately
                rgb_original = original_img.convert("RGB")
                alpha_original = original_img.split()[-1]  # Get alpha channel
                rgb_original.save(output_dir / f"{base_name}_original_rgb.png")
                alpha_original.save(output_dir / f"{base_name}_original_alpha.png")
            else:
                original_img.save(output_dir / f"{base_name}_original.png")
            
            print(f"  Saved test outputs to {output_dir}")
            
            # Analyze the results
            if original_img.mode == "RGBA":
                alpha_np = np.array(alpha_original)
                rgb_original_np = np.array(rgb_original.convert("L"))
                rgb_opencv_np = np.array(x_vis_opencv.convert("L"))
                
                # Count transparent pixels
                transparent_pixels = np.sum(alpha_np == 0)
                partial_transparent_pixels = np.sum((alpha_np > 0) & (alpha_np < 255))
                
                print(f"  Transparent pixels (alpha=0): {transparent_pixels}")
                print(f"  Partially transparent pixels (0<alpha<255): {partial_transparent_pixels}")
                
                # Compare brightness in non-transparent areas
                non_transparent_mask = alpha_np > 0
                if np.any(non_transparent_mask):
                    original_brightness = np.mean(rgb_original_np[non_transparent_mask])
                    opencv_brightness = np.mean(rgb_opencv_np[non_transparent_mask])
                    
                    print(f"  Original RGB brightness (non-transparent): {original_brightness:.1f}")
                    print(f"  OpenCV alpha-mult brightness (non-transparent): {opencv_brightness:.1f}")
                    print(f"  Brightness reduction: {((original_brightness - opencv_brightness) / original_brightness * 100):.1f}%")
                
                # Check edge smoothness (variance in edge regions)
                edge_region = (alpha_np > 0) & (alpha_np < 255)  # Partial transparency
                if np.any(edge_region):
                    original_edge_var = np.var(rgb_original_np[edge_region])
                    opencv_edge_var = np.var(rgb_opencv_np[edge_region])
                    
                    print(f"  Edge region variance - original: {original_edge_var:.1f}")
                    print(f"  Edge region variance - opencv alpha-mult: {opencv_edge_var:.1f}")
                    print(f"  Lower variance = smoother edges")
            
        except Exception as e:
            print(f"  Error processing {rgba_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTest completed! Check outputs in: {output_dir}")
    print("Files saved:")
    print("  *_original_rgb.png - Original RGB image")
    print("  *_original_alpha.png - Original alpha channel")
    print("  *_opencv_alpha_mult.png - RGB with alpha multiplication (OpenCV)")
    print("  *_alpha_channel.png - Alpha channel for reference")
    print("\nThe OpenCV approach multiplies RGB values with alpha values for smooth transparency!")

if __name__ == "__main__":
    test_rgba_processing()
