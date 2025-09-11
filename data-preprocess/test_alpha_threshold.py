#!/usr/bin/env python3
"""
Test script for different alpha thresholding approaches
Compares hard threshold (alpha > 0) vs soft threshold (alpha > 0.5)
"""

import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path

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

def preprocess_rgba_hard_threshold(image: Image.Image, size: int):
    """
    Preprocess RGBA image with hard threshold (alpha > 0)
    Returns:
      x: (1,3,H,W) in [-1,1] RGB only
      mask_img: PIL 'L' (H,W) binary mask (255 where alpha>0)
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    rgba = resize_center_crop(image, size)
    r, g, b, a = rgba.split()
    
    # Just use RGB channels directly, no alpha processing
    rgb = Image.merge("RGB", (r, g, b))
    x = TF.to_tensor(rgb) * 2.0 - 1.0  # (3,H,W) [-1,1]
    
    # Create binary mask with hard threshold
    a_np = np.array(a, dtype=np.uint8)
    mask_np = (a_np > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode="L")
    return x.unsqueeze(0), mask_img  # (1,3,H,W), PIL(L)

def preprocess_rgba_soft_threshold(image: Image.Image, size: int):
    """
    Preprocess RGBA image with soft threshold (alpha > 0.5)
    Returns:
      x: (1,3,H,W) in [-1,1] RGB only
      mask_img: PIL 'L' (H,W) binary mask (255 where alpha>0.5)
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    rgba = resize_center_crop(image, size)
    r, g, b, a = rgba.split()
    
    # Just use RGB channels directly, no alpha processing
    rgb = Image.merge("RGB", (r, g, b))
    x = TF.to_tensor(rgb) * 2.0 - 1.0  # (3,H,W) [-1,1]
    
    # Create binary mask with soft threshold
    a_np = np.array(a, dtype=np.uint8)
    print('a_np', a_np.shape, 'value range', a_np.min(), a_np.max())
    mask_np = (a_np > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode="L")
    return x.unsqueeze(0), mask_img  # (1,3,H,W), PIL(L)

def preprocess_rgba_soft_mask(image: Image.Image, size: int):
    """
    Preprocess RGBA image with soft mask (preserve alpha values)
    Returns:
      x: (1,3,H,W) in [-1,1] RGB only
      mask_img: PIL 'L' (H,W) grayscale mask (preserves alpha values)
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    rgba = resize_center_crop(image, size)
    r, g, b, a = rgba.split()
    
    # Just use RGB channels directly, no alpha processing
    rgb = Image.merge("RGB", (r, g, b))
    x = TF.to_tensor(rgb) * 2.0 - 1.0  # (3,H,W) [-1,1]
    
    # Create soft mask preserving alpha values
    a_np = np.array(a, dtype=np.uint8)
    mask_img = Image.fromarray(a_np, mode="L")  # Keep original alpha values
    return x.unsqueeze(0), mask_img  # (1,3,H,W), PIL(L)

def test_alpha_thresholding():
    """Test different alpha thresholding approaches"""
    
    # Test directory
    test_dir = Path("../../datasamples/nerf_synthetic/ship_latents_processed_test")
    output_dir = Path("../../datasamples/nerf_synthetic/ship_latents_processed_test/threshold_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Find RGB images (without _alpha suffix)
    rgb_images = [f for f in test_dir.glob("r_*.png") if not f.name.endswith("_alpha.png") and not f.name.endswith("_depth_0002.png")]
    if not rgb_images:
        print("No RGB images found!")
        return
    
    print(f"Found {len(rgb_images)} RGB images")
    
    # Test with first few images
    test_images = rgb_images[:3]  # Test first 3 images
    
    for i, rgb_path in enumerate(test_images):
        print(f"\nTesting image {i+1}: {rgb_path.name}")
        
        try:
            # Find corresponding alpha image
            base_name = rgb_path.stem
            alpha_path = test_dir / f"{base_name}_alpha.png"
            
            if not alpha_path.exists():
                print(f"  No alpha image found for {base_name}")
                continue
            
            # Load RGB and alpha images
            rgb_img = Image.open(rgb_path)
            alpha_img = Image.open(alpha_path)
            
            print(f"  RGB size: {rgb_img.size}, mode: {rgb_img.mode}")
            print(f"  Alpha size: {alpha_img.size}, mode: {alpha_img.mode}")
            
            # Combine RGB and alpha into RGBA
            if rgb_img.mode != "RGB":
                rgb_img = rgb_img.convert("RGB")
            if alpha_img.mode != "L":
                alpha_img = alpha_img.convert("L")
            
            # Create RGBA image
            rgba_img = Image.merge("RGBA", (*rgb_img.split(), alpha_img))
            print(f"  Combined RGBA size: {rgba_img.size}, mode: {rgba_img.mode}")
            
            # Test all three approaches
            print("  Testing hard threshold (alpha > 0)...")
            x_tensor_hard, mask_hard = preprocess_rgba_hard_threshold(rgba_img, size=512)
            
            print("  Testing soft threshold (alpha > 0.5)...")
            x_tensor_soft, mask_soft = preprocess_rgba_soft_threshold(rgba_img, size=512)
            
            print("  Testing soft mask (preserve alpha)...")
            x_tensor_soft_mask, mask_soft_mask = preprocess_rgba_soft_mask(rgba_img, size=512)
            
            print(f"  Processed tensor shape: {x_tensor_hard.shape}")
            print(f"  Tensor dtype: {x_tensor_hard.dtype}")
            print(f"  Tensor range: [{x_tensor_hard.min():.3f}, {x_tensor_hard.max():.3f}]")
            
            # Convert tensor back to image for visualization
            def tensor_to_image(tensor):
                vis = (tensor.squeeze(0).permute(1, 2, 0) + 1) / 2  # Convert from [-1,1] to [0,1]
                vis = torch.clamp(vis, 0, 1)
                vis_np = (vis.numpy() * 255).astype(np.uint8)
                return Image.fromarray(vis_np)
            
            x_vis = tensor_to_image(x_tensor_hard)  # RGB is same for all approaches
            
            # Save test outputs
            x_vis.save(output_dir / f"{base_name}_rgb.png")
            mask_hard.save(output_dir / f"{base_name}_mask_hard_threshold.png")
            mask_soft.save(output_dir / f"{base_name}_mask_soft_threshold.png")
            mask_soft_mask.save(output_dir / f"{base_name}_mask_soft_preserve.png")
            
            # Save original images for comparison
            rgb_img.save(output_dir / f"{base_name}_original_rgb.png")
            alpha_img.save(output_dir / f"{base_name}_original_alpha.png")
            
            print(f"  Saved test outputs to {output_dir}")
            
            # Analyze alpha distribution and threshold effects
            alpha_np = np.array(alpha_img)
            print(f"  Alpha value distribution:")
            print(f"    Min: {alpha_np.min()}, Max: {alpha_np.max()}")
            print(f"    Mean: {alpha_np.mean():.1f}, Std: {alpha_np.std():.1f}")
            
            # Detailed alpha histogram analysis
            unique_values, counts = np.unique(alpha_np, return_counts=True)
            print(f"  Alpha histogram:")
            for val, count in zip(unique_values, counts):
                percentage = count / alpha_np.size * 100
                print(f"    Alpha {val:3d}: {count:6d} pixels ({percentage:5.1f}%)")
            
            # Test different threshold values
            print(f"  Threshold analysis:")
            thresholds = [0, 1, 10, 50, 100, 127, 128, 200, 254]
            for thresh in thresholds:
                mask_count = np.sum(alpha_np > thresh)
                percentage = mask_count / alpha_np.size * 100
                print(f"    Threshold > {thresh:3d}: {mask_count:6d} pixels ({percentage:5.1f}%)")
            
            # Find optimal threshold based on data distribution
            # If binary (only 0 and 255), any threshold 0 < t < 255 will give same result
            if len(unique_values) == 2 and 0 in unique_values and 255 in unique_values:
                print(f"  → Binary alpha detected: Only values {unique_values}")
                print(f"  → Any threshold 0 < t < 255 will produce identical results")
                recommended_threshold = 127  # Middle value
            else:
                # For non-binary alpha, find threshold that captures most of the "solid" pixels
                # Look for the threshold that gives us ~90% of the non-zero pixels
                non_zero_pixels = np.sum(alpha_np > 0)
                target_pixels = int(non_zero_pixels * 0.9)
                
                # Find threshold that gives us close to target
                for thresh in sorted(unique_values[unique_values > 0]):
                    mask_count = np.sum(alpha_np > thresh)
                    if mask_count <= target_pixels:
                        recommended_threshold = thresh
                        break
                else:
                    recommended_threshold = 0
                
                print(f"  → Non-binary alpha detected")
                print(f"  → Recommended threshold: {recommended_threshold} (captures ~90% of solid pixels)")
            
            # Count pixels in different alpha ranges
            transparent = np.sum(alpha_np == 0)
            low_alpha = np.sum((alpha_np > 0) & (alpha_np <= 127))
            high_alpha = np.sum(alpha_np > 127)
            opaque = np.sum(alpha_np == 255)
            
            print(f"  Alpha distribution summary:")
            print(f"    Transparent (0): {transparent} pixels ({transparent/alpha_np.size*100:.1f}%)")
            print(f"    Low alpha (1-127): {low_alpha} pixels ({low_alpha/alpha_np.size*100:.1f}%)")
            print(f"    High alpha (128-254): {high_alpha} pixels ({high_alpha/alpha_np.size*100:.1f}%)")
            print(f"    Opaque (255): {opaque} pixels ({opaque/alpha_np.size*100:.1f}%)")
            
            # Compare mask statistics
            mask_hard_np = np.array(mask_hard)
            mask_soft_np = np.array(mask_soft)
            mask_soft_mask_np = np.array(mask_soft_mask)
            
            hard_white = np.sum(mask_hard_np == 255)
            soft_white = np.sum(mask_soft_np == 255)
            soft_mask_mean = mask_soft_mask_np.mean()
            
            print(f"  Mask comparison:")
            print(f"    Hard threshold white pixels: {hard_white} ({hard_white/mask_hard_np.size*100:.1f}%)")
            print(f"    Soft threshold white pixels: {soft_white} ({soft_white/mask_soft_np.size*100:.1f}%)")
            print(f"    Soft mask mean value: {soft_mask_mean:.1f}")
            print(f"    Difference (hard - soft): {hard_white - soft_white} pixels")
            
            # Test the recommended threshold
            if recommended_threshold != 0.5:
                recommended_mask = (alpha_np > recommended_threshold).astype(np.uint8) * 255
                recommended_count = np.sum(recommended_mask == 255)
                print(f"  Recommended threshold ({recommended_threshold}) result: {recommended_count} pixels ({recommended_count/alpha_np.size*100:.1f}%)")
            
        except Exception as e:
            print(f"  Error processing {rgb_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTest completed! Check outputs in: {output_dir}")
    print("Files saved:")
    print("  *_original_rgb.png - Original RGB image")
    print("  *_original_alpha.png - Original alpha channel")
    print("  *_rgb.png - RGB image (same for all approaches)")
    print("  *_mask_hard_threshold.png - Binary mask (alpha > 0)")
    print("  *_mask_soft_threshold.png - Binary mask (alpha > 0.5)")
    print("  *_mask_soft_preserve.png - Grayscale mask (preserves alpha values)")
    print("\nCompare the different mask approaches!")

if __name__ == "__main__":
    test_alpha_thresholding()
