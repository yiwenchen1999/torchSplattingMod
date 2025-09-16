#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
import cv2
from transformers import AutoImageProcessor, AutoModel

def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts])

def load_dino_model(model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", device="cuda", dtype=torch.float32):
    """Load DINOv3 model and processor"""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Keep model in its original precision to avoid dtype conflicts
    model.to(device).eval()
    return model, processor

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
    
    # Check if image has alpha channel
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        b, g, r, a = cv2.split(img_bgr)
        
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        # Create a fully opaque alpha channel
        h, w = img_bgr.shape[:2]
        a = np.ones((h, w), dtype=np.uint8) * 255
        b, g, r = cv2.split(img_bgr)
        
    else:
        # Grayscale image
        h, w = img_bgr.shape[:2]
        a = np.ones((h, w), dtype=np.uint8) * 255
        b = g = r = img_bgr
    
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
    
    # Save alpha channel for reference
    alpha_img = Image.fromarray(a_resized, mode="L")
    
    return rgb_pil, alpha_img

@torch.inference_mode()
def encode_image_dino(model, processor, img_pil: Image.Image, device="cuda", dtype=torch.float32):
    """Encode image using DINOv3 and return feature maps"""
    # Preprocess image using DINO's processor
    inputs = processor(images=img_pil, return_tensors="pt")
    # Convert input to match model's expected dtype
    pixel_values = inputs.pixel_values.to(device, dtype=next(model.parameters()).dtype)  # (1, 3, 224, 224)
    
    # Get features from DINOv3
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(pixel_values, output_hidden_states=True)
        
        # Get the last hidden state (patch features)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, num_patches + 1, hidden_size)
        
        # Remove CLS token to get patch features only
        patch_features = last_hidden_state[:, 1:, :]  # (batch_size, num_patches, hidden_size)
        
        # Reshape to spatial format
        # For DINO v2, the patch size is typically 14x14, so grid size is 16x16 (224/14 = 16)
        grid_size = int(np.sqrt(patch_features.shape[1]))
        spatial_features = patch_features.reshape(patch_features.shape[0], grid_size, grid_size, patch_features.shape[2])
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [batch, hidden_size, grid, grid]
    
    print(f'DINO feature maps shape: {spatial_features.shape}, dtype: {spatial_features.dtype}')
    return spatial_features  # (1, hidden_size, grid, grid)

def main():
    parser = argparse.ArgumentParser(description="Encode images to DINOv3 feature maps")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output folder for .npy features")
    parser.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="DINOv3 model name (e.g., facebook/dinov3-vits16-pretrain-lvd1689m, facebook/dinov3-vitb16-pretrain-lvd1689m, facebook/dinov3-vitl16-pretrain-lvd1689m)")
    parser.add_argument("--size", type=int, default=224, help="Input image size (DINOv3 typically uses 224)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Force float16 inference/storage")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--save_meta", action="store_true", help="Save metadata JSON file")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    images = list_images(args.input)
    # 只处理不在任何子文件夹中的图像
    images = [img for img in images if img.parent == Path(args.input)]
    if not images:
        print(f"No images found in {args.input}")
        return

    dtype = torch.float16 if args.device.startswith("cuda") or args.fp16 else torch.float32
    model, processor = load_dino_model(args.model, device=args.device, dtype=dtype)
    
    print(f"Using DINOv3 model: {args.model}")
    print(f"Model device: {next(model.parameters()).device}")

    for img_path in tqdm(images, desc="编码DINO特征中"):
        rel = Path(img_path).relative_to(args.input)
        out_base = Path(args.output) / rel
        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_features = out_base.with_suffix(".npy")
        out_mask = out_base.with_name(out_base.stem + "_mask").with_suffix(".npy")

        if out_features.exists() and out_mask.exists() and not args.overwrite:
            continue

        # Get original image size for metadata
        img = Image.open(img_path)
        W0, H0 = img.size  # original size

        # Preprocess using OpenCV with alpha multiplication
        rgb_pil, alpha_img = preprocess_rgba_opencv(str(img_path), size=args.size)
        
        # Encode using DINOv3
        features = encode_image_dino(model, processor, rgb_pil, device=args.device, dtype=dtype).squeeze(0)  # (hidden_size, grid, grid)
        h, w = features.shape[-2], features.shape[-1]

        # Save features
        feat_cpu = features.to("cpu")
        print(f'DINO features shape: {feat_cpu.shape}, dtype: {feat_cpu.dtype}')
        np.save(out_features, feat_cpu.half().numpy() if dtype == torch.float16 else feat_cpu.numpy())

        # Downscale alpha channel to feature resolution and save
        alpha_tensor = TF.to_tensor(alpha_img).unsqueeze(0).to(args.device)  # (1,1,H,W) in [0,1]
        alpha_bin = (alpha_tensor > 0).to(torch.float32)                     # strict alpha>0
        # Map full-resolution alpha to feature HxW with OR semantics
        alpha_down = torch.nn.functional.adaptive_max_pool2d(alpha_bin, output_size=(h, w))  # (1,1,h,w)
        alpha_np = (alpha_down.squeeze(0).squeeze(0).detach().cpu().numpy() > 0).astype(np.uint8)
        np.save(out_mask, alpha_np)

        if args.save_meta:
            meta = {
                "source": str(Path(img_path)),
                "orig_size_hw": [H0, W0],
                "size_input": args.size,
                "feature_shape": list(features.shape),
                "mask_shape": [h, w],
                "dtype": "float16" if dtype == torch.float16 else "float32",
                "model": args.model,
                "feature_type": "dinov3_patch_features",
                "preprocessing": "opencv_alpha_multiplication",
                "grid_size": h,
                "hidden_dim": features.shape[0],
            }
            with open(out_base.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)

    print(f"Done. Saved DINOv3 features and *_mask.npy to: {args.output}")

if __name__ == "__main__":
    main()
