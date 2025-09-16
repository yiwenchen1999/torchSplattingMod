#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
import clip
import cv2

def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts])

def load_clip_model(model_name: str = "ViT-L/14", device="cuda", dtype=torch.float16):
    """Load CLIP model and return both model and preprocess function"""
    model, preprocess = clip.load(model_name, device=device)
    # Convert model to half precision for consistency
    if dtype == torch.float16:
        model = model.half()
    model.eval()
    return model, preprocess

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
def encode_image_clip(model, preprocess, img_pil: Image.Image, device="cuda", dtype=torch.float16):
    """Encode image using CLIP ViT and return last layer feature maps"""
    # Preprocess image using CLIP's preprocessing
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device, dtype=dtype)  # (1, 3, 224, 224)
    print(f'[DEBUG] Input img_tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}')
    
    # Get features from the visual encoder
    with torch.no_grad():
        # Get the visual encoder directly
        visual_encoder = model.visual
        print(f'[DEBUG] Visual encoder device: {next(visual_encoder.parameters()).device}')
        print(f'[DEBUG] Visual encoder dtype: {next(visual_encoder.parameters()).dtype}')
        
        # Check conv1 layer specifically
        conv1_weight_dtype = visual_encoder.conv1.weight.dtype
        conv1_bias_dtype = visual_encoder.conv1.bias.dtype if visual_encoder.conv1.bias is not None else None
        print(f'[DEBUG] Conv1 weight dtype: {conv1_weight_dtype}')
        print(f'[DEBUG] Conv1 bias dtype: {conv1_bias_dtype}')
        
        # Forward through the visual encoder to get intermediate features
        # All components should now be float16
        print(f'[DEBUG] Starting forward pass through conv1...')
        x = visual_encoder.conv1(img_tensor)  # shape = [*, width, grid, grid]
        print(f'[DEBUG] After conv1 - x shape: {x.shape}, dtype: {x.dtype}')
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        print(f'[DEBUG] After reshape and permute - x shape: {x.shape}, dtype: {x.dtype}')
        
        # Add class token
        class_embedding_dtype = visual_encoder.class_embedding.dtype
        print(f'[DEBUG] Class embedding dtype: {class_embedding_dtype}')
        x = torch.cat([visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        print(f'[DEBUG] After adding class token - x shape: {x.shape}, dtype: {x.dtype}')
        
        # Add positional embedding
        pos_embedding_dtype = visual_encoder.positional_embedding.dtype
        print(f'[DEBUG] Positional embedding dtype: {pos_embedding_dtype}')
        x = x + visual_encoder.positional_embedding.to(x.dtype)
        print(f'[DEBUG] After adding positional embedding - x shape: {x.shape}, dtype: {x.dtype}')
        
        # Layer norm pre
        ln_pre_weight_dtype = visual_encoder.ln_pre.weight.dtype
        ln_pre_bias_dtype = visual_encoder.ln_pre.bias.dtype
        print(f'[DEBUG] LN_pre weight dtype: {ln_pre_weight_dtype}')
        print(f'[DEBUG] LN_pre bias dtype: {ln_pre_bias_dtype}')
        x = visual_encoder.ln_pre(x)
        print(f'[DEBUG] After ln_pre - x shape: {x.shape}, dtype: {x.dtype}')
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        print(f'[DEBUG] After permute to LND - x shape: {x.shape}, dtype: {x.dtype}')
        
        # Transformer layers
        print(f'[DEBUG] Starting transformer forward pass...')
        first_transformer_layer = visual_encoder.transformer.resblocks[0]
        transformer_weight_dtype = first_transformer_layer.attn.in_proj_weight.dtype
        print(f'[DEBUG] Transformer layer weight dtype: {transformer_weight_dtype}')
        x = visual_encoder.transformer(x)
        print(f'[DEBUG] After transformer - x shape: {x.shape}, dtype: {x.dtype}')
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        print(f'[DEBUG] After permute to NLD - x shape: {x.shape}, dtype: {x.dtype}')
        
        # Get the last layer features (before final layer norm and projection)
        last_layer_features = x  # shape = [batch_size, num_patches + 1, hidden_dim]
        print(f'[DEBUG] Last layer features shape: {last_layer_features.shape}, dtype: {last_layer_features.dtype}')
        
        # Remove class token to get patch features only
        patch_features = last_layer_features[:, 1:, :]  # shape = [batch_size, num_patches, hidden_dim]
        print(f'[DEBUG] Patch features shape: {patch_features.shape}, dtype: {patch_features.dtype}')
        
        # Reshape to spatial format
        # For ViT-L/14, the grid size is 16x16 (224/14 = 16)
        grid_size = int(np.sqrt(patch_features.shape[1]))
        print(f'[DEBUG] Grid size: {grid_size}')
        spatial_features = patch_features.reshape(patch_features.shape[0], grid_size, grid_size, patch_features.shape[2])
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [batch, hidden_dim, grid, grid]
        print(f'[DEBUG] Final spatial features shape: {spatial_features.shape}, dtype: {spatial_features.dtype}')
    
    print(f'CLIP feature maps shape: {spatial_features.shape}, dtype: {spatial_features.dtype}')
    return spatial_features  # (1, hidden_dim, grid, grid)

def main():
    parser = argparse.ArgumentParser(description="Encode images to CLIP ViT patch-level feature maps")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output folder for .npy features")
    parser.add_argument("--model", default="ViT-L/14", help="CLIP model name (e.g., ViT-L/14, ViT-B/32)")
    parser.add_argument("--size", type=int, default=224, help="Input image size (CLIP typically uses 224)")
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

    dtype = torch.float16 if args.device.startswith("cuda") or args.fp16 else torch.float16
    model, preprocess = load_clip_model(args.model, device=args.device, dtype=dtype)
    
    print(f"Using CLIP model: {args.model}")
    print(f"Model device: {next(model.parameters()).device}")

    for img_path in tqdm(images, desc="编码CLIP特征中"):
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
        
        # Encode using CLIP
        features = encode_image_clip(model, preprocess, rgb_pil, device=args.device, dtype=dtype).squeeze(0)  # (hidden_dim, grid, grid)
        h, w = features.shape[-2], features.shape[-1]

        # Save features
        feat_cpu = features.to("cpu")
        print(f'CLIP features shape: {feat_cpu.shape}, dtype: {feat_cpu.dtype}')
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
                "feature_type": "clip_vit_last_layer",
                "preprocessing": "opencv_alpha_multiplication",
                "grid_size": h,
                "hidden_dim": features.shape[0],
            }
            with open(out_base.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)

    print(f"Done. Saved CLIP features and *_mask.npy to: {args.output}")

if __name__ == "__main__":
    main()
