#!/usr/bin/env python3
import argparse, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL

def list_latents(folder):
    """List all .npy latent files in folder (excluding _mask files)"""
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() == ".npy" and not f.stem.endswith("_mask")])

def load_flux_vae(flux_repo: str = "black-forest-labs/FLUX.1-dev", device="cuda", dtype=torch.float16):
    """Load FLUX VAE model"""
    vae = AutoencoderKL.from_pretrained(flux_repo, subfolder="vae", torch_dtype=dtype)
    vae.to(device).eval()
    return vae

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL image"""
    t = (t.clamp(-1, 1) + 1) / 2
    t = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = (t * 255).round().astype("uint8")
    return Image.fromarray(t)

@torch.inference_mode()
def decode_latents_flux(vae: AutoencoderKL, latents: torch.Tensor):
    """Decode latents using FLUX VAE"""
    # FLUX VAE decoding: latents / scaling_factor + shift_factor
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(latents).sample  # (1,3, 8*h, 8*w) in [-1,1]

def overlay_mask(img: Image.Image, mask_up_np: np.ndarray, alpha: float = 0.35, color=(0, 255, 0)) -> Image.Image:
    """
    Overlay mask visualization on image
    img: PIL RGB
    mask_up_np: (H,W) uint8 {0,1}
    alpha: overlay opacity on masked pixels
    color: overlay color tuple
    """
    img_np = np.array(img).astype(np.float32)
    H, W = mask_up_np.shape
    overlay = np.zeros_like(img_np)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    m = mask_up_np.astype(np.float32)[..., None]  # (H,W,1)

    out = img_np * (1.0 - alpha * m) + overlay * (alpha * m)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def main():
    parser = argparse.ArgumentParser(description="Decode latent representations to images using FLUX VAE")
    parser.add_argument("--input", required=True, help="Input folder with .npy latents (and *_mask.npy)")
    parser.add_argument("--output", required=True, help="Output folder for decoded images and mask visualization")
    parser.add_argument("--flux", default="black-forest-labs/FLUX.1-dev", help="FLUX model path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Force float16 inference")
    parser.add_argument("--ext", default=".png", choices=[".png", ".jpg", ".webp"], help="Output image format")
    parser.add_argument("--no_mask_viz", action="store_true", help="Disable mask overlay visualization")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    latents_files = list_latents(args.input)
    if not latents_files:
        print(f"No .npy latents found in {args.input}")
        return

    dtype = torch.float16 if args.device.startswith("cuda") or args.fp16 else torch.float32
    vae = load_flux_vae(args.flux, device=args.device, dtype=dtype)
    
    print(f"Using FLUX VAE: {args.flux}")
    print(f"Scaling factor: {vae.config.scaling_factor}")
    print(f"Shift factor: {vae.config.shift_factor}")

    for lat_path in tqdm(latents_files, desc="解码中"):   
        rel = Path(lat_path).relative_to(args.input)
        out_img = (Path(args.output) / rel).with_suffix(args.ext)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        if out_img.exists():
            # 如果请求且缺失，仍然生成可视化
            pass

        # Load latent
        print(f'Processing file: {lat_path}')
        arr = np.load(lat_path, allow_pickle=True)  # (4,h,w)
        lat = torch.from_numpy(arr).to(args.device, dtype=dtype)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)  # (1,4,h,w)

        # Decode
        print(f"Decoding {lat_path} -> {out_img}")
        print(f"Latent shape: {lat.shape}")
        img_tensor = decode_latents_flux(vae, lat)   # (1,3, 8*h, 8*w)
        img = tensor_to_pil(img_tensor)
        img.save(out_img)

        # Try to load mask and visualize
        if not args.no_mask_viz:
            mask_path = lat_path.with_name(lat_path.stem + "_mask").with_suffix(".npy")
            if mask_path.exists():
                mask_lat = np.load(mask_path)  # (h,w) uint8 {0,1}
                h, w = mask_lat.shape
                H, W = img_tensor.shape[-2], img_tensor.shape[-1]  # ≈ 8*h, 8*w
                # nearest upsample to image resolution
                mask_up = torch.from_numpy(mask_lat[None, None].astype(np.float32)).to(img_tensor.device)
                mask_up = torch.nn.functional.interpolate(mask_up, size=(H, W), mode="nearest")
                mask_up_np = (mask_up.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)

                # Save raw mask PNG (+ overlay)
                raw_mask_png = out_img.with_name(out_img.stem + "_mask").with_suffix(".png")
                Image.fromarray((mask_up_np * 255).astype(np.uint8), mode="L").save(raw_mask_png)

                overlay_png = out_img.with_name(out_img.stem + "_maskviz").with_suffix(".png")
                overlay_img = overlay_mask(img, mask_up_np, alpha=0.35, color=(0, 255, 0))
                overlay_img.save(overlay_png)

    print(f"Done. Decoded images (and mask visualizations) saved to: {args.output}")

if __name__ == "__main__":
    main()
