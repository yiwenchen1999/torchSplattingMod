#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL

def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts])

def load_flux_vae(flux_repo: str = "black-forest-labs/FLUX.1-dev", device="cuda", dtype=torch.float16):
    """Load FLUX VAE model"""
    vae = AutoencoderKL.from_pretrained(flux_repo, subfolder="vae", torch_dtype=dtype)
    vae.to(device).eval()
    return vae

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

def preprocess_rgba(image: Image.Image, size: int):
    """
    Preprocess RGBA image
    Returns:
      x: (1,3,H,W) in [-1,1]
      mask_img: PIL 'L' (H,W) binary mask (255 where alpha>0)
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    rgba = resize_center_crop(image, size)
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    x = TF.to_tensor(rgb) * 2.0 - 1.0  # (3,H,W) [-1,1]
    # mask: 1 where alpha>0
    a_np = np.array(a, dtype=np.uint8)
    mask_np = (a_np > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode="L")
    return x.unsqueeze(0), mask_img  # (1,3,H,W), PIL(L)

@torch.inference_mode()
def encode_image_flux(vae: AutoencoderKL, img_tensor: torch.Tensor, sample: bool = False):
    """Encode image using FLUX VAE"""
    # FLUX VAE encoding
    posterior = vae.encode(img_tensor).latent_dist
    latents = posterior.sample() if sample else posterior.mean
    
    # FLUX VAE scaling: (latents - shift_factor) * scaling_factor
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    
    return latents  # (1,4,H/8,W/8)

def main():
    parser = argparse.ArgumentParser(description="Encode images to latent representations using FLUX VAE")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output folder for .npy latents")
    parser.add_argument("--flux", default="black-forest-labs/FLUX.1-dev", help="FLUX model path")
    parser.add_argument("--size", type=int, default=1024, help="Square size after resize+crop")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Force float16 inference/storage")
    parser.add_argument("--sample", action="store_true", help="Sample from latent distribution")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--save_meta", action="store_true", help="Save metadata JSON file")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    images = list_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    dtype = torch.float16 if args.device.startswith("cuda") or args.fp16 else torch.float32
    vae = load_flux_vae(args.flux, device=args.device, dtype=dtype)
    
    print(f"Using FLUX VAE: {args.flux}")
    print(f"Scaling factor: {vae.config.scaling_factor}")
    print(f"Shift factor: {vae.config.shift_factor}")

    for img_path in tqdm(images, desc="编码中"):
        rel = Path(img_path).relative_to(args.input)
        out_base = Path(args.output) / rel
        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_latent = out_base.with_suffix(".npy")
        out_mask = out_base.with_name(out_base.stem + "_mask").with_suffix(".npy")

        if out_latent.exists() and out_mask.exists() and not args.overwrite:
            continue

        img = Image.open(img_path)
        W0, H0 = img.size  # original size

        # Preprocess to square, keep alpha mask
        x, mask_img = preprocess_rgba(img, size=args.size)  # x:(1,3,H,W), mask PIL(L)
        x = x.to(args.device, dtype=dtype)

        # Encode
        latents = encode_image_flux(vae, x, sample=args.sample).squeeze(0)  # (4,h,w) on device
        h, w = latents.shape[-2], latents.shape[-1]

        # Save latent
        lat_cpu = latents.to("cpu")
        print(f'Latent shape: {lat_cpu.shape}, dtype: {lat_cpu.dtype}')
        np.save(out_latent, lat_cpu.half().numpy() if dtype == torch.float16 else lat_cpu.numpy())

        # Downscale mask to latent resolution and save
        mask_tensor = TF.to_tensor(mask_img).unsqueeze(0).to(args.device)  # (1,1,H,W) in [0,1]
        mask_bin = (mask_tensor > 0).to(torch.float32)                     # strict alpha>0
        # Map full-resolution mask to latent HxW with OR semantics
        mask_down = torch.nn.functional.adaptive_max_pool2d(mask_bin, output_size=(h, w))  # (1,1,h,w)
        mask_np = (mask_down.squeeze(0).squeeze(0).detach().cpu().numpy() > 0).astype(np.uint8)
        np.save(out_mask, mask_np)

        if args.save_meta:
            meta = {
                "source": str(Path(img_path)),
                "orig_size_hw": [H0, W0],
                "size_input": args.size,
                "latent_shape": [4, h, w],
                "mask_shape": [h, w],
                "dtype": "float16" if dtype == torch.float16 else "float32",
                "scale_factor": vae.config.scaling_factor,
                "shift_factor": vae.config.shift_factor,
                "vae_type": "flux",
                "mode": "sample" if args.sample else "mean",
                "vae": args.flux,
                "mask_note": "mask is 1 where input alpha>0, resized with nearest to latent resolution",
            }
            with open(out_base.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)

    print(f"Done. Saved latents and *_mask.npy to: {args.output}")

if __name__ == "__main__":
    main()
