import os
import json
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import sys

# Try to import OpenEXR for .exr file reading
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not available. Install with: pip install OpenEXR")

# Try to import cv2 as fallback for .exr files
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available. Install with: pip install opencv-python")

def read_exr_depth(file_path):
    """
    Read depth from .exr file using OpenEXR
    """
    if not HAS_OPENEXR:
        raise ImportError("OpenEXR is required to read .exr files")
    
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    
    # Get the data window
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Read the depth channel (usually 'R' for single channel depth)
    channels = header['channels'].keys()
    if 'R' in channels:
        depth_channel = 'R'
    elif 'Y' in channels:
        depth_channel = 'Y'
    elif 'Z' in channels:
        depth_channel = 'Z'
    else:
        # Use the first available channel
        depth_channel = list(channels)[0]
    
    # Read the depth data
    depth_data = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
    depth_array = np.frombuffer(depth_data, dtype=np.float32)
    depth_array = depth_array.reshape(height, width)
    
    exr_file.close()
    return depth_array

# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)

def read_camera(folder):
    """
    read camera from json file
    """
    scene_info = json.load(open(os.path.join(folder, 'info.json')))
    max_depth = 1
    try:
        max_depth = scene_info['images'][0]['max_depth']
    except:
        pass

    rgb_files = []
    poses = []
    intrinsics = []
    for item in scene_info['images']:
        rgb_files.append(os.path.join(folder, item['rgb']))
        c2w = item['pose']
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        poses.append(np.array(c2w_opencv))
        intrinsics.append(np.array(item['intrinsic']))
    return rgb_files, poses, intrinsics, max_depth

def read_all(folder, resize_factor=1., latent_model=False, image_size=128):
    """
    read source images from a folder
    """
    # scene_src_dir = os.path.join(self.folder_path_src, scene_id)
    print('reading folder', folder)
    src_rgb_files, src_poses, src_intrinsics, max_depth = read_camera(folder)

    src_cameras = []
    src_rgbs = []
    src_alphas = []
    src_depths = []
    src_latents = []
    i = 0
    for src_rgb_file, src_pose, intrinsic in zip(src_rgb_files, src_poses, src_intrinsics):
        src_rgb , src_depth, src_alpha, src_camera = \
        read_image(src_rgb_file, src_pose, 
            intrinsic, max_depth=max_depth, resize_factor=resize_factor)
        file_name = src_rgb_file.split('/')[-1]
        latent_file = os.path.join(folder, f'vae_latents_{image_size}', file_name.replace('png','npy'))
        # latent_file = latent_file.replace('png','npy')
        if latent_model:
            src_latent = torch.from_numpy(np.load(latent_file))
            src_latent = src_latent.permute(1,2,0)
        if i<1000000:
            src_rgbs.append(src_rgb)
            src_depths.append(src_depth)
            src_alphas.append(src_alpha)
            src_cameras.append(src_camera)
            if latent_model:
                src_latents.append(src_latent)
        i += 1
    src_alphas = torch.stack(src_alphas, axis=0)
    src_depths = torch.stack(src_depths, axis=0)
    src_rgbs = torch.stack(src_rgbs, axis=0)
    src_cameras = torch.stack(src_cameras, axis=0)
    src_rgbs = src_alphas[..., None] * src_rgbs + (1-src_alphas)[..., None]
    if latent_model:
        src_latents = torch.stack(src_latents, axis=0)
    print('returning with data', src_rgbs.shape, src_cameras.shape, src_depths.shape, src_alphas.shape)
    if latent_model:
        return {
            "rgb": src_rgbs[..., :3],
            "camera": src_cameras,
            "depth": src_depths,
            "alpha": src_alphas,
            "latent": src_latents,
        }
    else:
        return {
            "rgb": src_rgbs[..., :3],
            "camera": src_cameras,
            "depth": src_depths,
            "alpha": src_alphas,
        }


def read_image(rgb_file, pose, intrinsic_, max_depth, resize_factor=1, white_bkgd=True):
    rgb = torch.from_numpy(imageio.imread(rgb_file).astype(np.float32) / 255.0)
    if "B075X65R3X" in rgb_file:
        depth = torch.from_numpy(imageio.imread(rgb_file[:-7]+'depth.png').astype(np.float32) / 255.0 * max_depth)
        alpha = torch.from_numpy(imageio.imread(rgb_file[:-7]+'alpha.png').astype(np.float32) / 255.0)
    elif "nerf_synthetic" in rgb_file:
        depth = torch.from_numpy(imageio.imread(rgb_file.replace('.png','_depth_0002.png')).astype(np.float32))
        # print(depth[400,400])
        depth = (255 - depth) / 255.0 * 8.0
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        alpha = torch.from_numpy(imageio.imread(rgb_file.replace('.png','_alpha.png')).astype(np.float32))/255.0
    elif "objaverse" in rgb_file:
        # Handle processed datasets with .exr depth files that don't need remapping
        base_name = rgb_file.replace('.png', '')
        depth_file = base_name.replace('gt_', 'depth_') + '.exr'
        
        # Check if depth file exists
        if not os.path.exists(depth_file):
            print(f"Warning: Depth file not found: {depth_file}")
            # Try alternative naming patterns
            alt_depth_file = base_name.replace('gt_', 'depth_') + '.exr'
            if os.path.exists(alt_depth_file):
                depth_file = alt_depth_file
                print(f"Using alternative depth file: {depth_file}")
            else:
                print(f"Could not find depth file for: {rgb_file}")
                # Create a default depth (all zeros)
                depth = torch.zeros((rgb.shape[0], rgb.shape[1]), dtype=torch.float32)
                alpha = torch.ones_like(depth)
                print("Created default depth and alpha")
                # Continue with camera setup
                image_size = rgb.shape[:2]
                intrinsic = np.eye(4,4)
                intrinsic[:3,:3] = intrinsic_
                if resize_factor != 1:
                    image_size = image_size[0] * resize_factor, image_size[1] * resize_factor 
                    intrinsic[:2,:3] *= resize_factor
                camera = torch.from_numpy(np.concatenate(
                    (list(image_size), intrinsic.flatten(), pose.flatten())
                ).astype(np.float32))
                if white_bkgd:
                    rgb = alpha[..., None] * rgb + (1-alpha)[..., None]
                return rgb, depth, alpha, camera
        
        # Read depth from .exr file using proper EXR reader
        depth_array = read_exr_depth(depth_file)
        depth = torch.from_numpy(depth_array.astype(np.float32))
        # Create alpha mask (all pixels are opaque for now)
        alpha = torch.ones_like(depth)
        alpha[depth > 5] = 0
        depth[depth > 5] = 0

    else:
        sys.exit("Unknown dataset")

        # depth = depth / 255.0 * 5.0
        # print(depth.shape)
    
    
    image_size = rgb.shape[:2]
    intrinsic = np.eye(4,4)
    intrinsic[:3,:3] = intrinsic_

    if resize_factor != 1:
        image_size = image_size[0] * resize_factor, image_size[1] * resize_factor 
        intrinsic[:2,:3] *= resize_factor
        resize_fn = lambda img, resize_factor: F.interpolate(
                img.permute(0, 3, 1, 2), scale_factor=resize_factor, mode='bilinear',
            ).permute(0, 2, 3, 1)
        
        rgb = rearrange(resize_fn(rearrange(rgb, 'h w c -> 1 h w c'), resize_factor), '1 h w c -> h w c')
            
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth = rearrange(resize_fn(rearrange(depth, 'h w -> 1 h w 1'), resize_factor), '1 h w 1 -> h w')
        if len(alpha.shape) == 3:
            alpha = alpha[:, :, 0]
        alpha = rearrange(resize_fn(rearrange(alpha, 'h w -> 1 h w 1'), resize_factor), '1 h w 1 -> h w')

    camera = torch.from_numpy(np.concatenate(
        (list(image_size), intrinsic.flatten(), pose.flatten())
    ).astype(np.float32))
    
    if white_bkgd:
        rgb = alpha[..., None] * rgb + (1-alpha)[..., None]

    return rgb, depth, alpha, camera